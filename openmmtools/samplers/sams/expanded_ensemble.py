"""
Expanded ensemble samplers.

"""

__author__ = 'John D. Chodera'

################################################################################
# IMPORTS
################################################################################

from simtk import openmm, unit
from simtk.openmm import app
import os, os.path
import sys, math
import numpy as np
import copy
import time
from scipy.misc import logsumexp

from openmmtools import testsystems
from openmmtools.constants import kB

################################################################################
# LOGGER
################################################################################

import logging
logger = logging.getLogger(__name__)

################################################################################
# EXPANDED ENSEMBLE SAMPLER
################################################################################

class ExpandedEnsemble(object):
    """
    Sampler for the method of expanded ensembles.

    A Gibbs sampling framework is used to alternate between updates of the SamplersState
    (using the provided MCMCSampler) and the current ThermodynamicState.

    Properties
    ----------
    sampler : MCMCSampler
        The MCMC sampler used for updating positions
    thermodynamic_states : list of ThermodynamicState
        All thermodynamic states that can be sampled
    thermodynamic_state_index : int
        Current thermodynamic state index
    iteration : int
        Iterations completed
    naccepted : int
        Number of accepted thermodynamic/chemical state changes
    nrejected : int
        Number of rejected thermodynamic/chemical state changes
    number_of_state_visits : np.array of shape [nstates]
        Cumulative counts of visited states
    verbose : bool
        If True, verbose output is printed

    References
    ----------
    [1] Lyubartsev AP, Martsinovski AA, Shevkunov SV, and Vorontsov-Velyaminov PN. New approach to Monte Carlo calculation of the free energy: Method of expanded ensembles. JCP 96:1776, 1992
    http://dx.doi.org/10.1063/1.462133
    [2] Chodera JD and Shirts MR. Replica exchange and expanded ensemble simulations as Gibbs sampling: Simple improvements for enhanced mixing. JCP 135:194110, 2011.
    http://dx.doi.org/10.1063/1.3660669

    Examples
    --------

    Run a simulated tempering simulation of alanine dipeptide in vacuum:

    >>> # Create an alanine dipeptide test system
    >>> from testsystem import AlanineDipeptideVacuum
    >>> testsystem = AlanineDipeptideVacuum()
    >>> # Specify a list of temperatures
    >>> from numpy import geomspace
    >>> from simtk.unit import kelvin
    >>> temperatures = geomspace(300, 400, 10) * kelvin
    >>> # Construct a set of thermodynamic states differing in temperature
    >>> from openmmtools.states import ThermodynamicState
    >>> thermodynamic_states = [ ThermodynamicState(testsystem.system, temperature=temperature) for temperature in temperatures ]
    >>> # Define the initial sampler state in terms of positions only
    >>> from openmmtools.states import SamplerState
    >>> sampler_state = SamplerState(testsystem.positions)
    >>> # Create an MCMCSampler to propagate the SamplerState
    >>> from openmmtools.mcmc import MCMCSampler, GCMCMove
    >>> ghmc_move = GHMCMove(timestep=1.0*unit.femtosecond, n_steps=500)
    >>> mcmc_sampler = MCMCSampler(thermodynamic_state, sampler_state, move=ghmc_move)
    >>> # Create an expanded ensembles sampler with initial zero log weights guess
    >>> from openmmtools.samplers import ExpandedEnsemble
    >>> exen_sampler = ExpandedEnsemble(mcmc_sampler, thermodynamic_states)
    >>> # Run the sampler
    >>> exen_sampler.run(niterations=10)
    >>> # Analyze the data
    >>> [Delta_f_ij, dDelta_f_ij] = exen_sampler.compute_free_energies()

    """
    def __init__(self, sampler, thermodynamic_states, log_weights=None, update_scheme='global-jump', locality=1):
        """
        Create an expanded ensemble sampler.

        p(x,k) \propto \exp[-u_k(x) + g_k]

        where g_k is the log weight.

        Parameters
        ----------
        sampler : MCMCSampler
            MCMCSampler initialized with current SamplerState
        state : hashable object
            Current chemical state
        log_weights : dict of object : float
            Log weights to use for expanded ensemble biases
        update_scheme : str, optional, default='global-jump'
            Thermodynamic state update scheme, one of ['global-jump', 'local-jump', 'restricted-range']
        locality : int, optional, default=1
            Number of neighboring states on either side to consider for local update schemes

        """
        self.supported_update_schemes = ['local-jump', 'global-jump', 'restricted-range']
        if update_scheme not in self.supported_update_schemes:
            raise Exception("Update scheme '%s' not in list of supported update schemes: %s" % (update_scheme, str(self.supported_update_schemes)))

        self.sampler = sampler
        self.thermodynamic_states = thermodynamic_states
        self.nstates = len(self.thermodynamic_states)
        self.log_weights = log_weights
        self.update_scheme = update_scheme
        self.locality = locality

        # Determine which thermodynamic state is currently active
        self.thermodynamic_state_index = thermodynamic_states.index(sampler.thermodynamic_state)

        if self.log_weights is None:
            self.log_weights = np.zeros([self.nstates], np.float64)

        # Initialize
        self.iteration = 0
        self.naccepted = 0
        self.nrejected = 0
        self.number_of_state_visits = np.zeros([self.nstates], np.float64)
        self.verbose = False

        self._timing = self.sampler._timing
        self._initializeNetCDF(self.sampler.ncfile)

    def _initializeNetCDF(self, ncfile):
        self.ncfile = ncfile
        if self.ncfile == None:
            return

        nstates = self.nstates
        if self.update_scheme == 'global-jump':
            self.locality = self.nstates

        self.ncfile.createDimension('states', nstates)

        self.ncfile.createVariable('state_index', 'i4', dimensions=('iterations',), chunksizes=(1,))
        self.ncfile.createVariable('log_weights', 'f4', dimensions=('iterations', 'states'), chunksizes=(1,nstates))
        self.ncfile.createVariable('log_P_k', 'f4', dimensions=('iterations', 'states'), chunksizes=(1,nstates))
        self.ncfile.createVariable('u_k', 'f4', dimensions=('iterations', 'states'), chunksizes=(1,nstates))
        self.ncfile.createVariable('update_state_time', 'f4', dimensions=('iterations',), chunksizes=(1,))
        self.ncfile.createVariable('neighborhood', 'i1', dimensions=('iterations','states'), chunksizes=(1,nstates))

    def compute_free_energies(self, uncertainty_method=None):
        """
        Compute relative free energies between thermodynamic states and their associated uncertainties.

        Parameters
        ----------
        uncertainty_method : str, optional, default=None
            Specify uncertainty method to use for MBAR

        Returns
        -------
        Delta_f_ij : np.array of shape (nstates, nstates) of type np.float64
            Delta_f_ij[i,j] is the free energy difference f[j] - f[i] in units of kT
        dDelta_f_ij : np.array of shape (nstates, nstates) of type np.float64
            dDelta_f_ij[i,j] is an estimate of the statistical error in Delta_f_ij

        """
        # TODO
        raise NotImplementedError()

    def update_positions(self):
        """
        Sample new positions.
        """
        self.sampler.update()

    def update_state(self):
        """
        Sample the thermodynamic state.
        """
        initial_time = time.time()

        current_state_index = self.thermodynamic_state_index
        self.u_k = np.zeros([self.nstates], np.float64)
        self.log_P_k = np.zeros([self.nstates], np.float64)
        if self.update_scheme == 'local-jump':
            # Determine current neighborhood.
            neighborhood = range(max(0, current_state_index - self.locality), min(self.nstates, current_state_index + self.locality + 1))
            neighborhood_size = len(neighborhood)
            # Propose a move from the current neighborhood.
            proposed_state_index = np.random.choice(neighborhood, p=np.ones([len(neighborhood)], np.float64) / float(neighborhood_size))
            # Determine neighborhood for proposed state.
            proposed_neighborhood = range(max(0, proposed_state_index - self.locality), min(self.nstates, proposed_state_index + self.locality + 1))
            proposed_neighborhood_size = len(proposed_neighborhood)
            # Compute state log weights.
            log_Gamma_j_L = - float(proposed_neighborhood_size) # log probability of proposing return
            log_Gamma_L_j = - float(neighborhood_size)          # log probability of proposing new state
            L = current_state_index
            self.neighborhood = neighborhood
            # Compute potential for all states in neighborhood
            for j in self.neighborhood:
                self.u_k[j] = self.thermodynamic_states[j].reduced_potential(self.sampler.context)
            # Compute log of probability of selecting each state in neighborhood
            for j in self.neighborhood:
                if j != L:
                    self.log_P_k[j] = log_Gamma_L_j + min(0.0, log_Gamma_j_L - log_Gamma_L_j + (self.log_weights[j] - self.u_k[j]) - (self.log_weights[L] - self.u_k[L]))
            P_k = np.zeros([self.nstates], np.float64)
            P_k[self.neighborhood] = np.exp(self.log_P_k[self.neighborhood])
            # Compute probability to return to current state L
            P_k[L] = 0.0
            P_k[L] = 1.0 - P_k[self.neighborhood].sum()
            print('P_k = ', P_k) # DEBUG
            self.log_P_k[L] = np.log(P_k[L])
            # Update context.
            self.thermodynamic_state_index = np.random.choice(self.neighborhood, p=P_k[neighborhood])
        elif self.update_scheme == 'global-jump':
            #
            # Global jump scheme.
            # This method is described after Eq. 3 in [1]
            #

            # Compute unnormalized log probabilities for all thermodynamic states
            self.neighborhood = range(self.nstates)
            for state_index in self.neighborhood:
                self.u_k[state_index] = self.thermodynamic_states[state_index].reduced_potential(self.sampler.context)
                self.log_P_k[state_index] = self.log_weights[state_index] - self.u_k[state_index]
            self.log_P_k -= logsumexp(self.log_P_k)
            # Update sampler Context to current thermodynamic state.
            P_k = np.exp(self.log_P_k[self.neighborhood])
            self.thermodynamic_state_index = np.random.choice(self.neighborhood, p=P_k)
        elif self.update_scheme == 'restricted-range':
            # Propose new state from current neighborhood.
            self.neighborhood = range(max(0, current_state_index - self.locality), min(self.nstates, current_state_index + self.locality + 1))
            for j in self.neighborhood:
                self.u_k[j] = self.thermodynamic_states[j].reduced_potential(self.sampler.context)
                self.log_P_k[j] = self.log_weights[j] - self.u_k[j]
            self.log_P_k[self.neighborhood] -= logsumexp(self.log_P_k[self.neighborhood])
            P_k = np.exp(self.log_P_k[self.neighborhood])
            proposed_state_index = np.random.choice(self.neighborhood, p=P_k)
            # Determine neighborhood of proposed state.
            proposed_neighborhood = range(max(0, proposed_state_index - self.locality), min(self.nstates, proposed_state_index + self.locality + 1))
            for j in proposed_neighborhood:
                if j not in self.neighborhood:
                    self.u_k[j] = self.thermodynamic_states[j].reduced_potential(self.sampler.context)
            # Accept or reject.
            log_P_accept = logsumexp(self.log_weights[self.neighborhood] - self.u_k[self.neighborhood]) - logsumexp(self.log_weights[proposed_neighborhood] - self.u_k[proposed_neighborhood])
            if (log_P_accept >= 0.0) or (np.random.rand() < np.exp(log_P_accept)):
                self.thermodynamic_state_index = proposed_state_index
        else:
            raise Exception("Update scheme '%s' not implemented." % self.update_scheme)

        # Update context.
        self.thermodynamic_states[self.thermodynamic_state_index].update_context(self.sampler.context, integrator=self.sampler.integrator)

        # Track statistics.
        if (self.thermodynamic_state_index != current_state_index):
            self.naccepted += 1
        else:
            self.nrejected += 1

        if self.verbose:
            print('Current thermodynamic state index is %d' % self.thermodynamic_state_index)
            Neff = (P_k / P_k.max()).sum()
            print('Effective number of states with probability: %10.5f' % Neff)

        # Update timing
        final_time = time.time()
        elapsed_time = final_time - initial_time
        self._timing['update state time'] = elapsed_time
        print('elapsed time %8.3f s' % elapsed_time)

        # Update statistics.
        self.update_statistics()

    def update(self):
        """
        Update the sampler with one step of sampling.
        """
        if self.verbose:
            print("-" * 80)
            print("Expanded Ensemble sampler iteration %8d" % self.iteration)

        self.update_positions()
        self.update_state()

        if self.ncfile:
            self.ncfile.variables['state_index'][self.iteration] = self.thermodynamic_state_index
            self.ncfile.variables['log_weights'][self.iteration,:] = self.log_weights[:]
            self.ncfile.variables['log_P_k'][self.iteration,:] = 0
            self.ncfile.variables['log_P_k'][self.iteration,:] = self.log_P_k[:]
            self.ncfile.variables['u_k'][self.iteration,:] = 0
            self.ncfile.variables['u_k'][self.iteration,:] = self.u_k[:]
            self.ncfile.variables['neighborhood'][self.iteration,:] = 0
            self.ncfile.variables['neighborhood'][self.iteration,self.neighborhood] = 1
            self.ncfile.variables['update_state_time'][self.iteration] = self._timing['update state time']

        self.iteration += 1

        if self.verbose:
            print("-" * 80)

    def run(self, niterations=1):
        """
        Run the sampler for the specified number of iterations

        Parameters
        ----------
        niterations : int, optional, default=1
            Number of iterations to run the sampler for.
        """
        for iteration in range(niterations):
            self.update()

    def update_statistics(self):
        """
        Update sampler statistics.
        """
        self.number_of_state_visits[self.thermodynamic_state_index] += 1.0
