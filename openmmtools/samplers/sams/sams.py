"""
Self-adjusted mixture sample (SAMS) samplers.

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

from openmmtools.constants import kB

from . import ExpandedEnsemble

################################################################################
# LOGGER
################################################################################

import logging
logger = logging.getLogger(__name__)

################################################################################
# SAMS SAMPLER
################################################################################

class SAMS(object):
    """
    Self-adjusted mixture sampling.

    Properties
    ----------
    state_keys : set of objects
        The names of states sampled by the sampler.
    logZ : numpy array of shape [nstates]
        logZ[index] is the log partition function (up to an additive constant) estimate for thermodynamic state 'index'
    update_method : str
        Update method.  One of ['default']
    iteration : int
        Iterations completed.
    verbose : bool
        If True, verbose debug output is printed.

    References
    ----------
    [1] Tan, Z. Optimally adjusted mixture sampling and locally weighted histogram analysis, Journal of Computational and Graphical Statistics 26:54, 2017.
    http://dx.doi.org/10.1080/10618600.2015.1113975

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
    >>> # Create a SAMS sampler
    >>> from openmmtools.samplers import SAMS
    >>> sams_sampler = SAMS(exen_sampler)
    >>> # Run the simulation
    >>> sams_sampler.run(niterations=10)
    >>> # Estimate relative free energies between the thermodynamic states
    >>> [Delta_f_ij, dDelta_f_ij] = sams_sampler.compute_free_energies()

    """
    def __init__(self, sampler, logZ=None, log_target_probabilities=None, update_stages='two-stage', update_method='rao-blackwellized', adapt_target_probabilities=False,
        logZ_initialization_method=False, mbar_update_interval=None):
        """
        Create a SAMS Sampler.

        Parameters
        ----------
        sampler : ExpandedEnsemble
            The expanded ensemble sampler used to sample both configurations and discrete thermodynamic states.
        logZ : dict of key : float, optional, default=None
            If specified, the log partition functions for each state will be initialized to the specified dictionary.
        log_target_probabilities : dict of key : float, optional, default=None
            If specified, unnormalized target probabilities; default is all 0.
        update_stages : str, optional, default='two-stage'
            Number of stages to use for update. One of ['one-stage', 'two-stage']
        update_method : str, optional, default='optimal'
            SAMS update algorithm. One of ['optimal', 'rao-blackwellized']
        adapt_target_probabilities : bool, optional, default=False
            If True, target probabilities will be adapted to achieve minimal thermodynamic length between terminal thermodynamic states.
        logZ_initialization_method : bool, optional, default=False
            Method to use to guess logZ at start. [False, 'energies', 'nonequilibrium']
        mbar_update_interval : int, optional, default=False
            If set to a positive integer, MBAR will be used to update the estimates with the specified interval until histograms are flat.

        """
        # Check input arguments.
        self.supported_update_methods = ['optimal', 'rao-blackwellized']
        if update_method not in self.supported_update_methods:
            raise Exception("Update method '%s' not in supported update schemes: %s" % (update_method, str(self.supported_update_methods)))

        # Keep copies of initializing arguments.
        self.sampler = sampler
        self.logZ = logZ
        self.log_target_probabilities = log_target_probabilities
        self.update_stages = update_stages
        self.update_method = update_method

        if self.logZ is None:
            self.logZ = np.zeros([self.sampler.nstates], np.float64)
        if self.log_target_probabilities is None:
            self.log_target_probabilities = np.zeros([self.sampler.nstates], np.float64)
        self.log_target_probabilities -= logsumexp(self.log_target_probabilities)
        self.update_log_weights()

        # Initialize.
        self.iteration = 0
        self.verbose = False

        if adapt_target_probabilities:
            raise Exception('Not implemented yet.')

        if logZ_initialization_method is not False:
            self.guess_logZ(logZ_initialization_method)

        self.mbar_update_interval = mbar_update_interval

        #self._timing = self.sampler._timing # TODO: Handle timing
        if hasattr(self.sampler, 'ncfile'):
            self._initializeNetCDF(self.sampler.ncfile)

    def _initializeNetCDF(self, ncfile):
        self.ncfile = ncfile
        if self.ncfile == None:
            return

        nstates = self.sampler.nstates
        self.ncfile.createVariable('logZ', 'f4', dimensions=('iterations', 'states'), zlib=True, chunksizes=(1,nstates))
        self.ncfile.createVariable('log_target_probabilities', 'f4', dimensions=('iterations', 'states'), chunksizes=(1,nstates))
        self.ncfile.createVariable('update_logZ_time', 'f4', dimensions=('iterations',), chunksizes=(1,))

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

    def guess_logZ(self, method):
        # Initialize sampler
        self.sampler.update()

        # Guess logZ
        if method == 'energies':
            # Compute guess of all energies.
            for state_index in range(self.sampler.nstates):
                self.logZ[state_index] = - self.sampler.thermodynamic_states[state_index].reduced_potential(self.sampler.sampler.context)
            # Restore thermodynamic state.
            self.sampler.thermodynamic_states[self.sampler.thermodynamic_state_index].update_context(self.sampler.sampler.context, self.sampler.integrator)
        elif method == 'nonequilibrium':
            for state_index in range(self.sampler.nstates):
                print('Driving through state %d / %d' % (state_index, self.sampler.nstates))
                # Force thermodynamic state
                self.sampler.thermodynamic_state_index = state_index
                self.sampler.thermodynamic_states[self.sampler.thermodynamic_state_index].update_context(self.sampler.sampler.context, self.sampler.sampler.integrator)
                # Sample a bit
                self.sampler.sampler.update()
                # Compute an energy
                self.logZ[state_index] = - self.sampler.thermodynamic_states[state_index].reduced_potential(self.sampler.sampler.context)
        else:
            raise Exception("logZ initialization method '%s' unknown" % method)

        self.logZ[:] -= self.logZ[0]
        print('initialized logZ:')
        print(self.logZ)

    def update_sampler(self):
        """
        Update the underlying expanded ensembles sampler.
        """
        self.sampler.update()

    def update_logZ_estimates(self):
        """
        Update the logZ estimates according to selected SAMS update method.

        References
        ----------
        [1] http://www.stat.rutgers.edu/home/ztan/Publication/SAMS_redo4.pdf

        """
        initial_time = time.time()

        current_state = self.sampler.thermodynamic_state_index
        log_pi_k = self.log_target_probabilities
        pi_k = np.exp(self.log_target_probabilities)
        log_P_k = self.sampler.log_P_k # log probabilities of selecting states in neighborhood during update

        gamma0 = 1.0
        if self.update_stages == 'one-stage':
            gamma = gamma0 / float(self.iteration+1) # prefactor in Eq. 9 and 12 from [1]
            if self.ncfile: self.ncfile.variables['stage'][self.iteration] = 1
        elif self.update_stages == 'two-stage':
            if hasattr(self, 'second_stage_iteration_start'):
                # Use second stage scheme
                if self.ncfile: self.ncfile.variables['stage'][self.iteration] = 2
                # We flattened at iteration t0. Use this to compute gamma
                t0 = self.second_stage_iteration_start
                gamma = 1.0 / float(self.iteration - t0 + 1./gamma0)
            else:
                # Use first stage scheme.
                if self.ncfile: self.ncfile.variables['stage'][self.iteration] = 1
                #beta_factor = 0.6
                beta_factor = 0.4
                t = self.iteration + 1.0
                gamma = min(pi_k[current_state], t**(-beta_factor)) # Eq. 15
                #gamma = t**(-beta_factor) # Modified version of Eq. 15

                # Check if all state histograms are "flat" within 10% so we can enter the second stage
                RELATIVE_HISTOGRAM_ERROR_THRESHOLD = 0.10
                N_k = self.sampler.number_of_state_visits[:]
                print('N_k:') # DEBUG
                print(N_k) # DEBUG
                empirical_pi_k = N_k[:] / N_k.sum()
                pi_k = np.exp(self.log_target_probabilities)
                relative_error_k = np.abs(pi_k - empirical_pi_k) / pi_k
                if np.all(relative_error_k < RELATIVE_HISTOGRAM_ERROR_THRESHOLD):
                    self.second_stage_iteration_start = self.iteration
                    # Record start of second stage
                    setattr(self.ncfile, 'second_stage_start', self.second_stage_iteration_start)
        else:
            raise Exception("update_stages method '%s' unknown" % self.update_stages)

        # Record gamma for this iteration
        print('gamma = %f' % gamma)
        if self.ncfile: self.ncfile.variables['gamma'][self.iteration] = gamma

        if self.update_method == 'optimal':
            # Based on Eq. 9 of Ref. [1]
            self.logZ[current_state] += gamma * np.exp(-log_pi_k[current_state])
        elif self.update_method == 'rao-blackwellized':
            # Based on Eq. 12 of Ref [1]
            neighborhood = self.sampler.neighborhood # indices of states for expanded ensemble update
            self.logZ[neighborhood] += gamma * np.exp(log_P_k[neighborhood] - log_pi_k[neighborhood])
        else:
            raise Exception("SAMS update method '%s' unknown." % self.update_method)

        # Subtract off logZ[0] to prevent logZ from growing without bound
        self.logZ[:] -= self.logZ[0]

        # TODO: Handle timing`
        #final_time = time.time()
        #elapsed_time = final_time - initial_time
        #self._timing['update logZ time'] = elapsed_time
        #if self.verbose:
        #    print('time elapsed %8.3f s' % elapsed_time)

    def update_logZ_with_mbar(self):
        """
        Use MBAR to update logZ estimates.
        """
        if not self.ncfile:
            raise Exception("Cannot update logZ using MBAR since no NetCDF file is storing history.")

        if not self.sampler.update_scheme == 'global-jump':
            raise Exception("Only global jump is implemented right now.")

        if not self.ncfile:
            raise Exception("Must have a storage file attached to use MBAR updates")

        # Extract relative energies.
        if self.verbose:
            print('Updating logZ estimate with MBAR...')
        initial_time = time.time()
        from pymbar import MBAR
        #first = int(self.iteration / 2)
        first = 0
        u_kn = np.array(self.ncfile.variables['u_k'][first:,:]).T
        [N_k, bins] = np.histogram(self.ncfile.variables['state_index'][first:], bins=(np.arange(self.sampler.nstates+1) - 0.5))
        mbar = MBAR(u_kn, N_k)
        Deltaf_ij, dDeltaf_ij, Theta_ij = mbar.getFreeEnergyDifferences(compute_uncertainty=True, uncertainty_method='approximate')
        self.logZ[:] = -mbar.f_k[:]
        self.logZ -= self.logZ[0]

        # TODO: Handle timing
        #final_time = time.time()
        #elapsed_time = final_time - initial_time
        #self._timing['MBAR time'] = elapsed_time
        #if self.verbose:
        #    print('MBAR time    %8.3f s' % elapsed_time)

    def update_log_weights(self):
        """
        Update log weights for expanded ensemble sampler.
        """
        # Update log weights for expanded ensemble sampler sampler
        self.sampler.log_weights[:] = self.log_target_probabilities[:] - self.logZ[:]

    def update(self):
        """
        Update the sampler with one step of sampling.
        """
        initial_time = time.time()

        if self.verbose:
            print('')
            print("=" * 80)
            print("SAMS sampler iteration %5d" % self.iteration)

        # Update positions and state index with expanded ensemble sampler.
        self.update_sampler()

        # Update logZ estimates and expanded ensemble sampler log weights.
        self.update_logZ_estimates()
        if self.mbar_update_interval and (not hasattr(self, 'second_stage_iteration_start')) and (((self.iteration+1) % self.mbar_update_interval) == 0):
            self.update_logZ_with_mbar()

        self.update_log_weights()

        if self.ncfile:
            self.ncfile.variables['logZ'][self.iteration,:] = self.logZ[:]
            self.ncfile.variables['log_target_probabilities'][self.iteration,:] = self.log_target_probabilities[:]
            #self.ncfile.variables['update_logZ_time'][self.iteration] = self._timing['update logZ time'] # TODO: Handle timing
            self.ncfile.sync()

        # TODO: Handle timing
        #final_time = time.time()
        #elapsed_time = final_time - initial_time
        #self._timing['sams time'] = elapsed_time
        #if self.verbose:
        #    print('total time   %8.3f s' % elapsed_time)

        self.iteration += 1
        if self.verbose:
            print("=" * 80)

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
