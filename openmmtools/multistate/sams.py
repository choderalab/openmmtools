#!/usr/local/bin/env python

# ==============================================================================
# MODULE DOCSTRING
# ==============================================================================

"""
SamsSampler
===========

Self-adjusted mixture sampling (SAMS), also known as optimally-adjusted mixture sampling.

This implementation uses stochastic approximation to allow one or more replicas to sample the whole range of thermodynamic states
for rapid online computation of free energies.

COPYRIGHT

Written by John D. Chodera <john.chodera@choderalab.org> while at Memorial Sloan Kettering Cancer Center.

LICENSE

This code is licensed under the latest available version of the MIT License.

"""

import logging
import numpy as np
import openmmtools as mmtools
from scipy.special import logsumexp

from openmmtools import multistate, utils
from openmmtools.multistate.multistateanalyzer import MultiStateSamplerAnalyzer
import mpiplus


logger = logging.getLogger(__name__)

# ==============================================================================
# PARALLEL TEMPERING
# ==============================================================================


class SAMSSampler(multistate.MultiStateSampler):
    """Self-adjusted mixture sampling (SAMS), also known as optimally-adjusted mixture sampling.

    This class provides a facility for self-adjusted mixture sampling simulations.
    One or more replicas use the method of expanded ensembles [1] to sample multiple thermodynamic states within each replica,
    with log weights for each thermodynamic state adapted on the fly [2] to achieve the desired target probabilities for each state.

    Attributes
    ----------
    log_target_probabilities : array-like
        log_target_probabilities[state_index] is the log target probability for state ``state_index``
    state_update_scheme : str
        Thermodynamic state sampling scheme. One of ['global-jump', 'local-jump', 'restricted-range']
    locality : int
        Number of neighboring states on either side to consider for local update schemes
    update_stages : str
        Number of stages to use for update. One of ['one-stage', 'two-stage']
    weight_update_method : str
        Method to use for updating log weights in SAMS. One of ['optimal', 'rao-blackwellized']
    adapt_target_probabilities : bool
        If True, target probabilities will be adapted to achieve minimal thermodynamic length between terminal thermodynamic states.
    gamma0 : float, optional, default=0.0
        Initial weight adaptation rate.
    logZ_guess : array-like of shape [n_states] of floats, optional, default=None
        Initial guess for logZ for all states, if available.

    References
    ----------
    [1] Lyubartsev AP, Martsinovski AA, Shevkunov SV, and Vorontsov-Velyaminov PN. New approach to Monte Carlo calculation of the free energy: Method of expanded ensembles. JCP 96:1776, 1992
    http://dx.doi.org/10.1063/1.462133

    [2] Tan, Z. Optimally adjusted mixture sampling and locally weighted histogram analysis, Journal of Computational and Graphical Statistics 26:54, 2017.
    http://dx.doi.org/10.1080/10618600.2015.1113975

    Examples
    --------
    SAMS simulation of alanine dipeptide in implicit solvent at different temperatures.

    Create the system:

    >>> import math
    >>> from simtk import unit
    >>> from openmmtools import testsystems, states, mcmc
    >>> testsystem = testsystems.AlanineDipeptideVacuum()
    >>> import os
    >>> import tempfile

    Create thermodynamic states for parallel tempering with exponentially-spaced schedule:

    >>> n_replicas = 3  # Number of temperature replicas.
    >>> T_min = 298.0 * unit.kelvin  # Minimum temperature.
    >>> T_max = 600.0 * unit.kelvin  # Maximum temperature.
    >>> temperatures = [T_min + (T_max - T_min) * (math.exp(float(i) / float(n_replicas-1)) - 1.0) / (math.e - 1.0)
    ...                 for i in range(n_replicas)]
    >>> thermodynamic_states = [states.ThermodynamicState(system=testsystem.system, temperature=T)
    ...                         for T in temperatures]

    Initialize simulation object with options. Run with a GHMC integrator:

    >>> move = mcmc.GHMCMove(timestep=2.0*unit.femtoseconds, n_steps=50)
    >>> simulation = SAMSSampler(mcmc_moves=move, number_of_iterations=2,
    ...                          state_update_scheme='global-jump', locality=5,
    ...                          update_stages='two-stage', flatness_criteria='logZ-flatness',
    ...                          flatness_threshold=0.2, weight_update_method='rao-blackwellized',
    ...                          adapt_target_probabilities=False)


    Create a single-replica SAMS simulation bound to a storage file and run:

    >>> storage_path = tempfile.NamedTemporaryFile(delete=False).name + '.nc'
    >>> reporter = multistate.MultiStateReporter(storage_path, checkpoint_interval=1)
    >>> simulation.create(thermodynamic_states=thermodynamic_states,
    ...                   sampler_states=[states.SamplerState(testsystem.positions)],
    ...                   storage=reporter)
    Please cite the following:
    <BLANKLINE>
            Friedrichs MS, Eastman P, Vaidyanathan V, Houston M, LeGrand S, Beberg AL, Ensign DL, Bruns CM, and Pande VS. Accelerating molecular dynamic simulations on graphics processing unit. J. Comput. Chem. 30:864, 2009. DOI: 10.1002/jcc.21209
            Eastman P and Pande VS. OpenMM: A hardware-independent framework for molecular simulations. Comput. Sci. Eng. 12:34, 2010. DOI: 10.1109/MCSE.2010.27
            Eastman P and Pande VS. Efficient nonbonded interactions for molecular dynamics on a graphics processing unit. J. Comput. Chem. 31:1268, 2010. DOI: 10.1002/jcc.21413
            Eastman P and Pande VS. Constant constraint matrix approximation: A robust, parallelizable constraint method for molecular simulations. J. Chem. Theor. Comput. 6:434, 2010. DOI: 10.1021/ct900463w
    >>> simulation.run()  # This runs for a maximum of 2 iterations.
    >>> simulation.iteration
    2
    >>> simulation.run(n_iterations=1)
    >>> simulation.iteration
    2

    To resume a simulation from an existing storage file and extend it beyond
    the original number of iterations.

    >>> del simulation
    >>> simulation = SAMSSampler.from_storage(reporter)
    Please cite the following:
    <BLANKLINE>
            Friedrichs MS, Eastman P, Vaidyanathan V, Houston M, LeGrand S, Beberg AL, Ensign DL, Bruns CM, and Pande VS. Accelerating molecular dynamic simulations on graphics processing unit. J. Comput. Chem. 30:864, 2009. DOI: 10.1002/jcc.21209
            Eastman P and Pande VS. OpenMM: A hardware-independent framework for molecular simulations. Comput. Sci. Eng. 12:34, 2010. DOI: 10.1109/MCSE.2010.27
            Eastman P and Pande VS. Efficient nonbonded interactions for molecular dynamics on a graphics processing unit. J. Comput. Chem. 31:1268, 2010. DOI: 10.1002/jcc.21413
            Eastman P and Pande VS. Constant constraint matrix approximation: A robust, parallelizable constraint method for molecular simulations. J. Chem. Theor. Comput. 6:434, 2010. DOI: 10.1021/ct900463w
    >>> simulation.extend(n_iterations=1)
    >>> simulation.iteration
    3

    You can extract several information from the NetCDF file using the Reporter
    class while the simulation is running. This reads the SamplerStates of every
    run iteration.

    >>> reporter = multistate.MultiStateReporter(storage=storage_path, open_mode='r', checkpoint_interval=1)
    >>> sampler_states = reporter.read_sampler_states(iteration=3)
    >>> len(sampler_states)
    1
    >>> sampler_states[-1].positions.shape  # Alanine dipeptide has 22 atoms.
    (22, 3)

    Clean up.

    >>> os.remove(storage_path)

    See Also
    --------
    ReplicaExchangeSampler

    """

    _TITLE_TEMPLATE = ('Self-adjusted mixture sampling (SAMS) simulation using SAMSSampler '
                       'class of yank.multistate on {}')

    def __init__(self,
                 number_of_iterations=1,
                 log_target_probabilities=None,
                 state_update_scheme='global-jump',
                 locality=5,
                 update_stages='two-stage',
                 flatness_criteria='logZ-flatness',
                 flatness_threshold=0.2,
                 weight_update_method='rao-blackwellized',
                 adapt_target_probabilities=False,
                 gamma0=1.0,
                 logZ_guess=None,
                 **kwargs):
        """Initialize a SAMS sampler.

        Parameters
        ----------
        log_target_probabilities : array-like or None
            ``log_target_probabilities[state_index]`` is the log target probability for thermodynamic state ``state_index``
            When converged, each state should be sampled with the specified log probability.
            If None, uniform probabilities for all states will be assumed.
        state_update_scheme : str, optional, default='global-jump'
            Specifies the scheme used to sample new thermodynamic states given fixed sampler states.
            One of ['global-jump', 'local-jump', 'restricted-range-jump']
            ``global_jump`` will allow the sampler to access any thermodynamic state
            ``local-jump`` will propose a move to one of the local neighborhood states, and accept or reject.
            ``restricted-range`` will compute the probabilities for each of the states in the local neighborhood, increasing jump probability
        locality : int, optional, default=1
            Number of neighboring states on either side to consider for local update schemes.
        update_stages : str, optional, default='two-stage'
            One of ['one-stage', 'two-stage']
            ``one-stage`` will use the asymptotically optimal scheme throughout the entire simulation (not recommended due to slow convergence)
            ``two-stage`` will use a heuristic first stage to achieve flat histograms before switching to the asymptotically optimal scheme
        flatness_criteria : string, optiona, default='logZ-flatness'
            Method of assessing when to switch to asymptotically optimal scheme
             One of ['logZ-flatness','minimum-visits','histogram-flatness']
        flatness_threshold : float, optional, default=0.2
            Histogram relative flatness threshold to use for first stage of two-stage scheme.
        weight_update_method : str, optional, default='rao-blackwellized'
            Method to use for updating log weights in SAMS. One of ['optimal', 'rao-blackwellized']
            ``rao-blackwellized`` will update log free energy estimate for all states for which energies were computed
            ``optimal`` will use integral counts to update log free energy estimate of current state only
        adapt_target_probabilities : bool, optional, default=False
            If True, target probabilities will be adapted to achieve minimal thermodynamic length between terminal thermodynamic states.
            (EXPERIMENTAL)
        gamma0 : float, optional, default=0.0
            Initial weight adaptation rate.
        logZ_guess : array-like of shape [n_states] of floats, optiona, default=None
            Initial guess for logZ for all states, if available.
        """
        # Initialize multi-state sampler
        super(SAMSSampler, self).__init__(number_of_iterations=number_of_iterations, **kwargs)
        # Options
        self.log_target_probabilities = log_target_probabilities
        self.state_update_scheme = state_update_scheme
        self.locality = locality
        self.update_stages = update_stages
        self.flatness_criteria = flatness_criteria
        self.flatness_threshold = flatness_threshold
        self.weight_update_method = weight_update_method
        self.adapt_target_probabilities = adapt_target_probabilities
        self.gamma0 = gamma0
        self.logZ_guess = logZ_guess
        # Private variables
        # self._replica_neighbors[replica_index] is a list of states that form the neighborhood of ``replica_index``
        self._replica_neighbors = None
        self._cached_state_histogram = None

    class _StoredProperty(multistate.MultiStateSampler._StoredProperty):

        @staticmethod
        def _state_update_scheme_validator(instance, scheme):
            supported_schemes = ['global-jump', 'local-jump', 'restricted-range-jump']
            supported_schemes = ['global-jump'] # TODO: Eliminate this after release
            if scheme not in supported_schemes:
                raise ValueError("Unknown update scheme '{}'. Supported values "
                                 "are {}.".format(scheme, supported_schemes))
            return scheme

        @staticmethod
        def _update_stages_validator(instance, scheme):
            supported_schemes = ['one-stage', 'two-stage']
            if scheme not in supported_schemes:
                raise ValueError("Unknown update scheme '{}'. Supported values "
                                 "are {}.".format(scheme, supported_schemes))
            return scheme

        @staticmethod
        def _flatness_criteria_validator(instance, scheme):
            supported_schemes = ['minimum-visits', 'logZ-flatness', 'histogram-flatness']
            if scheme not in supported_schemes:
                raise ValueError("Unknown update scheme '{}'. Supported values "
                                 "are {}.".format(scheme, supported_schemes))
            return scheme

        @staticmethod
        def _weight_update_method_validator(instance, scheme):
            supported_schemes = ['optimal', 'rao-blackwellized']
            if scheme not in supported_schemes:
                raise ValueError("Unknown update scheme '{}'. Supported values "
                                 "are {}.".format(scheme, supported_schemes))
            return scheme

        @staticmethod
        def _adapt_target_probabilities_validator(instance, scheme):
            supported_schemes = [False]
            if scheme not in supported_schemes:
                raise ValueError("Unknown update scheme '{}'. Supported values "
                                 "are {}.".format(scheme, supported_schemes))
            return scheme

    log_target_probabilities = _StoredProperty('log_target_probabilities', validate_function=None)
    state_update_scheme = _StoredProperty('state_update_scheme', validate_function=_StoredProperty._state_update_scheme_validator)
    locality = _StoredProperty('locality', validate_function=None)
    update_stages = _StoredProperty('update_stages', validate_function=_StoredProperty._update_stages_validator)
    flatness_criteria = _StoredProperty('flatness_criteria', validate_function=_StoredProperty._flatness_criteria_validator)
    flatness_threshold = _StoredProperty('flatness_threshold', validate_function=None)
    weight_update_method = _StoredProperty('weight_update_method', validate_function=_StoredProperty._weight_update_method_validator)
    adapt_target_probabilities = _StoredProperty('adapt_target_probabilities', validate_function=_StoredProperty._adapt_target_probabilities_validator)
    gamma0 = _StoredProperty('gamma0', validate_function=None)
    logZ_guess = _StoredProperty('logZ_guess', validate_function=None)

    def _initialize_stage(self):
        self._t0 = 0  # reference iteration to subtract
        if self.update_stages == 'one-stage':
            self._stage = 1 # start with asymptotically-optimal stage
        elif self.update_stages == 'two-stage':
            self._stage = 0 # start with rapid heuristic adaptation initial stage

    def _pre_write_create(self, thermodynamic_states: list, sampler_states: list, storage,
                          **kwargs):
        """Initialize SAMS sampler.

        Parameters
        ----------
        thermodynamic_states : list of openmmtools.states.ThermodynamicState
            Thermodynamic states to simulate, where one replica is allocated per state.
            Each state must have a system with the same number of atoms.
        sampler_states : list of openmmtools.states.SamplerState
            One or more sets of initial sampler states.
            The number of replicas is determined by the number of sampler states provided,
            and does not need to match the number of thermodynamic states.
            Most commonly, a single sampler state is provided.
        storage : str or Reporter
            If str: path to the storage file, checkpoint options are default
            If Reporter: Instanced :class:`Reporter` class, checkpoint information is read from
            In the future this will be able to take a Storage class as well.
        initial_thermodynamic_states : None or list or array-like of int of length len(sampler_states), optional,
            default: None.
            Initial thermodynamic_state index for each sampler_state.
            If no initial distribution is chosen, ``sampler_states`` are distributed between the
            ``thermodynamic_states`` following these rules:

                * If ``len(thermodynamic_states) == len(sampler_states)``: 1-to-1 distribution

                * If ``len(thermodynamic_states) > len(sampler_states)``: First and last state distributed first
                  remaining ``sampler_states`` spaced evenly by index until ``sampler_states`` are depleted.
                  If there is only one ``sampler_state``, then the only first ``thermodynamic_state`` will be chosen

                * If ``len(thermodynamic_states) < len(sampler_states)``, each ``thermodynamic_state`` receives an
                  equal number of ``sampler_states`` until there are insufficient number of ``sampler_states`` remaining
                  to give each ``thermodynamic_state`` an equal number. Then the rules from the previous point are
                  followed.
        metadata : dict, optional
           Simulation metadata to be stored in the file.
        """
        # Initialize replica-exchange simulation.
        super()._pre_write_create(thermodynamic_states, sampler_states, storage=storage, **kwargs)

        if self.state_update_scheme == 'global-jump':
            self.locality = None  # override locality to be global
        if self.locality is not None:
            if self.locality < 1:
                raise Exception('locality must be >= 1')
            elif self.locality >= self.n_states:
                self.locality = None

        # Record current weight update stage
        self._initialize_stage()

        # Update log target probabilities
        if self.log_target_probabilities is None:
            self.log_target_probabilities = np.zeros([self.n_states], np.float64) - np.log(self.n_states) # log(1/n_states)
            #logger.debug('Setting log target probabilities: %s' % str(self.log_target_probabilities))
            #logger.debug('Target probabilities: %s' % str(np.exp(self.log_target_probabilities)))

        # Record initial logZ estimates
        self._logZ = np.zeros([self.n_states], np.float64)
        if self.logZ_guess is not None:
            if len(self.logZ_guess) != self.n_states:
                raise Exception('Initial logZ_guess (dim {}) must have same number of states as n_states ({})'.format(
                    len(self.logZ_guess), self.n_states))
            self._logZ = np.array(self.logZ_guess, np.float64)

        # Update log weights
        self._update_log_weights()

    def _restore_sampler_from_reporter(self, reporter):
        super()._restore_sampler_from_reporter(reporter)
        self._cached_state_histogram = self._compute_state_histogram(reporter=reporter)
        logger.debug('Restored state histogram: {}'.format(self._cached_state_histogram))
        data = reporter.read_online_analysis_data(self._iteration, 'logZ', 'stage', 't0')
        self._logZ = data['logZ']
        self._stage = int(data['stage'][0])
        self._t0 = int(data['t0'][0])

        # Compute log weights from log target probability and logZ estimate
        self._update_log_weights()

        # Determine t0
        self._update_stage()

    @mpiplus.on_single_node(rank=0, broadcast_result=False, sync_nodes=False)
    @mpiplus.delayed_termination
    def _report_iteration_items(self):
        super(SAMSSampler, self)._report_iteration_items()

        self._reporter.write_online_data_dynamic_and_static(self._iteration, logZ=self._logZ, stage=self._stage, t0=self._t0)
        # Split into which states and how many samplers are in each state
        # Trying to do histogram[replica_thermo_states] += 1 does not correctly handle multiple
        # replicas in the same state.
        states, counts = np.unique(self._replica_thermodynamic_states, return_counts=True)
        if self._cached_state_histogram is None:
            self._cached_state_histogram = np.zeros(self.n_states, dtype=int)
        self._cached_state_histogram[states] += counts

    @mpiplus.on_single_node(0, broadcast_result=True)
    def _mix_replicas(self):
        """Update thermodynamic states according to user-specified scheme."""
        logger.debug("Updating thermodynamic states using %s scheme..." % self.state_update_scheme)

        # Reset storage to keep track of swap attempts this iteration.
        self._n_accepted_matrix[:, :] = 0
        self._n_proposed_matrix[:, :] = 0

        # Perform swap attempts according to requested scheme.
        # TODO: We may be able to refactor this to simply have different update schemes compute neighborhoods differently.
        # TODO: Can we allow "plugin" addition of new update schemes that can be registered externally?
        with mmtools.utils.time_it('Mixing of replicas'):
            # Initialize statistics. This matrix is modified by the jump function and used when updating the logZ estimates.
            replicas_log_P_k = np.zeros([self.n_replicas, self.n_states], np.float64)
            if self.state_update_scheme == 'global-jump':
                self._global_jump(replicas_log_P_k)
            elif self.state_update_scheme == 'local-jump':
                self._local_jump(replicas_log_P_k)
            elif self.state_update_scheme == 'restricted-range-jump':
                self._restricted_range_jump(replicas_log_P_k)
            else:
                raise Exception('Programming error: Unreachable code')

        # Determine fraction of swaps accepted this iteration.
        n_swaps_proposed = self._n_proposed_matrix.sum()
        n_swaps_accepted = self._n_accepted_matrix.sum()
        swap_fraction_accepted = 0.0
        if n_swaps_proposed > 0:
            # TODO drop casting to float when dropping Python 2 support.
            swap_fraction_accepted = float(n_swaps_accepted) / n_swaps_proposed
        logger.debug("Accepted {}/{} attempted swaps ({:.1f}%)".format(n_swaps_accepted, n_swaps_proposed,
                                                                       swap_fraction_accepted * 100.0))

        # Update logZ estimates
        self._update_logZ_estimates(replicas_log_P_k)

        # Update log weights based on target probabilities
        self._update_log_weights()

    def _local_jump(self, replicas_log_P_k):
        n_replica, n_states, locality = self.n_replicas, self.n_states, self.locality
        for (replica_index, current_state_index) in enumerate(self._replica_thermodynamic_states):
            u_k = np.zeros([n_states], np.float64)
            log_P_k = np.zeros([n_states], np.float64)
            # Determine current neighborhood.
            neighborhood = self._neighborhood()
            neighborhood_size = len(neighborhood)
            # Propose a move from the current neighborhood.
            proposed_state_index = np.random.choice(neighborhood, p=np.ones([neighborhood_size], np.float64) / float(neighborhood_size))
            # Determine neighborhood for proposed state.
            proposed_neighborhood = self._neighborhood(proposed_state_index)
            proposed_neighborhood_size = len(proposed_neighborhood)
            # Compute state log weights.
            log_Gamma_j_L = - float(proposed_neighborhood_size) # log probability of proposing return
            log_Gamma_L_j = - float(neighborhood_size)          # log probability of proposing new state
            L = current_state_index
            # Compute potential for all states in neighborhood
            for j in neighborhood:
                u_k[j] = self._energy_thermodynamic_states[replica_index, j]
            # Compute log of probability of selecting each state in neighborhood
            for j in neighborhood:
                if j != L:
                    log_P_k[j] = log_Gamma_L_j + min(0.0, log_Gamma_j_L - log_Gamma_L_j + (self.log_weights[j] - u_k[j]) - (self.log_weights[L] - u_k[L]))
            P_k = np.zeros([n_states], np.float64)
            P_k[neighborhood] = np.exp(log_P_k[neighborhood])
            # Compute probability to return to current state L
            P_k[L] = 0.0
            P_k[L] = 1.0 - P_k[neighborhood].sum()
            log_P_k[L] = np.log(P_k[L])
            # Update context.
            new_state_index = np.random.choice(neighborhood, p=P_k[neighborhood])
            self._replica_thermodynamic_states[replica_index] = new_state_index
            # Accumulate statistics
            replicas_log_P_k[replica_index,:] = log_P_k[:]
            self._n_proposed_matrix[current_state_index, neighborhood] += 1
            self._n_accepted_matrix[current_state_index, new_state_index] += 1

    def _global_jump(self, replicas_log_P_k):
        """
        Global jump scheme.
        This method is described after Eq. 3 in [2]
        """
        n_replica, n_states = self.n_replicas, self.n_states
        for replica_index, current_state_index in enumerate(self._replica_thermodynamic_states):
            neighborhood = self._neighborhood(current_state_index)

            # Compute unnormalized log probabilities for all thermodynamic states.
            log_P_k = np.zeros([n_states], np.float64)
            for state_index in neighborhood:
                u_k = self._energy_thermodynamic_states[replica_index, :]
                log_P_k[state_index] =  - u_k[state_index] + self.log_weights[state_index]
            log_P_k -= logsumexp(log_P_k)

            # Update sampler Context to current thermodynamic state.
            P_k = np.exp(log_P_k[neighborhood])
            new_state_index = np.random.choice(neighborhood, p=P_k)
            self._replica_thermodynamic_states[replica_index] = new_state_index

            # Accumulate statistics.
            replicas_log_P_k[replica_index,:] = log_P_k[:]
            self._n_proposed_matrix[current_state_index, neighborhood] += 1
            self._n_accepted_matrix[current_state_index, new_state_index] += 1

    def _restricted_range_jump(self, replicas_log_P_k):
        # TODO: This has an major bug in that we also need to compute energies in `proposed_neighborhood`.
        # I'm working on a way to make this work.
        n_replica, n_states, locality = self.n_replicas, self.n_states, self.locality
        logger.debug('Using restricted range jump with locality %s' % str(self.locality))
        for (replica_index, current_state_index) in enumerate(self._replica_thermodynamic_states):
            u_k = self._energy_thermodynamic_states[replica_index, :]
            log_P_k = np.zeros([n_states], np.float64)
            # Propose new state from current neighborhood.
            neighborhood = self._neighborhood(current_state_index)
            logger.debug('  Current state       : %d' % current_state_index)
            logger.debug('  Neighborhood        : %s' % str(neighborhood))
            logger.debug('  Relative     u_k    : %s' % str(u_k[neighborhood] - u_k[current_state_index]))
            log_P_k[neighborhood] = self.log_weights[neighborhood] - u_k[neighborhood]
            log_P_k[neighborhood] -= logsumexp(log_P_k[neighborhood])
            logger.debug('  Neighborhood log_P_k: %s' % str(log_P_k[neighborhood]))
            P_k = np.exp(log_P_k[neighborhood])
            logger.debug('  Neighborhood P_k    : %s' % str(P_k))
            proposed_state_index = np.random.choice(neighborhood, p=P_k)
            logger.debug('  Proposed state      : %d' % proposed_state_index)
            # Determine neighborhood of proposed state.
            proposed_neighborhood = self._neighborhood(proposed_state_index)
            logger.debug('  Proposed neighborhood        : %s' % str(proposed_neighborhood))
            # Accept or reject.
            log_P_accept = logsumexp(self.log_weights[neighborhood] - u_k[neighborhood]) - logsumexp(self.log_weights[proposed_neighborhood] - u_k[proposed_neighborhood])
            logger.debug('  log_P_accept        : %f' % log_P_accept)
            logger.debug('     logsumexp(g[forward] - u[forward]) : %f' % logsumexp(self.log_weights[neighborhood] - u_k[neighborhood]))
            logger.debug('     logsumexp(g[reverse] - u[reverse]) : %f' % logsumexp(self.log_weights[proposed_neighborhood] - u_k[proposed_neighborhood]))
            new_state_index = current_state_index
            if (log_P_accept >= 0.0) or (np.random.rand() < np.exp(log_P_accept)):
                new_state_index = proposed_state_index
            logger.debug('  new_state_index     : %d' % new_state_index)
            self._replica_thermodynamic_states[replica_index] = new_state_index
            # Accumulate statistics
            replicas_log_P_k[replica_index,:] = log_P_k[:]
            self._n_proposed_matrix[current_state_index, neighborhood] += 1
            self._n_accepted_matrix[current_state_index, new_state_index] += 1

    @property
    def _state_histogram(self):
        """
        Compute the histogram for the number of times each state has been visited.

        Returns
        -------
        N_k : array-like of shape [n_states] of int
            N_k[state_index] is the number of times a replica has visited state ``state_index``
        """
        if self._cached_state_histogram is None:
            self._cached_state_histogram = self._compute_state_histogram()
        return self._cached_state_histogram

    def _compute_state_histogram(self, reporter=None):
        """ Compute state histogram from disk"""
        if reporter is None:
            reporter = self._reporter
        replica_thermodynamic_states = reporter.read_replica_thermodynamic_states()
        logger.debug('Read replica thermodynamic states: {}'.format(replica_thermodynamic_states))
        n_k, _ = np.histogram(replica_thermodynamic_states, bins=np.arange(-0.5, self.n_states + 0.5))
        return n_k

    def _update_stage(self):
        """
        Determine which adaptation stage we're in by checking histogram flatness.

        """
        # TODO: Make minimum_visits a user option
        minimum_visits = 1
        N_k = self._state_histogram
        logger.debug('    state histogram counts ({} total): {}'.format(self._cached_state_histogram.sum(), self._cached_state_histogram))
        if (self.update_stages == 'two-stage') and (self._stage == 0):
            advance = False
            if N_k.sum() == 0:
                # No samples yet; don't do anything.
                return

            if self.flatness_criteria == 'minimum-visits':
                # Advance if every state has been visited at least once
                if np.all(N_k >= minimum_visits):
                    advance = True
            elif self.flatness_criteria == 'histogram-flatness':
                # Check histogram flatness
                empirical_pi_k = N_k[:] / N_k.sum()
                pi_k = np.exp(self.log_target_probabilities)
                relative_error_k = np.abs(pi_k - empirical_pi_k) / pi_k
                if np.all(relative_error_k < self.flatness_threshold):
                    advance = True
            elif self.flatness_criteria == 'logZ-flatness':
                # TODO: Advance to asymptotically optimal scheme when logZ update fractional counts per state exceed threshold
                # for all states.
                criteria = abs(self._logZ / self.gamma0) > self.flatness_threshold
                logger.debug('logZ-flatness criteria met (%d total): %s' % (np.sum(criteria), str(np.array(criteria, 'i1'))))
                if np.all(criteria):
                    advance = True
            else:
                raise ValueError("Unknown flatness_criteria %s" % flatness_criteria)

            if advance or ((self._t0 > 0) and (self._iteration > self._t0)):
                # Histograms are sufficiently flat; switch to asymptotically optimal scheme
                self._stage = 1 # asymptotically optimal
                # TODO: On resuming, we need to recompute or restore t0, or use some other way to compute it
                self._t0 = self._iteration - 1

    def _update_logZ_estimates(self, replicas_log_P_k):
        """
        Update the logZ estimates according to selected SAMS update method

        References
        ----------
        [1] http://www.stat.rutgers.edu/home/ztan/Publication/SAMS_redo4.pdf

        """
        logger.debug('Updating logZ estimates...')

        # Store log weights used at the beginning of this iteration
        self._reporter.write_online_analysis_data(self._iteration, log_weights=self.log_weights)

        # Retrieve target probabilities
        log_pi_k = self.log_target_probabilities
        pi_k = np.exp(self.log_target_probabilities)
        #logger.debug('  log target probabilities log_pi_k: %s' % str(log_pi_k))
        #logger.debug('  target probabilities pi_k: %s' % str(pi_k))

        # Update which stage we're in, checking histogram flatness
        self._update_stage()

        logger.debug('  stage: %s' % self._stage)

        # Update logZ estimates from all replicas
        for (replica_index, state_index) in enumerate(self._replica_thermodynamic_states):
            logger.debug(' Replica %d state %d' % (replica_index, state_index))
            # Compute attenuation factor gamma
            beta_factor = 0.8
            pi_star = pi_k.min()
            t = float(self._iteration)
            if self._stage == 0: # initial stage
                gamma = self.gamma0 * min(pi_star, t**(-beta_factor)) # Eq. 15 of [1]
            elif self._stage == 1:
                gamma = self.gamma0 * min(pi_star, (t - self._t0 + self._t0**beta_factor)**(-1)) # Eq. 15 of [1]
            else:
                raise Exception('stage {} unknown'.format(self._stage))

            #logger.debug('  gamma: %s' % gamma)

            # Update online logZ estimate
            if self.weight_update_method == 'optimal':
                # Based on Eq. 9 of Ref. [1]
                logZ_update = gamma * np.exp(-log_pi_k[state_index])
                #logger.debug('  optimal logZ increment: %s' % str(logZ_update))
                self._logZ[state_index] += logZ_update
            elif self.weight_update_method == 'rao-blackwellized':
                # Based on Eq. 12 of Ref [1]
                # TODO: This has to be the previous state index and log_P_k used before update; store neighborhood?
                # TODO: Can we use masked arrays for this purpose?
                log_P_k = replicas_log_P_k[replica_index,:]
                neighborhood = np.where(self._neighborhoods[replica_index,:])[0] # compact list of states defining neighborhood
                #logger.debug('  using neighborhood: %s' % str(neighborhood))
                #logger.debug('  using log_P_k : %s' % str(log_P_k[neighborhood]))
                #logger.debug('  using log_pi_k: %s' % str(log_pi_k[neighborhood]))
                logZ_update = gamma * np.exp(log_P_k[neighborhood] - log_pi_k[neighborhood])
                #logger.debug('  Rao-Blackwellized logZ increment: %s' % str(logZ_update))
                self._logZ[neighborhood] += logZ_update
            else:
                raise Exception('Programming error: Unreachable code')

        # Subtract off logZ[0] to prevent logZ from growing without bound once we reach the asymptotically optimal stage
        if self._stage == 1: # asymptotically optimal or one-stage
            self._logZ[:] -= self._logZ[0]

        # Format logZ
        msg = '  logZ: ['
        for i, val in enumerate(self._logZ):
            if i > 0: msg += ', '
            msg += '%6.1f' % val
        msg += ']'
        logger.debug(msg)

        # Store gamma
        self._reporter.write_online_analysis_data(self._iteration, gamma=gamma)

    def _update_log_weights(self):
        """
        Update the log weights based on current online logZ estimates

        """
        # TODO: Add option to adapt target probabilities as well
        # TODO: If target probabilities are adapted, we need to store them as well

        self.log_weights = self.log_target_probabilities[:] - self._logZ[:]


class SAMSAnalyzer(MultiStateSamplerAnalyzer):
    """
    The SAMSAnalyzer is the analyzer for a simulation generated from a SAMSSampler simulation.

    See Also
    --------
    ReplicaExchangeAnalyzer
    PhaseAnalyzer

    """
    pass

# ==============================================================================
# MAIN AND TESTS
# ==============================================================================


if __name__ == "__main__":
    import doctest
    doctest.testmod()
