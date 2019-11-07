#!/usr/local/bin/env python

# ==============================================================================
# MODULE DOCSTRING
# ==============================================================================

"""
ReplicaExchangeSampler
======================

Derived multi-thermodynamic state multistate class with exchanging configurations between replicas

COPYRIGHT

Current version by Andrea Rizzi <andrea.rizzi@choderalab.org>, Levi N. Naden <levi.naden@choderalab.org> and
John D. Chodera <john.chodera@choderalab.org> while at Memorial Sloan Kettering Cancer Center.

Original version by John D. Chodera <jchodera@gmail.com> while at the University of
California Berkeley.

LICENSE

This code is licensed under the latest available version of the MIT License.

"""

# ==============================================================================
# GLOBAL IMPORTS
# ==============================================================================

import os
import math
import copy
import logging

import numpy as np
import mdtraj as md

from openmmtools import multistate, utils
from openmmtools.multistate.multistateanalyzer import MultiStateSamplerAnalyzer
import mpiplus


logger = logging.getLogger(__name__)


# ==============================================================================
# REPLICA-EXCHANGE SIMULATION
# ==============================================================================

class ReplicaExchangeSampler(multistate.MultiStateSampler):
    """Replica-exchange simulation facility.

    This MultiStateSampler class provides a general replica-exchange simulation facility,
    allowing any set of thermodynamic states to be specified, along with a
    set of initial positions to be assigned to the replicas in a round-robin
    fashion.

    No distinction is made between one-dimensional and multidimensional replica
    layout. By default, the replica mixing scheme attempts to mix *all* replicas
    to minimize slow diffusion normally found in multidimensional replica exchange
    simulations (Modification of the 'replica_mixing_scheme' setting will allow
    the traditional 'neighbor swaps only' scheme to be used.)

    Stored configurations, energies, swaps, and restart information are all written
    to a single output file using the platform portable, robust, and efficient
    NetCDF4 library.

    Parameters
    ----------
    mcmc_moves : MCMCMove or list of MCMCMove, optional
        The MCMCMove used to propagate the states. If a list of MCMCMoves,
        they will be assigned to the correspondent thermodynamic state on
        creation. If None is provided, Langevin dynamics with 2fm timestep, 5.0/ps collision rate,
        and 500 steps per iteration will be used.
    number_of_iterations : int or infinity, optional, default: 1
        The number of iterations to perform. Both ``float('inf')`` and
        ``numpy.inf`` are accepted for infinity. If you set this to infinity,
        be sure to set also ``online_analysis_interval``.
    replica_mixing_scheme : 'swap-all', 'swap-neighbors' or None, Default: 'swap-all'
        The scheme used to swap thermodynamic states between replicas.
    online_analysis_interval : None or Int >= 1, optional, default None
        Choose the interval at which to perform online analysis of the free energy.

        After every interval, the simulation will be stopped and the free energy estimated.

        If the error in the free energy estimate is at or below ``online_analysis_target_error``, then the simulation
        will be considered completed.

    online_analysis_target_error : float >= 0, optional, default 0.2
        The target error for the online analysis measured in kT per phase.

        Once the free energy is at or below this value, the phase will be considered complete.

        If ``online_analysis_interval`` is None, this option does nothing.

    online_analysis_minimum_iterations : int >= 0, optional, default 50
        Set the minimum number of iterations which must pass before online analysis is carried out.

        Since the initial samples likely not to yield a good estimate of free energy, save time and just skip them
        If ``online_analysis_interval`` is None, this does nothing

    Attributes
    ----------
    n_replicas
    iteration
    mcmc_moves
    sampler_states
    metadata
    is_completed

    Examples
    --------
    Parallel tempering simulation of alanine dipeptide in implicit solvent (replica
    exchange among temperatures). This is just an illustrative example; use :class:`ParallelTempering`
    class for actual production parallel tempering simulations.

    Create the system.

    >>> import math
    >>> from simtk import unit
    >>> from openmmtools import testsystems, states, mcmc
    >>> testsystem = testsystems.AlanineDipeptideImplicit()
    >>> import os
    >>> import tempfile

    Create thermodynamic states for parallel tempering with exponentially-spaced schedule.

    >>> n_replicas = 3  # Number of temperature replicas.
    >>> T_min = 298.0 * unit.kelvin  # Minimum temperature.
    >>> T_max = 600.0 * unit.kelvin  # Maximum temperature.
    >>> temperatures = [T_min + (T_max - T_min) * (math.exp(float(i) / float(n_replicas-1)) - 1.0) / (math.e - 1.0)
    ...                 for i in range(n_replicas)]
    >>> thermodynamic_states = [states.ThermodynamicState(system=testsystem.system, temperature=T)
    ...                         for T in temperatures]

    Initialize simulation object with options. Run with a GHMC integrator.

    >>> move = mcmc.GHMCMove(timestep=2.0*unit.femtoseconds, n_steps=50)
    >>> simulation = ReplicaExchangeSampler(mcmc_moves=move, number_of_iterations=2)

    Create simulation with its storage file (in a temporary directory) and run.

    >>> storage_path = tempfile.NamedTemporaryFile(delete=False).name + '.nc'
    >>> reporter = multistate.MultiStateReporter(storage_path, checkpoint_interval=1)
    >>> simulation.create(thermodynamic_states=thermodynamic_states,
    ...                   sampler_states=states.SamplerState(testsystem.positions),
    ...                   storage=reporter)
    Please cite the following:
    <BLANKLINE>
            Friedrichs MS, Eastman P, Vaidyanathan V, Houston M, LeGrand S, Beberg AL, Ensign DL, Bruns CM, and Pande VS. Accelerating molecular dynamic simulations on graphics processing unit. J. Comput. Chem. 30:864, 2009. DOI: 10.1002/jcc.21209
            Eastman P and Pande VS. OpenMM: A hardware-independent framework for molecular simulations. Comput. Sci. Eng. 12:34, 2010. DOI: 10.1109/MCSE.2010.27
            Eastman P and Pande VS. Efficient nonbonded interactions for molecular dynamics on a graphics processing unit. J. Comput. Chem. 31:1268, 2010. DOI: 10.1002/jcc.21413
            Eastman P and Pande VS. Constant constraint matrix approximation: A robust, parallelizable constraint method for molecular simulations. J. Chem. Theor. Comput. 6:434, 2010. DOI: 10.1021/ct900463w
            Chodera JD and Shirts MR. Replica exchange and expanded ensemble simulations as Gibbs multistate: Simple improvements for enhanced mixing. J. Chem. Phys., 135:194110, 2011. DOI:10.1063/1.3660669
    <BLANKLINE>
    >>> simulation.run()  # This runs for a maximum of 2 iterations.
    >>> simulation.iteration
    2
    >>> simulation.run(n_iterations=1)
    >>> simulation.iteration
    2

    To resume a simulation from an existing storage file and extend it beyond
    the original number of iterations.

    >>> del simulation
    >>> simulation = ReplicaExchangeSampler.from_storage(reporter)
    Please cite the following:
    <BLANKLINE>
            Friedrichs MS, Eastman P, Vaidyanathan V, Houston M, LeGrand S, Beberg AL, Ensign DL, Bruns CM, and Pande VS. Accelerating molecular dynamic simulations on graphics processing unit. J. Comput. Chem. 30:864, 2009. DOI: 10.1002/jcc.21209
            Eastman P and Pande VS. OpenMM: A hardware-independent framework for molecular simulations. Comput. Sci. Eng. 12:34, 2010. DOI: 10.1109/MCSE.2010.27
            Eastman P and Pande VS. Efficient nonbonded interactions for molecular dynamics on a graphics processing unit. J. Comput. Chem. 31:1268, 2010. DOI: 10.1002/jcc.21413
            Eastman P and Pande VS. Constant constraint matrix approximation: A robust, parallelizable constraint method for molecular simulations. J. Chem. Theor. Comput. 6:434, 2010. DOI: 10.1021/ct900463w
            Chodera JD and Shirts MR. Replica exchange and expanded ensemble simulations as Gibbs multistate: Simple improvements for enhanced mixing. J. Chem. Phys., 135:194110, 2011. DOI:10.1063/1.3660669
    <BLANKLINE>
    >>> simulation.extend(n_iterations=1)
    >>> simulation.iteration
    3

    You can extract several information from the NetCDF file using the Reporter
    class while the simulation is running. This reads the SamplerStates of every
    run iteration.

    >>> reporter = multistate.MultiStateReporter(storage=storage_path, open_mode='r', checkpoint_interval=1)
    >>> sampler_states = reporter.read_sampler_states(iteration=1)
    >>> len(sampler_states)
    3
    >>> sampler_states[0].positions.shape  # Alanine dipeptide has 22 atoms.
    (22, 3)

    Clean up.

    >>> os.remove(storage_path)


    :param number_of_iterations: Maximum number of integer iterations that will be run

    :param replica_mixing_scheme: Scheme which describes how replicas are exchanged each iteration as string

    :param online_analysis_interval: How frequently to carry out online analysis in number of iterations

    :param online_analysis_target_error: Target free energy difference error float at which simulation will be stopped during online analysis, in dimensionless energy

    :param online_analysis_minimum_iterations: Minimum number of iterations needed before online analysis is run as int

    """

    # -------------------------------------------------------------------------
    # Constructors.
    # -------------------------------------------------------------------------

    def __init__(self, replica_mixing_scheme='swap-all', **kwargs):

        # Initialize multi-state sampler simulation.
        super(ReplicaExchangeSampler, self).__init__(**kwargs)
        self.replica_mixing_scheme = replica_mixing_scheme

    class _StoredProperty(multistate.MultiStateSampler._StoredProperty):

        @staticmethod
        def _repex_mixing_scheme_validator(instance, replica_mixing_scheme):
            supported_schemes = ['swap-all', 'swap-neighbors', None]
            if replica_mixing_scheme not in supported_schemes:
                raise ValueError("Unknown replica mixing scheme '{}'. Supported values "
                                 "are {}.".format(replica_mixing_scheme, supported_schemes))
            if instance.locality is not None:
                if replica_mixing_scheme not in ['swap-neighbors']:
                    raise ValueError("replica_mixing_scheme must be 'swap-neighbors' if locality is used")
            return replica_mixing_scheme

    replica_mixing_scheme = _StoredProperty('replica_mixing_scheme',
                                            validate_function=_StoredProperty._repex_mixing_scheme_validator)

    _TITLE_TEMPLATE = ('Replica-exchange sampler simulation created using ReplicaExchangeSampler class '
                       'of openmmtools.multistate on {}')

    def _pre_write_create(self, thermodynamic_states, sampler_states, *args, **kwargs):
        """Overwrite parent implementation to make sure the number of
        thermodynamic states is equal to the number of sampler states.
        """
        # Make sure there are no more sampler states than thermodynamic states.
        n_states = len(thermodynamic_states)
        if len(sampler_states) > n_states:
            raise ValueError('Passed {} SamplerStates but only {} ThermodynamicStates'.format(
                len(sampler_states), n_states))

        # Distribute sampler states to replicas in a round-robin fashion.
        # The sampler states are deep-copied inside super()._pre_write_create().
        sampler_states = [sampler_states[i % len(sampler_states)] for i in range(n_states)]

        super()._pre_write_create(thermodynamic_states, sampler_states, *args, **kwargs)

    @mpiplus.on_single_node(0, broadcast_result=True)
    def _mix_replicas(self):
        """Attempt to swap replicas according to user-specified scheme."""
        logger.debug("Mixing replicas...")

        # Reset storage to keep track of swap attempts this iteration.
        self._n_accepted_matrix[:, :] = 0
        self._n_proposed_matrix[:, :] = 0

        # Perform swap attempts according to requested scheme.
        with utils.time_it('Mixing of replicas'):
            if self.replica_mixing_scheme == 'swap-neighbors':
                self._mix_neighboring_replicas()
            elif self.replica_mixing_scheme == 'swap-all':
                # Try to use cython-accelerated mixing code if possible,
                # otherwise fall back to Python-accelerated code.
                try:
                    self._mix_all_replicas_cython()
                except (ValueError, ImportError) as e:
                    logger.warning(str(e))
                    self._mix_all_replicas()
            else:
                assert self.replica_mixing_scheme is None

        # Determine fraction of swaps accepted this iteration.
        n_swaps_proposed = self._n_proposed_matrix.sum()
        n_swaps_accepted = self._n_accepted_matrix.sum()
        swap_fraction_accepted = 0.0
        if n_swaps_proposed > 0:
            # TODO drop casting to float when dropping Python 2 support.
            swap_fraction_accepted = float(n_swaps_accepted) / n_swaps_proposed
        logger.debug("Accepted {}/{} attempted swaps ({:.1f}%)".format(n_swaps_accepted, n_swaps_proposed,
                                                                       swap_fraction_accepted * 100.0))

    def _mix_all_replicas_cython(self):
        """Exchange all replicas with Cython-accelerated code."""
        from openmmtools.multistate.mixing._mix_replicas import _mix_replicas_cython

        replica_states = md.utils.ensure_type(self._replica_thermodynamic_states, np.int64, 1, "Replica States")
        u_kl = md.utils.ensure_type(self._energy_thermodynamic_states, np.float64, 2, "Reduced Potentials")
        n_proposed_matrix = md.utils.ensure_type(self._n_proposed_matrix, np.int64, 2, "Nij Proposed Swaps")
        n_accepted_matrix = md.utils.ensure_type(self._n_accepted_matrix, np.int64, 2, "Nij Accepted Swaps")
        _mix_replicas_cython(self.n_replicas**4, self.n_replicas, replica_states,
                             u_kl, n_proposed_matrix, n_accepted_matrix)

        self._replica_thermodynamic_states = replica_states
        self._n_proposed_matrix = n_proposed_matrix
        self._n_accepted_matrix = n_accepted_matrix

    def _mix_all_replicas(self):
        """Exchange all replicas with Python."""
        # Determine number of swaps to attempt to ensure thorough mixing.
        # TODO: Replace this with analytical result computed to guarantee sufficient mixing, or
        # TODO:     adjust it  based on how many we can afford to do and not have mixing take a
        # TODO:     substantial fraction of iteration time.
        nswap_attempts = self.n_replicas**5  # Number of swaps to attempt (ideal, but too slow!)
        nswap_attempts = self.n_replicas**3  # Best compromise for pure Python?

        logger.debug("Will attempt to swap all pairs of replicas, using a total of %d attempts." % nswap_attempts)

        # Attempt swaps to mix replicas.
        for swap_attempt in range(nswap_attempts):
            # Choose random replicas uniformly to attempt to swap.
            replica_i = np.random.randint(self.n_replicas)
            replica_j = np.random.randint(self.n_replicas)
            self._attempt_swap(replica_i, replica_j)

    def _mix_neighboring_replicas(self):
        """Attempt exchanges between neighboring replicas only."""
        logger.debug("Will attempt to swap only neighboring replicas.")

        # TODO: Extend this to allow more remote swaps or more thorough mixing if locality > 1.

        # Attempt swaps of pairs of replicas using traditional scheme (e.g. [0,1], [2,3], ...).
        offset = np.random.randint(2)  # Offset is 0 or 1.
        for thermodynamic_state_i in range(offset, self.n_replicas-1, 2):
            thermodynamic_state_j = thermodynamic_state_i + 1  # Neighboring state.

            # Determine which replicas currently hold the thermodynamic states.
            replica_i = np.where(self._replica_thermodynamic_states == thermodynamic_state_i)
            replica_j = np.where(self._replica_thermodynamic_states == thermodynamic_state_j)
            self._attempt_swap(replica_i, replica_j)

    def _attempt_swap(self, replica_i, replica_j):
        """Attempt a single exchange between two replicas."""
        # Determine the thermodynamic states associated to these replicas.
        thermodynamic_state_i = self._replica_thermodynamic_states[replica_i]
        thermodynamic_state_j = self._replica_thermodynamic_states[replica_j]

        # Compute log probability of swap.
        energy_ij = self._energy_thermodynamic_states[replica_i, thermodynamic_state_j]
        energy_ji = self._energy_thermodynamic_states[replica_j, thermodynamic_state_i]
        energy_ii = self._energy_thermodynamic_states[replica_i, thermodynamic_state_i]
        energy_jj = self._energy_thermodynamic_states[replica_j, thermodynamic_state_j]
        log_p_accept = - (energy_ij + energy_ji) + energy_ii + energy_jj

        # Record that this move has been proposed.
        self._n_proposed_matrix[thermodynamic_state_i, thermodynamic_state_j] += 1
        self._n_proposed_matrix[thermodynamic_state_j, thermodynamic_state_i] += 1

        # Accept or reject.
        if log_p_accept >= 0.0 or np.random.rand() < math.exp(log_p_accept):
            # Swap states in replica slots i and j.
            self._replica_thermodynamic_states[replica_i] = thermodynamic_state_j
            self._replica_thermodynamic_states[replica_j] = thermodynamic_state_i
            # Accumulate statistics.
            self._n_accepted_matrix[thermodynamic_state_i, thermodynamic_state_j] += 1
            self._n_accepted_matrix[thermodynamic_state_j, thermodynamic_state_i] += 1

    @mpiplus.on_single_node(rank=0, broadcast_result=False, sync_nodes=False)
    def _display_citations(self, overwrite_global=False, citation_stack=None):
        """
        Display papers to be cited.
        The overwrite_golbal command will force the citation to display even if the "have_citations_been_shown" variable
            is True
        """

        gibbs_citations = """\
        Chodera JD and Shirts MR. Replica exchange and expanded ensemble simulations as Gibbs multistate: Simple improvements for enhanced mixing. J. Chem. Phys., 135:194110, 2011. DOI:10.1063/1.3660669
        """
        if self.replica_mixing_scheme == 'swap-all':
            if citation_stack is None:
                citation_stack = [gibbs_citations]
            else:
                citation_stack = [gibbs_citations] + citation_stack
        super()._display_citations(overwrite_global=overwrite_global, citation_stack=citation_stack)


class ReplicaExchangeAnalyzer(MultiStateSamplerAnalyzer):

    """
    The ReplicaExchangeAnalyzer is the analyzer for a simulation generated from a Replica Exchange sampler simulation,
    implemented as an instance of the :class:`MultiStateSamplerAnalyzer`.

    See Also
    --------
    PhaseAnalyzer
    MultiStateSamplerAnalyzer

    """
    pass

# ==============================================================================
# MAIN AND TESTS
# ==============================================================================

if __name__ == "__main__":
    import doctest
    doctest.testmod()
