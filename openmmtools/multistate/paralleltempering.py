#!/usr/local/bin/env python

# ==============================================================================
# MODULE DOCSTRING
# ==============================================================================

"""
ParallelTemperingSampler
========================

Derived multi-thermodynamic state multistate class with exchanging configurations between replicas of different
temperatures. This is a special case which accepts a single thermodynamic_state and different temperatures to sample.
If you want different temperatures and Hamiltonians, use ReplicaExchangeSampler with temperatures pre-set.

COPYRIGHT

Current version by Andrea Rizzi <andrea.rizzi@choderalab.org>, Levi N. Naden <levi.naden@choderalab.org> and
John D. Chodera <john.chodera@choderalab.org> while at Memorial Sloan Kettering Cancer Center.

Original version by John D. Chodera <jchodera@gmail.com> while at the University of
California Berkeley.

LICENSE

This code is licensed under the latest available version of the MIT License.

"""

import copy
import math
import logging
import numpy as np

from openmmtools import states, cache, constants
from openmmtools.multistate import ReplicaExchangeSampler, ReplicaExchangeAnalyzer, MultiStateReporter

logger = logging.getLogger(__name__)


# ==============================================================================
# PARALLEL TEMPERING
# ==============================================================================

class ParallelTemperingSampler(ReplicaExchangeSampler):
    """Parallel tempering simulation facility.

    This class provides a facility for parallel tempering simulations. It
    is a subclass of :class:`ReplicaExchange`, but provides efficiency improvements
    for parallel tempering simulations, so should be preferred for this type
    of simulation. In particular, this makes use of the fact that the reduced
    potentials are linear in inverse temperature.

    Examples
    --------

    Create the system.

    >>> from simtk import unit
    >>> from openmmtools import testsystems, states, mcmc
    >>> import tempfile
    >>> testsystem = testsystems.AlanineDipeptideImplicit()
    >>> import os

    Create thermodynamic states for parallel tempering with exponentially-spaced schedule.

    >>> n_replicas = 3  # Number of temperature replicas.
    >>> T_min = 298.0 * unit.kelvin  # Minimum temperature.
    >>> T_max = 600.0 * unit.kelvin  # Maximum temperature.
    >>> reference_state = states.ThermodynamicState(system=testsystem.system, temperature=T_min)

    Initialize simulation object with options. Run with a GHMC integrator.

    >>> move = mcmc.GHMCMove(timestep=2.0*unit.femtoseconds, n_steps=50)
    >>> simulation = ParallelTemperingSampler(mcmc_moves=move, number_of_iterations=2)

    Create simulation with its storage file (in a temporary directory) and run.

    >>> storage_path = tempfile.NamedTemporaryFile(delete=False).name + '.nc'
    >>> reporter = MultiStateReporter(storage_path, checkpoint_interval=10)
    >>> simulation.create(reference_state,
    ...                   states.SamplerState(testsystem.positions),
    ...                   reporter, min_temperature=T_min,
    ...                   max_temperature=T_max, n_temperatures=n_replicas)
    Please cite the following:
    <BLANKLINE>
            Friedrichs MS, Eastman P, Vaidyanathan V, Houston M, LeGrand S, Beberg AL, Ensign DL, Bruns CM, and Pande VS. Accelerating molecular dynamic simulations on graphics processing unit. J. Comput. Chem. 30:864, 2009. DOI: 10.1002/jcc.21209
            Eastman P and Pande VS. OpenMM: A hardware-independent framework for molecular simulations. Comput. Sci. Eng. 12:34, 2010. DOI: 10.1109/MCSE.2010.27
            Eastman P and Pande VS. Efficient nonbonded interactions for molecular dynamics on a graphics processing unit. J. Comput. Chem. 31:1268, 2010. DOI: 10.1002/jcc.21413
            Eastman P and Pande VS. Constant constraint matrix approximation: A robust, parallelizable constraint method for molecular simulations. J. Chem. Theor. Comput. 6:434, 2010. DOI: 10.1021/ct900463w
            Chodera JD and Shirts MR. Replica exchange and expanded ensemble simulations as Gibbs multistate: Simple improvements for enhanced mixing. J. Chem. Phys., 135:194110, 2011. DOI:10.1063/1.3660669
    <BLANKLINE>
    >>> simulation.run(n_iterations=1)

    Clean up.

    >>> os.remove(storage_path)


    See Also
    --------
    MultiStateSampler
    ReplicaExchangeSampler

    """

    _TITLE_TEMPLATE = ('Parallel tempering simulation created using ParallelTempering '
                       'class of yank.multistate on {}')

    def create(self, thermodynamic_state, sampler_states: list, storage,
               min_temperature=None, max_temperature=None, n_temperatures=None,
               temperatures=None, **kwargs):
        """Initialize a parallel tempering simulation object.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
            Reference thermodynamic state that will be simulated at the given
            temperatures.

            WARNING: This is a SINGLE state, not a list of states!

        sampler_states : openmmtools.states.SamplerState or list
            One or more sets of initial sampler states. If a list of SamplerStates,
            they will be assigned to replicas in a round-robin fashion.
        storage : str or Reporter
            If str: path to the storage file, checkpoint options are default
            If Reporter: Instanced :class:`Reporter` class, checkpoint information is read from
            In the future this will be able to take a Storage class as well.
        min_temperature : simtk.unit.Quantity, optional
           Minimum temperature (units of temperature, default is None).
        max_temperature : simtk.unit.Quantity, optional
           Maximum temperature (units of temperature, default is None).
        n_temperatures : int, optional
           Number of exponentially-spaced temperatures between ``min_temperature``
           and ``max_temperature`` (default is None).
        temperatures : list of simtk.unit.Quantity, optional
           If specified, this list of temperatures will be used instead of
           ``min_temperature``, ``max_temperature``, and ``n_temperatures`` (units of temperature,
           default is None).
        metadata : dict, optional
           Simulation metadata to be stored in the file.

        Notes
        -----
        Either (``min_temperature``, ``max_temperature``, ``n_temperatures``) must all be
        specified or the list of '`temperatures`' must be specified.

        """
        # Create thermodynamic states from temperatures.
        if not isinstance(thermodynamic_state, states.ThermodynamicState):
            raise ValueError("ParallelTempering only accepts a single ThermodynamicState!\n"
                             "If you have already set temperatures in your list of states, please use the "
                             "standard ReplicaExchange class with your list of states.")
        if temperatures is not None:
            logger.debug("Using provided temperatures")
        elif min_temperature is not None and max_temperature is not None and n_temperatures is not None:
            temperatures = [min_temperature + (max_temperature - min_temperature) *
                            (math.exp(i / n_temperatures-1) - 1.0) / (math.e - 1.0)
                            for i in range(n_temperatures)]  # Python 3 uses true division for /
            logger.debug('using temperatures {}'.format(temperatures))
        else:
            raise ValueError("Either 'temperatures' or ('min_temperature', 'max_temperature', "
                             "and 'n_temperatures') must be provided.")

        thermodynamic_states = [copy.deepcopy(thermodynamic_state) for _ in range(n_temperatures)]
        for state, temperature in zip(thermodynamic_states, temperatures):
            state.temperature = temperature

        # Initialize replica-exchange simulation.
        super(ParallelTemperingSampler, self).create(thermodynamic_states, sampler_states, storage=storage, **kwargs)

    def _compute_replica_energies(self, replica_id):
        """Compute the energy for the replica at every temperature.

        Because only the temperatures differ among replicas, we replace the generic O(N^2)
        replica-exchange implementation with an O(N) implementation.

        """
        # Initialize replica energies for each thermodynamic state.
        energy_thermodynamic_states = np.zeros(self.n_states)
        energy_unsampled_states = np.zeros(len(self._unsampled_states))

        # Determine neighborhood
        state_index = self._replica_thermodynamic_states[replica_id]
        neighborhood = self._neighborhood(state_index)
        # Only compute energies over neighborhoods
        energy_neighborhood_states = energy_thermodynamic_states[neighborhood]  # Array, can be indexed like this
        neighborhood_thermodynamic_states = [self._thermodynamic_states[n] for n in neighborhood]  # List

        # Retrieve sampler states associated to this replica.
        sampler_state = self._sampler_states[replica_id]

        # Thermodynamic state differ only by temperatures.
        reference_thermodynamic_state = self._thermodynamic_states[0]

        # Get the context, any Integrator works.
        context, integrator = cache.global_context_cache.get_context(reference_thermodynamic_state)

        # Update positions and box vectors.
        sampler_state.apply_to_context(context)

        # Compute energy.
        reference_reduced_potential = reference_thermodynamic_state.reduced_potential(context)

        # Strip reference potential of reference state's beta.
        reference_beta = 1.0 / (constants.kB * reference_thermodynamic_state.temperature)
        reference_reduced_potential /= reference_beta

        # Update potential energy by temperature.
        for thermodynamic_state_id, thermodynamic_state in enumerate(neighborhood_thermodynamic_states):
            beta = 1.0 / (constants.kB * thermodynamic_state.temperature)
            energy_neighborhood_states[thermodynamic_state_id] = beta * reference_reduced_potential

        # Since no assumptions can be made about the unsampled thermodynamic states, do it the hard way
        for unsampled_id, state in enumerate(self._unsampled_states):
            if unsampled_id == 0 or not state.is_state_compatible(context_state):
                context_state = state

                # Get the context, any Integrator works.
                context, integrator = cache.global_context_cache.get_context(state)

                # Update positions and box vectors. We don't need
                # to set Context velocities for the potential.
                sampler_state.apply_to_context(context, ignore_velocities=True)
            else:
                # If this state is compatible with the context, just fix the
                # thermodynamic state as positions/box vectors are the same.
                state.apply_to_context(context)

            # Compute energy.
            energy_unsampled_states[unsampled_id] = state.reduced_potential(context)

        # Return the new energies.
        return energy_neighborhood_states, energy_unsampled_states


class ParallelTemperingAnalyzer(ReplicaExchangeAnalyzer):
    """
    The ParallelTemperingAnalyzer is the analyzer for a simulation generated from a Parallel Tempering sampler
    simulation, implemented as an instance of the :class:`ReplicaExchangeAnalyzer` as the sampler is a subclass of
    the :class:`yank.multistate.ReplicaExchangeSampler`

    See Also
    --------
    PhaseAnalyzer
    ReplicaExchangeAnalyzer

    """
    pass

# ==============================================================================
# MAIN AND TESTS
# ==============================================================================

if __name__ == "__main__":
    import doctest
    doctest.testmod()
