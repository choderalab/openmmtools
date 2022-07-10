#!/usr/bin/env python

"""

COPYRIGHT AND LICENSE

@author John D. Chodera <jchodera@gmail.com>

All code in this repository is released under the MIT License.

This program is free software: you can redistribute it and/or modify it under
the terms of the MIT License.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the MIT License for more details.

You should have received a copy of the MIT License along with this program.

"""


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""Markov chain Monte Carlo simulation framework.

This module provides a framework for equilibrium sampling from a given
thermodynamic state of a biomolecule using a Markov chain Monte Carlo scheme.

It currently offer supports for
* Langevin dynamics (assumed to be free of integration error; use at your own risk]),
* hybrid Monte Carlo,
* generalized hybrid Monte Carlo, and
* Monte Carlo barostat moves,
which can be combined through the SequenceMove and WeightedMove classes.

By default, the MCMCMoves use the fastest OpenMM platform available and a
shared global ContextCache that minimizes the number of OpenMM. The examples
below show how to configure these aspects.

References
----------
Jun S. Liu. Monte Carlo Strategies in Scientific Computing. Springer, 2008.

Examples
--------
>>> from openmm import unit
>>> from openmmtools import testsystems, cache
>>> from openmmtools.states import ThermodynamicState, SamplerState

Create the initial state (thermodynamic and microscopic) for an alanine
dipeptide system in vacuum.

>>> test = testsystems.AlanineDipeptideVacuum()
>>> thermodynamic_state = ThermodynamicState(system=test.system,
...                                          temperature=298*unit.kelvin)
>>> sampler_state = SamplerState(positions=test.positions)

Create an MCMC move to perform at every iteration of the simulation, and
initialize a sampler instance.

>>> ghmc_move = GHMCMove(timestep=1.0*unit.femtosecond, n_steps=50)
>>> langevin_move = LangevinDynamicsMove(n_steps=10)
>>> sampler = MCMCSampler(thermodynamic_state, sampler_state, move=ghmc_move)

You can combine them to form a sequence of moves

>>> sequence_move = SequenceMove([ghmc_move, langevin_move])
>>> sampler = MCMCSampler(thermodynamic_state, sampler_state, move=sequence_move)

or create a move that selects one of them at random with given probability
at each iteration.

>>> weighted_move = WeightedMove([(ghmc_move, 0.5), (langevin_move, 0.5)])
>>> sampler = MCMCSampler(thermodynamic_state, sampler_state, move=weighted_move)

By default the MCMCMove use a global ContextCache that creates Context on the
fastest available OpenMM platform. You can configure the default platform to use
before starting the simulation

>>> reference_platform = openmm.Platform.getPlatformByName('Reference')
>>> cache.global_context_cache.platform = reference_platform
>>> cache.global_context_cache.time_to_live = 10  # number of read/write operations

Minimize and run the simulation for few iterations.

>>> sampler.minimize()
>>> sampler.run(n_iterations=2)

If you don't want to use a global cache, you can create local ones.

>>> local_cache1 = cache.ContextCache(capacity=5, time_to_live=50)
>>> local_cache2 = cache.ContextCache(platform=reference_platform, capacity=1)
>>> sequence_move = SequenceMove([HMCMove(), LangevinDynamicsMove()],
...                              context_cache=local_cache1)
>>> ghmc_move = GHMCMove(context_cache=local_cache2)

If you don't want to cache Context at all but create one every time, you can use
the DummyCache.

>>> dummy_cache = cache.DummyContextCache(platform=reference_platform)
>>> ghmc_move = GHMCMove(context_cache=dummy_cache)

"""

# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import os
import abc
import copy
import logging

import numpy as np
try:
    import openmm
    from openmm import unit
except ImportError:  # OpenMM < 7.6
    from simtk import openmm, unit

from openmmtools import integrators, cache, utils
from openmmtools.utils import SubhookedABCMeta, Timer

logger = logging.getLogger(__name__)


# =============================================================================
# MODULE CONSTANTS
# =============================================================================

_RANDOM_SEED_MAX = np.iinfo(np.int32).max  # maximum random number seed value


# =============================================================================
# MARKOV CHAIN MOVE ABSTRACTION
# =============================================================================

class MCMCMove(SubhookedABCMeta):
    """Markov chain Monte Carlo (MCMC) move abstraction.

    To create a new MCMCMove class compatible with this framework, you
    will have to implement this abstraction. The instance can keep internal
    statistics such as number of attempted moves and acceptance rates.

    """
    def __init__(self, context_cache=None):
        if context_cache is not None:
            logger.warning("Ignoring context_cache argument. The MCMCMove.context_cache field has been deprecated."
                           " The API now requires context_cache be passed to apply(). Please update your code.'")

    @abc.abstractmethod
    def apply(self, thermodynamic_state, sampler_state, context_cache=None):
        """Apply the MCMC move.

        Depending on the implementation, this can alter the thermodynamic
        state and/or the sampler state.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
           The initial thermodynamic state before applying the move. This
           may be modified depending on the implementation.
        sampler_state : openmmtools.states.SamplerState
           The initial sampler state before applying the move. This may
           be modified depending on the implementation.
        context_cache : opemmtools.cache.ContextCache
            Context cache to be used in the propagation of the mcmc move. If None,
            it will use the global context cache.
        """
        pass

    @property
    def context_cache(self):
        logger.warning('The MCMCMove.context_cache field has been deprecated. The API now requires context_cache '
                       'be passed to apply(). Please update your code.')
        from openmmtools import cache
        return cache.global_context_cache

    @staticmethod
    def _get_context_cache(context_cache_input):
        """
        Method to return context to be used for move propagation.

        .. note:: centralized API point to deal with context cache behavior.

        Parameters
        ----------
        context_cache_input : openmmtools.cache.ContextCache or None
            Context cache to be used in the propagation of the mcmc move. If None,
            it will create a new unlimited ContextCache object.

        Returns
        -------
        context_cache : openmmtools.cache.ContextCache
            Context cache object to be used for propagation.
        """
        if context_cache_input is None:
            # Default behavior, gobal Context Cache
            context_cache = cache.global_context_cache
        elif isinstance(context_cache_input, cache.ContextCache):
            context_cache = context_cache_input
        else:
            raise ValueError("Context cache input is not a valid ContextCache or None type.")
        return context_cache


# =============================================================================
# MARKOV CHAIN MONTE CARLO SAMPLER
# =============================================================================

class MCMCSampler(object):
    """Basic Markov chain Monte Carlo sampler.

    Parameters
    ----------
    thermodynamic_state : openmmtools.states.ThermodynamicState
        Initial thermodynamic state.
    sampler_state : openmmtools.states.SamplerState
        Initial sampler state.
    move_set : container of MarkovChainMonteCarloMove objects
        Moves to attempt during MCMC run. If list or tuple, will run all moves each
        iteration in specified sequence (e.g. [move1, move2, move3]). If dict, will
        use specified unnormalized weights (e.g. { move1 : 0.3, move2 : 0.5, move3, 0.9 })

    Attributes
    ----------
    thermodynamic_state : openmmtools.states.ThermodynamicState
        Current thermodynamic state.
    sampler_state : openmmtools.states.SamplerState
        Current sampler state.
    move_set : container of MarkovChainMonteCarloMove objects
        Moves to attempt during MCMC run. If list or tuple, will run all moves each
        iteration in specified sequence (e.g. [move1, move2, move3]). If dict, will
        use specified unnormalized weights (e.g. { move1 : 0.3, move2 : 0.5, move3, 0.9 })

    Examples
    --------

    >>> import numpy as np
    >>> from openmm import unit
    >>> from openmmtools import testsystems
    >>> from openmmtools.states import ThermodynamicState, SamplerState

    Create and run an alanine dipeptide simulation with a weighted move.

    >>> test = testsystems.AlanineDipeptideVacuum()
    >>> thermodynamic_state = ThermodynamicState(system=test.system,
    ...                                          temperature=298*unit.kelvin)
    >>> sampler_state = SamplerState(positions=test.positions)
    >>> # Create a move set specifying probabilities fo each type of move.
    >>> move = WeightedMove([(HMCMove(n_steps=10), 0.5), (LangevinDynamicsMove(n_steps=10), 0.5)])
    >>> # Create an MCMC sampler instance and run 10 iterations of the simulation.
    >>> sampler = MCMCSampler(thermodynamic_state, sampler_state, move=move)
    >>> sampler.run(n_iterations=2)
    >>> np.allclose(sampler.sampler_state.positions, test.positions)
    False

    NPT ensemble simulation of a Lennard Jones fluid with a sequence of moves.

    >>> test = testsystems.LennardJonesFluid(nparticles=200)
    >>> thermodynamic_state = ThermodynamicState(system=test.system, temperature=298*unit.kelvin,
    ...                                          pressure=1*unit.atmospheres)
    >>> sampler_state = SamplerState(positions=test.positions)
    >>> # Create a move set that includes a Monte Carlo barostat move.
    >>> move = SequenceMove([GHMCMove(n_steps=50), MonteCarloBarostatMove(n_attempts=5)])
    >>> # Create an MCMC sampler instance and run 5 iterations of the simulation.
    >>> sampler = MCMCSampler(thermodynamic_state, sampler_state, move=move)
    >>> sampler.run(n_iterations=2)
    >>> np.allclose(sampler.sampler_state.positions, test.positions)
    False

    """

    def __init__(self, thermodynamic_state, sampler_state, move):
        # Make a deep copy of the state so that initial state is unchanged.
        self.thermodynamic_state = copy.deepcopy(thermodynamic_state)
        self.sampler_state = copy.deepcopy(sampler_state)
        self.move = move

    def run(self, n_iterations=1, context_cache=None):
        """
        Run the sampler for a specified number of iterations.

        Parameters
        ----------
        n_iterations : int
            Number of iterations of the sampler to run.
        context_cache : openmmtools.cache.ContextCache or None, optional, default None
            Context cache to be used for move/integrator propagation. If None, global context cache will be used.


        """
        # Handle context cache, fall back to global if None.
        if context_cache is None:
            context_cache = cache.global_context_cache
        # Apply move for n_iterations.
        for iteration in range(n_iterations):
            self.move.apply(self.thermodynamic_state, self.sampler_state, context_cache=context_cache)

    def minimize(self, tolerance=1.0*unit.kilocalories_per_mole/unit.angstroms,
                 max_iterations=100, context_cache=None):
        """Minimize the current configuration.

        Parameters
        ----------
        tolerance : openmm.unit.Quantity, optional
            Tolerance to use for minimization termination criterion (units of
            energy/(mole*distance), default is 1*kilocalories_per_mole/angstroms).
        max_iterations : int, optional
            Maximum number of iterations to use for minimization. If 0, the minimization
            will continue until convergence (default is 100).
        context_cache : openmmtools.cache.ContextCache, optional
            The ContextCache to use for Context creation. If None, the global cache
            openmmtools.cache.global_context_cache is used (default is None).

        """
        if context_cache is None:
            context_cache = cache.global_context_cache

        timer = Timer()

        # Use LocalEnergyMinimizer
        timer.start("Context request")
        integrator = openmm.VerletIntegrator(1.0*unit.femtosecond)
        context, integrator = context_cache.get_context(self.thermodynamic_state, integrator)
        self.sampler_state.apply_to_context(context)
        logger.debug("LocalEnergyMinimizer: platform is %s" % context.getPlatform().getName())
        logger.debug("Minimizing with tolerance %s and %d max. iterations." % (tolerance, max_iterations))
        timer.stop("Context request")

        timer.start("LocalEnergyMinimizer minimize")
        openmm.LocalEnergyMinimizer.minimize(context, tolerance, max_iterations)
        timer.stop("LocalEnergyMinimizer minimize")

        # Retrieve data.
        self.sampler_state.update_from_context(context)

        #timer.report_timing()


# =============================================================================
# MCMC MOVE CONTAINERS
# =============================================================================

class SequenceMove(MCMCMove):
    """A sequence of MCMC moves.

    Parameters
    ----------
    move_list : list-like of MCMCMove
        The sequence of MCMC moves to apply.

    Attributes
    ----------
    move_list : list of MCMCMove
        The sequence of MCMC moves to apply.

    Examples
    --------

    NPT ensemble simulation of a Lennard Jones fluid with a sequence of moves.

    >>> import numpy as np
    >>> from openmm import unit
    >>> from openmmtools import testsystems
    >>> from openmmtools.states import ThermodynamicState, SamplerState
    >>> test = testsystems.LennardJonesFluid(nparticles=200)
    >>> thermodynamic_state = ThermodynamicState(system=test.system, temperature=298*unit.kelvin,
    ...                                          pressure=1*unit.atmospheres)
    >>> sampler_state = SamplerState(positions=test.positions)
    >>> # Create a move set that includes a Monte Carlo barostat move.
    >>> move = SequenceMove([GHMCMove(n_steps=50), MonteCarloBarostatMove(n_attempts=5)])
    >>> # Create an MCMC sampler instance and run 5 iterations of the simulation.
    >>> sampler = MCMCSampler(thermodynamic_state, sampler_state, move=move)
    >>> sampler.run(n_iterations=2)
    >>> np.allclose(sampler.sampler_state.positions, test.positions)
    False

    """
    def __init__(self, move_list, **kwargs):
        super(SequenceMove, self).__init__(**kwargs)
        self.move_list = list(move_list)

    @property
    def statistics(self):
        """The statistics of all moves as a list of dictionaries."""
        stats = [None for _ in range(len(self.move_list))]
        for i, move in enumerate(self.move_list):
            try:
                stats[i] = move.statistics
            except AttributeError:
                stats[i] = {}
        return stats

    @statistics.setter
    def statistics(self, value):
        for i, move in enumerate(self.move_list):
            if hasattr(move, 'statistics'):
                move.statistics = value[i]

    def apply(self, thermodynamic_state, sampler_state, context_cache=None):
        """Apply the sequence of MCMC move in order.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
           The thermodynamic state to use to propagate dynamics.
        sampler_state : openmmtools.states.SamplerState
           The sampler state to apply the move to.
        context_cache : opemmtools.cache.ContextCache, optional

        """
        # Get context cache to be used
        local_context_cache = self._get_context_cache(context_cache)
        for move in self.move_list:
            # Apply each move with the specified local context
            move.apply(thermodynamic_state, sampler_state, context_cache=local_context_cache)

    def __str__(self):
        return str(self.move_list)

    def __iter__(self):
        return iter(self.move_list)

    def __getstate__(self):
        serialized_moves = [utils.serialize(move) for move in self.move_list]
        return dict(move_list=serialized_moves)

    def __setstate__(self, serialization):
        serialized_moves = serialization['move_list']
        self.move_list = [utils.deserialize(move) for move in serialized_moves]


class WeightedMove(MCMCMove):
    """Pick an MCMC move out of set with given probability at each iteration.

    Parameters
    ----------
    move_set : list of tuples (MCMCMove, float_
        Each tuple associate an MCMCMoves to its probability of being
        selected on apply().

    Attributes
    ----------
    move_set

    Examples
    --------
    Create and run an alanine dipeptide simulation with a weighted move.

    >>> import numpy as np
    >>> from openmm import unit
    >>> from openmmtools import testsystems
    >>> from openmmtools.states import ThermodynamicState, SamplerState
    >>> test = testsystems.AlanineDipeptideVacuum()
    >>> thermodynamic_state = ThermodynamicState(system=test.system,
    ...                                          temperature=298*unit.kelvin)
    >>> sampler_state = SamplerState(positions=test.positions)
    >>> # Create a move set specifying probabilities fo each type of move.
    >>> move = WeightedMove([(HMCMove(n_steps=10), 0.5),
    ...                      (LangevinDynamicsMove(n_steps=10), 0.5)])
    >>> # Create an MCMC sampler instance and run 10 iterations of the simulation.
    >>> sampler = MCMCSampler(thermodynamic_state, sampler_state, move=move)
    >>> sampler.run(n_iterations=2)
    >>> np.allclose(sampler.sampler_state.positions, test.positions)
    False

    """
    def __init__(self, move_set, **kwargs):
        super(WeightedMove, self).__init__(**kwargs)
        self.move_set = move_set

    @property
    def statistics(self):
        """The statistics of all moves as a list of dictionaries."""
        stats = [None for _ in range(len(self.move_set))]
        for i, (move, weight) in enumerate(self.move_set):
            try:
                stats[i] = move.statistics
            except AttributeError:
                stats[i] = {}
        return stats

    @statistics.setter
    def statistics(self, value):
        for i, (move, weight) in enumerate(self.move_set):
            if hasattr(move, 'statistics'):
                move.statistics = value[i]

    def apply(self, thermodynamic_state, sampler_state, context_cache=None):
        """Apply one of the MCMC moves in the set to the state.

        The probability that a move is picked is given by its weight.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
           The thermodynamic state to use to propagate dynamics.
        sampler_state : openmmtools.states.SamplerState
           The sampler state to apply the move to.
        context_cache : openmmtools.cache.ContextCache
            The ContextCache to use for Context creation.

        """
        moves, weights = zip(*self.move_set)
        move = np.random.choice(moves, p=weights)
        # Get context cache to be used
        local_context_cache = self._get_context_cache(context_cache)
        move.apply(thermodynamic_state, sampler_state, context_cache=local_context_cache)

    def __getstate__(self):
        serialized_moves = [utils.serialize(move) for move, _ in self.move_set]
        weights = [weight for _, weight in self.move_set]
        return dict(moves=serialized_moves, weights=weights)

    def __setstate__(self, serialization):
        serialized_moves = serialization['moves']
        weights = serialization['weights']
        self.move_set = [(utils.deserialize(move), weight)
                         for move, weight in zip(serialized_moves, weights)]

    def __str__(self):
        return str(self.move_set)

    def __iter__(self):
        return self.move_set


# =============================================================================
# INTEGRATOR MCMC MOVE BASE CLASS
# =============================================================================

class IntegratorMoveError(Exception):
    """An error raised when NaN is found after applying a move.

    Parameters
    ----------
    message : str
        A description of the error.
    move : MCMCMove
        The MCMCMove that raised the error.
    context : openmm.Context, optional
        The context after the integration.

    """
    def __init__(self, message, move, context=None):
        super(IntegratorMoveError, self).__init__(message)
        self.move = move
        self.context = context

    def serialize_error(self, path_files_prefix):
        """Serializes and save the state of the simulation causing the error.

        This creates several files:
        - path_files_prefix-move.yaml
            A YAML serialization of the MCMCMove.
        - path_files_prefix-system.xml
            The serialized system in the Context.
        - path_files_prefix-integrator.xml
            The serialized integrator in the Context.
        - path_files_prefix-state.xml
            The serialized OpenMM State object before the integration.

        Parameters
        ----------
        path_files_prefix : str
            The prefix (including eventually a directory) for the files. Existing
            files will be overwritten.

        """
        directory_path = os.path.dirname(path_files_prefix)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        # Serialize MCMCMove.
        import json

        # Create class to encode quantities
        class quantity_encoder(json.JSONEncoder):
            def default(self, o):
                if type(o) in [unit.quantity.Quantity, unit.unit.Unit]:
                    return str(o)
                super(quantity_encoder, self).default(o)
        serialized_move = utils.serialize(self.move)
        with open(os.path.join(path_files_prefix + '-move.json'), 'w') as f:
            json.dump(serialized_move, f, cls=quantity_encoder)

        # Serialize Context.
        openmm_state = self.context.getState(getPositions=True, getVelocities=True,
                                             getEnergy=True, getForces=True, getParameters=True)
        to_serialize = [self.context.getSystem(), self.context.getIntegrator(), openmm_state]
        for name, openmm_object in zip(['system', 'integrator', 'state'], to_serialize):
            serialized_object = openmm.XmlSerializer.serialize(openmm_object)
            with open(os.path.join(path_files_prefix + '-' + name + '.xml'), 'w') as f:
                f.write(serialized_object)


class BaseIntegratorMove(MCMCMove):
    """A general MCMC move that applies an integrator.

    This class is intended to be inherited by MCMCMoves that need to integrate
    the system. The child class has to implement the _get_integrator method.

    You can decide to override _before_integration() and _after_integration()
    to execute some code at specific points of the workflow, for example to
    read data from the Context before the it is destroyed.

    Parameters
    ----------
    n_steps : int
        The number of integration steps to take each time the move is applied.
    reassign_velocities : bool, optional
        If True, the velocities will be reassigned from the Maxwell-Boltzmann
        distribution at the beginning of the move (default is False).
    n_restart_attempts : int, optional
        When greater than 0, if after the integration there are NaNs in energies,
        the move will restart. When the integrator has a random component, this
        may help recovering. On the last attempt, the ``Context`` is
        re-initialized in a slower process, but better than the simulation
        crashing. An IntegratorMoveError is raised after the given number of
        attempts if there are still NaNs.

    Attributes
    ----------
    n_steps : int
    reassign_velocities : bool
    n_restart_attempts : int or None

    Examples
    --------
    Create a VerletIntegratorMove class.

    >>> from openmmtools import testsystems, states
    >>> from openmm import VerletIntegrator
    >>> class VerletMove(BaseIntegratorMove):
    ...     def __init__(self, timestep, n_steps, **kwargs):
    ...         super(VerletMove, self).__init__(n_steps, **kwargs)
    ...         self.timestep = timestep
    ...     def _get_integrator(self, thermodynamic_state):
    ...         return VerletIntegrator(self.timestep)
    ...     def _before_integration(self, context, thermodynamic_state):
    ...         print('Setting velocities')
    ...         context.setVelocitiesToTemperature(thermodynamic_state.temperature)
    ...     def _after_integration(self, context, thermodynamic_state):
    ...         print('Reading statistics')
    ...
    >>> alanine = testsystems.AlanineDipeptideVacuum()
    >>> sampler_state = states.SamplerState(alanine.positions)
    >>> thermodynamic_state = states.ThermodynamicState(alanine.system, 300*unit.kelvin)
    >>> move = VerletMove(timestep=1.0*unit.femtosecond, n_steps=2)
    >>> move.apply(thermodynamic_state,sampler_state,context_cache=context_cache)
    Setting velocities
    Reading statistics

    """

    def __init__(self, n_steps, reassign_velocities=False, n_restart_attempts=4, **kwargs):
        super(BaseIntegratorMove, self).__init__(**kwargs)
        self.n_steps = n_steps
        self.reassign_velocities = reassign_velocities
        self.n_restart_attempts = n_restart_attempts

    def apply(self, thermodynamic_state, sampler_state, context_cache=None):
        """Propagate the state through the integrator.

        This updates the SamplerState after the integration. It also logs
        benchmarking information through the utils.Timer class.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
           The thermodynamic state to use to propagate dynamics.
        sampler_state : openmmtools.states.SamplerState
           The sampler state to apply the move to. This is modified.
        context_cache : openmmtools.cache.ContextCache
            Context cache to be used during propagation with the integrator.

        See Also
        --------
        openmmtools.utils.Timer

        """
        move_name = self.__class__.__name__  # shortcut
        timer = Timer()

        # Create integrator.
        integrator = self._get_integrator(thermodynamic_state)

        # Get context cache
        local_context_cache = self._get_context_cache(context_cache)

        # Create context.
        timer.start("{}: Context request".format(move_name))
        # TODO: Is this still needed now that we are specifying the context?
        context, integrator = local_context_cache.get_context(thermodynamic_state, integrator)
        timer.stop("{}: Context request".format(move_name))
        #logger.debug("{}: Context obtained, platform is {}".format(
        #    move_name, context.getPlatform().getName()))

        # Perform the integration.
        for attempt_counter in range(self.n_restart_attempts + 1):

            # If we reassign velocities, we can ignore the ones in sampler_state.
            sampler_state.apply_to_context(context, ignore_velocities=self.reassign_velocities)
            if self.reassign_velocities:
                context.setVelocitiesToTemperature(thermodynamic_state.temperature)

            # Subclasses may implement _before_integration().
            self._before_integration(context, thermodynamic_state)

            try:
                # Run dynamics.
                timer.start("{}: step({})".format(move_name, self.n_steps))
                integrator.step(self.n_steps)
            except Exception:
                # Catches particle positions becoming nan during integration.
                restart = True
            else:
                timer.stop("{}: step({})".format(move_name, self.n_steps))

                # We get also velocities here even if we don't need them because we
                # will recycle this State to update the sampler state object. This
                # way we won't need a second call to Context.getState().
                context_state = context.getState(getPositions=True, getVelocities=True, getEnergy=True,
                                                 enforcePeriodicBox=thermodynamic_state.is_periodic)

                # Check for NaNs in energies.
                potential_energy = context_state.getPotentialEnergy()
                restart = np.isnan(potential_energy.value_in_unit(potential_energy.unit))

            # Restart the move if we found NaNs.
            if restart:
                err_msg = ('Potential energy is NaN after {} attempts of integration '
                           'with move {}'.format(attempt_counter, self.__class__.__name__))

                # If we are on our last chance before crash, try to re-initialize context
                if attempt_counter == self.n_restart_attempts - 1:
                    logger.error(err_msg + ' Trying to reinitialize Context as a last-resort restart attempt...')
                    context.reinitialize()
                    sampler_state.apply_to_context(context)
                    thermodynamic_state.apply_to_context(context)
                # If we have hit the number of restart attempts, raise an exception.
                elif attempt_counter == self.n_restart_attempts:
                    # Restore the context to the state right before the integration.
                    sampler_state.apply_to_context(context)
                    logger.error(err_msg)
                    raise IntegratorMoveError(err_msg, self, context)
                else:
                    logger.warning(err_msg + ' Attempting a restart...')
            else:
                break

        # Subclasses can read here info from the context to update internal statistics.
        self._after_integration(context, thermodynamic_state)

        # Updated sampler state.
        timer.start("{}: update sampler state".format(move_name))
        # This is an optimization around the fact that Collective Variables are not a part of the State,
        # but are a part of the Context. We do this call twice to minimize duplicating information fetched from
        # the State.
        # Update everything but the collective variables from the State object
        sampler_state.update_from_context(context_state, ignore_collective_variables=True)
        # Update only the collective variables from the Context
        sampler_state.update_from_context(context, ignore_positions=True, ignore_velocities=True,
                                          ignore_collective_variables=False)
        timer.stop("{}: update sampler state".format(move_name))

        #timer.report_timing()

    @abc.abstractmethod
    def _get_integrator(self, thermodynamic_state):
        """Create a new instance of the integrator to apply."""
        pass

    def _before_integration(self, context, thermodynamic_state):
        """Execute code after Context creation and before integration."""
        pass

    def _after_integration(self, context, thermodynamic_state):
        """Execute code after integration.

        After this point there are no guarantees that the Context will still
        exist, together with its bound integrator and system.
        """
        pass

    def __getstate__(self):
        return dict(n_steps=self.n_steps,
                    reassign_velocities=self.reassign_velocities,
                    n_restart_attempts=self.n_restart_attempts)

    def __setstate__(self, serialization):
        self.n_steps = serialization['n_steps']
        self.reassign_velocities = serialization['reassign_velocities']
        self.n_restart_attempts = serialization['n_restart_attempts']


# =============================================================================
# METROPOLIZED MOVE BASE CLASS
# =============================================================================

class MetropolizedMove(MCMCMove):
    """A base class for metropolized moves.

    This class is intended to be inherited by MCMCMoves that needs to
    accept or reject a proposed move with a Metropolis criterion. Only
    the proposal needs to be specified by subclasses through the method
    _propose_positions().

    Parameters
    ----------
    atom_subset : slice or list of int, optional
        If specified, the move is applied only to those atoms specified by these
        indices. If None, the move is applied to all atoms (default is None).

    Attributes
    ----------
    n_accepted : int
        The number of proposals accepted.
    n_proposed : int
        The total number of attempted moves.
    atom_subset

    Examples
    --------
    >>> from openmm import unit
    >>> from openmmtools import testsystems, states
    >>> class AddOneVector(MetropolizedMove):
    ...     def __init__(self, **kwargs):
    ...         super(AddOneVector, self).__init__(**kwargs)
    ...     def _propose_positions(self, initial_positions):
    ...         print('Propose new positions')
    ...         displacement = unit.Quantity(np.array([1.0, 1.0, 1.0]), initial_positions.unit)
    ...         return initial_positions + displacement
    ...
    >>> alanine = testsystems.AlanineDipeptideVacuum()
    >>> sampler_state = states.SamplerState(alanine.positions)
    >>> thermodynamic_state = states.ThermodynamicState(alanine.system, 300*unit.kelvin)
    >>> move = AddOneVector(atom_subset=list(range(sampler_state.n_particles)))
    >>> move.apply(thermodynamic_state,sampler_state,context_cache=context_cache)
    Propose new positions
    >>> move.n_accepted
    1
    >>> move.n_proposed
    1

    """
    def __init__(self, atom_subset=None, **kwargs):
        super(MetropolizedMove, self).__init__(**kwargs)
        self.n_accepted = 0
        self.n_proposed = 0
        self.atom_subset = atom_subset

    @property
    def statistics(self):
        """The acceptance statistics as a dictionary."""
        return dict(n_accepted=self.n_accepted, n_proposed=self.n_proposed)

    @statistics.setter
    def statistics(self, value):
        self.n_accepted = value['n_accepted']
        self.n_proposed = value['n_proposed']

    def apply(self, thermodynamic_state, sampler_state, context_cache=None):
        """Apply a metropolized move to the sampler state.

        Total number of acceptances and proposed move are updated.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
           The thermodynamic state to use to apply the move.
        sampler_state : openmmtools.states.SamplerState
           The initial sampler state to apply the move to. This is modified.
        context_cache : openmmtools.cache.ContextCache
            The ContextCache to use for Context creation.

        """
        timer = Timer()
        benchmark_id = 'Applying {}'.format(self.__class__.__name__ )
        timer.start(benchmark_id)

        # Get context cache
        local_context_cache = self._get_context_cache(context_cache)

        # TODO: Is this still needed now that we are specifying the context?
        # Create context, any integrator works.
        context, unused_integrator = local_context_cache.get_context(thermodynamic_state)

        # Compute initial energy. We don't need to set velocities to compute the potential.
        # TODO assume sampler_state.potential_energy is the correct potential if not None?
        sampler_state.apply_to_context(context, ignore_velocities=True)
        initial_energy = thermodynamic_state.reduced_potential(context)

        # Handle default and weird cases for atom_subset.
        if self.atom_subset is None:
            atom_subset = slice(None)
        elif not isinstance(self.atom_subset, slice) and len(self.atom_subset) == 1:
            # Slice so that initial_positions (below) will have a 2D shape.
            atom_subset = slice(self.atom_subset[0], self.atom_subset[0]+1)
        else:
            atom_subset = self.atom_subset

        # Store initial positions of the atoms that are moved.
        # We'll use this also to recover in case the move is rejected.
        if isinstance(atom_subset, slice):
            # Numpy array when sliced return a view, they are not copied.
            initial_positions = copy.deepcopy(sampler_state.positions[atom_subset])
        else:
            # This automatically creates a copy.
            initial_positions = sampler_state.positions[atom_subset]

        # Propose perturbed positions. Modifying the reference changes the sampler state.
        proposed_positions = self._propose_positions(initial_positions)

        # Compute the energy of the proposed positions.
        sampler_state.positions[atom_subset] = proposed_positions
        sampler_state.apply_to_context(context, ignore_velocities=True)
        proposed_energy = thermodynamic_state.reduced_potential(context)

        # Accept or reject with Metropolis criteria.
        delta_energy = proposed_energy - initial_energy
        if (not np.isnan(proposed_energy) and
                (delta_energy <= 0.0 or np.random.rand() < np.exp(-delta_energy))):
            self.n_accepted += 1
        else:
            # Restore original positions.
            sampler_state.positions[atom_subset] = initial_positions
        self.n_proposed += 1

        # Print timing information.
        timer.stop(benchmark_id)
        #timer.report_timing()

    def __getstate__(self):
        serialization = dict(atom_subset=self.atom_subset)
        serialization.update(self.statistics)
        return serialization

    def __setstate__(self, serialization):
        self.atom_subset = serialization['atom_subset']
        self.statistics = serialization

    @abc.abstractmethod
    def _propose_positions(self, positions):
        """Return new proposed positions.

        These method must be implemented in subclasses.

        Parameters
        ----------
        positions : nx3 numpy.ndarray
            The original positions of the subset of atoms that these move
            applied to.

        Returns
        -------
        proposed_positions : nx3 numpy.ndarray
            The new proposed positions.

        """
        pass


# =============================================================================
# GENERIC INTEGRATOR MOVE
# =============================================================================

class IntegratorMove(BaseIntegratorMove):
    """An MCMCMove that propagate the system with an integrator.

    This class makes it easy to convert OpenMM Integrator objects to
    MCMCMove objects.

    Parameters
    ----------
    integrator : openmm.Integrator
        An instance of an OpenMM Integrator object to use for propagation.
    n_steps : int
        The number of integration steps to take each time the move is applied.

    Attributes
    ----------
    integrator
    n_steps

    """
    def __init__(self, integrator, n_steps, **kwargs):
        super(IntegratorMove, self).__init__(n_steps=n_steps, **kwargs)
        self.integrator = integrator

    def _get_integrator(self, thermodynamic_state):
        """Implement BaseIntegratorMove._get_integrator abstract method."""
        # We copy the integrator to make sure that the MCMCMove
        # can be applied to multiple Contexts.
        copied_integrator = copy.deepcopy(self.integrator)
        # Restore eventual extra methods for custom forces.
        integrators.ThermostatedIntegrator.restore_interface(copied_integrator)
        return copied_integrator

    def __getstate__(self):
        serialization = super(IntegratorMove, self).__getstate__()
        serialization['integrator'] = openmm.XmlSerializer.serialize(self.integrator)
        return serialization

    def __setstate__(self, serialization):
        super(IntegratorMove, self).__setstate__(serialization)
        self.integrator = openmm.XmlSerializer.deserialize(serialization['integrator'])


# =============================================================================
# LANGEVIN DYNAMICS MOVES
# =============================================================================

class LangevinDynamicsMove(BaseIntegratorMove):
    """Langevin dynamics segment as a (pseudo) Monte Carlo move.

    This move assigns a velocity from the Maxwell-Boltzmann distribution
    and executes a number of Maxwell-Boltzmann steps to propagate dynamics.
    This is not a *true* Monte Carlo move, in that the generation of the
    correct distribution is only exact in the limit of infinitely small
    timestep; in other words, the discretization error is assumed to be
    negligible. Use HybridMonteCarloMove instead to ensure the exact
    distribution is generated.

    The OpenMM LangevinMiddleIntegrator, based on BAOAB [1],  is used.

    .. warning::
        The LangevinMiddleIntegrator generates velocities that are half a timestep lagged behind the positions.

    .. warning::
        No Metropolization is used to ensure the correct phase space
        distribution is sampled. This means that timestep-dependent errors
        will remain uncorrected, and are amplified with larger timesteps.
        Use this move at your own risk!

    References
    ----------
    [1] Leimkuhler B and Matthews C. Robust and efficient configurational molecular sampling via Langevin dynamics. https://doi.org/10.1063/1.4802990
    [2] Leimkuhler B and Matthews C. Efficient molecular dynamics using geodesic integration and solvent–solute splitting. https://doi.org/10.1098/rspa.2016.0138

    Parameters
    ----------
    timestep : openmm.unit.Quantity, optional
        The timestep to use for Langevin integration
        (time units, default is 1*openmm.unit.femtosecond).
    collision_rate : openmm.unit.Quantity, optional
        The collision rate with fictitious bath particles
        (1/time units, default is 10/openmm.unit.picoseconds).
    n_steps : int, optional
        The number of integration timesteps to take each time the
        move is applied (default is 1000).
    reassign_velocities : bool, optional
        If True, the velocities will be reassigned from the Maxwell-Boltzmann
        distribution at the beginning of the move (default is False).

    Attributes
    ----------
    timestep : openmm.unit.Quantity
        The timestep to use for Langevin integration (time units).
    collision_rate : openmm.unit.Quantity
        The collision rate with fictitious bath particles (1/time units).
    n_steps : int
        The number of integration timesteps to take each time the move
        is applied.
    reassign_velocities : bool
        If True, the velocities will be reassigned from the Maxwell-Boltzmann
        distribution at the beginning of the move.

    Examples
    --------
    First we need to create the thermodynamic state and the sampler
    state to propagate. Here we create an alanine dipeptide system
    in vacuum.

    >>> from openmm import unit
    >>> from openmmtools import testsystems
    >>> from openmmtools.states import SamplerState, ThermodynamicState
    >>> test = testsystems.AlanineDipeptideVacuum()
    >>> sampler_state = SamplerState(positions=test.positions)
    >>> thermodynamic_state = ThermodynamicState(system=test.system, temperature=298*unit.kelvin)

    Create a Langevin move with default parameters

    >>> move = LangevinDynamicsMove()

    or create a Langevin move with specified parameters.

    >>> move = LangevinDynamicsMove(timestep=0.5*unit.femtoseconds,
    ...                             collision_rate=20.0/unit.picoseconds, n_steps=10)

    Perform one update of the sampler state. The sampler state is updated
    with the new state.

    >>> move.apply(thermodynamic_state,sampler_state,context_cache=context_cache)
    >>> np.allclose(sampler_state.positions, test.positions)
    False

    The same move can be applied to a different state, here an ideal gas.

    >>> test = testsystems.IdealGas()
    >>> sampler_state = SamplerState(positions=test.positions)
    >>> thermodynamic_state = ThermodynamicState(system=test.system,
    ...                                          temperature=298*unit.kelvin)
    >>> move.apply(thermodynamic_state,sampler_state,context_cache=context_cache)
    >>> np.allclose(sampler_state.positions, test.positions)
    False

    """

    def __init__(self, timestep=1.0*unit.femtosecond, collision_rate=10.0/unit.picoseconds,
                 n_steps=1000, reassign_velocities=False, **kwargs):
        super(LangevinDynamicsMove, self).__init__(n_steps=n_steps,
                                                   reassign_velocities=reassign_velocities,
                                                   **kwargs)
        self.timestep = timestep
        self.collision_rate = collision_rate

    def apply(self, thermodynamic_state, sampler_state, context_cache=None):
        """Apply the Langevin dynamics MCMC move.

        This modifies the given sampler_state. The temperature of the
        thermodynamic state is used in Langevin dynamics.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
           The thermodynamic state to use to propagate dynamics.
        sampler_state : openmmtools.states.SamplerState
           The sampler state to apply the move to. This is modified.
        context_cache : openmmtools.cache.ContextCache
            Context cache to be used during propagation with the integrator.

        """
        # Explicitly implemented just to have more specific docstring.
        super(LangevinDynamicsMove, self).apply(thermodynamic_state, sampler_state,
                                                context_cache=context_cache)

    def __getstate__(self):
        serialization = super(LangevinDynamicsMove, self).__getstate__()
        serialization['timestep'] = self.timestep
        serialization['collision_rate'] = self.collision_rate
        return serialization

    def __setstate__(self, serialization):
        super(LangevinDynamicsMove, self).__setstate__(serialization)
        self.timestep = serialization['timestep']
        self.collision_rate = serialization['collision_rate']

    def _get_integrator(self, thermodynamic_state):
        """Implement BaseIntegratorMove._get_integrator()."""
        return openmm.LangevinMiddleIntegrator(thermodynamic_state.temperature,
                                               self.collision_rate, self.timestep)


class LangevinSplittingDynamicsMove(LangevinDynamicsMove):
    """
    Langevin dynamics segment with custom splitting of the operators and optional Metropolized Monte Carlo validation.

    Besides all the normal properties of the :class:`LangevinDynamicsMove`, this class implements the custom splitting
    sequence of the :class:`openmmtools.integrators.LangevinIntegrator`. Additionally, the steps can be wrapped around
    a proper Generalized Hybrid Monte Carlo step to ensure that the exact distribution is generated.

    Parameters
    ----------
    timestep : openmm.unit.Quantity, optional
        The timestep to use for Langevin integration
        (time units, default is 1*openmm.unit.femtosecond).
    collision_rate : openmm.unit.Quantity, optional
        The collision rate with fictitious bath particles
        (1/time units, default is 10/openmm.unit.picoseconds).
    n_steps : int, optional
        The number of integration timesteps to take each time the
        move is applied (default is 1000).
    reassign_velocities : bool, optional
        If True, the velocities will be reassigned from the Maxwell-Boltzmann
        distribution at the beginning of the move (default is False).

    splitting : string, default: "V R O R V"
        Sequence of "R", "V", "O" (and optionally "{", "}", "V0", "V1", ...) substeps to be executed each timestep.

        Forces are only used in V-step. Handle multiple force groups by appending the force group index
        to V-steps, e.g. "V0" will only use forces from force group 0. "V" will perform a step using all forces.
        "{" will cause metropolization, and must be followed later by a "}".

    constraint_tolerance : float, default: 1.0e-8
        Tolerance for constraint solver

    measure_shadow_work : boolean, default: False
        Accumulate the shadow work performed by the symplectic substeps, in the global `shadow_work`

    measure_heat : boolean, default: False
        Accumulate the heat exchanged with the bath in each step, in the global `heat`

    Attributes
    ----------
    timestep : openmm.unit.Quantity
        The timestep to use for Langevin integration (time units).
    collision_rate : openmm.unit.Quantity
        The collision rate with fictitious bath particles (1/time units).
    n_steps : int
        The number of integration timesteps to take each time the move
        is applied.
    reassign_velocities : bool
        If True, the velocities will be reassigned from the Maxwell-Boltzmann
        distribution at the beginning of the move.
    splitting : str
        Splitting applied to this integrator represented as a string.
    constraint_tolerance : float, default: 1.0e-8
        Tolerance for constraint solver
    measure_shadow_work : boolean, default: False
        Accumulate the shadow work performed by the symplectic substeps, in the global `shadow_work`
    measure_heat : boolean, default: False
        Accumulate the heat exchanged with the bath in each step, in the global `heat`

    Examples
    --------
    First we need to create the thermodynamic state and the sampler
    state to propagate. Here we create an alanine dipeptide system
    in vacuum.

    >>> from openmm import unit
    >>> from openmmtools import testsystems
    >>> from openmmtools.states import SamplerState, ThermodynamicState
    >>> test = testsystems.AlanineDipeptideVacuum()
    >>> sampler_state = SamplerState(positions=test.positions)
    >>> thermodynamic_state = ThermodynamicState(system=test.system, temperature=298*unit.kelvin)

    Create a Langevin move with default parameters

    >>> move = LangevinSplittingDynamicsMove()

    or create a Langevin move with specified splitting.

    >>> move = LangevinSplittingDynamicsMove(splitting="O { V R V } O")

    Where this splitting is a 5 step symplectic integrator:

        *. Ornstein-Uhlenbeck (O) interactions with the stochastic heat bath interactions
        *. Hybrid Metropolized step around the half-step velocity updates (V) with full position updates (R).

    Perform one update of the sampler state. The sampler state is updated
    with the new state.

    >>> move.apply(thermodynamic_state,sampler_state,context_cache=context_cache)
    >>> np.allclose(sampler_state.positions, test.positions)
    False

    The same move can be applied to a different state, here an ideal gas.

    >>> test = testsystems.IdealGas()
    >>> sampler_state = SamplerState(positions=test.positions)
    >>> thermodynamic_state = ThermodynamicState(system=test.system,
    ...                                          temperature=298*unit.kelvin)
    >>> move.apply(thermodynamic_state,sampler_state,context_cache=context_cache)
    >>> np.allclose(sampler_state.positions, test.positions)
    False

    """

    def __init__(self, timestep=1.0 * unit.femtosecond, collision_rate=10.0 / unit.picoseconds,
                 n_steps=1000, reassign_velocities=False, splitting="V R O R V", constraint_tolerance=1.0e-8,
                 measure_shadow_work=False, measure_heat=False, **kwargs):
        super(LangevinSplittingDynamicsMove, self).__init__(n_steps=n_steps,
                                                            reassign_velocities=reassign_velocities,
                                                            timestep=timestep,
                                                            collision_rate=collision_rate,
                                                            **kwargs)
        self.splitting = splitting
        self.constraint_tolerance = constraint_tolerance
        self.measure_shadow_work = measure_shadow_work
        self.measure_heat = measure_heat

    def __getstate__(self):
        serialization = super(LangevinSplittingDynamicsMove, self).__getstate__()
        serialization['splitting'] = self.splitting
        serialization['constraint_tolerance'] = self.constraint_tolerance
        serialization['measure_shadow_work'] = self.measure_shadow_work
        serialization['measure_heat'] = self.measure_heat
        return serialization

    def __setstate__(self, serialization):
        super(LangevinSplittingDynamicsMove, self).__setstate__(serialization)
        self.splitting = serialization['splitting']
        self.constraint_tolerance = serialization['constraint_tolerance']
        self.measure_shadow_work = serialization['measure_shadow_work']
        self.measure_heat = serialization['measure_heat']

    def _get_integrator(self, thermodynamic_state):
        """Implement BaseIntegratorMove._get_integrator()."""
        return integrators.LangevinIntegrator(temperature=thermodynamic_state.temperature,
                                              collision_rate=self.collision_rate,
                                              timestep=self.timestep,
                                              splitting=self.splitting,
                                              constraint_tolerance=self.constraint_tolerance,
                                              measure_shadow_work=self.measure_shadow_work,
                                              measure_heat=self.measure_heat)


# =============================================================================
# GENERALIZED HYBRID MONTE CARLO MOVE
# =============================================================================

class GHMCMove(BaseIntegratorMove):
    """Generalized hybrid Monte Carlo (GHMC) Markov chain Monte Carlo move.

    This move uses generalized Hybrid Monte Carlo (GHMC), a form of Metropolized
    Langevin dynamics, to propagate the system.

    Parameters
    ----------
    timestep : openmm.unit.Quantity, optional
        The timestep to use for Langevin integration (time units, default
        is 1*openmm.unit.femtoseconds).
    collision_rate : openmm.unit.Quantity, optional
        The collision rate with fictitious bath particles (1/time units,
        default is 20/openmm.unit.picoseconds).
    n_steps : int, optional
        The number of integration timesteps to take each time the move
        is applied (default is 1000).

    Attributes
    ----------
    timestep : openmm.unit.Quantity
        The timestep to use for Langevin integration (time units).
    collision_rate : openmm.unit.Quantity
        The collision rate with fictitious bath particles (1/time units).
    n_steps : int
        The number of integration timesteps to take each time the move
        is applied.
    n_accepted : int
        The number of accepted steps.
    n_proposed : int
        The number of attempted steps.
    fraction_accepted

    References
    ----------
    Lelievre T, Stoltz G, Rousset M. Free energy computations: A mathematical
    perspective. World Scientific, 2010.

    Examples
    --------
    First we need to create the thermodynamic state and the sampler
    state to propagate. Here we create an alanine dipeptide system
    in vacuum.

    >>> from openmm import unit
    >>> from openmmtools import testsystems
    >>> from openmmtools.states import ThermodynamicState, SamplerState
    >>> test = testsystems.AlanineDipeptideVacuum()
    >>> sampler_state = SamplerState(positions=test.positions)
    >>> thermodynamic_state = ThermodynamicState(system=test.system, temperature=298*unit.kelvin)

    Create a GHMC move with default parameters.

    >>> move = GHMCMove()

    or create a GHMC move with specified parameters.

    >>> move = GHMCMove(timestep=0.5*unit.femtoseconds,
    ...                 collision_rate=20.0/unit.picoseconds, n_steps=10)

    Perform one update of the sampler state. The sampler state is updated
    with the new state.

    >>> move.apply(thermodynamic_state,sampler_state,context_cache=context_cache)
    >>> np.allclose(sampler_state.positions, test.positions)
    False

    The same move can be applied to a different state, here an ideal gas.

    >>> test = testsystems.IdealGas()
    >>> sampler_state = SamplerState(positions=test.positions)
    >>> thermodynamic_state = ThermodynamicState(system=test.system,
    ...                                          temperature=298*unit.kelvin)
    >>> move.apply(thermodynamic_state,sampler_state,context_cache=context_cache)
    >>> np.allclose(sampler_state.positions, test.positions)
    False

    """

    def __init__(self, timestep=1.0*unit.femtosecond, collision_rate=20.0/unit.picoseconds,
                 n_steps=1000, **kwargs):
        super(GHMCMove, self).__init__(n_steps=n_steps, **kwargs)
        self.timestep = timestep
        self.collision_rate = collision_rate
        self.n_accepted = 0  # Number of accepted steps.
        self.n_proposed = 0  # Number of attempted steps.

    @property
    def fraction_accepted(self):
        """Ratio between accepted over attempted moves (read-only).

        If the number of attempted steps is 0, this is numpy.NaN.

        """
        if self.n_proposed == 0:
            return np.NaN
        # TODO drop the casting when stop Python2 support
        return float(self.n_accepted) / self.n_proposed

    @property
    def statistics(self):
        """The acceptance statistics as a dictionary."""
        return dict(n_accepted=self.n_accepted, n_proposed=self.n_proposed)

    @statistics.setter
    def statistics(self, value):
        self.n_accepted = value['n_accepted']
        self.n_proposed = value['n_proposed']

    def reset_statistics(self):
        """Reset the internal statistics of number of accepted and attempted moves."""
        self.n_accepted = 0
        self.n_proposed = 0

    def apply(self, thermodynamic_state, sampler_state, context_cache=None):
        """Apply the GHMC MCMC move.

        This modifies the given sampler_state. The temperature of the
        thermodynamic state is used.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
            The thermodynamic state to use when applying the MCMC move.
        sampler_state : openmmtools.states.SamplerState
            The sampler state to apply the move to. This is modified.
        context_cache : openmmtools.cache.ContextCache
            Context cache to be used during propagation with the integrator.

        """
        # Explicitly implemented just to have more specific docstring.
        super(GHMCMove, self).apply(thermodynamic_state, sampler_state, context_cache=context_cache)

    def __getstate__(self):
        serialization = super(GHMCMove, self).__getstate__()
        serialization['timestep'] = self.timestep
        serialization['collision_rate'] = self.collision_rate
        serialization.update(self.statistics)
        return serialization

    def __setstate__(self, serialization):
        super(GHMCMove, self).__setstate__(serialization)
        self.timestep = serialization['timestep']
        self.collision_rate = serialization['collision_rate']
        self.statistics = serialization

    def _get_integrator(self, thermodynamic_state):
        """Implement BaseIntegratorMove._get_integrator()."""
        # Store lastly generated integrator to collect statistics.
        return integrators.GHMCIntegrator(temperature=thermodynamic_state.temperature,
                                          collision_rate=self.collision_rate,
                                          timestep=self.timestep)

    def _after_integration(self, context, thermodynamic_state):
        """Implement BaseIntegratorMove._after_integration()."""
        integrator = context.getIntegrator()

        # Accumulate acceptance statistics.
        ghmc_global_variables = {integrator.getGlobalVariableName(index): index
                                 for index in range(integrator.getNumGlobalVariables())}
        n_accepted = integrator.getGlobalVariable(ghmc_global_variables['naccept'])
        n_proposed = integrator.getGlobalVariable(ghmc_global_variables['ntrials'])
        self.n_accepted += n_accepted
        self.n_proposed += n_proposed


# =============================================================================
# HYBRID MONTE CARLO MOVE
# =============================================================================

class HMCMove(BaseIntegratorMove):
    """Hybrid Monte Carlo dynamics.

    This move assigns a velocity from the Maxwell-Boltzmann distribution
    and executes a number of velocity Verlet steps to propagate dynamics.

    Parameters
    ----------
    timestep : openmm.unit.Quantity, optional
       The timestep to use for HMC dynamics, which uses velocity Verlet following
       velocity randomization (time units, default is 1*openmm.unit.femtosecond)
    n_steps : int, optional
       The number of dynamics steps to take before Metropolis acceptance/rejection
       (default is 1000).

    Attributes
    ----------
    timestep : openmm.unit.Quantity
       The timestep to use for HMC dynamics, which uses velocity Verlet following
       velocity randomization (time units).
    n_steps : int
       The number of dynamics steps to take before Metropolis acceptance/rejection.

    Examples
    --------
    First we need to create the thermodynamic state and the sampler
    state to propagate. Here we create an alanine dipeptide system
    in vacuum.

    >>> from openmm import unit
    >>> from openmmtools import testsystems
    >>> from openmmtools.states import ThermodynamicState, SamplerState
    >>> test = testsystems.AlanineDipeptideVacuum()
    >>> sampler_state = SamplerState(positions=test.positions)
    >>> thermodynamic_state = ThermodynamicState(system=test.system, temperature=298*unit.kelvin)

    Create a GHMC move with default parameters.

    >>> move = HMCMove()

    or create a GHMC move with specified parameters.

    >>> move = HMCMove(timestep=0.5*unit.femtoseconds, n_steps=10)

    Perform one update of the sampler state. The sampler state is updated
    with the new state.

    >>> move.apply(thermodynamic_state,sampler_state,context_cache=context_cache)
    >>> np.allclose(sampler_state.positions, test.positions)
    False

    The same move can be applied to a different state, here an ideal gas.

    >>> test = testsystems.IdealGas()
    >>> sampler_state = SamplerState(positions=test.positions)
    >>> thermodynamic_state = ThermodynamicState(system=test.system,
    ...                                          temperature=298*unit.kelvin)
    >>> move.apply(thermodynamic_state,sampler_state,context_cache=context_cache)
    >>> np.allclose(sampler_state.positions, test.positions)
    False

    """

    def __init__(self, timestep=1.0*unit.femtosecond, n_steps=1000, **kwargs):
        super(HMCMove, self).__init__(n_steps=n_steps, **kwargs)
        self.timestep = timestep

    def apply(self, thermodynamic_state, sampler_state, context_cache=None):
        """Apply the MCMC move.

        This modifies the given sampler_state.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
           The thermodynamic state to use when applying the MCMC move.
        sampler_state : openmmtools.states.SamplerState
           The sampler state to apply the move to. This is modified.
        context_cache : openmmtools.cache.ContextCache
            Context cache to be used during propagation with the integrator.

        """
        # Explicitly implemented just to have more specific docstring.
        super(HMCMove, self).apply(thermodynamic_state, sampler_state, context_cache=context_cache)

    def __getstate__(self):
        serialization = super(HMCMove, self).__getstate__()
        serialization['timestep'] = self.timestep
        return serialization

    def __setstate__(self, serialization):
        super(HMCMove, self).__setstate__(serialization)
        self.timestep = serialization['timestep']

    def _get_integrator(self, thermodynamic_state):
        """Implement BaseIntegratorMove._get_integrator()."""
        return integrators.HMCIntegrator(temperature=thermodynamic_state.temperature,
                                         timestep=self.timestep, nsteps=self.n_steps)


# =============================================================================
# MONTE CARLO BAROSTAT MOVE
# =============================================================================

class MonteCarloBarostatMove(BaseIntegratorMove):
    """Monte Carlo barostat move.

    This move makes one or more attempts to update the box volume using
    Monte Carlo updates.

    Parameters
    ----------
    n_attempts : int, optional
        The number of Monte Carlo attempts to make to adjust the box
        volume (default is 5).

    Attributes
    ----------
    n_attempts

    Examples
    --------
    The thermodynamic state must be barostated by a MonteCarloBarostat
    force. The class ThermodynamicState takes care of adding one when
    we specify the pressure in its constructor.

    >>> from openmm import unit
    >>> from openmmtools import testsystems
    >>> from openmmtools.states import ThermodynamicState, SamplerState
    >>> test = testsystems.AlanineDipeptideExplicit()
    >>> sampler_state = SamplerState(positions=test.positions)
    >>> thermodynamic_state = ThermodynamicState(system=test.system, temperature=298*unit.kelvin,
    ...                                          pressure=1.0*unit.atmosphere)

    Create a MonteCarloBarostatMove move with default parameters.

    >>> move = MonteCarloBarostatMove()

    or create a GHMC move with specified parameters.

    >>> move = MonteCarloBarostatMove(n_attempts=2)

    Perform one update of the sampler state. The sampler state is updated
    with the new state.

    >>> move.apply(thermodynamic_state,sampler_state,context_cache=context_cache)
    >>> np.allclose(sampler_state.positions, test.positions)
    False

    """

    def __init__(self, n_attempts=5, **kwargs):
        super(MonteCarloBarostatMove, self).__init__(n_steps=n_attempts, **kwargs)

    @property
    def n_attempts(self):
        """The number of MC attempts to make to adjust the box volume."""
        return self.n_steps  # The number of steps of the dummy integrator.

    @n_attempts.setter
    def n_attempts(self, value):
        self.n_steps = value

    def apply(self, thermodynamic_state, sampler_state, context_cache=None):
        """Apply the MCMC move.

        The thermodynamic state must be barostated by a MonteCarloBarostat
        force. This modifies the given sampler_state.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
            The thermodynamic state to use when applying the MCMC move.
        sampler_state : openmmtools.states.SamplerState
            The sampler state to apply the move to. This is modified.
        context_cache : openmmtools.cache.ContextCache
            Context cache to be used during propagation with the integrator

        """
        # Make sure system contains a MonteCarlo barostat.
        barostat = thermodynamic_state.barostat
        if barostat is None:
            raise RuntimeError('Requested a MonteCarloBarostat move'
                               ' on a system at constant pressure')
        if not isinstance(barostat, openmm.MonteCarloBarostat):
            raise RuntimeError('Requested a MonteCarloBarostat move on a system '
                               'barostated with a {}'.format(barostat.__class__.__name__))

        # Set temporarily the frequency if needed.
        old_barostat_frequency = barostat.getFrequency()
        if old_barostat_frequency != 1:
            barostat.setFrequency(1)
        thermodynamic_state.barostat = barostat

        super(MonteCarloBarostatMove, self).apply(thermodynamic_state, sampler_state,
                                                  context_cache=context_cache)

        # Restore frequency of barostat.
        if old_barostat_frequency != 1:
            barostat.setFrequency(old_barostat_frequency)
            thermodynamic_state.barostat = barostat

    def _get_integrator(self, thermodynamic_state):
        """Implement BaseIntegratorMove._get_integrator()."""
        return integrators.DummyIntegrator()


# =============================================================================
# RANDOM DISPLACEMENT MOVE
# =============================================================================

class MCDisplacementMove(MetropolizedMove):
    """A metropolized move that randomly displace a subset of atoms.

    Parameters
    ----------
    displacement_sigma : openmm.unit.Quantity
        The standard deviation of the normal distribution used to propose the
        random displacement (units of length, default is 1.0*nanometer).
    atom_subset : slice or list of int, optional
        If specified, the move is applied only to those atoms specified by these
        indices. If None, the move is applied to all atoms (default is None).

    Attributes
    ----------
    n_accepted : int
        The number of proposals accepted.
    n_proposed : int
        The total number of attempted moves.
    displacement_sigma
    atom_subset

    See Also
    --------
    MetropolizedMove

    """

    def __init__(self, displacement_sigma=1.0*unit.nanometer, **kwargs):
        super(MCDisplacementMove, self).__init__(**kwargs)
        self.displacement_sigma = displacement_sigma

    @staticmethod
    def displace_positions(positions, displacement_sigma=1.0*unit.nanometer):
        """Return the positions after applying a random displacement to them.

        Parameters
        ----------
        positions : nx3 numpy.ndarray openmm.unit.Quantity
            The positions to displace.
        displacement_sigma : openmm.unit.Quantity
            The standard deviation of the normal distribution used to propose
            the random displacement (units of length, default is 1.0*nanometer).

        Returns
        -------
        rotated_positions : nx3 numpy.ndarray openmm.unit.Quantity
            The displaced positions.

        """
        positions_unit = positions.unit
        unitless_displacement_sigma = displacement_sigma / positions_unit
        displacement_vector = unit.Quantity(np.random.randn(3) * unitless_displacement_sigma,
                                            positions_unit)
        return positions + displacement_vector

    def __getstate__(self):
        serialization = super(MCDisplacementMove, self).__getstate__()
        serialization['displacement_sigma'] = self.displacement_sigma
        return serialization

    def __setstate__(self, serialization):
        super(MCDisplacementMove, self).__setstate__(serialization)
        self.displacement_sigma = serialization['displacement_sigma']

    def _propose_positions(self, initial_positions):
        """Implement MetropolizedMove._propose_positions for apply()."""
        return self.displace_positions(initial_positions, self.displacement_sigma)


# =============================================================================
# RANDOM ROTATION MOVE
# =============================================================================

class MCRotationMove(MetropolizedMove):
    """A metropolized move that randomly rotate a subset of atoms.

    Parameters
    ----------
    atom_subset : slice or list of int, optional
        If specified, the move is applied only to those atoms specified by these
        indices. If None, the move is applied to all atoms (default is None).

    Attributes
    ----------
    n_accepted : int
        The number of proposals accepted.
    n_proposed : int
        The total number of attempted moves.
    atom_subset

    See Also
    --------
    MetropolizedMove

    """

    def __init__(self, **kwargs):
        super(MCRotationMove, self).__init__(**kwargs)

    @classmethod
    def rotate_positions(cls, positions):
        """Return the positions after applying a random rotation to them.

        Parameters
        ----------
        positions : nx3 numpy.ndarray openmm.unit.Quantity
            The positions to rotate.

        Returns
        -------
        rotated_positions : nx3 numpy.ndarray openmm.unit.Quantity
            The rotated positions.

        """
        positions_unit = positions.unit
        x_initial = positions / positions_unit

        # Compute center of geometry of atoms to rotate.
        x_initial_mean = x_initial.mean(0)

        # Generate a random rotation matrix.
        rotation_matrix = cls.generate_random_rotation_matrix()

        # Apply rotation.
        x_proposed = (rotation_matrix * np.matrix(x_initial - x_initial_mean).T).T + x_initial_mean
        return unit.Quantity(x_proposed, positions_unit)

    @classmethod
    def generate_random_rotation_matrix(cls):
        """Return a random 3x3 rotation matrix.

        Returns
        -------
        Rq : 3x3 numpy.ndarray
            The random rotation matrix.

        """
        q = cls._generate_uniform_quaternion()
        return cls._rotation_matrix_from_quaternion(q)

    @staticmethod
    def _rotation_matrix_from_quaternion(q):
        """Compute a 3x3 rotation matrix from a given quaternion (4-vector).

        Parameters
        ----------
        q : 1x4 numpy.ndarray
            Quaterion (need not be normalized, zero norm OK).

        Returns
        -------
        Rq : 3x3 numpy.ndarray
            Orthogonal rotation matrix corresponding to quaternion q.

        Examples
        --------
        >>> q = np.array([0.1, 0.2, 0.3, -0.4])
        >>> Rq = MCRotationMove._rotation_matrix_from_quaternion(q)

        References
        ----------
        [1] http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion

        """

        w, x, y, z = q
        Nq = (q**2).sum()  # Squared norm.
        if Nq > 0.0:
            s = 2.0 / Nq
        else:
            s = 0.0

        X = x*s;   Y = y*s;  Z = z*s
        wX = w*X; wY = w*Y; wZ = w*Z
        xX = x*X; xY = x*Y; xZ = x*Z
        yY = y*Y; yZ = y*Z; zZ = z*Z

        Rq = np.matrix([[1.0-(yY+zZ),     xY-wZ,          xZ+wY],
                        [xY+wZ,        1.0-(xX+zZ),       yZ-wX],
                        [xZ-wY,           yZ+wX,    1.0-(xX+yY)]])

        return Rq

    @staticmethod
    def _generate_uniform_quaternion():
        """Generate a uniform normalized quaternion 4-vector.

        References
        ----------
        [1] K. Shoemake. Uniform random rotations. In D. Kirk, editor,
        Graphics Gems III, pages 124-132. Academic, New York, 1992.
        [2] Described briefly here: http://planning.cs.uiuc.edu/node198.html

        Examples
        --------
        >>> q = MCRotationMove._generate_uniform_quaternion()

        """
        u = np.random.rand(3)
        q = np.array([np.sqrt(1-u[0])*np.sin(2*np.pi*u[1]),
                      np.sqrt(1-u[0])*np.cos(2*np.pi*u[1]),
                      np.sqrt(u[0])*np.sin(2*np.pi*u[2]),
                      np.sqrt(u[0])*np.cos(2*np.pi*u[2])])
        return q

    def _propose_positions(self, initial_positions):
        """Implement MetropolizedMove._propose_positions for apply()"""
        return self.rotate_positions(initial_positions)


# =============================================================================
# MAIN AND TESTS
# =============================================================================

if __name__ == "__main__":
    import doctest
    doctest.testmod()
