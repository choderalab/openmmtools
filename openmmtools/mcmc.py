#!/usr/bin/env python

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
>>> from simtk import unit
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

>>> weighted_move = WeightedMove({ghmc_move: 0.5, langevin_move: 0.5})
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




COPYRIGHT AND LICENSE

@author John D. Chodera <jchodera@gmail.com>

All code in this repository is released under the GNU General Public License.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <http://www.gnu.org/licenses/>.

"""

# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import abc
import copy
import logging

import numpy as np
from simtk import openmm, unit

from openmmtools import integrators, cache
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

    @abc.abstractmethod
    def apply(self, thermodynamic_state, sampler_state):
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

        """
        pass


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
    >>> from simtk import unit
    >>> from openmmtools import testsystems
    >>> from openmmtools.states import ThermodynamicState, SamplerState

    Create and run an alanine dipeptide simulation with a weighted move.

    >>> test = testsystems.AlanineDipeptideVacuum()
    >>> thermodynamic_state = ThermodynamicState(system=test.system,
    ...                                          temperature=298*unit.kelvin)
    >>> sampler_state = SamplerState(positions=test.positions)
    >>> # Create a move set specifying probabilities fo each type of move.
    >>> move = WeightedMove({HMCMove(n_steps=10): 0.5, LangevinDynamicsMove(n_steps=10): 0.5})
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

    def run(self, n_iterations=1):
        """
        Run the sampler for a specified number of iterations.

        Parameters
        ----------
        n_iterations : int
            Number of iterations of the sampler to run.

        """
        # Apply move for n_iterations.
        for iteration in range(n_iterations):
            self.move.apply(self.thermodynamic_state, self.sampler_state)

    def minimize(self, tolerance=1.0*unit.kilocalories_per_mole/unit.angstroms,
                 max_iterations=100, context_cache=None):
        """Minimize the current configuration.

        Parameters
        ----------
        tolerance : simtk.unit.Quantity, optional
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

        timer.report_timing()


# =============================================================================
# MCMC MOVE CONTAINERS
# =============================================================================

class SequenceMove(object):
    """A sequence of MCMC moves.

    Parameters
    ----------
    move_list : list-like of MCMCMove
        The sequence of MCMC moves to apply.
    context_cache : openmmtools.cache.ContextCache, optional
        If not None, the context_cache of all the moves in the sequence
        will be set to this (default is None).

    Attributes
    ----------
    move_list : list of MCMCMove
        The sequence of MCMC moves to apply.

    Examples
    --------

    NPT ensemble simulation of a Lennard Jones fluid with a sequence of moves.

    >>> import numpy as np
    >>> from simtk import unit
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
    def __init__(self, move_list, context_cache=None):
        self.move_list = list(move_list)
        if context_cache is not None:
            for move in self.move_list:
                move.context_cache = context_cache

    def apply(self, thermodynamic_state, sampler_state):
        """Apply the sequence of MCMC move in order.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
           The thermodynamic state to use to propagate dynamics.
        sampler_state : openmmtools.states.SamplerState
           The sampler state to apply the move to.

        """
        for move in self.move_list:
            move.apply(thermodynamic_state, sampler_state)

    def __str__(self):
        return str(self.move_list)

    def __iter__(self):
        return iter(self.move_list)


class WeightedMove(object):
    """Pick an MCMC move out of set with given probability at each iteration.

    Parameters
    ----------
    move_set : dict of MCMCMove: float
        The dict of MCMCMoves: probability of being selected at an iteration.
    context_cache : openmmtools.cache.ContextCache, optional
        If not None, the context_cache of all the moves in the set will be
        set to this (default is None).

    Attributes
    ----------
    move_set : dict of MCMCMove: float
        The dict of MCMCMoves: probability of being selected at an iteration.

    Examples
    --------
    Create and run an alanine dipeptide simulation with a weighted move.

    >>> import numpy as np
    >>> from simtk import unit
    >>> from openmmtools import testsystems
    >>> from openmmtools.states import ThermodynamicState, SamplerState
    >>> test = testsystems.AlanineDipeptideVacuum()
    >>> thermodynamic_state = ThermodynamicState(system=test.system,
    ...                                          temperature=298*unit.kelvin)
    >>> sampler_state = SamplerState(positions=test.positions)
    >>> # Create a move set specifying probabilities fo each type of move.
    >>> move = WeightedMove({HMCMove(n_steps=10): 0.5, LangevinDynamicsMove(n_steps=10): 0.5})
    >>> # Create an MCMC sampler instance and run 10 iterations of the simulation.
    >>> sampler = MCMCSampler(thermodynamic_state, sampler_state, move=move)
    >>> sampler.run(n_iterations=2)
    >>> np.allclose(sampler.sampler_state.positions, test.positions)
    False

    """
    def __init__(self, move_set, context_cache=None):
        self.move_set = move_set
        if context_cache is not None:
            for move in self.move_set:
                move.context_cache = context_cache

    def apply(self, thermodynamic_state, sampler_state):
        """Apply one of the MCMC moves in the set to the state.

        The probability that a move is picked is given by its weight.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
           The thermodynamic state to use to propagate dynamics.
        sampler_state : openmmtools.states.SamplerState
           The sampler state to apply the move to.

        """
        moves, weights = zip(*self.move_set.items())
        move = np.random.choice(moves, p=weights)
        move.apply(thermodynamic_state, sampler_state)

    def __str__(self):
        return str(self.move_set)

    def __iter__(self):
        return self.move_set.items()


# =============================================================================
# INTEGRATOR MCMC MOVE BASE CLASS
# =============================================================================

class IntegratorMove(object):
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
    context_cache : openmmtools.cache.ContextCache, optional
        The ContextCache to use for Context creation. If None, the global cache
        openmmtools.cache.global_context_cache is used (default is None).

    Attributes
    ----------
    n_steps : int
        The number of integration steps to take each time the move is applied.
    context_cache : openmmtools.cache.ContextCache
        The ContextCache to use for Context creation. If None, the global cache
        openmmtools.cache.global_context_cache is used.

    Examples
    --------
    Create a VerletIntegratorMove class.

    >>> from openmmtools import testsystems, states
    >>> from simtk.openmm import VerletIntegrator
    >>> class VerletMove(IntegratorMove):
    ...     def __init__(self, timestep, n_steps, context_cache=None):
    ...         super(VerletMove, self).__init__(n_steps, context_cache)
    ...         self.timestep = timestep
    ...     def _get_integrator(self, thermodynamic_state):
    ...         return VerletIntegrator(self.timestep)
    ...     def _before_integration(self, context, thermodynamic_state):
    ...         print('Setting velocities')
    ...         context.setVelocitiesToTemperature(thermodynamic_state.temperature)
    ...     def _after_integration(self, context):
    ...         print('Reading statistics')
    ...
    >>> alanine = testsystems.AlanineDipeptideVacuum()
    >>> sampler_state = states.SamplerState(alanine.positions)
    >>> thermodynamic_state = states.ThermodynamicState(alanine.system, 300*unit.kelvin)
    >>> move = VerletMove(timestep=1.0*unit.femtosecond, n_steps=2)
    >>> move.apply(thermodynamic_state, sampler_state)
    Setting velocities
    Reading statistics

    """

    def __init__(self, n_steps, context_cache=None):
        self.n_steps = n_steps
        self.context_cache = context_cache

    def apply(self, thermodynamic_state, sampler_state):
        """Propagate the state through the integrator.

        This updates the SamplerState after the integration. It also logs
        benchmarking information through the utils.Timer class.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
           The thermodynamic state to use to propagate dynamics.
        sampler_state : openmmtools.states.SamplerState
           The sampler state to apply the move to. This is modified.

        See Also
        --------
        openmmtools.utils.Timer

        """
        move_name = self.__class__.__name__  # shortcut
        timer = Timer()

        # Check if we have to use the global cache.
        if self.context_cache is None:
            context_cache = cache.global_context_cache
        else:
            context_cache = self.context_cache

        # Create integrator.
        integrator = self._get_integrator(thermodynamic_state)

        # Create context.
        timer.start("{}: Context request".format(move_name))
        context, integrator = context_cache.get_context(thermodynamic_state, integrator)
        sampler_state.apply_to_context(context)
        timer.stop("{}: Context request".format(move_name))
        logger.debug("{}: Context obtained, platform is {}".format(
            move_name, context.getPlatform().getName()))

        self._before_integration(context, thermodynamic_state)

        # Run dynamics.
        timer.start("{}: step({})".format(move_name, self.n_steps))
        integrator.step(self.n_steps)
        timer.stop("{}: step({})".format(move_name, self.n_steps))

        self._after_integration(context)

        # Get updated sampler state.
        timer.start("{}: update sampler state".format(move_name))
        sampler_state.update_from_context(context)
        timer.stop("{}: update sampler state".format(move_name))

        timer.report_timing()

    @abc.abstractmethod
    def _get_integrator(self, thermodynamic_state):
        """Create a new instance of the integrator to apply."""
        pass

    def _before_integration(self, context, thermodynamic_state):
        """Execute code after Context creation and before integration."""
        pass

    def _after_integration(self, context):
        """Execute code after integration.

        After this point there are no guarantees that the Context will still
        exist, together with its bound integrator and system.
        """
        pass

# =============================================================================
# LANGEVIN DYNAMICS MOVE
# =============================================================================

class LangevinDynamicsMove(IntegratorMove):
    """Langevin dynamics segment as a (pseudo) Monte Carlo move.

    This move assigns a velocity from the Maxwell-Boltzmann distribution
    and executes a number of Maxwell-Boltzmann steps to propagate dynamics.
    This is not a *true* Monte Carlo move, in that the generation of the
    correct distribution is only exact in the limit of infinitely small
    timestep; in other words, the discretization error is assumed to be
    negligible. Use HybridMonteCarloMove instead to ensure the exact
    distribution is generated.

    .. warning::
        No Metropolization is used to ensure the correct phase space
        distribution is sampled. This means that timestep-dependent errors
        will remain uncorrected, and are amplified with larger timesteps.
        Use this move at your own risk!

    Parameters
    ----------
    timestep : simtk.unit.Quantity, optional
        The timestep to use for Langevin integration
        (time units, default is 1*simtk.unit.femtosecond).
    collision_rate : simtk.unit.Quantity, optional
        The collision rate with fictitious bath particles
        (1/time units, default is 10/simtk.unit.picoseconds).
    n_steps : int, optional
        The number of integration timesteps to take each time the
        move is applied (default is 1000).
    reassign_velocities : bool, optional
        If True, the velocities will be reassigned from the Maxwell-Boltzmann
        distribution at the beginning of the move (default is False).
    context_cache : openmmtools.cache.ContextCache, optional
        The ContextCache to use for Context creation. If None, the global cache
        openmmtools.cache.global_context_cache is used (default is None).

    Attributes
    ----------
    timestep : simtk.unit.Quantity
        The timestep to use for Langevin integration (time units).
    collision_rate : simtk.unit.Quantity
        The collision rate with fictitious bath particles (1/time units).
    n_steps : int
        The number of integration timesteps to take each time the move
        is applied.
    reassign_velocities : bool
        If True, the velocities will be reassigned from the Maxwell-Boltzmann
        distribution at the beginning of the move.
    context_cache : openmmtools.cache.ContextCache
        The ContextCache to use for Context creation. If None, the global
        cache openmmtools.cache.global_context_cache is used.

    Examples
    --------
    First we need to create the thermodynamic state and the sampler
    state to propagate. Here we create an alanine dipeptide system
    in vacuum.

    >>> from simtk import unit
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

    >>> move.apply(thermodynamic_state, sampler_state)
    >>> np.allclose(sampler_state.positions, test.positions)
    False

    The same move can be applied to a different state, here an ideal gas.

    >>> test = testsystems.IdealGas()
    >>> sampler_state = SamplerState(positions=test.positions)
    >>> thermodynamic_state = ThermodynamicState(system=test.system,
    ...                                          temperature=298*unit.kelvin)
    >>> move.apply(thermodynamic_state, sampler_state)
    >>> np.allclose(sampler_state.positions, test.positions)
    False

    """

    def __init__(self, timestep=1.0*unit.femtosecond, collision_rate=10.0/unit.picoseconds,
                 n_steps=1000, reassign_velocities=False, context_cache=None):
        super(LangevinDynamicsMove, self).__init__(n_steps=n_steps, context_cache=context_cache)
        self.timestep = timestep
        self.collision_rate = collision_rate
        self.reassign_velocities = reassign_velocities

    def apply(self, thermodynamic_state, sampler_state):
        """Apply the Langevin dynamics MCMC move.

        This modifies the given sampler_state. The temperature of the
        thermodynamic state is used in Langevin dynamics.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
           The thermodynamic state to use to propagate dynamics.
        sampler_state : openmmtools.states.SamplerState
           The sampler state to apply the move to. This is modified.

        """
        # Explicitly implemented just to have more specific docstring.
        super(LangevinDynamicsMove, self).apply(thermodynamic_state, sampler_state)

    def _get_integrator(self, thermodynamic_state):
        """Implement IntegratorMove._get_integrator()."""
        return openmm.LangevinIntegrator(thermodynamic_state.temperature,
                                         self.collision_rate, self.timestep)

    def _before_integration(self, context, thermodynamic_state):
        """Override IntegratorMove._before_integration()."""
        if self.reassign_velocities:
            # Assign Maxwell-Boltzmann velocities.
            context.setVelocitiesToTemperature(thermodynamic_state.temperature)


# =============================================================================
# GENERALIZED HYBRID MONTE CARLO MOVE
# =============================================================================

class GHMCMove(IntegratorMove):
    """Generalized hybrid Monte Carlo (GHMC) Markov chain Monte Carlo move.

    This move uses generalized Hybrid Monte Carlo (GHMC), a form of Metropolized
    Langevin dynamics, to propagate the system.

    Parameters
    ----------
    timestep : simtk.unit.Quantity, optional
        The timestep to use for Langevin integration (time units, default
        is 1*simtk.unit.femtoseconds).
    collision_rate : simtk.unit.Quantity, optional
        The collision rate with fictitious bath particles (1/time units,
        default is 20/simtk.unit.picoseconds).
    n_steps : int, optional
        The number of integration timesteps to take each time the move
        is applied (default is 1000).
    context_cache : openmmtools.cache.ContextCache, optional
        The ContextCache to use for Context creation. If None, the global cache
        openmmtools.cache.global_context_cache is used (default is None).

    Attributes
    ----------
    timestep : simtk.unit.Quantity
        The timestep to use for Langevin integration (time units).
    collision_rate : simtk.unit.Quantity
        The collision rate with fictitious bath particles (1/time units).
    n_steps : int
        The number of integration timesteps to take each time the move
        is applied.
    context_cache : openmmtools.cache.ContextCache
        The ContextCache to use for Context creation. If None, the global
        cache openmmtools.cache.global_context_cache is used.
    n_accepted : int
        The number of accepted steps.
    n_attempted : int
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

    >>> from simtk import unit
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

    >>> move.apply(thermodynamic_state, sampler_state)
    >>> np.allclose(sampler_state.positions, test.positions)
    False

    The same move can be applied to a different state, here an ideal gas.

    >>> test = testsystems.IdealGas()
    >>> sampler_state = SamplerState(positions=test.positions)
    >>> thermodynamic_state = ThermodynamicState(system=test.system,
    ...                                          temperature=298*unit.kelvin)
    >>> move.apply(thermodynamic_state, sampler_state)
    >>> np.allclose(sampler_state.positions, test.positions)
    False

    """

    def __init__(self, timestep=1.0*unit.femtosecond, collision_rate=20.0/unit.picoseconds,
                 n_steps=1000, context_cache=None):
        super(GHMCMove, self).__init__(n_steps=n_steps, context_cache=context_cache)
        self.timestep = timestep
        self.collision_rate = collision_rate
        self.reset_statistics()

    def reset_statistics(self):
        """Reset the internal statistics of number of accepted and attempted moves."""
        self.n_accepted = 0  # number of accepted steps
        self.n_attempted = 0  # number of attempted steps

    @property
    def fraction_accepted(self):
        """Ratio between accepted over attempted moves (read-only).

        If the number of attempted steps is 0, this is numpy.NaN.

        """
        if self.n_attempted == 0:
            return np.NaN
        # TODO drop the casting when stop Python2 support
        return float(self.n_accepted) / self.n_attempted

    def apply(self, thermodynamic_state, sampler_state):
        """Apply the GHMC MCMC move.

        This modifies the given sampler_state. The temperature of the
        thermodynamic state is used.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
           The thermodynamic state to use when applying the MCMC move.
        sampler_state : openmmtools.states.SamplerState
           The sampler state to apply the move to. This is modified.

        """
        # Explicitly implemented just to have more specific docstring.
        super(GHMCMove, self).apply(thermodynamic_state, sampler_state)

    def _get_integrator(self, thermodynamic_state):
        """Implement IntegratorMove._get_integrator()."""
        # Store lastly generated integrator to collect statistics.
        return integrators.GHMCIntegrator(temperature=thermodynamic_state.temperature,
                                          collision_rate=self.collision_rate,
                                          timestep=self.timestep)

    def _after_integration(self, context):
        """Implement IntegratorMove._after_integration()."""
        integrator = context.getIntegrator()

        # Accumulate acceptance statistics.
        ghmc_global_variables = {integrator.getGlobalVariableName(index): index
                                 for index in range(integrator.getNumGlobalVariables())}
        n_accepted = integrator.getGlobalVariable(ghmc_global_variables['naccept'])
        n_attempted = integrator.getGlobalVariable(ghmc_global_variables['ntrials'])
        self.n_accepted += n_accepted
        self.n_attempted += n_attempted


# =============================================================================
# HYBRID MONTE CARLO MOVE
# =============================================================================

class HMCMove(IntegratorMove):
    """Hybrid Monte Carlo dynamics.

    This move assigns a velocity from the Maxwell-Boltzmann distribution
    and executes a number of velocity Verlet steps to propagate dynamics.

    Parameters
    ----------
    timestep : simtk.unit.Quantity, optional
       The timestep to use for HMC dynamics, which uses velocity Verlet following
       velocity randomization (time units, default is 1*simtk.unit.femtosecond)
    n_steps : int, optional
       The number of dynamics steps to take before Metropolis acceptance/rejection
       (default is 1000).
    context_cache : openmmtools.cache.ContextCache, optional
        The ContextCache to use for Context creation. If None, the global cache
        openmmtools.cache.global_context_cache is used (default is None).

    Attributes
    ----------
    timestep : simtk.unit.Quantity
       The timestep to use for HMC dynamics, which uses velocity Verlet following
       velocity randomization (time units).
    n_steps : int
       The number of dynamics steps to take before Metropolis acceptance/rejection.
    context_cache : openmmtools.cache.ContextCache
        The ContextCache to use for Context creation. If None, the global cache
        openmmtools.cache.global_context_cache is used.

    Examples
    --------
    First we need to create the thermodynamic state and the sampler
    state to propagate. Here we create an alanine dipeptide system
    in vacuum.

    >>> from simtk import unit
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

    >>> move.apply(thermodynamic_state, sampler_state)
    >>> np.allclose(sampler_state.positions, test.positions)
    False

    The same move can be applied to a different state, here an ideal gas.

    >>> test = testsystems.IdealGas()
    >>> sampler_state = SamplerState(positions=test.positions)
    >>> thermodynamic_state = ThermodynamicState(system=test.system,
    ...                                          temperature=298*unit.kelvin)
    >>> move.apply(thermodynamic_state, sampler_state)
    >>> np.allclose(sampler_state.positions, test.positions)
    False

    """

    def __init__(self, timestep=1.0*unit.femtosecond, n_steps=1000, context_cache=None):
        super(HMCMove, self).__init__(n_steps=n_steps, context_cache=context_cache)
        self.timestep = timestep

    def apply(self, thermodynamic_state, sampler_state):
        """Apply the MCMC move.

        This modifies the given sampler_state.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
           The thermodynamic state to use when applying the MCMC move.
        sampler_state : openmmtools.states.SamplerState
           The sampler state to apply the move to. This is modified.

        """
        # Explicitly implemented just to have more specific docstring.
        super(HMCMove, self).apply(thermodynamic_state, sampler_state)

    def _get_integrator(self, thermodynamic_state):
        """Implement IntegratorMove._get_integrator()."""
        return integrators.HMCIntegrator(temperature=thermodynamic_state.temperature,
                                         timestep=self.timestep, nsteps=self.n_steps)


# =============================================================================
# MONTE CARLO BAROSTAT MOVE
# =============================================================================

class MonteCarloBarostatMove(IntegratorMove):
    """Monte Carlo barostat move.

    This move makes one or more attempts to update the box volume using
    Monte Carlo updates.

    Parameters
    ----------
    n_attempts : int, optional
        The number of Monte Carlo attempts to make to adjust the box
        volume (default is 5).
    context_cache : openmmtools.cache.ContextCache, optional
        The ContextCache to use for Context creation. If None, the global
        cache openmmtools.cache.global_context_cache is used (default is
        None).

    Attributes
    ----------
    n_attempts
    context_cache : openmmtools.cache.ContextCache
        The ContextCache to use for Context creation. If None, the global
        cache openmmtools.cache.global_context_cache is used.

    Examples
    --------
    The thermodynamic state must be barostated by a MonteCarloBarostat
    force. The class ThermodynamicState takes care of adding one when
    we specify the pressure in its constructor.

    >>> from simtk import unit
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

    >>> move.apply(thermodynamic_state, sampler_state)
    >>> np.allclose(sampler_state.positions, test.positions)
    False

    """

    def __init__(self, n_attempts=5, context_cache=None):
        super(MonteCarloBarostatMove, self).__init__(n_steps=n_attempts,
                                                     context_cache=context_cache)

    @property
    def n_attempts(self):
        """The number of MC attempts to make to adjust the box volume."""
        return self.n_steps  # The number of steps of the dummy integrator.

    @n_attempts.setter
    def n_attempts(self, value):
        self.n_steps = value

    def apply(self, thermodynamic_state, sampler_state):
        """Apply the MCMC move.

        The thermodynamic state must be barostated by a MonteCarloBarostat
        force. This modifies the given sampler_state.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
           The thermodynamic state to use when applying the MCMC move.
        sampler_state : openmmtools.states.SamplerState
           The sampler state to apply the move to. This is modified.

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

        super(MonteCarloBarostatMove, self).apply(thermodynamic_state, sampler_state)

        # Restore frequency of barostat.
        if old_barostat_frequency != 1:
            barostat.setFrequency(old_barostat_frequency)
            thermodynamic_state.barostat = barostat

    def _get_integrator(self, thermodynamic_state):
        """Implement IntegratorMove._get_integrator()."""
        return integrators.DummyIntegrator()


# =============================================================================
# MAIN AND TESTS
# =============================================================================

if __name__ == "__main__":
    import doctest
    doctest.testmod()
