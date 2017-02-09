#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Markov chain Monte Carlo simulation framework.

DESCRIPTION

This module provides a framework for equilibrium sampling from a given thermodynamic state of
a biomolecule using a Markov chain Monte Carlo scheme.

CAPABILITIES
* Langevin dynamics [assumed to be free of integration error; use at your own risk]
* hybrid Monte Carlo
* generalized hybrid Monte Carlo

NOTES

This is still in development.

REFERENCES

[1] Jun S. Liu. Monte Carlo Strategies in Scientific Computing. Springer, 2008.

EXAMPLES

Construct a simple MCMC simulation using Langevin dynamics moves.

>>> # Create a test system
>>> from openmmtools import testsystems
>>> test = testsystems.AlanineDipeptideVacuum()
>>> # Create a thermodynamic state.
>>> import simtk.unit as u
>>> from openmmmcmc.thermodynamics import ThermodynamicState
>>> thermodynamic_state = ThermodynamicState(system=test.system, temperature=298*u.kelvin)
>>> # Create a sampler state.
>>> sampler_state = SamplerState(system=test.system, positions=test.positions)
>>> # Create a move set.
>>> move_set = [ HMCMove(nsteps=10), LangevinDynamicsMove(nsteps=10) ]
>>> # Create MCMC sampler
>>> sampler = MCMCSampler(thermodynamic_state, move_set=move_set)
>>> # Run a number of iterations of the sampler.
>>> updated_sampler_state = sampler.run(sampler_state, 10)

TODO

* Split this into a separate package, with individual files for each move type.

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

TODO
----
* Recognize when MonteCarloBarostat is in use with system.

"""

# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import copy
import logging
from abc import abstractmethod

import numpy as np
from simtk import openmm, unit

from openmmtools import integrators
from openmmtools.utils import SubhookedABCMeta
from openmmmcmc.timing import Timer  # TODO move this in openmmtools.utils

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

    @abstractmethod
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
    """
    Markov chain Monte Carlo sampler.

    >>> # Create a test system
    >>> from openmmtools import testsystems
    >>> test = testsystems.AlanineDipeptideVacuum()
    >>> # Create a thermodynamic state.
    >>> import simtk.unit as u
    >>> from openmmmcmc.thermodynamics import ThermodynamicState
    >>> thermodynamic_state = ThermodynamicState(system=test.system, temperature=298*u.kelvin)
    >>> # Create a sampler state.
    >>> sampler_state = SamplerState(system=test.system, positions=test.positions)
    >>> # Create a move set specifying probabilities fo each type of move.
    >>> move_set = { HMCMove(nsteps=10) : 0.5, LangevinDynamicsMove(nsteps=10) : 0.5 }
    >>> # Create MCMC sampler
    >>> sampler = MCMCSampler(thermodynamic_state, move_set=move_set)
    >>> # Run a number of iterations of the sampler.
    >>> updated_sampler_state = sampler.run(sampler_state, 10)


    >>> # Create a test system
    >>> from openmmtools import testsystems
    >>> test = testsystems.LennardJonesFluid(nparticles=200)
    >>> # Create a sampler state.
    >>> sampler_state = SamplerState(system=test.system, positions=test.positions,
    ...                              box_vectors=test.system.getDefaultPeriodicBoxVectors())
    >>> # Create a thermodynamic state.
    >>> from openmmmcmc.thermodynamics import ThermodynamicState
    >>> thermodynamic_state = ThermodynamicState(system=test.system, temperature=298*u.kelvin,
    ...                                          pressure=1*u.atmospheres)
    >>> # Create a move set that includes a Monte Carlo barostat move.
    >>> move_set = [ GHMCMove(nsteps=50), MonteCarloBarostatMove(nattempts=5) ]
    >>> # Simulate on Reference platform.
    >>> import simtk.openmm as mm
    >>> platform = mm.Platform.getPlatformByName('Reference')
    >>> sampler = MCMCSampler(thermodynamic_state, move_set=move_set, platform=platform)
    >>> # Run a number of iterations of the sampler.
    >>> updated_sampler_state = sampler.run(sampler_state, 2)

    """

    def __init__(self, thermodynamic_state, move_set=None, platform=None):
        """
        Initialize a Markov chain Monte Carlo sampler.

        Parameters
        ----------
        thermodynamic_state : ThermodynamicState
            Thermodynamic state to sample during MCMC run.
        move_set : container of MarkovChainMonteCarloMove objects
            Moves to attempt during MCMC run.
            If list or tuple, will run all moves each iteration in specified sequence. (e.g. [move1, move2, move3])
            if dict, will use specified unnormalized weights (e.g. { move1 : 0.3, move2 : 0.5, move3, 0.9 })
        platform : simtk.openmm.Platform, optional, default = None
            If specified, the Platform to use for simulations.

        Examples
        --------

        >>> # Create a test system
        >>> from openmmtools import testsystems
        >>> test = testsystems.AlanineDipeptideVacuum()
        >>> # Create a thermodynamic state.
        >>> import simtk.unit as u
        >>> from openmmmcmc.thermodynamics import ThermodynamicState
        >>> thermodynamic_state = ThermodynamicState(system=test.system, temperature=298*u.kelvin)
        >>> # Create a sampler state.
        >>> sampler_state = SamplerState(system=test.system, positions=test.positions)

        Create a move set specifying probabilities for each type of move.

        >>> move_set = { HMCMove() : 0.5, LangevinDynamicsMove() : 0.5 }
        >>> # Create MCMC sampler
        >>> sampler = MCMCSampler(thermodynamic_state, move_set=move_set)

        Create a move set specifying an order of moves.

        >>> move_set = [ HMCMove(), LangevinDynamicsMove(), HMCMove() ]
        >>> # Create MCMC sampler
        >>> sampler = MCMCSampler(thermodynamic_state, move_set=move_set)

        """

        # Store thermodynamic state.
        self.thermodynamic_state = thermodynamic_state

        # Store the move set.
        if type(move_set) not in [list, dict]:
            raise Exception("move_set must be list or dict")
        # TODO: Make deep copy of the move set?
        self.move_set = move_set
        self.platform = platform

        return

    def run(self, sampler_state, niterations=1):
        """
        Run the sampler for a specified number of iterations.

        Parameters
        ----------
        sampler_state : SamplerState
            The current state of the sampler.
        niterations : int
            Number of iterations of the sampler to run.

        Examples
        --------

        >>> # Create a test system
        >>> from openmmtools import testsystems
        >>> test = testsystems.AlanineDipeptideVacuum()
        >>> # Create a thermodynamic state.
        >>> import simtk.unit as u
        >>> from openmmmcmc.thermodynamics import ThermodynamicState
        >>> thermodynamic_state = ThermodynamicState(system=test.system, temperature=298*u.kelvin)
        >>> # Create a sampler state.
        >>> sampler_state = SamplerState(system=test.system, positions=test.positions)
        >>> # Create a move set specifying probabilities fo each type of move.
        >>> move_set = { HMCMove(nsteps=10) : 0.5, LangevinDynamicsMove(nsteps=10) : 0.5 }
        >>> # Create MCMC sampler
        >>> sampler = MCMCSampler(thermodynamic_state, move_set=move_set)
        >>> # Run a number of iterations of the sampler.
        >>> updated_sampler_state = sampler.run(sampler_state, 10)

        """

        # Make a deep copy of the sampler state so that initial state is unchanged.
        # TODO: This seems to cause problems.  Let's figure this out later.
        sampler_state = copy.deepcopy(sampler_state)

        # Generate move sequence.
        move_sequence = list()
        if type(self.move_set) == list:
            # Sequential moves.
            for iteration in range(niterations):
                for move in self.move_set:
                    move_sequence.append(move)
        elif type(self.move_set) == dict:
            # Random moves.
            moves = list(self.move_set)
            weights = np.array([self.move_set[move] for move in moves])
            weights /= weights.sum()  # normalize
            move_sequence = np.random.choice(moves, size=niterations, p=weights)

        sampler_state.system = self.thermodynamic_state.system  # HACK!

        # Apply move sequence.
        for move in move_sequence:
            sampler_state = move.apply(self.thermodynamic_state, sampler_state, platform=self.platform)

        # Return the updated sampler state.
        return sampler_state

    def minimize(self, tolerance=None, max_iterations=None, platform=None):
        """
        Minimize the current configuration.

        Parameters
        ----------
        tolerance : simtk.unit.Quantity compatible with kilocalories_per_mole/anstroms, optional, default = 1*kilocalories_per_mole/anstrom
           Tolerance to use for minimization termination criterion.

        max_iterations : int, optional, default = 100
           Maximum number of iterations to use for minimization.

        platform : simtk.openmm.Platform, optional
           Platform to use for minimization.

        Examples
        --------

        >>> # Create a test system
        >>> from openmmtools import testsystems
        >>> test = testsystems.AlanineDipeptideVacuum()
        >>> # Create a sampler state.
        >>> sampler_state = SamplerState(system=test.system, positions=test.positions)
        >>> # Minimize
        >>> sampler_state.minimize()

        """
        timer = Timer()

        if tolerance is None:
            tolerance = 1.0 * unit.kilocalories_per_mole / unit.angstroms

        if max_iterations is None:
            max_iterations = 100

        # Use LocalEnergyMinimizer
        timer.start("Context creation")
        integrator = openmm.VerletIntegrator(1.0*unit.femtosecond)
        context = self.thermodynamic_state.create_context(integrator=integrator,
                                                          platform=platform)
        logger.debug("LocalEnergyMinimizer: platform is %s" % context.getPlatform().getName())
        logger.debug("Minimizing with tolerance %s and %d max. iterations." % (tolerance, max_iterations))
        timer.stop("Context creation")
        timer.start("LocalEnergyMinimizer minimize")
        openmm.LocalEnergyMinimizer.minimize(context, tolerance, max_iterations)
        timer.stop("LocalEnergyMinimizer minimize")

        # Retrieve data.
        self.sampler_state.update_from_context(context)

        timer.report_timing()

    def update_thermodynamic_state(self, thermodynamic_state):
        """
        Update the thermodynamic state.

        Parameters
        ----------
        thermodynamic_state : ThermodynamicState
            Thermodynamic state to sample during MCMC run.

        Examples
        --------

        >>> # Create a test system
        >>> from openmmtools import testsystems
        >>> test = testsystems.AlanineDipeptideVacuum()
        >>> # Create a thermodynamic state.
        >>> import simtk.unit as u
        >>> from openmmmcmc.thermodynamics import ThermodynamicState
        >>> thermodynamic_state = ThermodynamicState(system=test.system, temperature=298*u.kelvin)
        >>> # Create a sampler state.
        >>> sampler_state = SamplerState(system=test.system, positions=test.positions)
        >>> # Create a move set specifying probabilities fo each type of move.
        >>> move_set = { HMCMove(nsteps=10) : 0.5, LangevinDynamicsMove(nsteps=10) : 0.5 }
        >>> # Create MCMC sampler
        >>> sampler = MCMCSampler(thermodynamic_state, move_set=move_set)
        >>> # Run a number of iterations of the sampler.
        >>> updated_sampler_state = sampler.run(sampler_state, 10)

        Update the thermodynamic state.

        >>> thermodynamic_state = ThermodynamicState(system=test.system, temperature=310*u.kelvin)
        >>> sampler.update_thermodynamic_state(thermodynamic_state)

        """

        # Store thermodynamic state.
        self.thermodynamic_state = thermodynamic_state


# =============================================================================
# LANGEVIN DYNAMICS MOVE
# =============================================================================

class LangevinDynamicsMove(object):
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
    platform : simtk.openmm.Platform, optional
        Platform to use for Context creation. If None, OpenMM selects
        the fastest available (default is None).

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
    platform : simtk.openmm.Platform
        Platform to use for Context creation. If None, OpenMM selects
        the fastest available.

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
    >>> sampler_state = SamplerState(system=test.system, positions=test.positions)
    >>> thermodynamic_state = ThermodynamicState(system=test.system,
    ...                                          temperature=298*unit.kelvin)
    >>> move.apply(thermodynamic_state, sampler_state)
    >>> np.allclose(sampler_state.positions, test.positions)
    False

    """

    def __init__(self, timestep=1.0*unit.femtosecond, collision_rate=10.0/unit.picoseconds,
                 n_steps=1000, reassign_velocities=False, platform=None):
        self.timestep = timestep
        self.collision_rate = collision_rate
        self.n_steps = n_steps
        self.reassign_velocities = reassign_velocities
        self.platform = platform

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

        timer = Timer()

        # Create integrator.
        integrator = openmm.LangevinIntegrator(thermodynamic_state.temperature,
                                               self.collision_rate, self.timestep)

        # Random number seed.
        seed = np.random.randint(_RANDOM_SEED_MAX)
        integrator.setRandomNumberSeed(seed)

        # Create context.
        timer.start("Context Creation")
        context = thermodynamic_state.create_context(integrator, self.platform)
        timer.stop("Context Creation")
        logger.debug("LangevinDynamicMove: Context created, platform is {}".format(
            context.getPlatform().getName()))

        if self.reassign_velocities:
            # Assign Maxwell-Boltzmann velocities.
            context.setVelocitiesToTemperature(thermodynamic_state.temperature)

        # Run dynamics.
        timer.start("step()")
        integrator.step(self.n_steps)
        timer.stop("step()")

        # Get updated sampler state.
        timer.start("update_sampler_state")
        sampler_state.update_from_context(context)
        timer.start("update_sampler_state")

        # Clean up.
        del context

        timer.report_timing()


# =============================================================================
# GENERALIZED HYBRID MONTE CARLO MOVE
# =============================================================================

class GHMCMove(object):
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
    platform : simtk.openmm.Platform, optional
        Platform to use for Context creation. If None, OpenMM selects
        the fastest available (default is None).

    Attributes
    ----------
    timestep : simtk.unit.Quantity
        The timestep to use for Langevin integration (time units).
    collision_rate : simtk.unit.Quantity
        The collision rate with fictitious bath particles (1/time units).
    n_steps : int
        The number of integration timesteps to take each time the move
        is applied.
    platform : simtk.openmm.Platform
        Platform to use for Context creation. If None, OpenMM selects
        the fastest available.
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
    >>> sampler_state = SamplerState(system=test.system, positions=test.positions)
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
    >>> sampler_state = SamplerState(system=test.system, positions=test.positions)
    >>> thermodynamic_state = ThermodynamicState(system=test.system,
    ...                                          temperature=298*unit.kelvin)
    >>> move.apply(thermodynamic_state, sampler_state)
    >>> np.allclose(sampler_state.positions, test.positions)
    False

    """

    def __init__(self, timestep=1.0*unit.femtosecond, collision_rate=20.0/unit.picoseconds,
                 n_steps=1000, platform=None):
        self.timestep = timestep
        self.collision_rate = collision_rate
        self.n_steps = n_steps
        self.platform = platform

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

        timer = Timer()

        # Create integrator.
        integrator = integrators.GHMCIntegrator(temperature=thermodynamic_state.temperature,
                                                collision_rate=self.collision_rate,
                                                timestep=self.timestep)

        # Random number seed.
        seed = np.random.randint(_RANDOM_SEED_MAX)
        integrator.setRandomNumberSeed(seed)

        # Create context.
        timer.start("Context Creation")
        context = thermodynamic_state.create_context(integrator, platform=self.platform)
        timer.stop("Context Creation")

        # TODO: Enforce constraints?
        # tol = 1.0e-8
        # context.applyConstraints(tol)
        # context.applyVelocityConstraints(tol)

        # Run dynamics.
        timer.start("step()")
        integrator.step(self.n_steps)
        timer.stop("step()")

        # Get updated sampler state.
        timer.start("update_sampler_state")
        sampler_state.update_from_context(context)
        timer.start("update_sampler_state")

        # Accumulate acceptance statistics.
        ghmc_global_variables = {integrator.getGlobalVariableName(index): index
                                 for index in range(integrator.getNumGlobalVariables())}
        n_accepted = integrator.getGlobalVariable(ghmc_global_variables['naccept'])
        n_attempted = integrator.getGlobalVariable(ghmc_global_variables['ntrials'])
        self.n_accepted += n_accepted
        self.n_attempted += n_attempted

        # Clean up.
        del context

        timer.report_timing()


# =============================================================================
# HYBRID MONTE CARLO MOVE
# =============================================================================

class HMCMove(object):
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
    platform : simtk.openmm.Platform, optional
        Platform to use for Context creation. If None, OpenMM selects the fastest
        available (default is None).

    Attributes
    ----------
    timestep : simtk.unit.Quantity
       The timestep to use for HMC dynamics, which uses velocity Verlet following
       velocity randomization (time units).
    n_steps : int
       The number of dynamics steps to take before Metropolis acceptance/rejection.
    platform : simtk.openmm.Platform
        Platform to use for Context creation. If None, OpenMM selects the fastest
        available.

    Examples
    --------
    First we need to create the thermodynamic state and the sampler
    state to propagate. Here we create an alanine dipeptide system
    in vacuum.

    >>> from simtk import unit
    >>> from openmmtools import testsystems
    >>> from openmmtools.states import ThermodynamicState, SamplerState
    >>> test = testsystems.AlanineDipeptideVacuum()
    >>> sampler_state = SamplerState(system=test.system, positions=test.positions)
    >>> thermodynamic_state = ThermodynamicState(system=test.system, temperature=298*unit.kelvin)

    Create a GHMC move with default parameters.

    >>> move = HMCMove()

    or create a GHMC move with specified parameters.

    >>> move = HMCMove(timestep=0.5*unit.femtoseconds, nsteps=10)

    Perform one update of the sampler state. The sampler state is updated
    with the new state.

    >>> move.apply(thermodynamic_state, sampler_state)
    >>> np.allclose(sampler_state.positions, test.positions)
    False

    The same move can be applied to a different state, here an ideal gas.

    >>> test = testsystems.IdealGas()
    >>> sampler_state = SamplerState(system=test.system, positions=test.positions)
    >>> thermodynamic_state = ThermodynamicState(system=test.system,
    ...                                          temperature=298*unit.kelvin)
    >>> move.apply(thermodynamic_state, sampler_state)
    >>> np.allclose(sampler_state.positions, test.positions)
    False

    """

    def __init__(self, timestep=1.0*unit.femtosecond, n_steps=1000, platform=None):
        self.timestep = timestep
        self.n_steps = n_steps
        self.platform = platform

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
        timer = Timer()

        # Create integrator.
        integrator = integrators.HMCIntegrator(temperature=thermodynamic_state.temperature,
                                               timestep=self.timestep, n_steps=self.n_steps)

        # Random number seed.
        seed = np.random.randint(_RANDOM_SEED_MAX)
        integrator.setRandomNumberSeed(seed)

        # Create context.
        timer.start("Context Creation")
        context = thermodynamic_state.create_context(integrator, platform=self.platform)
        timer.stop("Context Creation")

        # Run dynamics.
        # Note that ONE step of this integrator is equal to self.n_steps
        # of velocity Verlet dynamics followed by Metropolis accept/reject.
        timer.start("HMC integration")
        integrator.step(1)
        timer.stop("HMC integration")

        # Get sampler state.
        timer.start("updated_sampler_state")
        sampler_state.update_from_context(context)
        timer.stop("updated_sampler_state")

        # Clean up.
        del context

        timer.report_timing()


# =============================================================================
# MONTE CARLO BAROSTAT MOVE
# =============================================================================

class MonteCarloBarostatMove(object):
    """Monte Carlo barostat move.

    This move makes one or more attempts to update the box volume using
    Monte Carlo updates.

    Parameters
    ----------
    n_attempts : int, optional
        The number of Monte Carlo attempts to make to adjust the box
        volume (default is 5).
    platform : simtk.openmm.Platform, optional
        Platform to use for Context creation. If None, OpenMM selects
        the fastest available (default is None).

    Attributes
    ----------
    n_attempts : int
        The number of Monte Carlo attempts to make to adjust the box
        volume..
    platform : simtk.openmm.Platform
        Platform to use for Context creation. If None, OpenMM selects
        the fastest available.

    Examples
    --------
    The thermodynamic state must be barostated by a MonteCarloBarostat
    force. The class ThermodynamicState takes care of adding one when
    we specify the pressure in its constructor.

    >>> from simtk import unit
    >>> from openmmtools import testsystems
    >>> from openmmtools.states import ThermodynamicState, SamplerState
    >>> test = testsystems.AlanineDipeptideExplicit()
    >>> sampler_state = SamplerState(system=test.system, positions=test.positions)
    >>> thermodynamic_state = ThermodynamicState(system=test.system, temperature=298*unit.kelvin,
    ...                                          pressure=1.0*unit.atmosphere)

    Create a MonteCarloBarostatMove move with default parameters.

    >>> move = MonteCarloBarostatMove()

    or create a GHMC move with specified parameters.

    >>> MonteCarloBarostatMove(n_attempts=2)

    Perform one update of the sampler state. The sampler state is updated
    with the new state.

    >>> move.apply(thermodynamic_state, sampler_state)
    >>> np.allclose(sampler_state.positions, test.positions)
    False

    """

    def __init__(self, n_attempts=5, platform=None):
        self.n_attempts = n_attempts
        self.platform = platform

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
        timer = Timer()

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

        # Random number seed.
        seed = np.random.randint(_RANDOM_SEED_MAX)
        barostat.setRandomNumberSeed(seed)
        thermodynamic_state.barostat = barostat

        # Create integrator.
        integrator = integrators.DummyIntegrator()

        # Create context.
        timer.start("Context Creation")
        context = thermodynamic_state.create_context(integrator, platform=self.platform)
        timer.stop("Context Creation")

        # Run update.
        # Note that ONE step of this integrator is equal to self.nsteps
        # of velocity Verlet dynamics followed by Metropolis accept/reject.
        timer.start("step(1)")
        integrator.step(self.n_attempts)
        timer.stop("step(1)")

        # Get sampler state.
        timer.start("update_sampler_state")
        sampler_state.update_from_context(context)
        timer.stop("update_sampler_state")

        # Clean up.
        del context

        # Restore frequency of barostat.
        if old_barostat_frequency != 1:
            barostat.setFrequency(old_barostat_frequency)
            thermodynamic_state.barostat = barostat

        timer.report_timing()


# =============================================================================
# MAIN AND TESTS
# =============================================================================

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
