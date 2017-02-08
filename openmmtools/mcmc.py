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

import numpy as np

from simtk import openmm, unit

from openmmtools import integrators
from openmmmcmc import thermodynamics
from openmmmcmc.timing import Timer

from abc import abstractmethod

import logging
logger = logging.getLogger(__name__)


# =============================================================================
# MODULE CONSTANTS
# =============================================================================

_RANDOM_SEED_MAX = np.iinfo(np.int32).max  # maximum random number seed value


# =============================================================================
# MCMC sampler state
# =============================================================================

class SamplerState(object):
    """
    Sampler state for MCMC move representing everything that may be allowed to
    change during the simulation.

    Parameters
    ----------
    system : simtk.openmm.System
       Current system specifying force calculations.
    positions : array of simtk.unit.Quantity compatible with nanometers
       Particle positions.
    velocities : optional, array of simtk.unit.Quantity compatible with nanometers/picoseconds
       Particle velocities.
    box_vectors : optional, 3x3 array of simtk.unit.Quantity compatible with nanometers
       Current box vectors.

    Fields
    ------
    system : simtk.openmm.System
       Current system specifying force calculations.
    positions : array of simtk.unit.Quantity compatible with nanometers
       Particle positions.
    velocities : optional, array of simtk.unit.Quantity compatible with nanometers/picoseconds
       Particle velocities.
    box_vectors : optional, 3x3 array of simtk.unit.Quantity compatible with nanometers
       Current box vectors.
    potential_energy : optional, simtk.unit.Quantity compatible with kilocalories_per_mole
       Current potential energy.
    kinetic_energy : optional, simtk.unit.Quantity compatible with kilocalories_per_mole
       Current kinetic energy.
    total_energy : optional, simtk.unit.Quantity compatible with kilocalories_per_mole
       Current total energy.
    platform : optional, simtk.openmm.Platform
       Platform to use for Context creation to initialize sampler state.

    Examples
    --------

    Create a sampler state for a system with box vectors.

    >>> # Create a test system
    >>> from openmmtools import testsystems
    >>> test = testsystems.LennardJonesFluid()
    >>> # Create a sampler state manually.
    >>> box_vectors = test.system.getDefaultPeriodicBoxVectors()
    >>> sampler_state = SamplerState(system=test.system, positions=test.positions, box_vectors=box_vectors)

    Create a sampler state for a system without box vectors.

    >>> # Create a test system
    >>> from openmmtools import testsystems
    >>> test = testsystems.LennardJonesCluster()
    >>> # Create a sampler state manually.
    >>> sampler_state = SamplerState(system=test.system, positions=test.positions)

    TODO:
    * Can we remove the need to create a Context in initializing the sampler state by using the Reference platform and skipping energy calculations?

    """
    def __init__(self, system, positions, velocities=None, box_vectors=None, platform=None):
        self.system = copy.deepcopy(system)
        self.positions = positions
        self.velocities = velocities
        self.box_vectors = box_vectors

        # Create Context.
        context = self.createContext(platform=platform)

        # Get state.
        openmm_state = context.getState(getPositions=True, getVelocities=True, getEnergy=True)

        # Populate context.
        self.positions = openmm_state.getPositions(asNumpy=True)
        self.velocities = openmm_state.getVelocities(asNumpy=True)
        self.box_vectors = openmm_state.getPeriodicBoxVectors(asNumpy=False)
        self.potential_energy = openmm_state.getPotentialEnergy()
        self.kinetic_energy = openmm_state.getKineticEnergy()
        self.total_energy = self.potential_energy + self.kinetic_energy
        self.volume = thermodynamics.volume(self.box_vectors)

        # Clean up.
        del context

    @classmethod
    def createFromContext(cls, context):
        """
        Create an SamplerState object from the information in a current OpenMM Context object.

        Parameters
        ----------
        context : simtk.openmm.Context
           The Context object from which to create a sampler state.

        Returns
        -------
        sampler_state : SamplerState
           The sampler state containing positions, velocities, and box vectors.

        Examples
        --------

        >>> # Create a test system
        >>> from openmmtools import testsystems
        >>> test = testsystems.AlanineDipeptideVacuum()
        >>> # Create a Context.
        >>> import simtk.openmm as mm
        >>> import simtk.unit as u
        >>> integrator = mm.VerletIntegrator(1.0 * u.femtoseconds)
        >>> platform = mm.Platform.getPlatformByName('Reference')
        >>> context = mm.Context(test.system, integrator, platform)
        >>> # Set positions and velocities.
        >>> context.setPositions(test.positions)
        >>> context.setVelocitiesToTemperature(298 * u.kelvin)
        >>> # Create a sampler state from the Context.
        >>> sampler_state = SamplerState.createFromContext(context)
        >>> # Clean up.
        >>> del context, integrator

        """
        # Get state.
        openmm_state = context.getState(getPositions=True, getVelocities=True, getEnergy=True)

        # Create new object, bypassing init.
        self = SamplerState.__new__(cls)

        # Populate context.
        self.system = copy.deepcopy(context.getSystem())
        self.positions = openmm_state.getPositions(asNumpy=True)
        self.velocities = openmm_state.getVelocities(asNumpy=True)
        self.box_vectors = openmm_state.getPeriodicBoxVectors(asNumpy=True)
        self.potential_energy = openmm_state.getPotentialEnergy()
        self.kinetic_energy = openmm_state.getKineticEnergy()
        self.total_energy = self.potential_energy + self.kinetic_energy
        self.volume = thermodynamics.volume(self.box_vectors)

        return self

    def createContext(self, integrator=None, platform=None):
        """
        Create an OpenMM Context object from the current sampler state.

        Parameters
        ----------
        integrator : simtk.openmm.Integrator, optional, default=None
           The integrator to use for Context creation.
           If not specified, a VerletIntegrator with 1 fs timestep is created.
        platform : simtk.openmm.Platform, optional, default=None
           If specified, the Platform to use for context creation.

        Returns
        -------
        context : simtk.openmm.Context
           The created OpenMM Context object

        Notes
        -----
        If the selected or default platform fails, the CPU and Reference platforms will be tried, in that order.

        Examples
        --------

        Create a context for a system with periodic box vectors.

        >>> # Create a test system
        >>> from openmmtools import testsystems
        >>> test = testsystems.LennardJonesFluid()
        >>> # Create a sampler state manually.
        >>> box_vectors = test.system.getDefaultPeriodicBoxVectors()
        >>> sampler_state = SamplerState(positions=test.positions, box_vectors=box_vectors, system=test.system)
        >>> # Create a Context.
        >>> import simtk.openmm as mm
        >>> import simtk.unit as u
        >>> integrator = mm.VerletIntegrator(1.0*u.femtoseconds)
        >>> context = sampler_state.createContext(integrator)
        >>> # Clean up.
        >>> del context

        Create a context for a system without periodic box vectors.

        >>> # Create a test system
        >>> from openmmtools import testsystems
        >>> test = testsystems.LennardJonesCluster()
        >>> # Create a sampler state manually.
        >>> sampler_state = SamplerState(positions=test.positions, system=test.system)
        >>> # Create a Context.
        >>> import simtk.openmm as mm
        >>> import simtk.unit as u
        >>> integrator = mm.VerletIntegrator(1.0*u.femtoseconds)
        >>> context = sampler_state.createContext(integrator)
        >>> # Clean up.
        >>> del context

        TODO
        ----
        * Generalize fallback platform order to [CUDA, OpenCL, CPU, Reference] ordering.

        """

        if not self.system:
            raise Exception("SamplerState must have a 'system' object specified to create a Context")

        # Use a Verlet integrator if none is specified.
        if integrator is None:
            integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)

        # Create a Context.
        if platform:
            context = openmm.Context(self.system, integrator, platform)
        else:
            context = openmm.Context(self.system, integrator)

        # Set box vectors, if specified.
        if (self.box_vectors is not None):
            try:
                # try tuple of box vectors
                context.setPeriodicBoxVectors(self.box_vectors[0], self.box_vectors[1], self.box_vectors[2])
            except:
                # try numpy 3x3 matrix of box vectors
                context.setPeriodicBoxVectors(self.box_vectors[0,:], self.box_vectors[1,:], self.box_vectors[2,:])

        # Set positions.
        context.setPositions(self.positions)

        # Set velocities, if specified.
        if (self.velocities is not None):
            context.setVelocities(self.velocities)

        return context

    def minimize(self, tolerance=None, maxIterations=None, platform=None):
        """
        Minimize the current configuration.

        Parameters
        ----------
        tolerance : simtk.unit.Quantity compatible with kilocalories_per_mole/anstroms, optional, default = 1*kilocalories_per_mole/anstrom
           Tolerance to use for minimization termination criterion.

        maxIterations : int, optional, default = 100
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

        if (tolerance is None):
            tolerance = 1.0 * unit.kilocalories_per_mole / u.angstroms

        if (maxIterations is None):
            maxIterations = 100

        # Use LocalEnergyMinimizer
        from simtk.openmm import LocalEnergyMinimizer
        timer.start("Context creation")
        context = self.createContext(platform=platform)
        logger.debug("LocalEnergyMinimizer: platform is %s" % context.getPlatform().getName())
        logger.debug("Minimizing with tolerance %s and %d max. iterations." % (tolerance, maxIterations))
        timer.stop("Context creation")
        timer.start("LocalEnergyMinimizer minimize")
        LocalEnergyMinimizer.minimize(context, tolerance, maxIterations)
        timer.stop("LocalEnergyMinimizer minimize")

        # Retrieve data.
        sampler_state = SamplerState.createFromContext(context)
        self.positions = sampler_state.positions
        self.potential_energy = sampler_state.potential_energy
        self.total_energy = sampler_state.total_energy

        del context

        timer.report_timing()

        return

    def has_nan(self):
        """Return True if any of the generalized coordinates are nan.

        Notes
        -----

        Currently checks only the positions.
        """
        x = self.positions / unit.nanometers

        if np.any(np.isnan(x)):
            return True
        else:
            return False


# =============================================================================
# Monte Carlo Move abstract base class
# =============================================================================

class MCMCMove(object):
    """
    Markov chain Monte Carlo (MCMC) move abstract base class.

    Markov chain Monte Carlo (MCMC) simulations are constructed from a set of derived objects.

    """

    @abstractmethod
    def apply(self, thermodynamic_state, sampler_state, platform=None):
        """
        Apply the MCMC move.

        Parameters
        ----------
        thermodynamic_state : ThermodynamicState
           The thermodynamic state to use when applying the MCMC move
        sampler_state : SamplerState
           The sampler state to apply the move to
        platform : simtk.openmm.Platform, optional, default = None
           The platform to use.

        Returns
        -------
        updated_sampler_state : SamplerState
           The updated sampler state


        """
        pass


# =============================================================================
# Markov chain Monte Carlo sampler
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
# Langevin dynamics move
# =============================================================================

class LangevinDynamicsMove(MCMCMove):
    """
    Langevin dynamics segment as a (pseudo) Monte Carlo move.

    This move assigns a velocity from the Maxwell-Boltzmann distribution and executes a number
    of Maxwell-Boltzmann steps to propagate dynamics.  This is not a *true* Monte Carlo move,
    in that the generation of the correct distribution is only exact in the limit of infinitely
    small timestep; in other words, the discretization error is assumed to be negligible. Use
    HybridMonteCarloMove instead to ensure the exact distribution is generated.

    Warning
    -------
    No Metropolization is used to ensure the correct phase space distribution is sampled.
    This means that timestep-dependent errors will remain uncorrected, and are amplified with larger timesteps.
    Use this move at your own risk!

    Examples
    --------

    >>> # Create a test system
    >>> from openmmtools import testsystems
    >>> test = testsystems.AlanineDipeptideVacuum()
    >>> # Create a sampler state.
    >>> sampler_state = SamplerState(system=test.system, positions=test.positions)
    >>> # Create a thermodynamic state.
    >>> from openmmmcmc.thermodynamics import ThermodynamicState
    >>> thermodynamic_state = ThermodynamicState(system=test.system, temperature=298*u.kelvin)
    >>> # Create a LangevinDynamicsMove
    >>> move = LangevinDynamicsMove(nsteps=10)
    >>> # Perform one update of the sampler state.
    >>> updated_sampler_state = move.apply(thermodynamic_state, sampler_state)

    """

    def __init__(self, timestep=1.0*unit.femtosecond, collision_rate=10.0/unit.picoseconds,
                 nsteps=1000, reassign_velocities=False):
        """
        Parameters
        ----------
        timestep : simtk.unit.Quantity compatible with femtoseconds, optional, default = 1*simtk.unit.femtoseconds
            The timestep to use for Langevin integration.
        collision_rate : simtk.unit.Quantity compatible with 1/picoseconds, optional, default = 10/simtk.unit.picoseconds
            The collision rate with fictitious bath particles.
        nsteps : int, optional, default = 1000
            The number of integration timesteps to take each time the move is applied.
        reassign_velocities : bool, optional, default = False
            If True, the velocities will be reassigned from the Maxwell-Boltzmann distribution at the beginning of the move.

        Note
        ----
        The temperature of the thermodynamic state is used in Langevin dynamics.
        If a barostat is present, the temperature and pressure will be set to the thermodynamic state being simulated.

        Examples
        --------

        Create a Langevin move with default parameters.

        >>> move = LangevinDynamicsMove()

        Create a Langevin move with specified parameters.

        >>> move = LangevinDynamicsMove(timestep=0.5*u.femtoseconds, collision_rate=20.0/u.picoseconds, nsteps=100)

        """

        self.timestep = timestep
        self.collision_rate = collision_rate
        self.nsteps = nsteps
        self.reassign_velocities = reassign_velocities

        return

    def apply(self, thermodynamic_state, sampler_state, platform=None):
        """
        Apply the Langevin dynamics MCMC move.

        Parameters
        ----------
        thermodynamic_state : ThermodynamicState
           The thermodynamic state to use when applying the MCMC move
        sampler_state : SamplerState
           The sampler state to apply the move to
        platform : simtk.openmm.Platform, optional, default = None
           If not None, the specified platform will be used.

        Returns
        -------
        updated_sampler_state : SamplerState
           The updated sampler state

        Examples
        --------

        Alanine dipeptide in vacuum.

        >>> # Create a test system
        >>> from openmmtools import testsystems
        >>> test = testsystems.AlanineDipeptideVacuum()
        >>> # Create a sampler state.
        >>> sampler_state = SamplerState(system=test.system, positions=test.positions)
        >>> # Create a thermodynamic state.
        >>> from openmmmcmc.thermodynamics import ThermodynamicState
        >>> thermodynamic_state = ThermodynamicState(system=test.system, temperature=298*u.kelvin)
        >>> # Create a LangevinDynamicsMove
        >>> move = LangevinDynamicsMove(nsteps=10, timestep=0.5*u.femtoseconds, collision_rate=20.0/u.picoseconds)
        >>> # Perform one update of the sampler state.
        >>> updated_sampler_state = move.apply(thermodynamic_state, sampler_state)

        Ideal gas.

        >>> # Create a test system
        >>> from openmmtools import testsystems
        >>> test = testsystems.IdealGas()
        >>> # Create a sampler state.
        >>> sampler_state = SamplerState(system=test.system, positions=test.positions)
        >>> # Create a thermodynamic state.
        >>> from openmmmcmc.thermodynamics import ThermodynamicState
        >>> thermodynamic_state = ThermodynamicState(system=test.system, temperature=298*u.kelvin)
        >>> # Create a LangevinDynamicsMove
        >>> move = LangevinDynamicsMove(nsteps=500, timestep=0.5*u.femtoseconds, collision_rate=20.0/u.picoseconds)
        >>> # Perform one update of the sampler state.
        >>> updated_sampler_state = move.apply(thermodynamic_state, sampler_state)

        """

        timer = Timer()

        # Check if the system contains a barostat.
        system = sampler_state.system
        forces = {system.getForce(index).__class__.__name__: system.getForce(index)
                  for index in range(system.getNumForces())}
        barostat = None
        if 'MonteCarloBarostat' in forces:
            barostat = forces['MonteCarloBarostat']
            barostat.setDefaultTemperature(thermodynamic_state.temperature)
            parameter_name = barostat.Pressure()
            if thermodynamic_state.pressure is None:
                raise Exception('MonteCarloBarostat is present but no pressure specified in thermodynamic state.')

        # Create integrator.
        integrator = openmm.LangevinIntegrator(thermodynamic_state.temperature, self.collision_rate, self.timestep)

        # Random number seed.
        seed = np.random.randint(_RANDOM_SEED_MAX)
        integrator.setRandomNumberSeed(seed)

        # Create context.
        timer.start("Context Creation")
        context = sampler_state.createContext(integrator, platform=platform)
        timer.stop("Context Creation")
        logger.debug("LangevinDynamicMove: Context created, platform is %s" % context.getPlatform().getName())

        # Set pressure, if barostat is included.
        if barostat is not None:
            context.setParameter(parameter_name,
                                 thermodynamic_state.pressure.value_in_unit_system(unit.md_unit_system))

        if self.reassign_velocities:
            # Assign Maxwell-Boltzmann velocities.
            context.setVelocitiesToTemperature(thermodynamic_state.temperature)

        # Run dynamics.
        timer.start("step()")
        integrator.step(self.nsteps)
        timer.stop("step()")

        # Get updated sampler state.
        timer.start("update_sampler_state")
        updated_sampler_state = SamplerState.createFromContext(context)
        timer.start("update_sampler_state")

        # Clean up.
        del context

        timer.report_timing()

        return updated_sampler_state


# =============================================================================
# Genaralized Hybrid Monte Carlo (a form of Metropolized Langevin dynamics)
# =============================================================================

class GHMCMove(MCMCMove):
    """
    Generalized hybrid Monte Carlo (GHMC) Markov chain Monte Carlo move

    This move uses generalized Hybrid Monte Carlo (GHMC), a form of Metropolized Langevin
    dynamics, to propagate the system.

    References
    ----------
    Lelievre T, Stoltz G, and Rousset M. Free Energy Computations: A Mathematical Perspective
    http://www.amazon.com/Free-Energy-Computations-Mathematical-Perspective/dp/1848162472

    Examples
    --------

    >>> # Create a test system
    >>> from openmmtools import testsystems
    >>> test = testsystems.AlanineDipeptideVacuum()
    >>> # Create a sampler state.
    >>> sampler_state = SamplerState(system=test.system, positions=test.positions)
    >>> # Minimize.
    >>> sampler_state.minimize()
    >>> # Create a thermodynamic state.
    >>> from openmmmcmc.thermodynamics import ThermodynamicState
    >>> thermodynamic_state = ThermodynamicState(system=test.system, temperature=298*u.kelvin)
    >>> # Create a GHMC move
    >>> move = GHMCMove(nsteps=10)
    >>> # Perform one update of the sampler state.
    >>> updated_sampler_state = move.apply(thermodynamic_state, sampler_state)

    """

    def __init__(self, timestep=1.0*unit.femtosecond,
                 collision_rate=20.0/unit.picoseconds, nsteps=1000):
        """
        Parameters
        ----------
        timestep : simtk.unit.Quantity compatible with femtoseconds, optional, default = 1*simtk.unit.femtoseconds
            The timestep to use for Langevin integration.
        collision_rate : simtk.unit.Quantity compatible with 1/picoseconds, optional, default = 10/simtk.unit.picoseconds
            The collision rate with fictitious bath particles.
        nsteps : int, optional, default = 1000
            The number of integration timesteps to take each time the move is applied.

        Note
        ----
        The temperature of the thermodynamic state is used.

        Examples
        --------

        Create a GHMC move with default parameters.

        >>> move = GHMCMove()

        Create a GHMC move with specified parameters.

        >>> move = GHMCMove(timestep=0.5*u.femtoseconds, collision_rate=20.0/u.picoseconds, nsteps=100)

        """

        self.timestep = timestep
        self.collision_rate = collision_rate
        self.nsteps = nsteps

        self.reset_statistics()

        return

    def reset_statistics(self):
        """
        Reset the internal statistics of number of accepted and attempted moves.

        Examples
        --------

        >>> # Create a test system
        >>> from openmmtools import testsystems
        >>> test = testsystems.AlanineDipeptideVacuum()
        >>> # Create a sampler state.
        >>> sampler_state = SamplerState(system=test.system, positions=test.positions)
        >>> # Create a thermodynamic state.
        >>> from openmmmcmc.thermodynamics import ThermodynamicState
        >>> thermodynamic_state = ThermodynamicState(system=test.system, temperature=298*u.kelvin)
        >>> # Create a LangevinDynamicsMove
        >>> move = GHMCMove(nsteps=10, timestep=1.0*u.femtoseconds, collision_rate=20.0/u.picoseconds)
        >>> # Perform one update of the sampler state.
        >>> updated_sampler_state = move.apply(thermodynamic_state, sampler_state)

        Reset statistics.

        >>> move.reset_statistics()

        """

        self.naccepted = 0  # number of accepted steps
        self.nattempted = 0  # number of attempted steps

        return

    def get_statistics(self):
        """
        Return the current acceptance/rejection statistics of the sampler.

        Returns
        -------
        naccepted : int
           The number of accepted steps
        nattempted : int
           The number of attempted steps
        faction_accepted : float
           The fraction of steps accepted.

        Examples
        --------

        >>> # Create a test system
        >>> from openmmtools import testsystems
        >>> test = testsystems.AlanineDipeptideVacuum()
        >>> # Create a sampler state.
        >>> sampler_state = SamplerState(system=test.system, positions=test.positions)
        >>> # Create a thermodynamic state.
        >>> from openmmmcmc.thermodynamics import ThermodynamicState
        >>> thermodynamic_state = ThermodynamicState(system=test.system, temperature=298*u.kelvin)
        >>> # Create a LangevinDynamicsMove
        >>> move = GHMCMove(nsteps=10, timestep=1.0*u.femtoseconds, collision_rate=20.0/u.picoseconds)
        >>> # Perform one update of the sampler state.
        >>> updated_sampler_state = move.apply(thermodynamic_state, sampler_state)

        Get statistics.

        >>> [naccepted, nattempted, fraction_accepted] = move.get_statistics()

        """
        return self.naccepted, self.nattempted, float(self.naccepted) / float(self.nattempted)

    def apply(self, thermodynamic_state, sampler_state, platform=None):
        """
        Apply the GHMC MCMC move.

        Parameters
        ----------
        thermodynamic_state : ThermodynamicState
           The thermodynamic state to use when applying the MCMC move
        sampler_state : SamplerState
           The sampler state to apply the move to
        platform : simtk.openmm.Platform, optional, default = None
           If not None, the specified platform will be used.

        Returns
        -------
        updated_sampler_state : SamplerState
           The updated sampler state

        Examples
        --------

        >>> # Create a test system
        >>> from openmmtools import testsystems
        >>> test = testsystems.AlanineDipeptideVacuum()
        >>> # Create a sampler state.
        >>> sampler_state = SamplerState(system=test.system, positions=test.positions)
        >>> # Create a thermodynamic state.
        >>> from openmmmcmc.thermodynamics import ThermodynamicState
        >>> thermodynamic_state = ThermodynamicState(system=test.system, temperature=298*u.kelvin)
        >>> # Create a LangevinDynamicsMove
        >>> move = GHMCMove(nsteps=10, timestep=1.0*u.femtoseconds, collision_rate=20.0/u.picoseconds)
        >>> # Perform one update of the sampler state.
        >>> updated_sampler_state = move.apply(thermodynamic_state, sampler_state)

        """

        timer = Timer()

        # Create integrator.
        integrator = integrators.GHMCIntegrator(temperature=thermodynamic_state.temperature,
                                                collision_rate=self.collision_rate, timestep=self.timestep)

        # Random number seed.
        seed = np.random.randint(_RANDOM_SEED_MAX)
        integrator.setRandomNumberSeed(seed)

        # Create context.
        timer.start("Context Creation")
        context = sampler_state.createContext(integrator, platform=platform)
        timer.stop("Context Creation")

        # TODO: Enforce constraints?
        # tol = 1.0e-8
        # context.applyConstraints(tol)
        # context.applyVelocityConstraints(tol)

        # Run dynamics.
        timer.start("step()")
        integrator.step(self.nsteps)
        timer.stop("step()")

        # Get updated sampler state.
        timer.start("update_sampler_state")
        updated_sampler_state = SamplerState.createFromContext(context)
        timer.start("update_sampler_state")

        # Accumulate acceptance statistics.
        ghmc_global_variables = {integrator.getGlobalVariableName(index): index
                                 for index in range(integrator.getNumGlobalVariables())}
        naccepted = integrator.getGlobalVariable(ghmc_global_variables['naccept'])
        nattempted = integrator.getGlobalVariable(ghmc_global_variables['ntrials'])
        self.naccepted += naccepted
        self.nattempted += nattempted

        # Clean up.
        del context

        timer.report_timing()

        return updated_sampler_state


# =============================================================================
# Hybrid Monte Carlo move
# =============================================================================

class HMCMove(MCMCMove):
    """
    Hybrid Monte Carlo dynamics.

    This move assigns a velocity from the Maxwell-Boltzmann distribution and executes a number
    of velocity Verlet steps to propagate dynamics.

    Examples
    --------

    >>> # Create a test system
    >>> from openmmtools import testsystems
    >>> test = testsystems.AlanineDipeptideVacuum()
    >>> # Create a sampler state.
    >>> sampler_state = SamplerState(system=test.system, positions=test.positions)
    >>> # Create a thermodynamic state.
    >>> from openmmmcmc.thermodynamics import ThermodynamicState
    >>> thermodynamic_state = ThermodynamicState(system=test.system, temperature=298*u.kelvin)
    >>> # Create an HMC move.
    >>> move = HMCMove(nsteps=10)
    >>> # Perform one update of the sampler state.
    >>> updated_sampler_state = move.apply(thermodynamic_state, sampler_state)

    """

    def __init__(self, timestep=1.0*unit.femtosecond, nsteps=1000):
        """
        Parameters
        ----------
        timestep : simtk.unit.Quantity compatible with femtoseconds, optional, default = 1*femtosecond
           The timestep to use for HMC dynamics (which uses velocity Verlet following velocity randomization)
        nsteps : int, optional, default = 1000
           The number of dynamics steps to take before Metropolis acceptance/rejection.

        Examples
        --------

        Create an HMC move with default timestep and number of steps.

        >>> move = HMCMove()

        Create an HMC move with specified timestep and number of steps.

        >>> move = HMCMove(timestep=0.5*u.femtoseconds, nsteps=500)

        """

        self.timestep = timestep
        self.nsteps = nsteps

        return

    def apply(self, thermodynamic_state, sampler_state, platform=None):
        """
        Apply the MCMC move.

        Parameters
        ----------
        thermodynamic_state : ThermodynamicState
           The thermodynamic state to use when applying the MCMC move
        sampler_state : SamplerState
           The sampler state to apply the move to
        platform : simtk.openmm.Platform, optional, default = None
           If not None, the specified platform will be used.

        Returns
        -------
        updated_sampler_state : SamplerState
           The updated sampler state

        Examples
        --------

        >>> # Create a test system
        >>> from openmmtools import testsystems
        >>> test = testsystems.AlanineDipeptideVacuum()
        >>> # Create a sampler state.
        >>> sampler_state = SamplerState(system=test.system, positions=test.positions)
        >>> # Create a thermodynamic state.
        >>> from openmmmcmc.thermodynamics import ThermodynamicState
        >>> thermodynamic_state = ThermodynamicState(system=test.system, temperature=298*u.kelvin)
        >>> # Create an HMC move.
        >>> move = HMCMove(nsteps=10, timestep=0.5*u.femtoseconds)
        >>> # Perform one update of the sampler state.
        >>> updated_sampler_state = move.apply(thermodynamic_state, sampler_state)

        """

        timer = Timer()

        # Create integrator.
        integrator = integrators.HMCIntegrator(temperature=thermodynamic_state.temperature,
                                               timestep=self.timestep, nsteps=self.nsteps)

        # Random number seed.
        seed = np.random.randint(_RANDOM_SEED_MAX)
        integrator.setRandomNumberSeed(seed)

        # Create context.
        timer.start("Context Creation")
        context = sampler_state.createContext(integrator, platform=platform)
        timer.stop("Context Creation")

        # Run dynamics.
        # Note that ONE step of this integrator is equal to self.nsteps
        # of velocity Verlet dynamics followed by Metropolis accept/reject.
        timer.start("HMC integration")
        integrator.step(1)
        timer.stop("HMC integration")

        # Get sampler state.
        timer.start("updated_sampler_state")
        updated_sampler_state = SamplerState.createFromContext(context)
        timer.stop("updated_sampler_state")

        # Clean up.
        del context

        timer.report_timing()

        # Return updated sampler state.
        return updated_sampler_state


# =============================================================================
# Monte Carlo barostat move
# =============================================================================

class MonteCarloBarostatMove(MCMCMove):
    """
    Monte Carlo barostat move.

    This move makes one or more attempts to update the box volume using Monte Carlo updates.

    Examples
    --------

    >>> # Create a test system
    >>> from openmmtools import testsystems
    >>> test = testsystems.LennardJonesFluid(nparticles=200)
    >>> # Create a sampler state.
    >>> sampler_state = SamplerState(system=test.system, positions=test.positions, box_vectors=test.system.getDefaultPeriodicBoxVectors())
    >>> # Create a thermodynamic state.
    >>> from openmmmcmc.thermodynamics import ThermodynamicState
    >>> thermodynamic_state = ThermodynamicState(system=test.system, temperature=298*u.kelvin, pressure=1*u.atmospheres)
    >>> # Create a move set that includes a Monte Carlo barostat move.
    >>> move_set = [ GHMCMove(nsteps=50), MonteCarloBarostatMove(nattempts=5) ]
    >>> # Simulate on Reference platform.
    >>> import simtk.openmm as mm
    >>> platform = mm.Platform.getPlatformByName('Reference')
    >>> sampler = MCMCSampler(thermodynamic_state, move_set=move_set, platform=platform)
    >>> # Run a number of iterations of the sampler.
    >>> updated_sampler_state = sampler.run(sampler_state, 2)

    """

    def __init__(self, nattempts=5):
        """
        Parameters
        ----------
        nattempts : int
           The number of Monte Carlo attempts to make to adjust the box volume.

        Examples
        --------

        Create a Monte Carlo barostat move with default parameters.

        >>> move = MonteCarloBarostatMove()

        Create a Monte Carlo barostat move with specified parameters.

        >>> move = MonteCarloBarostatMove(nattempts=10)

        """
        self.nattempts = nattempts

        return

    def apply(self, thermodynamic_state, sampler_state, platform=None):
        """
        Apply the MCMC move.

        Parameters
        ----------
        thermodynamic_state : ThermodynamicState
           The thermodynamic state to use when applying the MCMC move
        sampler_state : SamplerState
           The sampler state to apply the move to
        platform : simtk.openmm.Platform, optional, default = None
           If not None, the specified platform will be used.

        Returns
        -------
        updated_sampler_state : SamplerState
           The updated sampler state

        Examples
        --------

        >>> # Create a test system
        >>> from openmmtools import testsystems
        >>> test = testsystems.LennardJonesFluid()
        >>> # Create a sampler state.
        >>> sampler_state = SamplerState(system=test.system, positions=test.positions, box_vectors=test.system.getDefaultPeriodicBoxVectors())
        >>> # Create a thermodynamic state.
        >>> from openmmmcmc.thermodynamics import ThermodynamicState
        >>> thermodynamic_state = ThermodynamicState(system=test.system, temperature=298*u.kelvin, pressure=1*u.atmospheres)
        >>> # Create a Monte Carlo Barostat move.
        >>> move = MonteCarloBarostatMove(nattempts=5)
        >>> # Perform one update of the sampler state.
        >>> updated_sampler_state = move.apply(thermodynamic_state, sampler_state)

        """

        timer = Timer()

        # Make sure system contains a barostat.
        system = sampler_state.system
        forces = {system.getForce(index).__class__.__name__: system.getForce(index)
                  for index in range(system.getNumForces())}
        old_barostat_frequency = None
        if 'MonteCarloBarostat' in forces:
            force = forces['MonteCarloBarostat']
            force.setDefaultTemperature(thermodynamic_state.temperature)
            old_barostat_frequency = force.getFrequency()
            force.setFrequency(1)
            parameter_name = force.Pressure()
        else:
            # Add MonteCarloBarostat.
            force = openmm.MonteCarloBarostat(thermodynamic_state.pressure, thermodynamic_state.temperature, 1)
            system.addForce(force)
            parameter_name = force.Pressure()

        # Create integrator.
        integrator = integrators.DummyIntegrator()

        # Random number seed.
        seed = np.random.randint(_RANDOM_SEED_MAX)
        force.setRandomNumberSeed(seed)

        # Create context.
        timer.start("Context Creation")
        context = sampler_state.createContext(integrator, platform=platform)
        timer.stop("Context Creation")

        # Set pressure.
        context.setParameter(parameter_name, thermodynamic_state.pressure)

        # Run update.
        # Note that ONE step of this integrator is equal to self.nsteps
        # of velocity Verlet dynamics followed by Metropolis accept/reject.
        timer.start("step(1)")
        integrator.step(self.nattempts)
        timer.stop("step(1)")

        # Get sampler state.
        timer.start("update_sampler_state")
        updated_sampler_state = SamplerState.createFromContext(context)
        timer.stop("update_sampler_state")

        # Clean up.
        del context

        # Restore frequency of barostat.
        if old_barostat_frequency:
            force.setFrequency(old_barostat_frequency)

        timer.report_timing()

        # Return updated sampler state.
        return updated_sampler_state


# =============================================================================
# MAIN AND TESTS
# =============================================================================

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
