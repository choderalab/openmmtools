#!/usr/local/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Test custom integrators.

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import re
import numpy

from simtk import unit
from simtk import openmm
from openmmtools import integrators, testsystems
from functools import partial

#=============================================================================================
# CONSTANTS
#=============================================================================================

kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA

#=============================================================================================
# UTILITY SUBROUTINES
#=============================================================================================

def check_stability(integrator, test, platform=None, nsteps=100, temperature=300.0*unit.kelvin):
   """
   Check that the simulation does not explode over a number integration steps.

   Parameters
   ----------
   integrator : simtk.openmm.Integrator
      The integrator to test.
   test : testsystem
      The testsystem to test.

   """
   kT = kB * temperature

   # Create Context and initialize positions.
   if platform:
      context = openmm.Context(test.system, integrator, platform)
   else:
      context = openmm.Context(test.system, integrator)
   context.setPositions(test.positions)
   context.setVelocitiesToTemperature(temperature) # TODO: Make deterministic.

   # Take a number of steps.
   integrator.step(nsteps)

   # Check that simulation has not exploded.
   state = context.getState(getEnergy=True)
   potential = state.getPotentialEnergy() / kT
   if numpy.isnan(potential):
      raise Exception("Potential energy for integrator %s became NaN." % integrator.__doc__)

   del context

   return

#=============================================================================================
# TESTS
#=============================================================================================

def test_bitwise_reversible_velocity_verlet():
   """
   Test bitwise-reversible velocity Verlet integrator.

   """
   from openmmtools import testsystems
   from simtk.openmm import app

   # Select test system.
   testsystem_list = [
      testsystems.LennardJonesCluster(),
      testsystems.LennardJonesFluid(),
      testsystems.UnconstrainedDiatomicFluid(),
      testsystems.FlexibleWaterBox(nonbondedMethod=app.CutoffPeriodic),
      testsystems.AlanineDipeptideImplicit(constraints=None),
      testsystems.AlanineDipeptideExplicit(constraints=None)
      ]
   for testsystem in testsystem_list:
      f = partial(check_bitwise_reversible_velocity_verlet, testsystem)
      f.description = "Testing bitwise reversible velocity Verlet integration with %s" % testsystem.__class__.__name__
      yield f

   return

def check_bitwise_reversible_velocity_verlet(testsystem):
   from simtk import openmm, unit
   from simtk.openmm import app
   from openmmtools import integrators, testsystems
   import numpy as np

   # Create a bitwise-reversible velocity Verlet integrator.
   timestep = 1.0 * unit.femtoseconds
   integrator = integrators.BitwiseReversibleVelocityVerletIntegrator(timestep)
   nsteps = 10
   # Demonstrate bitwise reversibility for a simple harmonic oscillator.
   platform = openmm.Platform.getPlatformByName('Reference')
   context = openmm.Context(testsystem.system, integrator, platform)
   context.setPositions(testsystem.positions)
   # Select velocity.
   context.setVelocitiesToTemperature(300*unit.kelvin)
   # Truncate accuracy and store initial positions.
   integrator.truncatePrecision(context)
   initial_positions = context.getState(getPositions=True).getPositions(asNumpy=True)
   # Integrate forward in time.
   integrator.step(nsteps)
   # Negate velocity and integrate backwards
   context.setVelocities(-context.getState(getVelocities=True).getVelocities(asNumpy=True))
   integrator.step(nsteps)
   # Compare positions.
   final_positions = context.getState(getPositions=True).getPositions(asNumpy=True)
   # Make sure differences are identically zero.
   delta = final_positions - initial_positions
   delta = delta / delta.unit
   if not np.all(delta==0.0):
      # TODO: Use bitstring representations for positions.
      string  = "Final positions do not match initial positions after %d steps of forward/backward integration.\n\n" % nsteps
      string += "Initial positions:\n"
      string += str(initial_positions) + '\n'
      string += "Final positions:\n"
      string += str(final_positions) + '\n'
      string += "Delta:\n"
      string += str(final_positions - initial_positions) + '\n'
      raise Exception(string)

   return

def test_stabilities_harmonic_oscillator():
   """
   Test integrators for stability over a short number of steps of a harmonic oscillator.

   """
   test = testsystems.HarmonicOscillator()

   for methodname in dir(integrators):
      if re.match('.*Integrator$', methodname):
         integrator = getattr(integrators, methodname)()
         integrator.__doc__ = methodname
         f = partial(check_stability, integrator, test)
         f.description = "Testing %s for stability over a short number of integration steps of a harmonic oscillator." % methodname
         yield f

def test_stabilities_alanine_dipeptide():
   """
   Test integrators for stability over a short number of steps of a harmonic oscillator.

   """
   test = testsystems.AlanineDipeptideImplicit()

   for methodname in dir(integrators):
      if re.match('.*Integrator$', methodname):
         integrator = getattr(integrators, methodname)()
         integrator.__doc__ = methodname
         f = partial(check_stability, integrator, test)
         f.description = "Testing %s for stability over a short number of integration steps of alanine dipeptide in implicit solvent." % methodname
         yield f

def test_integrator_decorators():
    integrator = integrators.HMCIntegrator(timestep=0.05 * unit.femtoseconds)
    testsystem = testsystems.IdealGas()
    nsteps = 25

    context = openmm.Context(testsystem.system, integrator)
    context.setPositions(testsystem.positions)
    context.setVelocitiesToTemperature(300 * unit.kelvin)

    integrator.step(nsteps)

    assert integrator.n_accept == nsteps
    assert integrator.n_trials == nsteps
    assert integrator.acceptance_rate == 1.0
