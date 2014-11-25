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
import sys
import math
import doctest
import numpy
import time

import simtk.unit as unit
import simtk.openmm as openmm
from simtk.openmm import app

from openmmtools import testsystems
from openmmtools import integrators

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

def test_stabilities_harmonic_oscillator():
   """
   Test integrators for stability over a short number of steps of a harmonic oscillator.

   """
   from openmmtools import integrators, testsystems

   test = testsystems.HarmonicOscillator()

   for methodname in dir(integrators):
      if re.match('.*Integrator$', methodname):
         integrator = getattr(integrators, methodname)()
         integrator.__doc__ = methodname
         check_stability.description = "Testing %s for stability over a short number of integration steps of a harmonic oscillator." % methodname
         yield check_stability, integrator, test

def test_stabilities_alanine_dipeptide():
   """
   Test integrators for stability over a short number of steps of a harmonic oscillator.

   """
   from openmmtools import integrators, testsystems

   test = testsystems.AlanineDipeptideImplicit()

   for methodname in dir(integrators):
      if re.match('.*Integrator$', methodname):
         integrator = getattr(integrators, methodname)()
         integrator.__doc__ = methodname
         check_stability.description = "Testing %s for stability over a short number of integration steps of alanine dipeptide in implicit solvent." % methodname
         yield check_stability, integrator, test

