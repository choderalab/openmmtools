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
   test = testsystems.AlanineDipeptideImplicit()

   for methodname in dir(integrators):
      if re.match('.*Integrator$', methodname):
         integrator = getattr(integrators, methodname)()
         integrator.__doc__ = methodname
         check_stability.description = "Testing %s for stability over a short number of integration steps of alanine dipeptide in implicit solvent." % methodname
         yield check_stability, integrator, test

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

def test_vvvr_shadow_work_accumulation():
   ''' When `monitor_work==True`, assert that global `shadow_work` is initialized to zero and
   reaches a nonzero value after integrating a few dozen steps.
   
   By default (`monitor_work=False`), assert that there is no global name for `shadow_work`. '''
   
   # test `monitor_work=True` --> accumulation of a nonzero value in global `shadow_work`
   testsystem = testsystems.HarmonicOscillator()
   system, topology = testsystem.system, testsystem.topology
   temperature = 298.0 * unit.kelvin
   integrator = integrators.VVVRIntegrator(temperature, monitor_work=True)
   context = openmm.Context(system, integrator)
   context.setPositions(testsystem.positions)
   context.setVelocitiesToTemperature(temperature)
   assert(integrator.getGlobalVariableByName('shadow_work') == 0)
   integrator.step(25)
   assert(integrator.getGlobalVariableByName('shadow_work') != 0)
   
   # test default (`monitor_work=False`, `monitor_heat=False`) --> absence of a global `shadow_work`
   integrator = integrators.VVVRIntegrator(temperature)
   context = openmm.Context(system, integrator)
   context.setPositions(testsystem.positions)
   context.setVelocitiesToTemperature(temperature)
   integrator.step(25)
   # get the names of all global variables
   n_globals = integrator.getNumGlobalVariables()
   names_of_globals = [integrator.getGlobalVariableName(i) for i in range(n_globals)]
   assert('shadow_work' not in names_of_globals)
   
def test_vvvr_getset_temperature():
   temperature = 298.0 * unit.kelvin
   integrator = integrators.VVVRIntegrator(temperature, monitor_work=True)

   assert(integrator.getTemperature() == temperature)

   new_temperature = 500.0 * unit.kelvin
   integrator.setTemperature(new_temperature)

   assert(integrator.getTemperature() == new_temperature)

   new_kT = new_temperature * kB / unit.kilojoule_per_mole
   assert(integrator.getGlobalVariableByName('kT') == new_kT)
