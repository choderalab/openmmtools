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

import numpy
import inspect

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
 
   # Set integrator temperature
   if hasattr(integrator, 'setTemperature'):
      integrator.setTemperature(temperature)

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

def test_stabilities():
   """
   Test integrators for stability over a short number of steps.

   """
   ts = testsystems  # shortcut
   test_cases = {'harmonic oscillator': ts.HarmonicOscillator(),
                 'alanine dipeptide in implicit solvent': ts.AlanineDipeptideImplicit()}

   # Get all the CustomIntegrators in the integrators module.
   is_integrator = lambda x: inspect.isclass(x) and issubclass(x, openmm.CustomIntegrator)
   custom_integrators = inspect.getmembers(integrators, predicate=is_integrator)

   for test_name, test in test_cases.items():
      for integrator_name, integrator_class in custom_integrators:
         integrator = integrator_class()
         integrator.__doc__ = integrator_name
         check_stability.description = ("Testing {} for stability over a short number of "
                                        "integration steps of a {}.").format(integrator_name, test_name)
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


def test_temperature_getter_setter():
    """Test that temperature setter and getter modify integrator variables."""

    def check_temperature(temperature, has_changed):
        kT = (temperature * integrators.kB).value_in_unit_system(unit.md_unit_system)
        temperature = temperature / unit.kelvin
        assert numpy.isclose(integrator.getTemperature() / unit.kelvin, temperature)
        assert integrator.getGlobalVariableByName('kT') == kT
        try:
            has_kT_changed = integrator.getGlobalVariableByName('has_kT_changed')
        except Exception:
            has_kT_changed = False
        if has_kT_changed is not False:
            assert has_kT_changed == has_changed

    test = testsystems.HarmonicOscillator()

    # Find all integrators with temperature setter/getter.
    is_metropolized = lambda x: (inspect.isclass(x) and
                                 issubclass(x, openmm.CustomIntegrator) and
                                 issubclass(x, integrators.MetropolizedIntegrator))
    metropolized_integrators = inspect.getmembers(integrators, predicate=is_metropolized)

    temperature1 = 300*unit.kelvin
    temperature2 = temperature1 + 100*unit.kelvin
    for integrator_name, integrator_class in metropolized_integrators:
        check_temperature.description = 'Test temperature setter and getter of {}'.format(integrator_name)

        # Initialization set temperature correctly.
        integrator = integrator_class(temperature=temperature1)
        yield check_temperature, temperature1, 1

        # At the first step step, the temperature-dependent constants are computed.
        context = openmm.Context(test.system, integrator)
        context.setPositions(test.positions)
        integrator.step(1)
        yield check_temperature, temperature1, 0

        # Setting temperature update kT and has_kT_changed.
        integrator.setTemperature(temperature2)
        yield check_temperature, temperature2, 1

        # At the next step, temperature-dependent constants are recomputed.
        integrator.step(1)
        yield check_temperature, temperature2, 0

        del context
