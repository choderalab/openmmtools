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
from openmmtools.integrators import ThermostatedIntegrator

#=============================================================================================
# CONSTANTS
#=============================================================================================

kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA


#=============================================================================================
# UTILITY SUBROUTINES
#=============================================================================================

def get_all_custom_integrators(only_thermostated=False):
    """Return all CustomIntegrators in integrators.

    Parameters
    ----------
    only_thermostated : bool
        If True, only the CustomIntegrators inheriting from
        ThermostatedIntegrator are returned.

    Returns
    -------
    custom_integrators : list of tuples
        A list of tuples ('IntegratorName', IntegratorClass)

    """
    predicate = lambda x: (inspect.isclass(x) and
                           issubclass(x, openmm.CustomIntegrator) and
                           x != integrators.ThermostatedIntegrator)
    if only_thermostated:
        old_predicate = predicate  # Avoid infinite recursion.
        predicate = lambda x: old_predicate(x) and issubclass(x, integrators.ThermostatedIntegrator)
    custom_integrators = inspect.getmembers(integrators, predicate=predicate)
    return custom_integrators


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


def check_integrator_temperature(integrator, temperature, has_changed):
    """Check integrator temperature has has_kT_changed variables."""
    kT = (temperature * integrators.kB).value_in_unit_system(unit.md_unit_system)
    temperature = temperature / unit.kelvin
    assert numpy.isclose(integrator.getTemperature() / unit.kelvin, temperature)
    assert numpy.isclose(integrator.getGlobalVariableByName('kT'), kT)
    try:
        has_kT_changed = integrator.getGlobalVariableByName('has_kT_changed')
    except Exception:
        has_kT_changed = False
    if has_kT_changed is not False:
        assert has_kT_changed == has_changed


def check_integrator_temperature_getter_setter(integrator):
    """Check that temperature setter/getter works correctly.

    Parameters
    ----------
    integrator : ThermostatedIntegrator
        An integrator just created and already bound to a context.

    """
    # The variable has_kT_changed is initialized correctly.
    temperature = integrator.getTemperature()
    check_integrator_temperature(integrator, temperature, 1)

    # At the first step step, the temperature-dependent constants are computed.
    integrator.step(1)
    check_integrator_temperature(integrator, temperature, 0)

    # Setting temperature update kT and has_kT_changed.
    temperature += 100*unit.kelvin
    integrator.setTemperature(temperature)
    check_integrator_temperature(integrator, temperature, 1)

    # At the next step, temperature-dependent constants are recomputed.
    integrator.step(1)
    check_integrator_temperature(integrator, temperature, 0)


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
    custom_integrators = get_all_custom_integrators()

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
    """When `measure_shadow_work==True`, assert that global `shadow_work` is initialized to zero and
    reaches a nonzero value after integrating a few dozen steps.

    By default (`measure_shadow_work=False`), assert that there is no global name for `shadow_work`."""

    # test `measure_shadow_work=True` --> accumulation of a nonzero value in global `shadow_work`
    testsystem = testsystems.HarmonicOscillator()
    system, topology = testsystem.system, testsystem.topology
    temperature = 298.0 * unit.kelvin
    integrator = integrators.VVVRIntegrator(temperature, measure_shadow_work=True)
    context = openmm.Context(system, integrator)
    context.setPositions(testsystem.positions)
    context.setVelocitiesToTemperature(temperature)
    assert(integrator.getGlobalVariableByName('shadow_work') == 0)
    integrator.step(25)
    assert(integrator.getGlobalVariableByName('shadow_work') != 0)

    # test default (`measure_shadow_work=False`, `measure_heat=True`) --> absence of a global `shadow_work`
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
    temperature = 350*unit.kelvin
    test = testsystems.HarmonicOscillator()
    custom_integrators = get_all_custom_integrators()
    thermostated_integrators = dict(get_all_custom_integrators(only_thermostated=True))

    for integrator_name, integrator_class in custom_integrators:

        # If this is not a ThermostatedIntegrator, the interface should not be added.
        if integrator_name not in thermostated_integrators:
            integrator = integrator_class()
            assert ThermostatedIntegrator.is_thermostated(integrator) is False
            assert ThermostatedIntegrator.restore_interface(integrator) is False
            assert not hasattr(integrator, 'getTemperature')
            continue

        # Test original integrator.
        check_integrator_temperature_getter_setter.description = ('Test temperature setter and '
                                                                  'getter of {}').format(integrator_name)
        integrator = integrator_class(temperature=temperature)
        context = openmm.Context(test.system, integrator)
        context.setPositions(test.positions)

        # Integrator temperature is initialized correctly.
        check_integrator_temperature(integrator, temperature, 1)
        yield check_integrator_temperature_getter_setter, integrator
        del context

        # Test Context integrator wrapper.
        check_integrator_temperature_getter_setter.description = ('Test temperature wrapper '
                                                                  'of {}').format(integrator_name)
        integrator = integrator_class()
        context = openmm.Context(test.system, integrator)
        context.setPositions(test.positions)
        integrator = context.getIntegrator()

        # Setter and getter should be added successfully.
        assert ThermostatedIntegrator.is_thermostated(integrator) is True
        assert ThermostatedIntegrator.restore_interface(integrator) is True
        assert isinstance(integrator, integrator_class)
        yield check_integrator_temperature_getter_setter, integrator
        del context


def test_thermostated_integrator_hash():
    """Check hash collisions between ThermostatedIntegrators."""
    thermostated_integrators = get_all_custom_integrators(only_thermostated=True)
    all_hashes = set()
    for integrator_name, integrator_class in thermostated_integrators:
        hash_float = ThermostatedIntegrator._compute_class_hash(integrator_class)
        all_hashes.add(hash_float)
        integrator = integrator_class()
        assert integrator.getGlobalVariableByName('thermostated_class_hash') == hash_float
    assert len(all_hashes) == len(thermostated_integrators)
