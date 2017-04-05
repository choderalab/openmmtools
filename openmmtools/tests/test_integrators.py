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
import pymbar

from simtk import unit
from simtk import openmm

from openmmtools import integrators, testsystems, alchemy
from openmmtools.integrators import RestorableIntegrator, ThermostatedIntegrator, AlchemicalLangevinSplittingIntegrator, GHMCIntegrator, GeodesicBAOABIntegrator

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
                           x != integrators.ThermostatedIntegrator and
                           x != integrators.RestorableIntegrator)
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
            # Need an alchemical system to test this
            if issubclass(integrator_class, integrators.AlchemicalLangevinSplittingIntegrator):
                continue
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


def test_external_protocol_work_accumulation():
    """When `measure_protocol_work==True`, assert that global `protocol_work` is initialized to zero and
    reaches a zero value after integrating a few dozen steps without perturbation.

    By default (`measure_protocol_work=False`), assert that there is no global name for `protocol_work`."""

    testsystem = testsystems.HarmonicOscillator()
    system, topology = testsystem.system, testsystem.topology
    temperature = 298.0 * unit.kelvin
    integrator = integrators.ExternalPerturbationLangevinSplittingIntegrator(splitting="O V R V O", temperature=temperature)
    context = openmm.Context(system, integrator)
    context.setPositions(testsystem.positions)
    context.setVelocitiesToTemperature(temperature)
    assert(integrator.getGlobalVariableByName('protocol_work') == 0), "Protocol work should be 0 initially"
    integrator.step(25)
    assert(integrator.getGlobalVariableByName('protocol_work') == 0), "There should be no protocol work."

    pe_1 = context.getState(getEnergy=True).getPotentialEnergy()
    perturbed_K=99.0 * unit.kilocalories_per_mole / unit.angstroms**2
    context.setParameter('testsystems_HarmonicOscillator_K', perturbed_K)
    pe_2 = context.getState(getEnergy=True).getPotentialEnergy()
    integrator.step(1)
    assert (integrator.getGlobalVariableByName('protocol_work') != 0), "There should be protocol work after perturbing."
    assert (integrator.getGlobalVariableByName('protocol_work') * unit.kilojoule_per_mole == (pe_2 - pe_1)), \
        "The potential energy difference should be equal to protocol work."

    # test default (`measure_protocol_work=False`, `measure_heat=True`) --> absence of a global `protocol_work`
    integrator = integrators.VVVRIntegrator(temperature)
    context = openmm.Context(system, integrator)
    context.setPositions(testsystem.positions)
    context.setVelocitiesToTemperature(temperature)
    integrator.step(25)
    # get the names of all global variables
    n_globals = integrator.getNumGlobalVariables()
    names_of_globals = [integrator.getGlobalVariableName(i) for i in range(n_globals)]
    assert('protocol_work' not in names_of_globals), "Protocol work should not be defined."


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
        if issubclass(integrator_class, integrators.AlchemicalLangevinSplittingIntegrator):
            integrator = integrator_class(dict(),temperature=temperature)
        else:
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
        if issubclass(integrator_class, integrators.AlchemicalLangevinSplittingIntegrator):
            integrator = integrator_class(dict())
        else:
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
        hash_float = RestorableIntegrator._compute_class_hash(integrator_class)
        all_hashes.add(hash_float)
        if issubclass(integrator_class, integrators.AlchemicalLangevinSplittingIntegrator):
            integrator = integrator_class(dict())
        else:
            integrator = integrator_class()
        assert integrator.getGlobalVariableByName('_restorable__class_hash') == hash_float
    assert len(all_hashes) == len(thermostated_integrators)


def test_alchemical_langevin_integrator():
    """Check that the AlchemicalLangevinSplittingIntegrator, when performing nonequilibrium switching from
    AlanineDipeptideImplicit to the same with nonbonded forces decoupled and back, results in an approximately
    zero free energy difference (using BAR). Up to 6*sigma is tolerated for error.
    """
    nsteps = 1000000
    #These are the alchemical functions we will use to switch the sterics and electrostatics
    default_functions = {
    'lambda_sterics' : 'lambda',
    }

    alchemical_integrator_forward = AlchemicalLangevinSplittingIntegrator(default_functions,
                                                                      splitting="O { V R H R V } O",
                                                                      nsteps_neq=nsteps,
                                                                      direction="forward")
    alchemical_integrator_reverse = AlchemicalLangevinSplittingIntegrator(default_functions,
                                                                  splitting="O { V R H R V } O",
                                                                  nsteps_neq=nsteps,
                                                                  direction="reverse")

    platform = openmm.Platform.getPlatformByName("Reference")

    # Do 100 iterations of each direction
    n_iterations = 100

    # Instantiate the testsystem
    lj = testsystems.LennardJonesCluster()

    # Alchemically modify everything:
    alchemical_factory = alchemy.AlchemicalFactory(consistent_exceptions=False)
    alchemical_region = alchemy.AlchemicalRegion([1], alchemical_bonds=False, alchemical_angles=False,
                                                 alchemical_torsions=False)
    modified_system = alchemical_factory.create_alchemical_system(reference_system=lj.system,
                                                                  alchemical_regions=alchemical_region,
                                                                  )

    alchemical_ctx_forward = openmm.Context(modified_system, alchemical_integrator_forward, platform)
    alchemical_ctx_reverse = openmm.Context(modified_system, alchemical_integrator_reverse, platform)

    # Get the forward work values:
    positions = lj.positions
    w_f = numpy.zeros([n_iterations])
    for i in range(n_iterations):
        w_f[i], eq_positions = run_nonequilibrium_switching(modified_system, positions,
                                              default_functions, alchemical_integrator_forward, 100, alchemical_ctx_forward,
                                              direction="forward")
        positions = eq_positions

        print(i)

    # Get the reverse work values:
    w_r = numpy.zeros([n_iterations])
    for i in range(n_iterations):
        w_r[i], eq_positions = run_nonequilibrium_switching(modified_system, positions,
                                              default_functions, alchemical_integrator_reverse, 100, alchemical_ctx_reverse,
                                              direction="reverse")
        positions = eq_positions
        print(i)

    deltaF, ddeltaF = pymbar.BAR(-w_f, -w_r)

    print(deltaF)
    print(ddeltaF)



def run_nonequilibrium_switching(system, positions, alchemical_functions, alchemical_integrator, nsteps, alchemical_ctx, direction="forward"):
    """Equilibrate and then run some nonequilibrium switching simulations

    Parameters
    ----------
    system
    positions
    alchemical_integrator
    direction

    Returns
    -------
    protocol_work : float
        Work performed by protocol
    """

    # Make a ghmc integrator with the default parameters
    ghmc = GHMCIntegrator()

    # Use the reference platform, since this is for a test
    platform = openmm.Platform.getPlatformByName("Reference")

    # Make a context
    context = openmm.Context(system, ghmc, platform)
    context.setPositions(positions)

    # Set the initial alchemical state for the equilibration simulation:
    for parameter in alchemical_functions.keys():
        context.setParameter(parameter, 0.0) if direction == "forward" else context.setParameter(parameter, 1.0)

    # Run some steps to equilibrate
    ghmc.step(100)

    eq_positions = context.getState(getPositions=True).getPositions(asNumpy=True)

    alchemical_ctx.setPositions(eq_positions)

    alchemical_integrator.step(nsteps)

    return alchemical_integrator.getGlobalVariableByName("protocol_work"), eq_positions


if __name__=="__main__":
    test_alchemical_langevin_integrator()