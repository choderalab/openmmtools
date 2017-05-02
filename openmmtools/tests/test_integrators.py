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

from tqdm import tqdm

from functools import partial
from unittest import TestCase

from simtk import unit
from simtk import openmm

from openmmtools import integrators, testsystems, alchemy
from openmmtools.integrators import RestorableIntegrator, ThermostatedIntegrator, NonequilibriumLangevinIntegrator, GHMCIntegrator, GeodesicBAOABIntegrator

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
            if issubclass(integrator_class, integrators.NonequilibriumLangevinIntegrator):
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
    integrator = integrators.ExternalPerturbationLangevinIntegrator(splitting="O V R V O", temperature=temperature)
    context = openmm.Context(system, integrator)
    context.setPositions(testsystem.positions)
    context.setVelocitiesToTemperature(temperature)
    # Check that initial step accumulates no protocol work
    assert(integrator.getGlobalVariableByName('protocol_work') == 0), "Protocol work should be 0 initially"
    integrator.step(1)
    assert(integrator.getGlobalVariableByName('protocol_work') == 0), "There should be no protocol work."
    # Check that a single step accumulates protocol work
    pe_1 = context.getState(getEnergy=True).getPotentialEnergy()
    perturbed_K=99.0 * unit.kilocalories_per_mole / unit.angstroms**2
    context.setParameter('testsystems_HarmonicOscillator_K', perturbed_K)
    pe_2 = context.getState(getEnergy=True).getPotentialEnergy()
    integrator.step(1)
    assert (integrator.getGlobalVariableByName('protocol_work') != 0), "There should be protocol work after perturbing."
    assert (integrator.getGlobalVariableByName('protocol_work') * unit.kilojoule_per_mole == (pe_2 - pe_1)), \
        "The potential energy difference should be equal to protocol work."
    del context, integrator

    # Test default (`measure_protocol_work=False`, `measure_heat=True`) --> absence of a global `protocol_work`
    integrator = integrators.VVVRIntegrator(temperature)
    context = openmm.Context(system, integrator)
    context.setPositions(testsystem.positions)
    context.setVelocitiesToTemperature(temperature)
    integrator.step(25)
    # get the names of all global variables
    n_globals = integrator.getNumGlobalVariables()
    names_of_globals = [integrator.getGlobalVariableName(i) for i in range(n_globals)]
    assert('protocol_work' not in names_of_globals), "Protocol work should not be defined."
    del context, integrator

class TestExternalPerturbationLangevinIntegrator(TestCase):
    def test_protocol_work_accumulation_harmonic_oscillator(self):
        """Testing protocol work accumulation for ExternalPerturbationLangevinIntegrator with HarmonicOscillator
        """
        testsystem = testsystems.HarmonicOscillator()
        parameter_name = 'testsystems_HarmonicOscillator_x0'
        parameter_initial = 0.0 * unit.angstroms
        parameter_final = 10.0 * unit.angstroms
        for platform_name in ['Reference', 'CPU']:
            self.compare_external_protocol_work_accumulation(testsystem, parameter_name, parameter_initial, parameter_final, platform_name=platform_name)

    def test_protocol_work_accumulation_waterbox(self):
        """Testing protocol work accumulation for ExternalPerturbationLangevinIntegrator with AlchemicalWaterBox
        """
        from simtk.openmm import app
        parameter_name = 'lambda_electrostatics'
        parameter_initial = 1.0
        parameter_final = 0.0
        platform_names = [ openmm.Platform.getPlatform(index).getName() for index in range(openmm.Platform.getNumPlatforms()) ]
        for nonbonded_method in ['CutoffPeriodic', 'PME']:
            testsystem = testsystems.AlchemicalWaterBox(nonbondedMethod=getattr(app, nonbonded_method))
            for platform_name in platform_names:
                name = '%s %s %s' % (testsystem.name, nonbonded_method, platform_name)                
                self.compare_external_protocol_work_accumulation(testsystem, parameter_name, parameter_initial, parameter_final, platform_name=platform_name, name=name)

    def test_protocol_work_accumulation_waterbox_barostat(self):
        """
        Testing protocol work accumulation for ExternalPerturbationLangevinIntegrator with AlchemicalWaterBox
        with an active barostat. For brevity, only using CutoffPeriodic as the non-bonded method.
        """
        from simtk.openmm import app
        parameter_name = 'lambda_electrostatics'
        parameter_initial = 1.0
        parameter_final = 0.0
        platform_names = [ openmm.Platform.getPlatform(index).getName() for index in range(openmm.Platform.getNumPlatforms()) ]
        nonbonded_method = 'CutoffPeriodic'
        testsystem = testsystems.AlchemicalWaterBox(nonbondedMethod=getattr(app, nonbonded_method))

        # Adding the barostat with a high frequency
        testsystem.system.addForce(openmm.MonteCarloBarostat(1*unit.atmospheres, 300*unit.kelvin, 2))

        for platform_name in platform_names:
            name = '%s %s %s' % (testsystem.name, nonbonded_method, platform_name)
            self.compare_external_protocol_work_accumulation(testsystem, parameter_name, parameter_initial, parameter_final, platform_name=platform_name, name=name)

    def compare_external_protocol_work_accumulation(self, testsystem, parameter_name, parameter_initial, parameter_final, platform_name='Reference', name=None):
        """Compare external work accumulation between Reference and CPU platforms.
        """

        if name is None:
            name = testsystem.name

        from openmmtools.constants import kB
        system, topology = testsystem.system, testsystem.topology
        temperature = 298.0 * unit.kelvin
        platform = openmm.Platform.getPlatformByName(platform_name)

        # TODO: Set precision and determinism if platform is ['OpenCL', 'CUDA']

        nsteps = 20
        kT = kB * temperature
        integrator = integrators.ExternalPerturbationLangevinIntegrator(splitting="O V R V O", temperature=temperature)
        context = openmm.Context(system, integrator, platform)
        context.setParameter(parameter_name, parameter_initial)
        context.setPositions(testsystem.positions)
        context.setVelocitiesToTemperature(temperature)
        assert(integrator.getGlobalVariableByName('protocol_work') == 0), "Protocol work should be 0 initially"
        integrator.step(1)
        assert(integrator.getGlobalVariableByName('protocol_work') == 0), "There should be no protocol work."

        external_protocol_work = 0.0
        for step in tqdm(range(nsteps), desc=name):
            lambda_value = float(step+1) / float(nsteps)
            parameter_value = parameter_initial * (1-lambda_value) + parameter_final * lambda_value
            initial_energy = context.getState(getEnergy=True).getPotentialEnergy()
            context.setParameter(parameter_name, parameter_value)
            final_energy = context.getState(getEnergy=True).getPotentialEnergy()
            external_protocol_work += (final_energy - initial_energy) / kT

            integrator.step(1)
            integrator_protocol_work = integrator.getGlobalVariableByName('protocol_work') * unit.kilojoules_per_mole / kT

            message = '\n'
            message += 'protocol work discrepancy noted for %s on platform %s\n' % (name, platform_name)
            message += 'step %5d : external %16e kT | integrator %16e kT | difference %16e kT' % (step, external_protocol_work, integrator_protocol_work, external_protocol_work - integrator_protocol_work)
            self.assertAlmostEqual(external_protocol_work, integrator_protocol_work, msg=message)

        del context, integrator

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
        if issubclass(integrator_class, integrators.NonequilibriumLangevinIntegrator):
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
        if issubclass(integrator_class, integrators.NonequilibriumLangevinIntegrator):
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
        if issubclass(integrator_class, integrators.NonequilibriumLangevinIntegrator):
            integrator = integrator_class(dict())
        else:
            integrator = integrator_class()
        assert integrator.getGlobalVariableByName('_restorable__class_hash') == hash_float
    assert len(all_hashes) == len(thermostated_integrators)


def test_alchemical_langevin_integrator():
    """Check that the AlchemicalLangevinSplittingIntegrator, when performing nonequilibrium switching from
    LennardJonesCluster to the same with nonbonded forces decoupled and back, results in an approximately
    zero free energy difference (using BAR). Up to 6*sigma is tolerated for error.
    """

    #max deviation from the calculated free energy
    NSIGMA_MAX = 6
    n_iterations = 100  # number of forward and reverse protocols
    nsteps = 10 # number of steps within each protocol

    # These are the alchemical functions that will be used to control the system
    default_functions = {'lambda_sterics' : 'lambda^2 - lambda'}

    splitting = "O { V R H R V } O"
    alchemical_integrator = NonequilibriumLangevinIntegrator(default_functions,
                                                             splitting=splitting,
                                                             nsteps_neq=nsteps,
                                                             )

    platform = openmm.Platform.getPlatformByName("Reference")

    lj = testsystems.LennardJonesCluster()
    positions = lj.positions

    # Alchemically modify everything:
    alchemical_factory = alchemy.AlchemicalFactory(consistent_exceptions=False)
    alchemical_region = alchemy.AlchemicalRegion([1], alchemical_bonds=False, alchemical_angles=False,
                                                 alchemical_torsions=False)
    modified_system = alchemical_factory.create_alchemical_system(reference_system=lj.system,
                                                                  alchemical_regions=alchemical_region)

    alchemical_ctx = openmm.Context(modified_system, alchemical_integrator, platform)

    # Get equilibrium samples
    burn_in = 1000
    n_equil_samples = n_iterations
    thinning = 10

    samples_at_0, samples_at_1 = [], []
    get_acceptance_rate = lambda integrator : integrator.getGlobalVariableByName("naccept") / integrator.getGlobalVariableByName("ntrials")

    # Get equilibrium samples from the lambda=0 state
    ghmc = GHMCIntegrator()
    context = openmm.Context(modified_system, ghmc, platform)
    context.setPositions(positions)
    for parameter in default_functions.keys():
        context.setParameter(parameter, 0.0)
    ghmc.step(burn_in)
    for _ in range(n_equil_samples):
        ghmc.step(thinning)
        samples_at_0.append(context.getState(getPositions=True).getPositions(asNumpy=True))
    print("GHMC acceptance rate: {:.3f}".format(get_acceptance_rate(ghmc)))

    # Get the forward work values:
    w_f = numpy.zeros([n_iterations])
    f_acceptance_rates = []
    for i in range(n_iterations):
        init_x = samples_at_0[numpy.random.randint(len(samples_at_0))]
        w_f[i] = run_nonequilibrium_switching(init_x, alchemical_integrator, nsteps, alchemical_ctx)
        f_acceptance_rates.append(get_acceptance_rate(alchemical_integrator))

    dF, ddF = pymbar.EXP(w_f)
    print("DeltaF: {:.4f}, dDeltaF: {:.4f}".format(dF, ddF))
    if numpy.abs(dF) > NSIGMA_MAX * ddF:
        raise Exception("The free energy difference for the nonequilibrium switching is not correct.")

def run_nonequilibrium_switching(init_x, alchemical_integrator, nsteps, alchemical_ctx):
    """Perform a nonequilibrium switching protocol

    Parameters
    ----------
    init_x
    alchemical_integrator
    nsteps
    alchemical_ctx

    Returns
    -------
    protocol_work : float
        Work performed by protocol
    """

    alchemical_ctx.setPositions(init_x)
    alchemical_ctx.setVelocitiesToTemperature(298 * unit.kelvin)
    alchemical_integrator.reset_integrator()
    alchemical_integrator.step(nsteps)
    return alchemical_integrator.getGlobalVariableByName("protocol_work")


if __name__=="__main__":
    test_alchemical_langevin_integrator()
