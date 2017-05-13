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

import numpy as np
import inspect
import pymbar

from unittest import TestCase

from simtk import unit
from simtk import openmm

from openmmtools import integrators, testsystems, alchemy
from openmmtools.integrators import RestorableIntegrator, ThermostatedIntegrator, AlchemicalNonequilibriumLangevinIntegrator, GHMCIntegrator

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
    if np.isnan(potential):
        raise Exception("Potential energy for integrator %s became NaN." % integrator.__doc__)

    del context


def check_integrator_temperature(integrator, temperature, has_changed):
    """Check integrator temperature has has_kT_changed variables."""
    kT = (temperature * integrators.kB)
    temperature = temperature / unit.kelvin
    assert np.isclose(integrator.getTemperature() / unit.kelvin, temperature)
    assert np.isclose(integrator.getGlobalVariableByName('kT'), kT.value_in_unit_system(unit.md_unit_system))
    assert np.isclose(integrator.kT.value_in_unit_system(unit.md_unit_system), kT.value_in_unit_system(unit.md_unit_system))
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

def test_pretty_formatting():
    """
    Test pretty-printing and pretty-formatting of integrators.
    """
    custom_integrators = get_all_custom_integrators()
    for integrator_name, integrator_class in custom_integrators:
        # The NonequilibriumLangevinIntegrator requires an alchemical function.
        integrator = integrator_class()

        if hasattr(integrator, 'pretty_format'):
            # Check formatting as text
            text = integrator.pretty_format()
            # Check formatting as text with highlighted steps
            text = integrator.pretty_format(step_types_to_highlight=[5])
            # Check list format
            lines = integrator.pretty_format(as_list=True)
            msg = "integrator.pretty_format(as_list=True) has %d lines while integrator has %d steps" % (len(lines), integrator.getNumComputations())
            assert len(lines) == integrator.getNumComputations(), msg

def test_update_context_state_calls():
    """
    Ensure that all integrators only call addUpdateContextState() once.
    """
    custom_integrators = get_all_custom_integrators()
    for integrator_name, integrator_class in custom_integrators:
        # The NonequilibriumLangevinIntegrator requires an alchemical function.
        integrator = integrator_class()
        num_force_update = 0
        for i in range(integrator.getNumComputations()):
            step_type, target, expr = integrator.getComputationStep(i)

            if step_type == 5:
                num_force_update += 1

        msg = "Integrator '%s' has %d calls to addUpdateContextState(), while there should be only one." % (integrator_name, num_force_update)
        if hasattr(integrator, 'pretty_format'):
            msg += '\n' + integrator.pretty_format(step_types_to_highlight=[5])
        assert num_force_update == 1, msg

def test_vvvr_shadow_work_accumulation():
    """When `measure_shadow_work==True`, assert that global `shadow_work` is initialized to zero and
    reaches a nonzero value after integrating a few dozen steps."""

    # test `measure_shadow_work=True` --> accumulation of a nonzero value in global `shadow_work`
    testsystem = testsystems.HarmonicOscillator()
    system, topology = testsystem.system, testsystem.topology
    temperature = 298.0 * unit.kelvin
    integrator = integrators.VVVRIntegrator(temperature, measure_shadow_work=True)
    context = openmm.Context(system, integrator)
    context.setPositions(testsystem.positions)
    context.setVelocitiesToTemperature(temperature)
    assert(integrator.get_shadow_work(dimensionless=True) == 0), "Shadow work should initially be zero."
    assert(integrator.get_shadow_work() / unit.kilojoules_per_mole == 0), "integrator.get_shadow_work() should have units of energy."
    assert(integrator.shadow_work / unit.kilojoules_per_mole == 0), "integrator.shadow_work should have units of energy."
    integrator.step(25)
    assert(integrator.get_shadow_work(dimensionless=True) != 0), "integrator.get_shadow_work() should be nonzero after dynamics"

    # test default (`measure_shadow_work=False`, `measure_heat=True`) --> absence of a global `shadow_work`
    integrator = integrators.VVVRIntegrator(temperature)
    context = openmm.Context(system, integrator)
    context.setPositions(testsystem.positions)
    context.setVelocitiesToTemperature(temperature)
    integrator.step(25)

    del context, integrator

def test_baoab_heat_accumulation():
    """When `measure_heat==True`, assert that global `heat` is initialized to zero and
    reaches a nonzero value after integrating a few dozen steps."""

    # test `measure_shadow_work=True` --> accumulation of a nonzero value in global `shadow_work`
    testsystem = testsystems.HarmonicOscillator()
    system, topology = testsystem.system, testsystem.topology
    temperature = 298.0 * unit.kelvin
    integrator = integrators.BAOABIntegrator(temperature, measure_heat=True)
    context = openmm.Context(system, integrator)
    context.setPositions(testsystem.positions)
    context.setVelocitiesToTemperature(temperature)
    assert(integrator.get_heat(dimensionless=True) == 0), "Heat should initially be zero."
    assert(integrator.get_heat() / unit.kilojoules_per_mole == 0), "integrator.get_heat() should have units of energy."
    assert(integrator.heat / unit.kilojoules_per_mole == 0), "integrator.heat should have units of energy."
    integrator.step(25)
    assert(integrator.get_heat(dimensionless=True) != 0), "integrator.get_heat() should be nonzero after dynamics"

    # test default (`measure_shadow_work=False`, `measure_heat=True`) --> absence of a global `shadow_work`
    integrator = integrators.VVVRIntegrator(temperature)
    context = openmm.Context(system, integrator)
    context.setPositions(testsystem.positions)
    context.setVelocitiesToTemperature(temperature)
    integrator.step(25)

    del context, integrator

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
    assert(integrator.get_protocol_work(dimensionless=True) == 0), "Protocol work should be 0 initially"
    assert(integrator.get_protocol_work() / unit.kilojoules_per_mole == 0), "Protocol work should have units of energy"
    integrator.step(1)
    assert(integrator.get_protocol_work(dimensionless=True) == 0), "There should be no protocol work."
    # Check that a single step accumulates protocol work
    pe_1 = context.getState(getEnergy=True).getPotentialEnergy()
    perturbed_K=99.0 * unit.kilocalories_per_mole / unit.angstroms**2
    context.setParameter('testsystems_HarmonicOscillator_K', perturbed_K)
    pe_2 = context.getState(getEnergy=True).getPotentialEnergy()
    integrator.step(1)
    assert (integrator.get_protocol_work(dimensionless=True) != 0), "There should be protocol work after perturbing."
    assert (integrator.protocol_work == (pe_2 - pe_1)), "The potential energy difference should be equal to protocol work."
    del context, integrator

    # Test default (`measure_protocol_work=False`, `measure_heat=True`) --> absence of a global `protocol_work`
    integrator = integrators.VVVRIntegrator(temperature)
    context = openmm.Context(system, integrator)
    context.setPositions(testsystem.positions)
    context.setVelocitiesToTemperature(temperature)
    integrator.step(25)
    del context, integrator

class TestExternalPerturbationLangevinIntegrator(TestCase):

    def create_system(self, testsystem, parameter_name, parameter_initial, temperature = 298.0 * unit.kelvin, platform_name='Reference'):
        """
        Create an example system to be used by other tests
        """
        system, topology = testsystem.system, testsystem.topology
        integrator = integrators.ExternalPerturbationLangevinIntegrator(splitting="O V R V O", temperature=temperature)

        # Create the context
        platform = openmm.Platform.getPlatformByName(platform_name)
        if platform_name in ['CPU', 'CUDA']:
            platform.setPropertyDefaultValue('DeterministicForces', 'true')
        context = openmm.Context(system, integrator, platform)
        context.setParameter(parameter_name, parameter_initial)
        context.setPositions(testsystem.positions)
        context.setVelocitiesToTemperature(temperature)

        return context, integrator

    def run_ncmc(self, context, integrator, temperature, nsteps, parameter_name, parameter_initial, parameter_final):
        """
        A simple example of NCMC to be used with unit tests. The protocol work should be reset each time this command
        is called.

        Returns
        -------
        external_protocol_work: float
            the protocol work calculated with context.getState()
        integrator_protocol_work: float
            the protocol work calculated inside the integrator.
        """

        kT = kB * temperature

        external_protocol_work = 0.0
        integrator.step(1)
        for step in range(nsteps):
            lambda_value = float(step + 1) / float(nsteps)
            parameter_value = parameter_initial * (1 - lambda_value) + parameter_final * lambda_value
            initial_energy = context.getState(getEnergy=True).getPotentialEnergy()
            context.setParameter(parameter_name, parameter_value)
            final_energy = context.getState(getEnergy=True).getPotentialEnergy()
            external_protocol_work += (final_energy - initial_energy) / kT
            integrator.step(1)

        integrator_protocol_work = integrator.get_protocol_work(dimensionless=True)

        return external_protocol_work, integrator_protocol_work

    def test_initial_protocol_work(self):
        """
        Ensure the protocol work is initially zero and remains zero after a number of integrator steps.
        """
        from simtk.openmm import app
        parameter_name = 'lambda_electrostatics'
        temperature = 298.0 * unit.kelvin
        parameter_initial = 1.0
        platform_name = 'CPU'
        nonbonded_method = 'CutoffPeriodic'

        # Create the system
        testsystem = testsystems.AlchemicalWaterBox(nonbondedMethod=getattr(app, nonbonded_method))
        testsystem.system.addForce(openmm.MonteCarloBarostat(1 * unit.atmospheres, temperature, 2))
        context, integrator = self.create_system(testsystem, parameter_name, parameter_initial, temperature, platform_name)

        assert (integrator.get_protocol_work(dimensionless=True) == 0)
        integrator.step(5)
        assert(integrator.get_protocol_work(dimensionless=True) == 0)

    def test_reset_protocol_work(self):
        """
        Make sure the protocol work that is accumulated internally by the langevin integrator matches the protocol
        is correctly reset with the reset_protocol_work() command.
        """
        from simtk.openmm import app
        parameter_name = 'lambda_electrostatics'
        temperature = 298.0 * unit.kelvin
        parameter_initial = 1.0
        parameter_final = 0.0
        platform_name = 'CPU'
        nonbonded_method = 'CutoffPeriodic'

        # Creating the test system with a high frequency barostat.
        testsystem = testsystems.AlchemicalWaterBox(nonbondedMethod=getattr(app, nonbonded_method))
        testsystem.system.addForce(openmm.MonteCarloBarostat(1 * unit.atmospheres, temperature, 2))
        context, integrator = self.create_system(testsystem, parameter_name, parameter_initial, temperature, platform_name)

        # Number of NCMC steps
        nsteps = 20
        niterations = 3

        # Running several rounds of configuration updates and NCMC
        for i in range(niterations):
            integrator.step(5)
            # Reseting the protocol work inside the integrator
            integrator.reset_protocol_work()
            external_protocol_work, integrator_protocol_work = self.run_ncmc(context, integrator, temperature, nsteps, parameter_name, parameter_initial, parameter_final)
            assert abs(external_protocol_work - integrator_protocol_work) < 1.E-5

    def test_ncmc_update_parameters_in_context(self):
        """
        Testing that the protocol work is correctly calculated in cases when the parameters are updated using
        context.updateParametersInContext() and the integrator is a compound integrator. The NCMC scheme tested below
        is based on the one used by the saltswap and protons code-bases.
        """
        from simtk.openmm import app
        from openmmtools.constants import kB

        size = 20.0
        temperature = 298.0 * unit.kelvin
        kT = kB * temperature
        nonbonded_method = 'CutoffPeriodic'
        platform_name = 'CPU'
        timestep = 1. * unit.femtoseconds
        collision_rate = 90. / unit.picoseconds

        wbox = testsystems.WaterBox(box_edge=size*unit.angstrom, cutoff=9.*unit.angstrom, nonbondedMethod=getattr(app, nonbonded_method))

        integrator = integrators.ExternalPerturbationLangevinIntegrator(splitting="V R O R V", temperature=temperature, timestep=timestep, collision_rate=collision_rate)

        # Create context
        platform = openmm.Platform.getPlatformByName(platform_name)
        context = openmm.Context(wbox.system, integrator, platform)
        context.setPositions(wbox.positions)
        context.setPositions(wbox.positions)
        context.setVelocitiesToTemperature(temperature)

        def switchoff(force, context, frac=0.9):
            force.setParticleParameters(0, charge=-0.834 * frac, sigma=0.3150752406575124*frac, epsilon=0.635968 * frac)
            force.setParticleParameters(1, charge=0.417 * frac, sigma=0, epsilon=1 * frac)
            force.setParticleParameters(2, charge=0.417 * frac, sigma=0, epsilon=1 * frac)
            force.updateParametersInContext(context)

        def switchon(force, context):
            force.setParticleParameters(0, charge=-0.834, sigma=0.3150752406575124, epsilon=0.635968)
            force.setParticleParameters(1, charge=0.417, sigma=0, epsilon=1)
            force.setParticleParameters(2, charge=0.417, sigma=0, epsilon=1)
            force.updateParametersInContext(context)

        force = wbox.system.getForce(2)  # Non-bonded force.

        # Number of NCMC steps
        nsteps = 20
        niterations = 3

        for i in range(niterations):
            external_protocol_work = 0.0
            integrator.reset_protocol_work()
            integrator.step(1)
            for step in range(nsteps):
                fraction = float(step + 1) / float(nsteps)
                initial_energy = context.getState(getEnergy=True).getPotentialEnergy()
                switchoff(force, context, frac=fraction)
                final_energy = context.getState(getEnergy=True).getPotentialEnergy()
                external_protocol_work += (final_energy - initial_energy) / kT
                integrator.step(1)
            integrator_protocol_work = integrator.get_protocol_work(dimensionless=True)
            assert abs(external_protocol_work - integrator_protocol_work) < 1.E-5
            # Return to unperturbed state
            switchon(force, context)


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
        for nonbonded_method in ['CutoffPeriodic']:
            testsystem = testsystems.AlchemicalWaterBox(nonbondedMethod=getattr(app, nonbonded_method), box_edge=12.0*unit.angstroms, cutoff=5.0*unit.angstroms)
            for platform_name in platform_names:
                name = '%s %s %s' % (testsystem.name, nonbonded_method, platform_name)
                self.compare_external_protocol_work_accumulation(testsystem, parameter_name, parameter_initial, parameter_final, platform_name=platform_name, name=name)

    def test_protocol_work_accumulation_waterbox_barostat(self, temperature=300*unit.kelvin):
        """
        Testing protocol work accumulation for ExternalPerturbationLangevinIntegrator with AlchemicalWaterBox with barostat.
        For brevity, only using CutoffPeriodic as the non-bonded method.
        """
        from simtk.openmm import app
        parameter_name = 'lambda_electrostatics'
        parameter_initial = 1.0
        parameter_final = 0.0
        platform_names = [ openmm.Platform.getPlatform(index).getName() for index in range(openmm.Platform.getNumPlatforms()) ]
        nonbonded_method = 'CutoffPeriodic'
        testsystem = testsystems.AlchemicalWaterBox(nonbondedMethod=getattr(app, nonbonded_method), box_edge=12.0*unit.angstroms, cutoff=5.0*unit.angstroms)

        # Adding the barostat with a high frequency
        testsystem.system.addForce(openmm.MonteCarloBarostat(1*unit.atmospheres, temperature, 2))

        for platform_name in platform_names:
            name = '%s %s %s' % (testsystem.name, nonbonded_method, platform_name)
            self.compare_external_protocol_work_accumulation(testsystem, parameter_name, parameter_initial, parameter_final, platform_name=platform_name, name=name)

    def compare_external_protocol_work_accumulation(self, testsystem, parameter_name, parameter_initial, parameter_final, platform_name='Reference', name=None):
        """Compare external work accumulation between Reference and CPU platforms.
        """

        if name is None:
            name = testsystem.name

        from openmmtools.constants import kB

        temperature = 298.0 * unit.kelvin
        kT = kB * temperature

        context, integrator = self.create_system(testsystem, parameter_name, parameter_initial,
                                                 temperature=temperature, platform_name='Reference')

        external_protocol_work = 0.0
        nsteps = 20
        integrator.step(1)
        for step in range(nsteps):
            lambda_value = float(step+1) / float(nsteps)
            parameter_value = parameter_initial * (1-lambda_value) + parameter_final * lambda_value
            initial_energy = context.getState(getEnergy=True).getPotentialEnergy()
            context.setParameter(parameter_name, parameter_value)
            final_energy = context.getState(getEnergy=True).getPotentialEnergy()
            external_protocol_work += (final_energy - initial_energy) / kT

            integrator.step(1)
            integrator_protocol_work = integrator.get_protocol_work(dimensionless=True)

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
        hash_float = RestorableIntegrator._compute_class_hash(integrator_class)
        all_hashes.add(hash_float)
        integrator = integrator_class()
        assert integrator.getGlobalVariableByName('_restorable__class_hash') == hash_float
    assert len(all_hashes) == len(thermostated_integrators)


def run_alchemical_langevin_integrator(nsteps=0, splitting="O { V R H R V } O"):
    """Check that the AlchemicalLangevinSplittingIntegrator, when performing nonequilibrium switching from
    LennardJonesCluster to the same with nonbonded forces decoupled and back, results in an approximately
    zero free energy difference (using BAR). Up to 6*sigma is tolerated for error.

    The total work (protocol work + shadow work) is used.
    """

    #max deviation from the calculated free energy
    NSIGMA_MAX = 6
    n_iterations = 100  # number of forward and reverse protocols

    # These are the alchemical functions that will be used to control the system
    temperature = 298.0 * unit.kelvin
    sigma = 1.0 * unit.angstrom # stddev of harmonic oscillator
    kT = kB * temperature # thermal energy
    beta = 1.0 / kT # inverse thermal energy
    K = kT / sigma**2 # spring constant corresponding to sigma
    mass = 39.948 * unit.amu
    period = unit.sqrt(mass/K) # period of harmonic oscillator
    timestep = period / 20.0
    collision_rate = 1.0 / period
    dF_analytical = 1.0
    parameters = dict()
    parameters['testsystems_HarmonicOscillator_x0'] = (0 * sigma, 2 * sigma)
    parameters['testsystems_HarmonicOscillator_U0'] = (0 * kT, 1 * kT)
    forward_functions = { name : '(1-lambda)*%f + lambda*%f' % (value[0].value_in_unit_system(unit.md_unit_system), value[1].value_in_unit_system(unit.md_unit_system)) for (name, value) in parameters.items() }
    reverse_functions = { name : '(1-lambda)*%f + lambda*%f' % (value[1].value_in_unit_system(unit.md_unit_system), value[0].value_in_unit_system(unit.md_unit_system)) for (name, value) in parameters.items() }

    # Create harmonic oscillator testsystem
    testsystem = testsystems.HarmonicOscillator(K=K, mass=mass)
    system = testsystem.system
    positions = testsystem.positions

    # Get equilibrium samples from initial and final states
    burn_in = 1000
    thinning = 10 * 20 # number of steps between samples

    # Collect forward and reverse work values
    w_f = np.zeros([n_iterations], np.float64)
    w_r = np.zeros([n_iterations], np.float64)
    platform = openmm.Platform.getPlatformByName("Reference")
    for direction in ['forward', 'reverse']:
        positions = testsystem.positions
        for iteration in range(n_iterations):
            # Generate equilibrium sample
            equilibrium_integrator = GHMCIntegrator(temperature=temperature, collision_rate=collision_rate, timestep=timestep)
            equilibrium_context = openmm.Context(system, equilibrium_integrator, platform)
            for (name, value) in parameters.items():
                if direction == 'forward':
                    equilibrium_context.setParameter(name, value[0].value_in_unit_system(unit.md_unit_system))
                else:
                    equilibrium_context.setParameter(name, value[1].value_in_unit_system(unit.md_unit_system))
            equilibrium_context.setPositions(positions)
            equilibrium_integrator.step(thinning)
            positions = equilibrium_context.getState(getPositions=True).getPositions(asNumpy=True)
            del equilibrium_context, equilibrium_integrator
            # Generate nonequilibrium work sample
            if direction == 'forward':
                alchemical_functions = forward_functions
            else:
                alchemical_functions = reverse_functions
            nonequilibrium_integrator = AlchemicalNonequilibriumLangevinIntegrator(temperature=temperature, collision_rate=collision_rate, timestep=timestep,
                                                                                   alchemical_functions=alchemical_functions, splitting=splitting, nsteps_neq=nsteps,
                                                                                   measure_shadow_work=True)
            nonequilibrium_context = openmm.Context(system, nonequilibrium_integrator, platform)
            nonequilibrium_context.setPositions(positions)
            if nsteps == 0:
                nonequilibrium_integrator.step(1) # need to execute at least one step
            else:
                nonequilibrium_integrator.step(nsteps)
            if direction == 'forward':
                w_f[iteration] = nonequilibrium_integrator.get_total_work(dimensionless=True)
            else:
                w_r[iteration] = nonequilibrium_integrator.get_total_work(dimensionless=True)
            del nonequilibrium_context, nonequilibrium_integrator

    dF, ddF = pymbar.BAR(w_f, w_r)
    nsigma = np.abs(dF - dF_analytical) / ddF
    print("analytical DeltaF: {:12.4f}, DeltaF: {:12.4f}, dDeltaF: {:12.4f}, nsigma: {:12.1f}".format(dF_analytical, dF, ddF, nsigma))
    if nsigma > NSIGMA_MAX:
        raise Exception("The free energy difference for the nonequilibrium switching for splitting '%s' and %d steps is not zero within statistical error." % (splitting, nsteps))

def run_nonequilibrium_switching(init_x, alchemical_integrator, nsteps, alchemical_ctx, temperature=298 * unit.kelvin):
    """Perform a nonequilibrium switching protocol

    Parameters
    ----------
    init_x : simtk.openmm.Quantity of size [natoms,3] with units compatible with angstroms
        Initial positions
    alchemical_integrator : AlchemicalNonequilibriumLangevinIntegrator
        Integrator to use for switching
    alchemical_ctx : simtk.openmm.Context
        Context to use for alchemical switching.
    temperature : simtk.unit.Quantity, optional, default=298*kelvin
        Temperature to initialize simulation with

    Returns
    -------
    protocol_work : float
        Work performed by protocol
    """
    # Get number of NCMC steps
    nsteps = alchemical_integrator.getGlobalVariableByName("nsteps")
    # Set positions and velocities
    alchemical_ctx.setPositions(init_x)
    alchemical_ctx.setVelocitiesToTemperature(temperature)
    # Reset the integrator
    alchemical_integrator.reset_integrator()
    if (nsteps == 0):
        # We still need to take one step if nsteps == 0
        alchemical_integrator.step(1)
    else:
        alchemical_integrator.step(nsteps)
    # Get the protocol work in dimensionless units (kT)
    return alchemical_integrator.getGlobalVariableByName("protocol_work") # in kT

def test_alchemical_langevin_integrator():
    for splitting in ["O { V R H R V } O", "O V R H R V O", "R V O H O V R", "H R V O V R H"]:
        for nsteps in [0, 10, 50]:
            run_alchemical_langevin_integrator(nsteps=nsteps)

if __name__=="__main__":
    test_alchemical_langevin_integrator()
