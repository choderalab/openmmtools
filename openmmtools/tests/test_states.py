#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test State classes in states.py.

"""

# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import nose
import pickle
import operator

from openmmtools import testsystems
from openmmtools.states import *


# =============================================================================
# CONSTANTS
# =============================================================================

# We use CPU as OpenCL sometimes causes segfaults on Travis.
DEFAULT_PLATFORM = openmm.Platform.getPlatformByName('CPU')
DEFAULT_PLATFORM.setPropertyDefaultValue('DeterministicForces', 'true')


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_default_context(thermodynamic_state, integrator):
    """Shortcut to create a context from the thermodynamic state using the DEFAULT_PLATFORM."""
    return thermodynamic_state.create_context(integrator, DEFAULT_PLATFORM)


def get_barostat_temperature(barostat):
    """Backward-compatibly get barostat's temperature"""
    try:  # TODO drop this when we stop openmm7.0 support
        return barostat.getDefaultTemperature()
    except AttributeError:  # versions previous to OpenMM 7.1
        return barostat.getTemperature()


# =============================================================================
# TEST THERMODYNAMIC STATE
# =============================================================================

class TestThermodynamicState(object):
    """Test suite for states.ThermodynamicState class."""

    @classmethod
    def setup_class(cls):
        """Create the test systems used in the test suite."""
        cls.std_pressure = ThermodynamicState._STANDARD_PRESSURE
        cls.std_temperature = ThermodynamicState._STANDARD_TEMPERATURE

        alanine_explicit = testsystems.AlanineDipeptideExplicit()
        cls.alanine_positions = alanine_explicit.positions
        cls.alanine_no_thermostat = alanine_explicit.system

        toluene_implicit = testsystems.TolueneImplicit()
        cls.toluene_positions = toluene_implicit.positions
        cls.toluene_implicit = toluene_implicit.system
        cls.toluene_vacuum = testsystems.TolueneVacuum().system
        thermostat = openmm.AndersenThermostat(cls.std_temperature,
                                               1.0/unit.picosecond)
        cls.toluene_vacuum.addForce(thermostat)
        cls.alanine_explicit = copy.deepcopy(cls.alanine_no_thermostat)
        thermostat = openmm.AndersenThermostat(cls.std_temperature,
                                               1.0/unit.picosecond)
        cls.alanine_explicit.addForce(thermostat)

        # A system correctly barostated
        cls.barostated_alanine = copy.deepcopy(cls.alanine_explicit)
        barostat = openmm.MonteCarloBarostat(cls.std_pressure, cls.std_temperature)
        cls.barostated_alanine.addForce(barostat)

        # A non-periodic system barostated
        cls.barostated_toluene = copy.deepcopy(cls.toluene_vacuum)
        barostat = openmm.MonteCarloBarostat(cls.std_pressure, cls.std_temperature)
        cls.barostated_toluene.addForce(barostat)

        # A system with two identical MonteCarloBarostats
        cls.multiple_barostat_alanine = copy.deepcopy(cls.alanine_explicit)
        barostat = openmm.MonteCarloBarostat(cls.std_pressure, cls.std_temperature)
        cls.multiple_barostat_alanine.addForce(barostat)
        barostat = openmm.MonteCarloBarostat(cls.std_pressure, cls.std_temperature)
        cls.multiple_barostat_alanine.addForce(barostat)

        # A system with an unsupported MonteCarloAnisotropicBarostat
        cls.unsupported_barostat_alanine = copy.deepcopy(cls.alanine_explicit)
        pressure_in_bars = cls.std_pressure / unit.bar
        anisotropic_pressure = openmm.Vec3(pressure_in_bars, pressure_in_bars,
                                           pressure_in_bars)
        cls.anisotropic_barostat = openmm.MonteCarloAnisotropicBarostat(anisotropic_pressure,
                                                                        cls.std_temperature)
        cls.unsupported_barostat_alanine.addForce(cls.anisotropic_barostat)

        # A system with an inconsistent pressure in the barostat.
        cls.inconsistent_pressure_alanine = copy.deepcopy(cls.alanine_explicit)
        barostat = openmm.MonteCarloBarostat(cls.std_pressure + 0.2*unit.bar,
                                             cls.std_temperature)
        cls.inconsistent_pressure_alanine.addForce(barostat)

        # A system with an inconsistent temperature in the barostat.
        cls.inconsistent_temperature_alanine = copy.deepcopy(cls.alanine_no_thermostat)
        barostat = openmm.MonteCarloBarostat(cls.std_pressure,
                                             cls.std_temperature + 1.0*unit.kelvin)
        thermostat = openmm.AndersenThermostat(cls.std_temperature + 1.0*unit.kelvin,
                                               1.0/unit.picosecond)
        cls.inconsistent_temperature_alanine.addForce(barostat)
        cls.inconsistent_temperature_alanine.addForce(thermostat)

    @staticmethod
    def get_integrators(temperature):
        friction = 5.0/unit.picosecond
        time_step = 2.0*unit.femtosecond

        # Test cases
        verlet = openmm.VerletIntegrator(time_step)
        langevin = openmm.LangevinIntegrator(temperature,
                                             friction, time_step)
        velocity_verlet = integrators.VelocityVerletIntegrator()
        ghmc = integrators.GHMCIntegrator(temperature)
        # Copying a CustomIntegrator will make it lose any extra function
        # including the temperature getter/setter, so we must test this.
        custom_ghmc = copy.deepcopy(ghmc)

        compound_ghmc = openmm.CompoundIntegrator()
        compound_ghmc.addIntegrator(openmm.VerletIntegrator(time_step))
        compound_ghmc.addIntegrator(integrators.GHMCIntegrator(temperature))

        compound_verlet = openmm.CompoundIntegrator()
        compound_verlet.addIntegrator(openmm.VerletIntegrator(time_step))
        compound_verlet.addIntegrator(openmm.VerletIntegrator(time_step))

        return [(False, verlet), (False, velocity_verlet), (False, compound_verlet),
                (True, langevin), (True, ghmc), (True, custom_ghmc), (True, compound_ghmc)]

    def test_single_instance_standard_system(self):
        """ThermodynamicState should store only 1 System per compatible state."""
        state_nvt_300 = ThermodynamicState(system=self.alanine_explicit, temperature=300*unit.kelvin)
        state_nvt_350 = ThermodynamicState(system=self.alanine_explicit, temperature=350*unit.kelvin)
        state_npt_1 = ThermodynamicState(system=self.alanine_explicit, pressure=1.0*unit.atmosphere)
        state_npt_2 = ThermodynamicState(system=self.alanine_explicit, pressure=2.0*unit.atmosphere)
        assert state_nvt_300._standard_system == state_nvt_350._standard_system
        assert state_nvt_300._standard_system != state_npt_1._standard_system
        assert state_npt_1._standard_system == state_npt_2._standard_system

    def test_deepcopy(self):
        """Test that copy/deepcopy doesn't generate a new System instance."""
        state = ThermodynamicState(system=self.barostated_alanine)
        copied_state = copy.copy(state)
        deepcopied_state = copy.deepcopy(state)
        assert state._standard_system == copied_state._standard_system
        assert state._standard_system == deepcopied_state._standard_system

    def test_method_find_barostat(self):
        """ThermodynamicState._find_barostat() method."""
        barostat = ThermodynamicState._find_barostat(self.barostated_alanine)
        assert isinstance(barostat, openmm.MonteCarloBarostat)

        # Raise exception if multiple or unsupported barostats found
        TE = ThermodynamicsError  # shortcut
        test_cases = [(self.multiple_barostat_alanine, TE.MULTIPLE_BAROSTATS),
                      (self.unsupported_barostat_alanine, TE.UNSUPPORTED_BAROSTAT)]
        for system, err_code in test_cases:
            with nose.tools.assert_raises(ThermodynamicsError) as cm:
                ThermodynamicState._find_barostat(system)
            assert cm.exception.code == err_code

    def test_method_find_thermostat(self):
        """ThermodynamicState._find_thermostat() method."""
        system = copy.deepcopy(self.alanine_no_thermostat)
        assert ThermodynamicState._find_thermostat(system) is None
        thermostat = openmm.AndersenThermostat(self.std_temperature,
                                               1.0/unit.picosecond)
        system.addForce(thermostat)
        assert ThermodynamicState._find_thermostat(system) is not None

        # An error is raised with two thermostats.
        thermostat2 = openmm.AndersenThermostat(self.std_temperature,
                                                1.0/unit.picosecond)
        system.addForce(thermostat2)
        with nose.tools.assert_raises(ThermodynamicsError) as cm:
            ThermodynamicState._find_thermostat(system)
        cm.exception.code == ThermodynamicsError.MULTIPLE_THERMOSTATS

    def test_method_is_barostat_consistent(self):
        """ThermodynamicState._is_barostat_consistent() method."""
        temperature = self.std_temperature
        pressure = self.std_pressure
        state = ThermodynamicState(self.barostated_alanine, temperature)

        barostat = openmm.MonteCarloBarostat(pressure, temperature)
        assert state._is_barostat_consistent(barostat)
        barostat = openmm.MonteCarloBarostat(pressure + 0.2*unit.bar, temperature)
        assert not state._is_barostat_consistent(barostat)
        barostat = openmm.MonteCarloBarostat(pressure, temperature + 10*unit.kelvin)
        assert not state._is_barostat_consistent(barostat)

    def test_method_set_system_temperature(self):
        """ThermodynamicState._set_system_temperature() method."""
        system = copy.deepcopy(self.alanine_no_thermostat)
        assert ThermodynamicState._find_thermostat(system) is None

        # Add a thermostat to the system.
        ThermodynamicState._set_system_temperature(system, self.std_temperature)
        thermostat = ThermodynamicState._find_thermostat(system)
        assert thermostat.getDefaultTemperature() == self.std_temperature

        # Change temperature of thermostat and barostat.
        new_temperature = self.std_temperature + 1.0*unit.kelvin
        ThermodynamicState._set_system_temperature(system, new_temperature)
        assert thermostat.getDefaultTemperature() == new_temperature

    def test_property_temperature(self):
        """ThermodynamicState.temperature property."""
        state = ThermodynamicState(self.barostated_alanine,
                                   self.std_temperature)
        assert state.temperature == self.std_temperature

        temperature = self.std_temperature + 10.0*unit.kelvin
        state.temperature = temperature
        assert state.temperature == temperature
        assert get_barostat_temperature(state.barostat) == temperature

        # Setting temperature to None raise error.
        with nose.tools.assert_raises(ThermodynamicsError) as cm:
            state.temperature = None
        assert cm.exception.code == ThermodynamicsError.NONE_TEMPERATURE

    def test_method_set_system_pressure(self):
        """ThermodynamicState._set_system_pressure() method."""
        state = ThermodynamicState(self.alanine_explicit, self.std_temperature)
        system = state.system
        assert state._find_barostat(system) is None
        state._set_system_pressure(system, self.std_pressure)
        assert state._find_barostat(system).getDefaultPressure() == self.std_pressure
        state._set_system_pressure(system, None)
        assert state._find_barostat(system) is None

    def test_property_pressure_barostat(self):
        """ThermodynamicState.pressure and barostat properties."""
        # Vacuum and implicit system are read with no pressure
        nonperiodic_testcases = [self.toluene_vacuum, self.toluene_implicit]
        new_barostat = openmm.MonteCarloBarostat(1.0*unit.bar, self.std_temperature)
        for system in nonperiodic_testcases:
            state = ThermodynamicState(system, self.std_temperature)
            assert state.pressure is None
            assert state.barostat is None

            # We can't set the pressure on non-periodic systems
            with nose.tools.assert_raises(ThermodynamicsError) as cm:
                state.pressure = 1.0*unit.bar
            assert cm.exception.code == ThermodynamicsError.BAROSTATED_NONPERIODIC
            with nose.tools.assert_raises(ThermodynamicsError) as cm:
                state.barostat = new_barostat
            assert cm.exception.code == ThermodynamicsError.BAROSTATED_NONPERIODIC

            assert state.pressure is None
            assert state.barostat is None

        # Correctly reads and set system pressures
        periodic_testcases = [self.alanine_explicit]
        for system in periodic_testcases:
            state = ThermodynamicState(system, self.std_temperature)
            assert state.pressure is None
            assert state.barostat is None

            # Setting pressure adds a barostat
            state.pressure = self.std_pressure
            assert state.pressure == self.std_pressure
            assert state.barostat.getDefaultPressure() == self.std_pressure
            assert get_barostat_temperature(state.barostat) == self.std_temperature

            # Changing the exposed barostat doesn't affect the state.
            new_pressure = self.std_pressure + 1.0*unit.bar
            barostat = state.barostat
            barostat.setDefaultPressure(new_pressure)
            assert state.barostat.getDefaultPressure() == self.std_pressure

            # Setting new pressure changes the barostat parameters
            state.pressure = new_pressure
            assert state.pressure == new_pressure
            assert state.barostat.getDefaultPressure() == new_pressure
            assert get_barostat_temperature(state.barostat) == self.std_temperature

            # Assigning the barostat changes the pressure
            barostat = state.barostat
            barostat.setDefaultPressure(self.std_pressure)
            state.barostat = barostat
            assert state.pressure == self.std_pressure

            # Setting pressure of the assigned barostat doesn't change TS internals
            barostat.setDefaultPressure(new_pressure)
            assert state.pressure == self.std_pressure

            # Setting pressure to None removes barostat and viceversa.
            state.pressure = None
            assert state.barostat is None

            state.pressure = self.std_pressure
            state.barostat = None
            assert state.pressure is None

            # It is impossible to assign an unsupported barostat with incorrect temperature
            new_temperature = self.std_temperature + 10.0*unit.kelvin
            ThermodynamicState._set_barostat_temperature(barostat, new_temperature)
            with nose.tools.assert_raises(ThermodynamicsError) as cm:
                state.barostat = barostat
            assert cm.exception.code == ThermodynamicsError.INCONSISTENT_BAROSTAT

            # Assign incompatible barostat raise error
            with nose.tools.assert_raises(ThermodynamicsError) as cm:
                state.barostat = self.anisotropic_barostat
            assert cm.exception.code == ThermodynamicsError.UNSUPPORTED_BAROSTAT

            # After exception, state is left consistent
            assert state.pressure is None

    def test_property_volume(self):
        """Check that volume is computed correctly."""
        # For volume-fluctuating systems volume is None.
        state = ThermodynamicState(self.barostated_alanine, self.std_temperature)
        assert state.volume is None

        # For periodic systems in NVT, volume is correctly computed.
        system = self.alanine_explicit
        box_vectors = system.getDefaultPeriodicBoxVectors()
        volume = box_vectors[0][0] * box_vectors[1][1] * box_vectors[2][2]
        state = ThermodynamicState(system, self.std_temperature)
        assert state.volume == volume

        # For non-periodic systems, volume is None.
        state = ThermodynamicState(self.toluene_vacuum, self.std_temperature)
        assert state.volume is None

    def test_property_system(self):
        """Cannot set a system in a different thermodynamic state."""
        state = ThermodynamicState(self.barostated_alanine, self.std_temperature)
        assert state.pressure == self.std_pressure  # pre-condition

        inconsistent_barostat_temperature = copy.deepcopy(self.inconsistent_temperature_alanine)
        thermostat = state._find_thermostat(inconsistent_barostat_temperature)
        thermostat.setDefaultTemperature(self.std_temperature)

        TE = ThermodynamicsError  # shortcut
        test_cases = [(self.toluene_vacuum, TE.NO_BAROSTAT),
                      (self.barostated_toluene, TE.BAROSTATED_NONPERIODIC),
                      (self.multiple_barostat_alanine, TE.MULTIPLE_BAROSTATS),
                      (self.inconsistent_pressure_alanine, TE.INCONSISTENT_BAROSTAT),
                      (self.inconsistent_temperature_alanine, TE.INCONSISTENT_THERMOSTAT),
                      (inconsistent_barostat_temperature, TE.INCONSISTENT_BAROSTAT)]
        for i, (system, error_code) in enumerate(test_cases):
            with nose.tools.assert_raises(ThermodynamicsError) as cm:
                state.system = system
            assert cm.exception.code == error_code

        # It is possible to set an inconsistent system
        # if thermodynamic state is changed first.
        inconsistent_system = self.inconsistent_pressure_alanine
        state.pressure = self.std_pressure + 0.2*unit.bar
        state.system = self.inconsistent_pressure_alanine
        state_system_str = openmm.XmlSerializer.serialize(state.system)
        inconsistent_system_str = openmm.XmlSerializer.serialize(inconsistent_system)
        assert state_system_str == inconsistent_system_str

    def test_method_set_system(self):
        """ThermodynamicState.set_system() method."""
        system = copy.deepcopy(self.alanine_no_thermostat)
        state = ThermodynamicState(system, self.std_temperature)

        # We can't set the system without adding a thermostat.
        with nose.tools.assert_raises(ThermodynamicsError) as cm:
            state.set_system(system)
        assert cm.exception.code == ThermodynamicsError.NO_THERMOSTAT

        state.set_system(system, fix_state=True)
        system = state.system
        thermostat = state._find_thermostat(system)
        assert utils.is_quantity_close(thermostat.getDefaultTemperature(), self.std_temperature)
        assert state.barostat is None

        # In NPT, we can't set the system without adding a barostat.
        system = state.system  # System with thermostat.
        state.pressure = self.std_pressure
        with nose.tools.assert_raises(ThermodynamicsError) as cm:
            state.set_system(system)
        assert cm.exception.code == ThermodynamicsError.NO_BAROSTAT

        state.set_system(system, fix_state=True)
        assert state.barostat.getDefaultPressure() == self.std_pressure
        assert get_barostat_temperature(state.barostat) == self.std_temperature

    def test_method_get_system(self):
        """ThermodynamicState.get_system() method."""
        temperature = 400 * unit.kelvin
        pressure = 10 * unit.bar
        state = ThermodynamicState(self.alanine_explicit, temperature, pressure)

        # Normally a system has both barostat and thermostat
        system = state.get_system()
        assert state._find_barostat(system).getDefaultPressure() == pressure
        assert state._find_thermostat(system).getDefaultTemperature() == temperature

        # We can request a system without thermostat or barostat.
        system = state.get_system(remove_thermostat=True)
        assert state._find_thermostat(system) is None
        assert get_barostat_temperature(state._find_barostat(system)) == temperature
        system = state.get_system(remove_barostat=True)
        assert state._find_barostat(system) is None

    def test_constructor_unsupported_barostat(self):
        """Exception is raised on construction with unsupported barostats."""
        TE = ThermodynamicsError  # shortcut
        test_cases = [(self.barostated_toluene, TE.BAROSTATED_NONPERIODIC),
                      (self.multiple_barostat_alanine, TE.MULTIPLE_BAROSTATS),
                      (self.unsupported_barostat_alanine, TE.UNSUPPORTED_BAROSTAT)]
        for i, (system, err_code) in enumerate(test_cases):
            with nose.tools.assert_raises(TE) as cm:
                ThermodynamicState(system=system, temperature=self.std_temperature)
            assert cm.exception.code == err_code

    def test_constructor_barostat(self):
        """The system barostat is properly configured on construction."""
        system = self.alanine_explicit
        old_serialization = openmm.XmlSerializer.serialize(system)
        assert ThermodynamicState._find_barostat(system) is None  # Test precondition.

        # If we don't specify pressure, no barostat is added
        state = ThermodynamicState(system=system, temperature=self.std_temperature)
        assert state.barostat is None

        # If we specify pressure, barostat is added
        state = ThermodynamicState(system=system, temperature=self.std_temperature,
                                   pressure=self.std_pressure)
        assert state.barostat is not None

        # If we feed a barostat with an inconsistent temperature, it's fixed.
        state = ThermodynamicState(self.inconsistent_temperature_alanine,
                                   temperature=self.std_temperature)
        assert state._is_barostat_consistent(state.barostat)

        # If we feed a barostat with an inconsistent pressure, it's fixed.
        state = ThermodynamicState(self.inconsistent_pressure_alanine,
                                   temperature=self.std_temperature,
                                   pressure=self.std_pressure)
        assert state.pressure == self.std_pressure

        # The original system is unaltered.
        new_serialization = openmm.XmlSerializer.serialize(system)
        assert new_serialization == old_serialization

    def test_constructor_thermostat(self):
        """The system thermostat is properly configured on construction."""
        # If we don't specify a temperature without a thermostat, it complains.
        system = self.alanine_no_thermostat
        assert ThermodynamicState._find_thermostat(system) is None  # Test precondition.
        with nose.tools.assert_raises(ThermodynamicsError) as cm:
            ThermodynamicState(system=system)
        assert cm.exception.code == ThermodynamicsError.NO_THERMOSTAT

        # With thermostat, temperature is inferred correctly.
        system = copy.deepcopy(self.alanine_explicit)
        new_temperature = self.std_temperature + 1.0*unit.kelvin
        thermostat = ThermodynamicState._find_thermostat(system)
        thermostat.setDefaultTemperature(new_temperature)
        state = ThermodynamicState(system=system)
        assert state.temperature == new_temperature

        # If barostat is inconsistent, an error is raised.
        system.addForce(openmm.MonteCarloBarostat(self.std_pressure, self.std_temperature))
        with nose.tools.assert_raises(ThermodynamicsError) as cm:
            ThermodynamicState(system=system)
        assert cm.exception.code == ThermodynamicsError.INCONSISTENT_BAROSTAT

        # Specifying temperature overwrite thermostat.
        state = ThermodynamicState(system=system, temperature=self.std_temperature)
        assert state.temperature == self.std_temperature
        assert get_barostat_temperature(state.barostat) == self.std_temperature

    def test_method_is_integrator_thermostated(self):
        """ThermodynamicState._is_integrator_thermostated method."""
        state = ThermodynamicState(self.toluene_vacuum, self.std_temperature)
        test_cases = self.get_integrators(self.std_temperature)
        inconsistent_temperature = self.std_temperature + 1.0*unit.kelvin

        for thermostated, integrator in test_cases:
            # If integrator expose a getTemperature method, return True.
            assert state._is_integrator_thermostated(integrator) is thermostated

            # If temperature is different, it raises an exception.
            if thermostated:
                for _integrator in ThermodynamicState._loop_over_integrators(integrator):
                    try:
                        _integrator.setTemperature(inconsistent_temperature)
                    except AttributeError:  # handle CompoundIntegrator case
                        pass
                with nose.tools.assert_raises(ThermodynamicsError) as cm:
                    state._is_integrator_thermostated(integrator)
                assert cm.exception.code == ThermodynamicsError.INCONSISTENT_INTEGRATOR

    def test_method_set_integrator_temperature(self):
        """ThermodynamicState._set_integrator_temperature() method."""
        test_cases = self.get_integrators(self.std_temperature)
        new_temperature = self.std_temperature + 1.0*unit.kelvin
        state = ThermodynamicState(self.toluene_vacuum, new_temperature)

        for thermostated, integrator in test_cases:
            if thermostated:
                assert state._set_integrator_temperature(integrator)
                for _integrator in ThermodynamicState._loop_over_integrators(integrator):
                    try:
                        assert _integrator.getTemperature() == new_temperature
                    except AttributeError:  # handle CompoundIntegrator case
                        pass
            else:
                # It doesn't explode with integrators not coupled to a heat bath
                assert not state._set_integrator_temperature(integrator)

    def test_method_standardize_system(self):
        """ThermodynamicState._standardize_system() class method."""

        def check_barostat_thermostat(_system, cmp_op):
            barostat = ThermodynamicState._find_barostat(_system)
            thermostat = ThermodynamicState._find_thermostat(_system)
            assert cmp_op(barostat.getDefaultPressure(), self.std_pressure)
            assert cmp_op(get_barostat_temperature(barostat), self.std_temperature)
            assert cmp_op(thermostat.getDefaultTemperature(), self.std_temperature)

        # Create NPT system in non-standard state.
        npt_state = ThermodynamicState(self.inconsistent_pressure_alanine,
                                       self.std_temperature + 1.0*unit.kelvin)
        npt_system = npt_state.system
        check_barostat_thermostat(npt_system, operator.ne)

        # Standardize system sets pressure and temperature to standard.
        npt_state._standardize_system(npt_system)
        check_barostat_thermostat(npt_system, operator.eq)

    def test_method_create_context(self):
        """ThermodynamicState.create_context() method."""
        state = ThermodynamicState(self.toluene_vacuum, self.std_temperature)
        toluene_str = openmm.XmlSerializer.serialize(self.toluene_vacuum)
        test_integrators = self.get_integrators(self.std_temperature)
        inconsistent_temperature = self.std_temperature + 1.0*unit.kelvin

        # Divide test platforms among the integrators since we
        # can't bind the same integrator to multiple contexts.
        test_platforms = utils.get_available_platforms()
        test_platforms = [test_platforms[i % len(test_platforms)]
                          for i in range(len(test_integrators))]

        for (is_thermostated, integrator), platform in zip(test_integrators, test_platforms):
            context = state.create_context(integrator, platform)
            assert platform is None or platform.getName() == context.getPlatform().getName()
            assert isinstance(integrator, context.getIntegrator().__class__)

            if is_thermostated:
                assert state._find_thermostat(context.getSystem()) is None

                # create_context complains if integrator is inconsistent
                inconsistent_integrator = copy.deepcopy(integrator)
                for _integrator in ThermodynamicState._loop_over_integrators(inconsistent_integrator):
                    try:
                        _integrator.setTemperature(inconsistent_temperature)
                    except AttributeError:  # handle CompoundIntegrator case
                        pass
                with nose.tools.assert_raises(ThermodynamicsError) as cm:
                    state.create_context(inconsistent_integrator)
                assert cm.exception.code == ThermodynamicsError.INCONSISTENT_INTEGRATOR
            else:
                # The context system must have the thermostat.
                assert toluene_str == context.getSystem().__getstate__()
                assert state._find_thermostat(context.getSystem()) is not None

            # Get rid of old context. This test can create a lot of them.
            del context, integrator

    def test_method_is_compatible(self):
        """ThermodynamicState context and state compatibility methods."""

        def check_compatibility(state1, state2, is_compatible):
            """Check compatibility of contexts thermostated by force or integrator."""
            assert state1.is_state_compatible(state2) is is_compatible
            assert state2.is_state_compatible(state1) is is_compatible
            time_step = 1.0*unit.femtosecond
            friction = 5.0/unit.picosecond
            integrator1 = openmm.VerletIntegrator(time_step)
            integrator2 = openmm.LangevinIntegrator(state2.temperature, friction, time_step)
            context1 = create_default_context(state1, integrator1)
            context2 = create_default_context(state2, integrator2)
            assert state1.is_context_compatible(context2) is is_compatible
            assert state2.is_context_compatible(context1) is is_compatible

        toluene_vacuum = ThermodynamicState(self.toluene_vacuum, self.std_temperature)
        toluene_implicit = ThermodynamicState(self.toluene_implicit, self.std_temperature)
        alanine_explicit = ThermodynamicState(self.alanine_explicit, self.std_temperature)
        barostated_alanine = ThermodynamicState(self.barostated_alanine, self.std_temperature)

        # Different systems/ensembles are incompatible.
        check_compatibility(toluene_vacuum, toluene_vacuum, True)
        check_compatibility(toluene_vacuum, toluene_implicit, False)
        check_compatibility(toluene_implicit, alanine_explicit, False)
        check_compatibility(alanine_explicit, barostated_alanine, False)

        # System in same ensemble with different parameters are compatible.
        alanine_explicit2 = copy.deepcopy(alanine_explicit)
        alanine_explicit2.temperature = alanine_explicit.temperature + 1.0*unit.kelvin
        check_compatibility(alanine_explicit, alanine_explicit2, True)

        barostated_alanine2 = copy.deepcopy(barostated_alanine)
        barostated_alanine2.pressure = barostated_alanine.pressure + 0.2*unit.bars
        check_compatibility(barostated_alanine, barostated_alanine2, True)

    def test_method_apply_to_context(self):
        """ThermodynamicState.apply_to_context() method."""
        friction = 5.0/unit.picosecond
        time_step = 2.0*unit.femtosecond
        state0 = ThermodynamicState(self.barostated_alanine, self.std_temperature)

        langevin_integrator = openmm.LangevinIntegrator(self.std_temperature, friction, time_step)
        context = create_default_context(state0, langevin_integrator)

        verlet_integrator = openmm.VerletIntegrator(1.0*unit.femtosecond)
        thermostated_context = create_default_context(state0, verlet_integrator)

        # Change context pressure.
        barostat = state0._find_barostat(context.getSystem())
        assert barostat.getDefaultPressure() == self.std_pressure
        assert context.getParameter(barostat.Pressure()) == self.std_pressure / unit.bar
        new_pressure = self.std_pressure + 1.0*unit.bars
        state1 = ThermodynamicState(self.barostated_alanine, self.std_temperature,
                                    new_pressure)
        state1.apply_to_context(context)
        assert barostat.getDefaultPressure() == new_pressure
        assert context.getParameter(barostat.Pressure()) == new_pressure / unit.bar

        # Change context temperature.
        for c in [context, thermostated_context]:
            barostat = state0._find_barostat(c.getSystem())
            thermostat = state0._find_thermostat(c.getSystem())

            # Pre-conditions.
            assert get_barostat_temperature(barostat) == self.std_temperature
            # TODO remove try except when OpenMM 7.1 works on travis
            try:
                assert c.getParameter(barostat.Temperature()) == self.std_temperature / unit.kelvin
            except AttributeError:
                pass
            if thermostat is not None:
                assert c.getParameter(thermostat.Temperature()) == self.std_temperature / unit.kelvin
            else:
                assert context.getIntegrator().getTemperature() == self.std_temperature

            new_temperature = self.std_temperature + 10.0*unit.kelvin
            state2 = ThermodynamicState(self.barostated_alanine, new_temperature)
            state2.apply_to_context(c)

            assert get_barostat_temperature(barostat) == new_temperature
            # TODO remove try except when OpenMM 7.1 works on travis
            try:
                assert c.getParameter(barostat.Temperature()) == new_temperature / unit.kelvin
            except AttributeError:
                pass
            if thermostat is not None:
                assert c.getParameter(thermostat.Temperature()) == new_temperature / unit.kelvin
            else:
                assert context.getIntegrator().getTemperature() == new_temperature

        # Trying to apply to a system in a different ensemble raises an error.
        state2.pressure = None
        with nose.tools.assert_raises(ThermodynamicsError) as cm:
            state2.apply_to_context(context)
        assert cm.exception.code == ThermodynamicsError.INCOMPATIBLE_ENSEMBLE

        # Clean up contexts.
        del context, langevin_integrator
        del thermostated_context, verlet_integrator

        verlet_integrator = openmm.VerletIntegrator(time_step)
        nvt_context = create_default_context(state2, verlet_integrator)
        with nose.tools.assert_raises(ThermodynamicsError) as cm:
            state1.apply_to_context(nvt_context)
        assert cm.exception.code == ThermodynamicsError.INCOMPATIBLE_ENSEMBLE
        del nvt_context, verlet_integrator

    def test_method_reduced_potential(self):
        """ThermodynamicState.reduced_potential() method."""
        kj_mol = unit.kilojoule_per_mole
        beta = 1.0 / (unit.MOLAR_GAS_CONSTANT_R * self.std_temperature)
        state = ThermodynamicState(self.alanine_explicit, self.std_temperature)
        integrator = openmm.VerletIntegrator(1.0*unit.femtosecond)
        context = create_default_context(state, integrator)
        context.setPositions(self.alanine_positions)
        sampler_state = SamplerState.from_context(context)

        # Compute constant volume reduced potential.
        reduced_potential = state.reduced_potential(sampler_state)
        potential_energy = reduced_potential / beta / kj_mol
        assert np.isclose(sampler_state.potential_energy / kj_mol, potential_energy)
        assert np.isclose(reduced_potential, state.reduced_potential(context))

        # Compute constant pressure reduced potential.
        state.pressure = self.std_pressure
        reduced_potential = state.reduced_potential(sampler_state)
        pressure_volume_work = (self.std_pressure * sampler_state.volume *
                                unit.AVOGADRO_CONSTANT_NA)
        potential_energy = (reduced_potential / beta - pressure_volume_work) / kj_mol
        assert np.isclose(sampler_state.potential_energy / kj_mol, potential_energy)
        assert np.isclose(reduced_potential, state.reduced_potential(context))

        # Raise error if SamplerState is not compatible.
        incompatible_sampler_state = sampler_state[:-1]
        with nose.tools.assert_raises(ThermodynamicsError) as cm:
            state.reduced_potential(incompatible_sampler_state)
        assert cm.exception.code == ThermodynamicsError.INCOMPATIBLE_SAMPLER_STATE

    def test_method_reduced_potential_at_states(self):
        """ThermodynamicState.reduced_potential_at_states() method.

        Computing the reduced potential singularly and with the class
        method should give the same result.
        """
        # Build a mixed collection of compatible and incompatible thermodynamic states.
        thermodynamic_states = [
            ThermodynamicState(self.alanine_explicit, temperature=300*unit.kelvin,
                               pressure=1.0*unit.atmosphere),
            ThermodynamicState(self.toluene_implicit, temperature=200*unit.kelvin),
            ThermodynamicState(self.alanine_explicit, temperature=250*unit.kelvin,
                               pressure=1.2*unit.atmosphere)
        ]

        # Group thermodynamic states by compatibility.
        compatible_groups, original_indices = group_by_compatibility(thermodynamic_states)
        assert len(compatible_groups) == 2
        assert original_indices == [[0, 2], [1]]

        # Compute the reduced potentials.
        expected_energies = []
        obtained_energies = []
        for compatible_group in compatible_groups:
            # Create context.
            integrator = openmm.VerletIntegrator(2.0*unit.femtoseconds)
            context = create_default_context(compatible_group[0], integrator)
            if len(compatible_group) == 2:
                context.setPositions(self.alanine_positions)
            else:
                context.setPositions(self.toluene_positions)

            # Compute with single-state method.
            for state in compatible_group:
                state.apply_to_context(context)
                expected_energies.append(state.reduced_potential(context))

            # Compute with multi-state method.
            obtained_energies.extend(ThermodynamicState.reduced_potential_at_states(context, compatible_group))
        assert np.allclose(np.array(expected_energies), np.array(obtained_energies))


# =============================================================================
# TEST SAMPLER STATE
# =============================================================================

class TestSamplerState(object):
    """Test suite for states.SamplerState class."""

    @classmethod
    def setup_class(cls):
        """Create various variables shared by tests in suite."""
        temperature = 300*unit.kelvin
        alanine_vacuum = testsystems.AlanineDipeptideVacuum()
        cls.alanine_vacuum_positions = alanine_vacuum.positions
        cls.alanine_vacuum_state = ThermodynamicState(alanine_vacuum.system,
                                                      temperature)

        alanine_explicit = testsystems.AlanineDipeptideExplicit()
        cls.alanine_explicit_positions = alanine_explicit.positions
        cls.alanine_explicit_state = ThermodynamicState(alanine_explicit.system,
                                                        temperature)

    @staticmethod
    def is_sampler_state_equal_context(sampler_state, context):
        """Check sampler and openmm states in context are equal."""
        equal = True
        ss = sampler_state  # Shortcut.
        os = context.getState(getPositions=True, getEnergy=True,
                              getVelocities=True)
        equal = equal and np.allclose(ss.positions.value_in_unit(ss.positions.unit),
                                      os.getPositions().value_in_unit(ss.positions.unit))
        equal = equal and np.allclose(ss.velocities.value_in_unit(ss.velocities.unit),
                                      os.getVelocities().value_in_unit(ss.velocities.unit))
        equal = equal and np.allclose(ss.box_vectors.value_in_unit(ss.box_vectors.unit),
                                      os.getPeriodicBoxVectors().value_in_unit(ss.box_vectors.unit))
        equal = equal and np.isclose(ss.potential_energy.value_in_unit(ss.potential_energy.unit),
                                     os.getPotentialEnergy().value_in_unit(ss.potential_energy.unit))
        equal = equal and np.isclose(ss.kinetic_energy.value_in_unit(ss.kinetic_energy.unit),
                                     os.getKineticEnergy().value_in_unit(ss.kinetic_energy.unit))
        equal = equal and np.isclose(ss.volume.value_in_unit(ss.volume.unit),
                                     os.getPeriodicBoxVolume().value_in_unit(ss.volume.unit))
        return equal

    @staticmethod
    def create_context(thermodynamic_state):
        integrator = openmm.VerletIntegrator(1.0*unit.femtoseconds)
        return thermodynamic_state.create_context(integrator, DEFAULT_PLATFORM)

    def test_inconsistent_n_particles(self):
        """Exception raised with inconsistent positions and velocities."""
        positions = self.alanine_vacuum_positions
        sampler_state = SamplerState(positions)

        # If velocities have different length, an error is raised.
        velocities = [0.0 for _ in range(len(positions) - 1)]
        with nose.tools.assert_raises(SamplerStateError) as cm:
            sampler_state.velocities = velocities
        assert cm.exception.code == SamplerStateError.INCONSISTENT_VELOCITIES

        # The same happens in constructor.
        with nose.tools.assert_raises(SamplerStateError) as cm:
            SamplerState(positions, velocities)
        assert cm.exception.code == SamplerStateError.INCONSISTENT_VELOCITIES

        # The same happens if we update positions.
        with nose.tools.assert_raises(SamplerStateError) as cm:
            sampler_state.positions = positions[:-1]
        assert cm.exception.code == SamplerStateError.INCONSISTENT_POSITIONS

        # We cannot set positions to None.
        with nose.tools.assert_raises(SamplerStateError) as cm:
            sampler_state.positions = None
        assert cm.exception.code == SamplerStateError.INCONSISTENT_POSITIONS

    def test_constructor_from_context(self):
        """SamplerState.from_context constructor."""
        alanine_vacuum_context = self.create_context(self.alanine_vacuum_state)
        alanine_vacuum_context.setPositions(self.alanine_vacuum_positions)

        sampler_state = SamplerState.from_context(alanine_vacuum_context)
        assert self.is_sampler_state_equal_context(sampler_state, alanine_vacuum_context)

    def test_unitless_cache(self):
        """Test that the unitless cache for positions and velocities is invalidated."""
        positions = copy.deepcopy(self.alanine_vacuum_positions)

        alanine_vacuum_context = self.create_context(self.alanine_vacuum_state)
        alanine_vacuum_context.setPositions(copy.deepcopy(positions))

        test_cases = [
            SamplerState(positions),
            SamplerState.from_context(alanine_vacuum_context)
        ]

        pos_unit = unit.micrometer
        vel_unit = unit.micrometer / unit.nanosecond

        # Assigning an item invalidates the cache.
        for sampler_state in test_cases:
            old_unitless_positions = copy.deepcopy(sampler_state._unitless_positions)
            sampler_state.positions[5] = [1.0, 1.0, 1.0] * pos_unit
            assert sampler_state.positions.has_changed
            assert np.all(old_unitless_positions[5] != sampler_state._unitless_positions[5])
            sampler_state.positions = copy.deepcopy(positions)
            assert sampler_state._unitless_positions_cache is None

            if isinstance(sampler_state._positions._value, np.ndarray):
                old_unitless_positions = copy.deepcopy(sampler_state._unitless_positions)
                sampler_state.positions[5:8] = [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]] * pos_unit
                assert sampler_state.positions.has_changed
                assert np.all(old_unitless_positions[5:8] != sampler_state._unitless_positions[5:8])

            if sampler_state.velocities is not None:
                old_unitless_velocities = copy.deepcopy(sampler_state._unitless_velocities)
                sampler_state.velocities[5] = [1.0, 1.0, 1.0] * vel_unit
                assert sampler_state.velocities.has_changed
                assert np.all(old_unitless_velocities[5] != sampler_state._unitless_velocities[5])
                sampler_state.velocities = copy.deepcopy(sampler_state.velocities)
                assert sampler_state._unitless_velocities_cache is None

                if isinstance(sampler_state._velocities._value, np.ndarray):
                    old_unitless_velocities = copy.deepcopy(sampler_state._unitless_velocities)
                    sampler_state.velocities[5:8] = [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]] * vel_unit
                    assert sampler_state.velocities.has_changed
                    assert np.all(old_unitless_velocities[5:8] != sampler_state._unitless_velocities[5:8])
            else:
                assert sampler_state._unitless_velocities is None

    def test_method_is_context_compatible(self):
        """SamplerState.is_context_compatible() method."""
        # Vacuum.
        alanine_vacuum_context = self.create_context(self.alanine_vacuum_state)
        vacuum_sampler_state = SamplerState(self.alanine_vacuum_positions)

        # Explicit solvent.
        alanine_explicit_context = self.create_context(self.alanine_explicit_state)
        explicit_sampler_state = SamplerState(self.alanine_explicit_positions)

        assert vacuum_sampler_state.is_context_compatible(alanine_vacuum_context)
        assert not vacuum_sampler_state.is_context_compatible(alanine_explicit_context)
        assert explicit_sampler_state.is_context_compatible(alanine_explicit_context)
        assert not explicit_sampler_state.is_context_compatible(alanine_vacuum_context)

    def test_method_update_from_context(self):
        """SamplerState.update_from_context() method."""
        vacuum_context = self.create_context(self.alanine_vacuum_state)
        explicit_context = self.create_context(self.alanine_explicit_state)

        # Test that the update is successful
        vacuum_context.setPositions(self.alanine_vacuum_positions)
        sampler_state = SamplerState.from_context(vacuum_context)
        vacuum_context.getIntegrator().step(10)
        assert not self.is_sampler_state_equal_context(sampler_state, vacuum_context)
        sampler_state.update_from_context(vacuum_context)
        assert self.is_sampler_state_equal_context(sampler_state, vacuum_context)

        # Trying to update with an inconsistent context raise error.
        explicit_context.setPositions(self.alanine_explicit_positions)
        with nose.tools.assert_raises(SamplerStateError) as cm:
            sampler_state.update_from_context(explicit_context)
        assert cm.exception.code == SamplerStateError.INCONSISTENT_POSITIONS

    def test_method_apply_to_context(self):
        """SamplerState.apply_to_context() method."""
        explicit_context = self.create_context(self.alanine_explicit_state)
        explicit_context.setPositions(self.alanine_explicit_positions)
        sampler_state = SamplerState.from_context(explicit_context)

        explicit_context.getIntegrator().step(10)
        assert not self.is_sampler_state_equal_context(sampler_state, explicit_context)
        sampler_state.apply_to_context(explicit_context)
        assert self.is_sampler_state_equal_context(sampler_state, explicit_context)

    def test_operator_getitem(self):
        """SamplerState.__getitem__() method."""
        explicit_context = self.create_context(self.alanine_explicit_state)
        explicit_context.setPositions(self.alanine_explicit_positions)
        sampler_state = SamplerState.from_context(explicit_context)

        sliced_sampler_state = sampler_state[0]
        assert sliced_sampler_state.n_particles == 1
        assert len(sliced_sampler_state.velocities) == 1
        assert np.allclose(sliced_sampler_state.positions[0],
                           self.alanine_explicit_positions[0])

        # Modifying the sliced sampler state doesn't modify original.
        sliced_sampler_state.positions[0][0] += 1 * unit.angstrom
        assert sliced_sampler_state.positions[0][0] == sampler_state.positions[0][0] + 1 * unit.angstrom

        # SamplerState.__getitem__ should work for both slices and lists.
        for sliced_sampler_state in [sampler_state[2:10],
                                     sampler_state[list(range(2, 10))]]:
            assert sliced_sampler_state.n_particles == 8
            assert len(sliced_sampler_state.velocities) == 8
            assert np.allclose(sliced_sampler_state.positions,
                               self.alanine_explicit_positions[2:10])

        sliced_sampler_state = sampler_state[2:10:2]
        assert sliced_sampler_state.n_particles == 4
        assert len(sliced_sampler_state.velocities) == 4
        assert np.allclose(sliced_sampler_state.positions,
                           self.alanine_explicit_positions[2:10:2])

        # Modifying the sliced sampler state doesn't modify original. We check
        # this here too since the algorithm for slice objects is different.
        sliced_sampler_state.positions[0][0] += 1 * unit.angstrom
        assert sliced_sampler_state.positions[0][0] == sampler_state.positions[2][0] + 1 * unit.angstrom

        # The other attributes are copied correctly.
        assert sliced_sampler_state.volume == sampler_state.volume

        # Energies are undefined for as subset of atoms.
        assert sliced_sampler_state.kinetic_energy is None
        assert sliced_sampler_state.potential_energy is None

    def test_dict_representation(self):
        """Setting the state of the object should work when ignoring velocities."""
        alanine_vacuum_context = self.create_context(self.alanine_vacuum_state)
        alanine_vacuum_context.setPositions(self.alanine_vacuum_positions)
        alanine_vacuum_context.setVelocitiesToTemperature(300*unit.kelvin)

        # Test precondition.
        vacuum_sampler_state = SamplerState.from_context(alanine_vacuum_context)
        old_velocities = vacuum_sampler_state.velocities
        assert old_velocities is not None

        # Get a dictionary representation without velocities.
        serialization = vacuum_sampler_state.__getstate__(ignore_velocities=True)
        assert serialization['velocities'] is None

        # Do not overwrite velocities when setting a state.
        serialization['velocities'] = np.random.rand(*vacuum_sampler_state.positions.shape) * unit.nanometer/unit.picosecond
        vacuum_sampler_state.__setstate__(serialization, ignore_velocities=True)
        assert np.all(vacuum_sampler_state.velocities == old_velocities)

    def test_collective_variable(self):
        """Test that CV calculation is working"""
        # Setup the CV tests if we have a late enough OpenMM
        # alanine_explicit_cv = copy.deepcopy(self.alanine_explicit)
        system_cv = self.alanine_explicit_state.system
        cv_distance = openmm.CustomBondForce("r")
        cv_distance.addBond(0, 1, [])
        cv_angle = openmm.CustomAngleForce("theta")
        cv_angle.addAngle(0, 1, 2, [])
        # 3 unique CV names in the Context: BondCV, AngleCVSingle, AngleCV
        cv_single_1 = openmm.CustomCVForce("4*BondCV")
        # We are going to use this name later too
        cv_single_1.addCollectiveVariable('BondCV', copy.deepcopy(cv_distance))
        cv_single_2 = openmm.CustomCVForce("sin(AngleCVSingle)")  # This is suppose to be unique
        cv_single_2.addCollectiveVariable('AngleCVSingle', copy.deepcopy(cv_angle))
        cv_combined = openmm.CustomCVForce("4*BondCV + sin(AngleCV)")
        cv_combined.addCollectiveVariable("BondCV", cv_distance)
        cv_combined.addCollectiveVariable("AngleCV", cv_angle)
        for force in [cv_single_1, cv_single_2, cv_combined]:
            system_cv.addForce(force)
        thermo_state = ThermodynamicState(system_cv, self.alanine_explicit_state.temperature)
        context = self.create_context(thermo_state)
        context.setPositions(self.alanine_explicit_positions)
        sampler_state = SamplerState.from_context(context)
        collective_variables = sampler_state.collective_variables
        name_count = (('BondCV', 2), ('AngleCV', 1), ('AngleCVSingle', 1))
        # Ensure the CV's are all accounted for
        assert len(collective_variables.keys()) == 3
        for name, count in name_count:
            # Ensure the CV's show up in the Context the number of times we expect them to
            assert len(collective_variables[name].keys()) == count
        # Ensure CVs which are the same in different forces are equal
        assert len(set(collective_variables['BondCV'].values())) == 1  # Cast values of CV to set, make sure len == 1
        # Ensure invalidation with single replacement
        new_pos = copy.deepcopy(self.alanine_explicit_positions)
        new_pos[0] *= 2
        sampler_state.positions[0] = new_pos[0]
        assert sampler_state.collective_variables is None
        # Ensure CV's are read from context
        sampler_state.update_from_context(context)
        assert sampler_state.collective_variables is not None
        # Ensure invalidation with full variable swap
        sampler_state.positions = new_pos
        assert sampler_state.collective_variables is None

# =============================================================================
# TEST COMPOUND STATE
# =============================================================================

class TestCompoundThermodynamicState(object):
    """Test suite for states.CompoundThermodynamicState class."""

    class DummyState(object):
        """A state that keeps track of a useless system parameter."""

        standard_dummy_parameter = 1.0

        def __init__(self, dummy_parameter):
            self._dummy_parameter = dummy_parameter

        @property
        def dummy_parameter(self):
            return self._dummy_parameter

        @dummy_parameter.setter
        def dummy_parameter(self, value):
            self._dummy_parameter = value

        def _standardize_system(self, system):
            try:
                self.set_dummy_parameter(system, self.standard_dummy_parameter)
            except TypeError:  # No parameter to set.
                raise ComposableStateError()

        def apply_to_system(self, system):
            self.set_dummy_parameter(system, self.dummy_parameter)

        def check_system_consistency(self, system):
            dummy_parameter = TestCompoundThermodynamicState.get_dummy_parameter(system)
            if dummy_parameter != self.dummy_parameter:
                raise ComposableStateError()

        @staticmethod
        def is_context_compatible(context):
            parameters = context.getState(getParameters=True).getParameters()
            if 'dummy_parameters' in parameters.keys():
                return True
            else:
                return False

        def apply_to_context(self, context):
            context.setParameter('dummy_parameter', self.dummy_parameter)

        def _on_setattr(self, standard_system, attribute_name, old_dummy_state):
            return False

        def _find_force_groups_to_update(self, context, current_context_state, memo):
            if current_context_state.dummy_parameter == self.dummy_parameter:
                return {}
            force, _ = self._find_dummy_force(context.getSystem())
            return {force.getForceGroup()}

        @classmethod
        def add_dummy_parameter(cls, system):
            """Add to system a CustomBondForce with a dummy parameter."""
            force = openmm.CustomBondForce('dummy_parameter')
            force.addGlobalParameter('dummy_parameter', cls.standard_dummy_parameter)
            max_force_group = cls._find_max_force_group(system)
            force.setForceGroup(max_force_group + 1)
            system.addForce(force)

        @staticmethod
        def _find_dummy_force(system):
            for force in system.getForces():
                if isinstance(force, openmm.CustomBondForce):
                    for parameter_id in range(force.getNumGlobalParameters()):
                        parameter_name = force.getGlobalParameterName(parameter_id)
                        if parameter_name == 'dummy_parameter':
                            return force, parameter_id

        @classmethod
        def set_dummy_parameter(cls, system, value):
            force, parameter_id = cls._find_dummy_force(system)
            force.setGlobalParameterDefaultValue(parameter_id, value)

        @staticmethod
        def _find_max_force_group(system):
            max_force_group = 0
            for force in system.getForces():
                if max_force_group < force.getForceGroup():
                    max_force_group = force.getForceGroup()
            return max_force_group

    @classmethod
    def get_dummy_parameter(cls, system):
        force, parameter_id = cls.DummyState._find_dummy_force(system)
        return force.getGlobalParameterDefaultValue(parameter_id)

    @classmethod
    def setup_class(cls):
        """Create various variables shared by tests in suite."""
        cls.std_pressure = ThermodynamicState._STANDARD_PRESSURE
        cls.std_temperature = ThermodynamicState._STANDARD_TEMPERATURE

        cls.dummy_parameter = cls.DummyState.standard_dummy_parameter + 1.0
        cls.dummy_state = cls.DummyState(cls.dummy_parameter)

        alanine_explicit = testsystems.AlanineDipeptideExplicit().system
        cls.DummyState.add_dummy_parameter(alanine_explicit)
        cls.alanine_explicit = alanine_explicit

    def test_dynamic_inheritance(self):
        """ThermodynamicState is inherited dinamically."""
        thermodynamic_state = ThermodynamicState(self.alanine_explicit,
                                                 self.std_temperature)
        compound_state = CompoundThermodynamicState(thermodynamic_state, [])

        assert isinstance(compound_state, ThermodynamicState)

        # Attributes are correctly read.
        assert hasattr(compound_state, 'pressure')
        assert compound_state.pressure is None
        assert hasattr(compound_state, 'temperature')
        assert compound_state.temperature == self.std_temperature

        # Properties and attributes are correctly set.
        new_temperature = self.std_temperature + 1.0*unit.kelvin
        compound_state.pressure = self.std_pressure
        compound_state.temperature = new_temperature
        barostat = compound_state.barostat
        assert barostat.getDefaultPressure() == self.std_pressure
        assert get_barostat_temperature(barostat) == new_temperature

    def test_constructor_set_state(self):
        """IComposableState.set_state is called on construction."""
        thermodynamic_state = ThermodynamicState(self.alanine_explicit, self.std_temperature)

        assert self.get_dummy_parameter(thermodynamic_state.system) != self.dummy_parameter
        compound_state = CompoundThermodynamicState(thermodynamic_state, [self.dummy_state])
        assert self.get_dummy_parameter(compound_state.system) == self.dummy_parameter

    def test_property_forwarding(self):
        """Forward properties to IComposableStates and update system."""
        dummy_state = self.DummyState(self.dummy_parameter + 1)
        thermodynamic_state = ThermodynamicState(self.alanine_explicit, self.std_temperature)
        compound_state = CompoundThermodynamicState(thermodynamic_state, [dummy_state])

        # Properties are correctly read and set, and
        # the system is updated to the new value.
        assert compound_state.dummy_parameter != self.dummy_parameter
        assert self.get_dummy_parameter(compound_state.system) != self.dummy_parameter
        compound_state.dummy_parameter = self.dummy_parameter
        assert compound_state.dummy_parameter == self.dummy_parameter
        assert self.get_dummy_parameter(compound_state.system) == self.dummy_parameter

        # Default behavior for attribute error and monkey patching.
        with nose.tools.assert_raises(AttributeError):
            compound_state.temp
        compound_state.temp = 0
        assert 'temp' in compound_state.__dict__

        # If there are multiple composable states setting two different
        # values for the same attribute, an exception is raise.
        dummy_state2 = self.DummyState(dummy_state.dummy_parameter + 1)
        compound_state = CompoundThermodynamicState(thermodynamic_state, [dummy_state, dummy_state2])
        with nose.tools.assert_raises(RuntimeError):
            compound_state.dummy_parameter

    def test_set_system(self):
        """CompoundThermodynamicState.system and set_system method."""
        thermodynamic_state = ThermodynamicState(self.alanine_explicit, self.std_temperature)
        compound_state = CompoundThermodynamicState(thermodynamic_state, [self.dummy_state])

        # Setting an inconsistent system for the dummy raises an error.
        system = compound_state.system
        self.DummyState.set_dummy_parameter(system, self.dummy_parameter + 1.0)
        with nose.tools.assert_raises(ComposableStateError):
            compound_state.system = system

        # Same for set_system when called with default arguments.
        with nose.tools.assert_raises(ComposableStateError):
            compound_state.set_system(system)

        # This doesn't happen if we fix the state.
        compound_state.set_system(system, fix_state=True)

    def test_method_standardize_system(self):
        """CompoundThermodynamicState._standardize_system method."""
        alanine_explicit = copy.deepcopy(self.alanine_explicit)
        thermodynamic_state = ThermodynamicState(alanine_explicit, self.std_temperature)
        thermodynamic_state.pressure = self.std_pressure + 1.0*unit.bar
        compound_state = CompoundThermodynamicState(thermodynamic_state, [self.dummy_state])

        # Standardizing the system fixes both ThermodynamicState and DummyState parameters.
        system = compound_state.system
        barostat = ThermodynamicState._find_barostat(system)
        assert barostat.getDefaultPressure() != self.std_pressure
        assert self.get_dummy_parameter(system) == self.dummy_parameter
        compound_state._standardize_system(system)
        barostat = ThermodynamicState._find_barostat(system)
        assert barostat.getDefaultPressure() == self.std_pressure
        assert self.get_dummy_parameter(system) == self.DummyState.standard_dummy_parameter

        # Check that the standard system hash is correct.
        standard_hash = openmm.XmlSerializer.serialize(system).__hash__()
        assert standard_hash == compound_state._standard_system_hash

        # Check that is_state_compatible works.
        undummied_alanine = testsystems.AlanineDipeptideExplicit().system
        incompatible_state = ThermodynamicState(undummied_alanine, self.std_temperature)
        assert not compound_state.is_state_compatible(incompatible_state)

        integrator = openmm.VerletIntegrator(2.0*unit.femtoseconds)
        context = create_default_context(incompatible_state, integrator)
        assert not compound_state.is_context_compatible(context)

    def test_method_apply_to_context(self):
        """Test CompoundThermodynamicState.apply_to_context() method."""
        dummy_parameter = self.DummyState.standard_dummy_parameter
        thermodynamic_state = ThermodynamicState(self.alanine_explicit, self.std_temperature)
        thermodynamic_state.pressure = self.std_pressure
        self.DummyState.set_dummy_parameter(thermodynamic_state.system, dummy_parameter)

        integrator = openmm.VerletIntegrator(2.0*unit.femtoseconds)
        context = create_default_context(thermodynamic_state, integrator)
        barostat = ThermodynamicState._find_barostat(context.getSystem())
        assert context.getParameter('dummy_parameter') == dummy_parameter
        assert context.getParameter(barostat.Pressure()) == self.std_pressure / unit.bar

        compound_state = CompoundThermodynamicState(thermodynamic_state, [self.dummy_state])
        new_pressure = thermodynamic_state.pressure + 1.0*unit.bar
        compound_state.pressure = new_pressure
        compound_state.apply_to_context(context)
        assert context.getParameter('dummy_parameter') == self.dummy_parameter
        assert context.getParameter(barostat.Pressure()) == new_pressure / unit.bar

    def test_method_find_force_groups_to_update(self):
        """Test CompoundThermodynamicState._find_force_groups_to_update() method."""
        alanine_explicit = copy.deepcopy(self.alanine_explicit)
        thermodynamic_state = ThermodynamicState(alanine_explicit, self.std_temperature)
        compound_state = CompoundThermodynamicState(thermodynamic_state, [self.dummy_state])
        context = create_default_context(compound_state, openmm.VerletIntegrator(2.0*unit.femtoseconds))

        # No force group should be updated if the two states are identical.
        assert compound_state._find_force_groups_to_update(context, compound_state, memo={}) == set()

        # If the dummy parameter changes, there should be 1 force group to update.
        compound_state2 = copy.deepcopy(compound_state)
        compound_state2.dummy_parameter -= 0.5
        group = self.DummyState._find_max_force_group(context.getSystem())
        assert compound_state._find_force_groups_to_update(context, compound_state2, memo={}) == {group}


# =============================================================================
# TEST SERIALIZATION
# =============================================================================

def are_pickle_equal(state1, state2):
    """Check if they two ThermodynamicStates are identical."""
    # Pickle internally uses __getstate__ so we are effectively
    # comparing the serialization of the two objects.
    return pickle.dumps(state1) == pickle.dumps(state2)


def test_states_serialization():
    """Test serialization compatibility with utils.serialize."""
    test_system = testsystems.AlanineDipeptideImplicit()
    thermodynamic_state = ThermodynamicState(test_system.system, temperature=300*unit.kelvin)
    sampler_state = SamplerState(positions=test_system.positions)

    test_cases = [thermodynamic_state, sampler_state]
    for test_state in test_cases:
        serialization = utils.serialize(test_state)

        # First test serialization with cache. Copy
        # serialization so that we can use it again.
        deserialized_state = utils.deserialize(copy.deepcopy(serialization))
        assert are_pickle_equal(test_state, deserialized_state)

        # Now test without cache.
        ThermodynamicState._standard_system_cache = {}
        deserialized_state = utils.deserialize(serialization)
        assert are_pickle_equal(test_state, deserialized_state)


def test_uncompressed_thermodynamic_state_serialization():
    """Test for backwards compatibility.

    Until openmmtools 0.11.0, the ThermodynamicStates serialized
    system was not compressed.
    """
    system = testsystems.AlanineDipeptideImplicit().system
    state = ThermodynamicState(system, temperature=300 * unit.kelvin)
    compressed_serialization = utils.serialize(state)

    # Create uncompressed ThermodynamicState serialization.
    state._standardize_system(system)
    uncompressed_serialization = copy.deepcopy(compressed_serialization)
    uncompressed_serialization['standard_system'] = openmm.XmlSerializer.serialize(system)

    # First test serialization with cache. Copy
    # serialization so that we can use it again.
    deserialized_state = utils.deserialize(copy.deepcopy(uncompressed_serialization))
    assert are_pickle_equal(state, deserialized_state)

    # Now test without cache.
    ThermodynamicState._standard_system_cache = {}
    deserialized_state = utils.deserialize(uncompressed_serialization)
    assert are_pickle_equal(state, deserialized_state)


# =============================================================================
# TEST GLOBAL PARAMETER STATE
# =============================================================================

_GLOBAL_PARAMETER_STANDARD_VALUE = 1.0


class ParameterStateExample(GlobalParameterState):
    standard_value = _GLOBAL_PARAMETER_STANDARD_VALUE
    lambda_bonds = GlobalParameterState.GlobalParameter('lambda_bonds', standard_value)
    gamma = GlobalParameterState.GlobalParameter('gamma', standard_value)

    def set_defined_parameters(self, value):
        for parameter_name, parameter_value in self._parameters.items():
            if parameter_value is not None:
                self._parameters[parameter_name] = value


class TestGlobalParameterState(object):
    """Test GlobalParameterState stand-alone functionality.

    The compatibility with CompoundThermodynamicState is tested in the next
    test suite.
    """

    @classmethod
    def setup_class(cls):
        """Create test systems and shared objects."""
        # Define a diatomic molecule System with two custom forces
        # using the simple version and the suffix'ed version.
        r0 = 0.15*unit.nanometers

        # Make sure that there is a force without defining a parameter.
        cls.parameters_default_values = {
            'lambda_bonds': 1.0,
            'gamma': 2.0,
            'lambda_bonds_mysuffix': 0.5,
            'gamma_mysuffix': None,
        }

        r0_nanometers = r0.value_in_unit(unit.nanometers)  # Shortcut in OpenMM units.
        system = openmm.System()
        system.addParticle(40.0*unit.amu)
        system.addParticle(40.0*unit.amu)
        # Add a force defining lambda_bonds and gamma global parameters.
        custom_force = openmm.CustomBondForce('lambda_bonds^gamma*60000*(r-{})^2;'.format(r0_nanometers))
        custom_force.addGlobalParameter('lambda_bonds', cls.parameters_default_values['lambda_bonds'])
        custom_force.addGlobalParameter('gamma', cls.parameters_default_values['gamma'])
        custom_force.addBond(0, 1, [])
        system.addForce(custom_force)
        # Add a force defining the lambda_bonds_mysuffix global parameters.
        custom_force_suffix = openmm.CustomBondForce('lambda_bonds_mysuffix*20000*(r-{})^2;'.format(r0_nanometers))
        custom_force_suffix.addGlobalParameter('lambda_bonds_mysuffix', cls.parameters_default_values['lambda_bonds_mysuffix'])
        custom_force_suffix.addBond(0, 1, [])
        system.addForce(custom_force_suffix)

        # Create a thermodynamic and sampler states.
        cls.diatomic_molecule_ts = ThermodynamicState(system, temperature=300.0*unit.kelvin)
        pos1 = [0.0, 0.0, 0.0]
        pos2 = [0.0, 0.0, r0_nanometers]
        cls.diatomic_molecule_ss = SamplerState(positions=np.array([pos1, pos2]) * unit.nanometers)

        # Create a system with a duplicate force to test handling forces
        # defining the same parameters in different force groups.
        custom_force = copy.deepcopy(custom_force_suffix)
        custom_force.setForceGroup(30)
        system_force_groups = copy.deepcopy(system)
        system_force_groups.addForce(custom_force)
        cls.diatomic_molecule_force_groups_ts = ThermodynamicState(system_force_groups, temperature=300.0*unit.kelvin)

        # Create few incompatible systems for testing. An incompatible state
        # has a different set of defined global parameters.
        cls.incompatible_systems = [system]

        # System without suffixed or non-suffixed parameters.
        for i in range(2):
            cls.incompatible_systems.append(copy.deepcopy(system))
            cls.incompatible_systems[i+1].removeForce(i)

        # System with the global parameters duplicated in two different force groups.
        cls.incompatible_systems.append(copy.deepcopy(system_force_groups))

        # System with both lambda_bonds_suffix and gamma_bond_suffix defined (instead of only the former).
        cls.incompatible_systems.append(copy.deepcopy(system))
        custom_force = copy.deepcopy(cls.incompatible_systems[-1].getForce(1))
        energy_function = custom_force.getEnergyFunction()
        energy_function = energy_function.replace('lambda_bonds_mysuffix', 'lambda_bonds_mysuffix^gamma_mysuffix')
        custom_force.setEnergyFunction(energy_function)
        custom_force.addGlobalParameter('gamma_mysuffix', cls.parameters_default_values['gamma'])
        cls.incompatible_systems[-1].addForce(custom_force)

    def read_system_state(self, system):
        states = []
        for suffix in [None, 'mysuffix']:
            try:
                states.append(ParameterStateExample.from_system(system, parameters_name_suffix=suffix))
            except GlobalParameterError:
                continue
        return states

    @staticmethod
    def test_constructor_parameters():
        """Test GlobalParameterState constructor behave as expected."""
        class MyState(GlobalParameterState):
            lambda_angles = GlobalParameterState.GlobalParameter('lambda_angles', standard_value=1.0)
            lambda_sterics = GlobalParameterState.GlobalParameter('lambda_sterics', standard_value=1.0)

        # Raise an exception if parameter is not recognized.
        with nose.tools.assert_raises_regexp(GlobalParameterError, 'Unknown parameters'):
            MyState(lambda_steric=1.0)  # Typo.

        # Properties are initialized correctly.
        test_cases = [{},
                      {'lambda_angles': 1.0},
                      {'lambda_sterics': 0.5, 'lambda_angles': 0.5},
                      {'parameters_name_suffix': 'suffix'},
                      {'parameters_name_suffix': 'suffix', 'lambda_angles': 1.0},
                      {'parameters_name_suffix': 'suffix', 'lambda_sterics': 0.5, 'lambda_angles': 0.5}]

        for test_kwargs in test_cases:
            state = MyState(**test_kwargs)

            # Check which parameters are defined/undefined in the constructed state.
            for parameter in MyState._get_controlled_parameters():
                # Store whether parameter is defined before appending the suffix.
                is_defined = parameter in test_kwargs

                # The "unsuffixed" parameter should not be controlled by the state.
                if 'parameters_name_suffix' in test_kwargs:
                    with nose.tools.assert_raises_regexp(AttributeError, 'state does not control'):
                        getattr(state, parameter)
                    # The state exposes a "suffixed" version of the parameter.
                    state_attribute = parameter + '_' + test_kwargs['parameters_name_suffix']
                else:
                    state_attribute = parameter

                # Check if parameter should is defined or undefined (i.e. set to None) as expected.
                err_msg = 'Parameter: {} (Test case: {})'.format(parameter, test_kwargs)
                if is_defined:
                    assert getattr(state, state_attribute) == test_kwargs[parameter], err_msg
                else:
                    assert getattr(state, state_attribute) is None, err_msg

    def test_from_system_constructor(self):
        """Test GlobalParameterState.from_system constructor."""
        # A system exposing no global parameters controlled by the state raises an error.
        with nose.tools.assert_raises_regexp(GlobalParameterError, 'no global parameters'):
            GlobalParameterState.from_system(openmm.System())

        system = self.diatomic_molecule_ts.system
        state = ParameterStateExample.from_system(system)
        state_suffix = ParameterStateExample.from_system(system, parameters_name_suffix='mysuffix')

        for parameter_name, parameter_value in self.parameters_default_values.items():
            if 'suffix' in parameter_name:
                controlling_state = state_suffix
                noncontrolling_state = state
            else:
                controlling_state = state
                noncontrolling_state = state_suffix

            err_msg = '{}: {}'.format(parameter_name, parameter_value)
            assert getattr(controlling_state, parameter_name) == parameter_value, err_msg
            with nose.tools.assert_raises(AttributeError):
                getattr(noncontrolling_state, parameter_name), parameter_name

    def test_parameter_validator(self):
        """Test GlobalParameterState constructor behave as expected."""

        class MyState(GlobalParameterState):
            lambda_bonds = GlobalParameterState.GlobalParameter('lambda_bonds', standard_value=1.0)

            @lambda_bonds.validator
            def lambda_bonds(self, instance, new_value):
                if not (0.0 <= new_value <= 1.0):
                    raise ValueError('lambda_bonds must be between 0.0 and 1.0')
                return new_value

        # Create system with incorrect initial parameter.
        system = self.diatomic_molecule_ts.system
        system.getForce(0).setGlobalParameterDefaultValue(0, 2.0)  # lambda_bonds
        system.getForce(1).setGlobalParameterDefaultValue(0, -1.0)  # lambda_bonds_mysuffix

        for suffix in [None, 'mysuffix']:
            # Raise an exception on init.
            with nose.tools.assert_raises_regexp(ValueError, 'must be between'):
                MyState(parameters_name_suffix=suffix, lambda_bonds=-1.0)
            with nose.tools.assert_raises_regexp(ValueError, 'must be between'):
                MyState.from_system(system, parameters_name_suffix=suffix)

            # Raise an exception when properties are set.
            state = MyState(parameters_name_suffix=suffix, lambda_bonds=1.0)
            parameter_name = 'lambda_bonds' if suffix is None else 'lambda_bonds_' + suffix
            with nose.tools.assert_raises_regexp(ValueError, 'must be between'):
                setattr(state, parameter_name, 5.0)

    def test_equality_operator(self):
        """Test equality operator between GlobalParameterStates."""
        state1 = ParameterStateExample(lambda_bonds=1.0)
        state2 = ParameterStateExample(lambda_bonds=1.0)
        state3 = ParameterStateExample(lambda_bonds=0.9)
        state4 = ParameterStateExample(lambda_bonds=0.9, gamma=1.0)
        state5 = ParameterStateExample(lambda_bonds=0.9, parameters_name_suffix='suffix')
        state6 = ParameterStateExample(parameters_name_suffix='suffix', lambda_bonds=0.9, gamma=1.0)
        assert state1 == state2
        assert state2 != state3
        assert state3 != state4
        assert state3 != state5
        assert state4 != state6
        assert state5 != state6
        assert state6 == state6

        # States that control more variables are not equal.
        class MyState(ParameterStateExample):
            extra_parameter = GlobalParameterState.GlobalParameter('extra_parameter', standard_value=1.0)
        state7 = MyState(lambda_bonds=0.9)
        assert state3 != state7

        # States defined by global parameter functions are evaluated correctly.
        state8 = copy.deepcopy(state1)
        state8.set_function_variable('lambda1', state1.lambda_bonds*2.0)
        state8.lambda_bonds = GlobalParameterFunction('lambda1 / 2')
        assert state1 == state8
        state8.set_function_variable('lambda1', state1.lambda_bonds)
        assert state1 != state8

    def test_apply_to_system(self):
        """Test method GlobalParameterState.apply_to_system()."""
        system = self.diatomic_molecule_ts.system
        state = ParameterStateExample.from_system(system)
        state_suffix = ParameterStateExample.from_system(system, parameters_name_suffix='mysuffix')

        expected_system_values = copy.deepcopy(self.parameters_default_values)

        def check_system_values():
            state, state_suffix = self.read_system_state(system)
            for parameter_name, parameter_value in expected_system_values.items():
                err_msg = 'parameter: {}, expected_value: {}'.format(parameter_name, parameter_value)
                if 'suffix' in parameter_name:
                    assert getattr(state_suffix, parameter_name) == parameter_value, err_msg
                else:
                    assert getattr(state, parameter_name) == parameter_value, err_msg

        # Test precondition: all parameters have the expected default value.
        check_system_values()

        # apply_to_system() modifies the state.
        state.lambda_bonds /= 2
        expected_system_values['lambda_bonds'] /= 2
        state_suffix.lambda_bonds_mysuffix /= 2
        expected_system_values['lambda_bonds_mysuffix'] /= 2
        for s in [state, state_suffix]:
            s.apply_to_system(system)
        check_system_values()

        # Raise an error if an extra parameter is defined in the system.
        state.gamma = None
        err_msg = 'The system parameter gamma is not defined in this state.'
        with nose.tools.assert_raises_regexp(GlobalParameterError, err_msg):
            state.apply_to_system(system)

        # Raise an error if an extra parameter is defined in the state.
        state_suffix.gamma_mysuffix = 2.0
        err_msg = 'Could not find global parameter gamma_mysuffix in the system.'
        with nose.tools.assert_raises_regexp(GlobalParameterError, err_msg):
            state_suffix.apply_to_system(system)

    def test_check_system_consistency(self):
        """Test method GlobalParameterState.check_system_consistency()."""
        system = self.diatomic_molecule_ts.get_system(remove_thermostat=True)

        def check_not_consistency(states):
            for s in states:
                with nose.tools.assert_raises_regexp(GlobalParameterError, 'Consistency check failed'):
                    s.check_system_consistency(system)

        # A system is consistent with itself.
        state, state_suffix = self.read_system_state(system)
        for s in [state, state_suffix]:
            s.check_system_consistency(system)

        # Raise error if System defines global parameters that are undefined in the state.
        state, state_suffix = self.read_system_state(system)
        state.gamma = None
        state_suffix.lambda_bonds_mysuffix = None
        check_not_consistency([state, state_suffix])

        # Raise error if state defines global parameters that are undefined in the System.
        state, state_suffix = self.read_system_state(system)
        state_suffix.gamma_mysuffix = 1.0
        check_not_consistency([state_suffix])

        # Raise error if system has different lambda values.
        state, state_suffix = self.read_system_state(system)
        state.lambda_bonds /= 2
        state_suffix.lambda_bonds_mysuffix /=2
        check_not_consistency([state, state_suffix])

    def test_apply_to_context(self):
        """Test method GlobalParameterState.apply_to_context."""
        system = self.diatomic_molecule_ts.system
        integrator = openmm.VerletIntegrator(1.0*unit.femtosecond)
        context = create_default_context(self.diatomic_molecule_ts, integrator)

        def check_not_applicable(states, error, context):
            for s in states:
                with nose.tools.assert_raises_regexp(GlobalParameterError, error):
                    s.apply_to_context(context)

        # Raise error if the Context defines global parameters that are undefined in the state.
        state, state_suffix = self.read_system_state(system)
        state.lambda_bonds = None
        state_suffix.lambda_bonds_mysuffix = None
        check_not_applicable([state, state_suffix], 'undefined in this state', context)

        # Raise error if the state defines global parameters that are undefined in the Context.
        state, state_suffix = self.read_system_state(system)
        state_suffix.gamma_mysuffix = 2.0
        check_not_applicable([state_suffix], 'Could not find parameter', context)

        # Test-precondition: Context parameters are different than the value we'll test.
        tested_value = 0.2
        for parameter_value in context.getParameters().values():
            assert parameter_value != tested_value

        # Correctly sets Context's parameters.
        state, state_suffix = self.read_system_state(system)
        state.lambda_bonds = tested_value
        state.gamma = tested_value
        state_suffix.lambda_bonds_mysuffix = tested_value
        for s in [state, state_suffix]:
            s.apply_to_context(context)
            for parameter_name, parameter_value in context.getParameters().items():
                if parameter_name in s._parameters:
                    assert parameter_value == tested_value
        del context

    def test_standardize_system(self):
        """Test method GlobalParameterState.standardize_system."""
        system = self.diatomic_molecule_ts.system
        standard_value = _GLOBAL_PARAMETER_STANDARD_VALUE  # Shortcut.

        def check_is_standard(states, is_standard):
            for s in states:
                for parameter_name in s._get_controlled_parameters(s._parameters_name_suffix):
                    parameter_value = getattr(s, parameter_name)
                    err_msg = 'Parameter: {}; Value: {};'.format(parameter_name, parameter_value)
                    if parameter_value is not None:
                        assert (parameter_value  == standard_value) is is_standard, err_msg

        # Test pre-condition: The system is not in the standard state.
        system.getForce(0).setGlobalParameterDefaultValue(0, 0.9)
        states = self.read_system_state(system)
        check_is_standard(states, is_standard=False)

        # Check that _standardize_system() sets all parameters to the standard value.
        for state in states:
            state._standardize_system(system)
        states_standard = self.read_system_state(system)
        check_is_standard(states_standard, is_standard=True)

    def test_find_force_groups_to_update(self):
        """Test method GlobalParameterState._find_force_groups_to_update."""
        system = self.diatomic_molecule_force_groups_ts.system
        integrator = openmm.VerletIntegrator(2.0*unit.femtoseconds)
        # Test cases are (force_groups, force_groups_suffix)
        test_cases = [
            ([0], [0, 0]),
            ([1], [5, 5]),
            ([9], [4, 2])
        ]

        for test_case in test_cases:
            for i, force_group in enumerate(test_case[0] + test_case[1]):
                system.getForce(i).setForceGroup(force_group)
            states = self.read_system_state(system)
            context = openmm.Context(system, copy.deepcopy(integrator))

            # No force group should be updated if we don't change the global parameter.
            for state, force_groups in zip(states, test_case):
                assert state._find_force_groups_to_update(context, state, memo={}) == set()

                # Change the lambdas one by one and check that the method
                # recognizes that the force group energy must be updated.
                current_state = copy.deepcopy(state)
                for parameter_name in state._get_controlled_parameters(state._parameters_name_suffix):
                    # Check that the system defines the global variable.
                    parameter_value = getattr(state, parameter_name)
                    if parameter_value is None:
                        continue

                    # Change the current state.
                    setattr(current_state, parameter_name, parameter_value / 2)
                    assert state._find_force_groups_to_update(context, current_state, memo={}) == set(force_groups)
                    setattr(current_state, parameter_name, parameter_value)  # Reset current state.
            del context

    def test_global_parameters_functions(self):
        """Test function variables and global parameter functions work correctly."""
        system = copy.deepcopy(self.diatomic_molecule_ts.system)
        state = ParameterStateExample.from_system(system)

        # Add two function variables to the state.
        state.set_function_variable('lambda', 1.0)
        state.set_function_variable('lambda2', 0.5)
        assert state.get_function_variable('lambda') == 1.0
        assert state.get_function_variable('lambda2') == 0.5

        # Cannot call an function variable as a supported parameter.
        with nose.tools.assert_raises(GlobalParameterError):
            state.set_function_variable('lambda_bonds', 0.5)

        # Assign string global parameter functions to parameters.
        state.lambda_bonds = GlobalParameterFunction('lambda')
        state.gamma = GlobalParameterFunction('(lambda + lambda2) / 2.0')
        assert state.lambda_bonds == 1.0
        assert state.gamma == 0.75

        # Setting function variables updates global parameter as well.
        state.set_function_variable('lambda2', 0)
        assert state.gamma == 0.5

    # ---------------------------------------------------
    # Integration tests with CompoundThermodynamicStates
    # ---------------------------------------------------

    def test_constructor_compound_state(self):
        """The GlobalParameterState is set on construction of the CompoundState."""
        system = self.diatomic_molecule_ts.system

        # Create a system state different than the initial.
        composable_states = self.read_system_state(system)
        for state in composable_states:
            state.set_defined_parameters(0.222)

        # CompoundThermodynamicState set the system state in the constructor.
        compound_state = CompoundThermodynamicState(self.diatomic_molecule_ts, composable_states)
        new_system_states = self.read_system_state(compound_state.system)
        for state, new_state in zip(composable_states, new_system_states):
            assert state == new_state

        # Trying to set in the constructor undefined global parameters raise an exception.
        composable_states[1].gamma_mysuffix = 2.0
        err_msg = 'Could not find global parameter gamma_mysuffix in the system.'
        with nose.tools.assert_raises_regexp(GlobalParameterError, err_msg):
            CompoundThermodynamicState(self.diatomic_molecule_ts, composable_states)

    def test_global_parameters_compound_state(self):
        """Global parameters setters/getters work in the CompoundState system."""
        composable_states = self.read_system_state(self.diatomic_molecule_ts.system)
        for state in composable_states:
            state.set_defined_parameters(0.222)
        compound_state = CompoundThermodynamicState(self.diatomic_molecule_ts, composable_states)

        # Defined properties can be assigned and read, unless they are undefined.
        for parameter_name, default_value in self.parameters_default_values.items():
            if default_value is None:
                assert getattr(compound_state, parameter_name) is None
                # If undefined, setting the property should raise an error.
                err_msg = 'Cannot set the parameter gamma_mysuffix in the system'
                with nose.tools.assert_raises_regexp(GlobalParameterError, err_msg):
                    setattr(compound_state, parameter_name, 2.0)
                continue

            # Defined parameters should be gettable and settables.
            assert getattr(compound_state, parameter_name) == 0.222
            setattr(compound_state, parameter_name, 0.5)
            assert getattr(compound_state, parameter_name) == 0.5

        # System global variables are updated correctly
        system_states = self.read_system_state(compound_state.system)
        for state in system_states:
            for parameter_name in state._parameters:
                assert getattr(state, parameter_name) == getattr(compound_state, parameter_name)

        # Same for global parameter function variables.
        compound_state.set_function_variable('lambda', 0.25)
        defined_parameters = {name for name, value in self.parameters_default_values.items()
                              if value is not None}
        for parameter_name in defined_parameters:
            setattr(compound_state, parameter_name, GlobalParameterFunction('lambda'))
            parameter_value = getattr(compound_state, parameter_name)
            assert parameter_value == 0.25, '{}, {}'.format(parameter_name, parameter_value)

        system_states = self.read_system_state(compound_state.system)
        for state in system_states:
            for parameter_name in state._parameters:
                if parameter_name in defined_parameters:
                    parameter_value = getattr(compound_state, parameter_name)
                    assert parameter_value == 0.25, '{}, {}'.format(parameter_name, parameter_value)

    def test_set_system_compound_state(self):
        """Setting inconsistent system in compound state raise errors."""
        system = self.diatomic_molecule_ts.system
        composable_states = self.read_system_state(system)
        compound_state = CompoundThermodynamicState(self.diatomic_molecule_ts, composable_states)

        for parameter_name, default_value in self.parameters_default_values.items():
            if default_value is None:
                continue
            elif 'suffix' in parameter_name:
                original_state = composable_states[1]
            else:
                original_state = composable_states[0]

            # We create an incompatible state with the parameter set to a different value.
            incompatible_state = copy.deepcopy(original_state)
            original_value = getattr(incompatible_state, parameter_name)
            setattr(incompatible_state, parameter_name, original_value/2)
            incompatible_state.apply_to_system(system)

            # Setting an inconsistent system raise an error.
            with nose.tools.assert_raises_regexp(GlobalParameterError, parameter_name):
                compound_state.system = system

            # Same for set_system when called with default arguments.
            with nose.tools.assert_raises_regexp(GlobalParameterError, parameter_name):
                compound_state.set_system(system)

            # This doesn't happen if we fix the state.
            compound_state.set_system(system, fix_state=True)
            new_state = incompatible_state.from_system(compound_state.system, original_state._parameters_name_suffix)
            assert new_state == original_state, (str(new_state), str(incompatible_state))

            # Restore old value in system, and test next parameter.
            original_state.apply_to_system(system)

    def test_compatibility_compound_state(self):
        """Compatibility between states is handled correctly in compound state."""
        incompatible_systems = copy.deepcopy(self.incompatible_systems)

        # Build all compound states.
        compound_states = []
        for system in incompatible_systems:
            thermodynamic_state = ThermodynamicState(system, temperature=300*unit.kelvin)
            composable_states = self.read_system_state(system)
            compound_states.append(CompoundThermodynamicState(thermodynamic_state, composable_states))

        # Build all contexts for testing.
        integrator = openmm.VerletIntegrator(2.0*unit.femtoseconds)
        contexts = [create_default_context(s, copy.deepcopy(integrator)) for s in compound_states]

        for state_idx, (compound_state, context) in enumerate(zip(compound_states, contexts)):
            # The state is compatible with itself.
            assert compound_state.is_state_compatible(compound_state)
            assert compound_state.is_context_compatible(context)

            # Changing the values of the parameters do not affect
            # compatibility (only defined/undefined parameters do).
            altered_compound_state = copy.deepcopy(compound_state)
            for parameter_name in ['gamma', 'lambda_bonds_mysuffix']:
                try:
                    new_value = getattr(compound_state, parameter_name) / 2
                    setattr(altered_compound_state, parameter_name, new_value)
                except AttributeError:
                    continue
            assert altered_compound_state.is_state_compatible(compound_state)
            assert altered_compound_state.is_context_compatible(context)

            # All other states are incompatible. Test only those that we
            # haven't tested yet, but test transitivity.
            for incompatible_state_idx in range(state_idx+1, len(compound_states)):
                print(state_idx, incompatible_state_idx)
                incompatible_state = compound_states[incompatible_state_idx]
                incompatible_context = contexts[incompatible_state_idx]
                assert not compound_state.is_state_compatible(incompatible_state)
                assert not incompatible_state.is_state_compatible(compound_state)
                assert not compound_state.is_context_compatible(incompatible_context)
                assert not incompatible_state.is_context_compatible(context)

    def test_reduced_potential_compound_state(self):
        """Test CompoundThermodynamicState.reduced_potential_at_states() method.

        Computing the reduced potential singularly and with the class
        method should give the same result.
        """
        positions = copy.deepcopy(self.diatomic_molecule_ss.positions)
        # Build a mixed collection of compatible and incompatible thermodynamic states.
        thermodynamic_states = [
            copy.deepcopy(self.diatomic_molecule_ts),
            copy.deepcopy(self.diatomic_molecule_force_groups_ts)
        ]

        compound_states = []
        for ts_idx, ts in enumerate(thermodynamic_states):
            compound_state = CompoundThermodynamicState(ts, self.read_system_state(ts.system))
            for state in [dict(lambda_bonds=1.0, gamma=1.0, lambda_bonds_mysuffix=1.0, gamma_mysuffix=1.0),
                          dict(lambda_bonds=0.5, gamma=1.0, lambda_bonds_mysuffix=1.0, gamma_mysuffix=1.0),
                          dict(lambda_bonds=0.5, gamma=1.0, lambda_bonds_mysuffix=1.0, gamma_mysuffix=0.5),
                          dict(lambda_bonds=0.1, gamma=0.5, lambda_bonds_mysuffix=0.2, gamma_mysuffix=0.5)]:
                for parameter_name, parameter_value in state.items():
                    try:
                        setattr(compound_state, parameter_name, parameter_value)
                    except GlobalParameterError:
                        continue
                compound_states.append(copy.deepcopy(compound_state))

        # Group thermodynamic states by compatibility.
        compatible_groups, _ = group_by_compatibility(compound_states)
        assert len(compatible_groups) == 2

        # Compute the reduced potentials.
        expected_energies = []
        obtained_energies = []
        for compatible_group in compatible_groups:
            # Create context.
            integrator = openmm.VerletIntegrator(2.0*unit.femtoseconds)
            context = create_default_context(compatible_group[0], integrator)
            context.setPositions(positions)

            # Compute with single-state method.
            for state in compatible_group:
                state.apply_to_context(context)
                expected_energies.append(state.reduced_potential(context))

            # Compute with multi-state method.
            compatible_energies = ThermodynamicState.reduced_potential_at_states(context, compatible_group)

            # The first and the last state must be equal.
            assert np.isclose(compatible_energies[0], compatible_energies[-1])
            obtained_energies.extend(compatible_energies)

        assert np.allclose(np.array(expected_energies), np.array(obtained_energies))

    def test_serialization(self):
        """Test GlobalParameterState serialization alone and in a compound state."""
        composable_states = self.read_system_state(self.diatomic_molecule_ts.system)

        # Add a global parameter function to test if they are serialized correctly.
        composable_states[0].set_function_variable('lambda', 0.5)
        composable_states[0].gamma = GlobalParameterFunction('lambda**2')

        # Test serialization/deserialization of GlobalParameterState.
        for state in composable_states:
            serialization = utils.serialize(state)
            deserialized_state = utils.deserialize(serialization)
            are_pickle_equal(state, deserialized_state)

        # Test serialization/deserialization of GlobalParameterState in CompoundState.
        compound_state = CompoundThermodynamicState(self.diatomic_molecule_ts, composable_states)
        serialization = utils.serialize(compound_state)
        deserialized_state = utils.deserialize(serialization)
        are_pickle_equal(compound_state, deserialized_state)


def test_create_thermodynamic_state_protocol():
    """Test the method for efficiently creating a list of thermoydamic states."""

    system = testsystems.AlchemicalAlanineDipeptide().system
    thermo_state = ThermodynamicState(system, temperature=400*unit.kelvin)

    # The method raises an exception when the protocol is empty.
    with nose.tools.assert_raises_regexp(ValueError, 'No protocol'):
        create_thermodynamic_state_protocol(system, protocol={})

    # The method raises an exception when different parameters have different lengths.
    with nose.tools.assert_raises_regexp(ValueError, 'different lengths'):
        protocol = {'temperature': [1.0, 2.0],
                    'pressure': [4.0]}
        create_thermodynamic_state_protocol(system, protocol=protocol)

    # An exception is raised if the temperature is not specified with a System.
    with nose.tools.assert_raises_regexp(ValueError, 'must specify the temperature'):
        protocol = {'pressure': [5.0] * unit.atmosphere}
        create_thermodynamic_state_protocol(system, protocol=protocol)

    # An exception is raised if a parameter is specified both as constant and protocol.
    with nose.tools.assert_raises_regexp(ValueError, 'constants and protocol'):
        protocol = {'temperature': [5.0, 10.0] * unit.kelvin}
        const = {'temperature': 5.0 * unit.kelvin}
        create_thermodynamic_state_protocol(system, protocol=protocol, constants=const)

    # Method works as expected with a reference System or ThermodynamicState.
    protocol = {'temperature': [290, 310, 360]*unit.kelvin}
    for reference in [system, thermo_state]:
        states = create_thermodynamic_state_protocol(reference, protocol=protocol)
        for state, temp in zip(states, protocol['temperature']):
            assert state.temperature == temp
        assert len(states) == 3

    # Same with CompoundThermodynamicState.
    from openmmtools.alchemy import AlchemicalState
    alchemical_state = AlchemicalState.from_system(system)
    protocol = {'temperature': [290, 310, 360]*unit.kelvin,
                'lambda_sterics': [1.0, 0.5, 0.0],
                'lambda_electrostatics': [0.75, 0.5, 0.25]}
    for reference in [system, thermo_state]:
        states = create_thermodynamic_state_protocol(reference, protocol=protocol,
                                                     composable_states=alchemical_state)
        for state, temp, sterics, electro in zip(states, protocol['temperature'],
                                                 protocol['lambda_sterics'],
                                                 protocol['lambda_electrostatics']):
            assert state.temperature == temp
            assert state.lambda_sterics == sterics
            assert state.lambda_electrostatics == electro
        assert len(states) == 3

    # Check that constants work correctly.
    del protocol['temperature']
    const = {'temperature': 500*unit.kelvin}
    states = create_thermodynamic_state_protocol(thermo_state, protocol=protocol, constants=const,
                                                 composable_states=alchemical_state)
    for state in states:
        assert state.temperature == 500*unit.kelvin
