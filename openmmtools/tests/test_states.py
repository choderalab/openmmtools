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

from openmmtools import testsystems
from openmmtools.states import *


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

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

        cls.toluene_implicit = testsystems.TolueneImplicit().system
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
        cls.multiple_barostat_alanine = copy.deepcopy(cls.barostated_alanine)
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

    def test_method_is_thermostat_consistent(self):
        """ThermodynamicState._is_thermostat_consistent() method."""
        temperature = self.std_temperature
        collision_freq = 1.0/unit.picosecond
        state = ThermodynamicState(self.alanine_explicit, temperature)

        thermostat = openmm.AndersenThermostat(temperature, collision_freq)
        assert state._is_thermostat_consistent(thermostat)
        thermostat.setDefaultTemperature(temperature + 1.0*unit.kelvin)
        assert not state._is_thermostat_consistent(thermostat)

    def test_method_set_barostat_temperature(self):
        """ThermodynamicState._set_barostat_temperature() method."""
        barostat = openmm.MonteCarloBarostat(self.std_pressure, self.std_temperature)
        new_temperature = self.std_temperature + 10*unit.kelvin

        assert ThermodynamicState._set_barostat_temperature(barostat, new_temperature)
        assert get_barostat_temperature(barostat) == new_temperature
        assert not ThermodynamicState._set_barostat_temperature(barostat, new_temperature)

    def test_method_set_system_thermostat(self):
        """ThermodynamicState._set_system_thermostat() method."""
        system = copy.deepcopy(self.alanine_no_thermostat)
        assert ThermodynamicState._find_thermostat(system) is None

        # Add a thermostat to the system.
        assert ThermodynamicState._set_system_thermostat(system, self.std_temperature)
        thermostat = ThermodynamicState._find_thermostat(system)
        assert thermostat.getDefaultTemperature() == self.std_temperature

        # Change temperature of existing barostat.
        new_temperature = self.std_temperature + 1.0*unit.kelvin
        assert ThermodynamicState._set_system_thermostat(system, new_temperature)
        assert thermostat.getDefaultTemperature() == new_temperature
        assert not ThermodynamicState._set_system_thermostat(system, new_temperature)

        # Remove system thermostat.
        assert ThermodynamicState._set_system_thermostat(system, None)
        assert ThermodynamicState._find_thermostat(system) is None
        assert not ThermodynamicState._set_system_thermostat(system, None)

    def test_property_temperature(self):
        """ThermodynamicState.temperature property."""
        state = ThermodynamicState(self.barostated_alanine,
                                   self.std_temperature)
        assert state.temperature == self.std_temperature

        temperature = self.std_temperature + 10.0*unit.kelvin
        state.temperature = temperature
        assert state.temperature == temperature
        assert get_barostat_temperature(state._barostat) == temperature

        # Setting temperature to None raise error.
        with nose.tools.assert_raises(ThermodynamicsError) as cm:
            state.temperature = None
        assert cm.exception.code == ThermodynamicsError.NONE_TEMPERATURE

    def test_method_set_system_pressure(self):
        """ThermodynamicState._set_system_pressure() method."""
        state = ThermodynamicState(self.alanine_explicit, self.std_temperature)
        state._set_system_pressure(state._system, None)
        assert state._barostat is None
        state._set_system_pressure(state._system, self.std_pressure)
        assert state._barostat.getDefaultPressure() == self.std_pressure

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
            # Changing ensemble also reset the cached system hash.
            state.pressure = None
            state._standard_system_hash  # cause the system hash to be cached
            assert state.barostat is None
            state.pressure = self.std_pressure
            assert state._cached_standard_system_hash is None
            state._standard_system_hash
            state.pressure = None
            assert state._cached_standard_system_hash is None

            state._standard_system_hash
            state.barostat = barostat
            assert state._cached_standard_system_hash is None
            state._standard_system_hash
            state.barostat = None
            assert state.pressure is None
            assert state._cached_standard_system_hash is None

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
        assert state._thermostat.getDefaultTemperature() == self.std_temperature
        assert state._barostat is None

        # In NPT, we can't set the system without adding a barostat.
        system = state.system  # System with thermostat.
        state.pressure = self.std_pressure
        with nose.tools.assert_raises(ThermodynamicsError) as cm:
            state.set_system(system)
        assert cm.exception.code == ThermodynamicsError.NO_BAROSTAT

        state.set_system(system, fix_state=True)
        assert state._barostat.getDefaultPressure() == self.std_pressure
        assert get_barostat_temperature(state._barostat) == self.std_temperature

    def test_method_get_system(self):
        """ThermodynamicState.get_system() method."""
        state = ThermodynamicState(self.alanine_explicit, self.std_temperature,
                                   self.std_pressure)

        # Normally a system has both barostat and thermostat
        system = state.get_system()
        assert state._find_barostat(system) is not None
        assert state._find_thermostat(system) is not None

        # We can request a system without thermostat or barostat.
        system = state.get_system(remove_thermostat=True)
        assert state._find_thermostat(system) is None
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
        assert state._barostat is None

        # If we specify pressure, barostat is added
        state = ThermodynamicState(system=system, temperature=self.std_temperature,
                                   pressure=self.std_pressure)
        assert state._barostat is not None

        # If we feed a barostat with an inconsistent temperature, it's fixed.
        state = ThermodynamicState(self.inconsistent_temperature_alanine,
                                   temperature=self.std_temperature)
        assert state._is_barostat_consistent(state._barostat)

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

        # With thermostat, temperature is inferred correctly,
        # and the barostat temperature is set correctly as well.
        system = copy.deepcopy(self.barostated_alanine)
        new_temperature = self.std_temperature + 1.0*unit.kelvin
        barostat = ThermodynamicState._find_barostat(system)
        assert get_barostat_temperature(barostat) != new_temperature  # Precondition.
        thermostat = ThermodynamicState._find_thermostat(system)
        thermostat.setDefaultTemperature(new_temperature)
        state = ThermodynamicState(system=system)
        assert state.temperature == new_temperature
        assert get_barostat_temperature(state._barostat) == new_temperature

        # Specifying temperature overwrite thermostat.
        state = ThermodynamicState(system=system, temperature=self.std_temperature)
        assert state.temperature == self.std_temperature
        assert get_barostat_temperature(state._barostat) == self.std_temperature

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
                assert not state._set_integrator_temperature(integrator)
            else:
                # It doesn't explode with integrators not coupled to a heat bath
                assert not state._set_integrator_temperature(integrator)

    def test_method_standardize_system(self):
        """ThermodynamicState._standardize_system() class method."""
        # Nothing happens if system has neither barostat nor thermostat.
        nvt_system = copy.deepcopy(self.alanine_no_thermostat)
        ThermodynamicState._standardize_system(nvt_system)
        assert nvt_system.__getstate__() == self.alanine_no_thermostat.__getstate__()

        # Create NPT system in non-standard state.
        npt_state = ThermodynamicState(self.inconsistent_pressure_alanine,
                                       self.std_temperature + 1.0*unit.kelvin)
        barostat = npt_state._barostat
        thermostat = npt_state._thermostat
        assert barostat.getDefaultPressure() != self.std_pressure
        assert get_barostat_temperature(barostat) != self.std_temperature
        assert thermostat.getDefaultTemperature() != self.std_temperature

        # With NPT system, the barostat is set to standard
        # and the thermostat is removed.
        npt_system = npt_state.system
        ThermodynamicState._standardize_system(npt_system)
        barostat = ThermodynamicState._find_barostat(npt_system)
        assert barostat.getDefaultPressure() == self.std_pressure
        assert get_barostat_temperature(barostat) == self.std_temperature
        assert ThermodynamicState._find_thermostat(npt_system) is None

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
            del context

    def test_method_is_compatible(self):
        """ThermodynamicState context and state compatibility methods."""
        def check_compatibility(state1, state2, is_compatible):
            assert state1.is_state_compatible(state2) is is_compatible
            assert state2.is_state_compatible(state1) is is_compatible
            time_step = 1.0*unit.femtosecond
            integrator1 = openmm.VerletIntegrator(time_step)
            integrator2 = openmm.VerletIntegrator(time_step)
            context1 = state1.create_context(integrator1)
            context2 = state2.create_context(integrator2)
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

        # If we change system/ensemble, cached values are updated correctly.
        toluene_implicit.system = self.toluene_vacuum
        check_compatibility(toluene_vacuum, toluene_implicit, True)

        barostated_alanine2.pressure = None  # Switch to NVT.
        check_compatibility(barostated_alanine, barostated_alanine2, False)

    def test_method_apply_to_context(self):
        """ThermodynamicState.apply_to_context() method."""
        friction = 5.0/unit.picosecond
        time_step = 2.0*unit.femtosecond
        state0 = ThermodynamicState(self.barostated_alanine, self.std_temperature)

        integrator = openmm.LangevinIntegrator(self.std_temperature, friction, time_step)
        context = state0.create_context(integrator)

        integrator = openmm.VerletIntegrator(1.0*unit.femtosecond)
        thermostated_context = state0.create_context(integrator)

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

        nvt_context = state2.create_context(openmm.VerletIntegrator(time_step))
        with nose.tools.assert_raises(ThermodynamicsError) as cm:
            state1.apply_to_context(nvt_context)
        assert cm.exception.code == ThermodynamicsError.INCOMPATIBLE_ENSEMBLE

    def test_method_reduced_potential(self):
        """ThermodynamicState.reduced_potential() method."""
        kj_mol = unit.kilojoule_per_mole
        beta = 1.0 / (unit.MOLAR_GAS_CONSTANT_R * self.std_temperature)
        state = ThermodynamicState(self.alanine_explicit, self.std_temperature)
        integrator = openmm.VerletIntegrator(1.0*unit.femtosecond)
        context = state.create_context(integrator)
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
        return thermodynamic_state.create_context(integrator)

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

        sliced_sampler_state = sampler_state[2:10]
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

        @classmethod
        def _standardize_system(cls, system):
            try:
                cls.set_dummy_parameter(system, cls.standard_dummy_parameter)
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

        @classmethod
        def add_dummy_parameter(cls, system):
            """Add to system a CustomBondForce with a dummy parameter."""
            force = openmm.CustomBondForce('dummy_parameter')
            force.addGlobalParameter('dummy_parameter', cls.standard_dummy_parameter)
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
        assert compound_state._barostat.getDefaultPressure() == self.std_pressure
        assert compound_state._thermostat.getDefaultTemperature() == new_temperature

    def test_constructor_set_state(self):
        """IComposableState.set_state is called on construction."""
        thermodynamic_state = ThermodynamicState(self.alanine_explicit, self.std_temperature)

        assert self.get_dummy_parameter(thermodynamic_state.system) != self.dummy_parameter
        compound_state = CompoundThermodynamicState(thermodynamic_state, [self.dummy_state])
        assert self.get_dummy_parameter(compound_state.system) == self.dummy_parameter

    def test_property_forwarding(self):
        """Forward properties to IComposableStates and update system."""
        dummy_state = self.DummyState(self.dummy_parameter + 1.0)
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

        system = thermodynamic_state.system
        barostat = ThermodynamicState._find_barostat(system)
        assert barostat.getDefaultPressure() != self.std_pressure
        assert self.get_dummy_parameter(system) == self.dummy_parameter
        compound_state._standardize_system(system)
        assert barostat.getDefaultPressure() == self.std_pressure
        assert self.get_dummy_parameter(system) == self.DummyState.standard_dummy_parameter

        # We still haven't computed the ThermodynamicState system hash
        # (pre-condition). Check that the standard system hash is correct.
        assert thermodynamic_state._cached_standard_system_hash is None
        standard_hash = openmm.XmlSerializer.serialize(system).__hash__()
        assert standard_hash == compound_state._standard_system_hash

        # Check that is_state_compatible work.
        undummied_alanine = testsystems.AlanineDipeptideExplicit().system
        incompatible_state = ThermodynamicState(undummied_alanine, self.std_temperature)
        assert not compound_state.is_state_compatible(incompatible_state)

        integrator = openmm.VerletIntegrator(2.0*unit.femtoseconds)
        context = incompatible_state.create_context(integrator)
        assert not compound_state.is_context_compatible(context)

    def test_method_apply_to_context(self):
        """CompoundThermodynamicState.apply_to_context() method."""
        dummy_parameter = self.DummyState.standard_dummy_parameter
        thermodynamic_state = ThermodynamicState(self.alanine_explicit, self.std_temperature)
        thermodynamic_state.pressure = self.std_pressure
        self.DummyState.set_dummy_parameter(thermodynamic_state.system, dummy_parameter)

        integrator = openmm.VerletIntegrator(2.0*unit.femtoseconds)
        context = thermodynamic_state.create_context(integrator)
        barostat = ThermodynamicState._find_barostat(context.getSystem())
        assert context.getParameter('dummy_parameter') == dummy_parameter
        assert context.getParameter(barostat.Pressure()) == self.std_pressure / unit.bar

        compound_state = CompoundThermodynamicState(thermodynamic_state, [self.dummy_state])
        new_pressure = thermodynamic_state.pressure + 1.0*unit.bar
        compound_state.pressure = new_pressure
        compound_state.apply_to_context(context)
        assert context.getParameter('dummy_parameter') == self.dummy_parameter
        assert context.getParameter(barostat.Pressure()) == new_pressure / unit.bar


# =============================================================================
# TEST SERIALIZATION
# =============================================================================

def test_states_serialization():
    """Test serialization compatibility with utils.serialize."""

    test_system = testsystems.AlanineDipeptideImplicit()
    thermodynamic_state = ThermodynamicState(test_system.system, temperature=300*unit.kelvin)
    sampler_state = SamplerState(positions=test_system.positions)

    test_cases = [thermodynamic_state, sampler_state]
    for test_state in test_cases:
        serialization = utils.serialize(test_state)
        deserialized_state = utils.deserialize(serialization)
        original_pickle = pickle.dumps(test_state)
        deserialized_pickle = pickle.dumps(deserialized_state)
        assert original_pickle == deserialized_pickle
