#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Classes that represent a portion of the state of an OpenMM context.

"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import abc
import copy

import numpy as np
from simtk import openmm, unit

from yank import utils

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================

def _box_vectors_volume(box_vectors):
    """Return the volume of the box vectors.

    Support also triclinic boxes.

    """
    a, b, c = box_vectors
    box_matrix = np.array([a/a.unit, b/a.unit, c/a.unit])
    return np.linalg.det(box_matrix) * a.unit**3


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

class ThermodynamicsError(Exception):

    # TODO substitute this with enum when we drop Python 2.7 support
    (MULTIPLE_BAROSTATS,
     UNSUPPORTED_BAROSTAT,
     INCONSISTENT_BAROSTAT,
     BAROSTATED_NONPERIODIC,
     INCONSISTENT_INTEGRATOR) = range(5)

    error_messages = {
        MULTIPLE_BAROSTATS: "System has multiple barostats.",
        UNSUPPORTED_BAROSTAT: "Found unsupported barostat {} in system.",
        INCONSISTENT_BAROSTAT: "System barostat is inconsistent with thermodynamic state.",
        BAROSTATED_NONPERIODIC: "Non-periodic systems cannot have a barostat.",
        INCONSISTENT_INTEGRATOR: "Integrator is coupled to a heat bath at a different temperature."
    }

    def __init__(self, code, *args):
        error_message = self.error_messages[code].format(*args)
        super(ThermodynamicsError, self).__init__(error_message)
        self.code = code


class SamplerStateError(Exception):

    # TODO substitute this with enum when we drop Python 2.7 support
    (INCONSISTENT_VELOCITIES) = range(1)

    error_messages = {
        INCONSISTENT_VELOCITIES: "Velocities have different length than positions.",
    }

    def __init__(self, code, *args):
        error_message = self.error_messages[code].format(*args)
        super(SamplerStateError, self).__init__(error_message)
        self.code = code


# =============================================================================
# THERMODYNAMIC STATE
# =============================================================================


class ThermodynamicState(object):
    """The state of a Context that does not change with integration."""

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def __init__(self, system, temperature, pressure=None):
        """Constructor.

        Parameters
        ----------
        system : simtk.openmm.System
            An OpenMM system in a particular thermodynamic state.
        temperature : simtk.unit.Quantity
            The temperature for the system at constant temperature. If
            a MonteCarloBarostat is associated to the system, its
            temperature will be set to this.
        pressure : simtk.unit.Quantity, optional
            The pressure for the system at constant pressure. If this
            is specified, a MonteCarloBarostat is added to the system,
            or just set to this pressure in case it already exists.

        """
        # The standard system hash is cached and computed on-demand.
        self._cached_standard_system_hash = None

        # Do not modify original system.
        self._system = copy.deepcopy(system)

        # We cannot model the temperature as a pure property because
        # if the system has no barostat, it doesn't contain any info
        # on T, so we need to maintain consistency between this
        # internal variable and the barostat/integrator.
        self._temperature = temperature

        # Set barostat temperature and pressure.
        self.temperature = temperature
        if pressure is not None:
            self.pressure = pressure

        self._check_internal_consistency()

    @property
    def system(self):
        """A copy of the system in this thermodynamic state."""
        # TODO wrap system in a CallBackable class to avoid copying it
        return copy.deepcopy(self._system)

    @system.setter
    def system(self, value):
        self._check_system_consistency(value)
        self._system = copy.deepcopy(value)
        self._cached_standard_system_hash = None  # Invalidate cache.

    @property
    def temperature(self):
        """Constant temperature of the thermodynamic state."""
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        self._temperature = value
        barostat = self._barostat
        if barostat is not None:
            self._set_barostat_temperature(barostat)

    @property
    def pressure(self):
        """Constant pressure of the thermodynamic state.

        If the pressure is allowed to fluctuate, this is None.

        """
        barostat = self._barostat
        if barostat is None:
            return None
        return barostat.getDefaultPressure()

    @pressure.setter
    def pressure(self, value):
        self._set_system_pressure(self._system, value)

    @property
    def volume(self):
        """Constant volume of the thermodynamic state.

        If the volume is allowed to fluctuate, this is None.

        """
        if self.pressure is not None:  # Volume fluctuates.
            return None
        if not self._system.usesPeriodicBoundaryConditions():
            return None
        box_vectors = self._system.getDefaultPeriodicBoxVectors()
        return _box_vectors_volume(box_vectors)

    def reduced_potential(self, sampler_state):
        """Compute reduced potential."""
        beta = 1.0 / (unit.MOLAR_GAS_CONSTANT_R * self.temperature)
        reduced_potential = sampler_state.potential_energy
        pressure = self.pressure
        if pressure is not None:
            reduced_potential += (pressure * sampler_state.volume *
                                  unit.AVOGADRO_CONSTANT_NA)
        return beta * reduced_potential

    def is_state_compatible(self, thermodynamic_state):
        """Check compatibility between ThermodynamicStates.

        The state is compatible if a context created by state is
        compatible.

        Parameters
        ----------
        thermodynamic_state : ThermodynamicState
            The thermodynamic state to test.

        Returns
        -------
        is_compatible : bool
            True if the context created by thermodynamic_state can be
            converted to this state with apply_to_context().

        See Also
        --------
        ThermodynamicState.is_context_compatible

        """
        try:
            state_system_hash = thermodynamic_state._standard_system_hash
        except AttributeError:
            state_system = thermodynamic_state.system
            state_system_hash = self._get_standard_system_hash(state_system)
        return self._standard_system_hash == state_system_hash

    def is_context_compatible(self, context):
        """Check compatibility of the given context.

        The context is compatible if this ThermodynamicState can be
        applied to it.

        Parameters
        ----------
        context : simtk.openmm.Context
            The OpenMM context to test.

        Returns
        -------
        is_compatible : bool
            True if this ThermodynamicState can be applied to context.

        See Also
        --------
        ThermodynamicState.apply_to_context
        ThermodynamicState.is_state_compatible

        """
        context_system_hash = self._get_standard_system_hash(context.getSystem())
        is_compatible = self._standard_system_hash == context_system_hash
        return is_compatible

    def create_context(self, integrator, platform=None):
        """Create a context in this ThermodynamicState.

        The context contains a copy of the system. An exception is
        raised if the integrator is coupled to a heat bath set at a
        temperature different from the thermodynamic state's.

        Parameters
        ----------
        integrator : simtk.openmm.Integrator
           The integrator to use for Context creation. The eventual
           heat bath temperature must be consistent with the
           thermodynamic state.
        platform : simtk.openmm.Platform, optional
           Platform to use. If None, OpenMM tries to select the fastest
           available platform. Default is None.

        Returns
        -------
        context : simtk.openmm.Context
           The created OpenMM Context object.

        Raises
        ------
        ThermodynamicsError
            If the integrator has an inconsistent temperature.

        """
        # Check that integrator is consistent
        if not self._is_integrator_consistent(integrator):
            raise ThermodynamicsError(ThermodynamicsError.INCONSISTENT_INTEGRATOR)
        if platform is None:
            return openmm.Context(self.system, integrator)
        else:
            return openmm.Context(self.system, integrator, platform)

    def apply_to_context(self, context):
        """Apply this ThermodynamicState to the context.

        The user is responsible to test if the context is compatible.

        Parameters
        ----------
        context : simtk.openmm.Context
           The OpenMM Context to be set to this ThermodynamicState.

        See Also
        --------
        ThermodynamicState.is_state_compatible
        ThermodynamicState.is_context_compatible

        """
        system = context.getSystem()
        integrator = context.getIntegrator()
        has_changed = self._set_system_pressure(system, self.pressure)
        barostat = self._find_barostat(system)
        if barostat is not None:
            has_changed = self._set_barostat_temperature(barostat) or has_changed
        has_changed = self._set_integrator_temperature(integrator) or has_changed
        if has_changed:
            context.reinitialize()
        # TODO context.setVelocitiesToTemperature()?

    @classmethod
    def turn_to_standard_system(cls, system):
        """Return a copy of the system in a standard representation.

        The standard system can be used to test compatibility between
        different ThermodynamicState objects. Here the standard system
        simply removes the barostat, which makes the system instance
        serialization independent from temperature and pressure.

        """
        barostat_id = cls._find_barostat_index(system)
        if barostat_id is not None:
            system.removeForce(barostat_id)

    # -------------------------------------------------------------------------
    # Internal-usage: system handling
    # -------------------------------------------------------------------------

    _NONPERIODIC_NONBONDED_METHODS = {openmm.NonbondedForce.NoCutoff,
                                      openmm.NonbondedForce.CutoffNonPeriodic}

    def _check_internal_consistency(self):
        """Shortcut self._check_system_consistency(self._system)."""
        self._check_system_consistency(self._system)

    def _check_system_consistency(self, system):
        """Raise an error if the system is inconsistent.

        Current check that there's only 1 barostat, that is supported,
        that has the correct temperature and pressure, and that it is
        not associated to a non-periodic system.

        """
        TE = ThermodynamicsError  # shortcut

        # This raises MULTIPLE_BAROSTATS and UNSUPPORTED_BAROSTAT.
        barostat = self._find_barostat(system)
        if barostat is not None:
            if not self._is_barostat_consistent(barostat):
                raise TE(TE.INCONSISTENT_BAROSTAT)

            # Check that barostat is not added to non-periodic system. We
            # cannot use System.usesPeriodicBoundaryConditions() because
            # in OpenMM < 7.1 that returns True when a barostat is added.
            # TODO just use usesPeriodicBoundaryConditions when drop openmm7.0
            for force in system.getForces():
                if isinstance(force, openmm.NonbondedForce):
                    nonbonded_method = force.getNonbondedMethod()
                    if nonbonded_method in self._NONPERIODIC_NONBONDED_METHODS:
                        raise TE(TE.BAROSTATED_NONPERIODIC)

    @classmethod
    def _get_standard_system_hash(cls, system):
        """Return the serialization hash of the standard system."""
        standard_system = copy.deepcopy(system)
        cls.turn_to_standard_system(standard_system)
        system_serialization = openmm.XmlSerializer.serialize(standard_system)
        return system_serialization.__hash__()

    @property
    def _standard_system_hash(self):
        """Shortcut for _get_standard_system_hash(self._system)."""
        if self._cached_standard_system_hash is None:
            self._cached_standard_system_hash = self._get_standard_system_hash(self._system)
        return self._cached_standard_system_hash

    # -------------------------------------------------------------------------
    # Internal-usage: integrator handling
    # -------------------------------------------------------------------------

    def _is_integrator_consistent(self, integrator):
        """False if integrator is coupled to a heat bath at different T."""
        if isinstance(integrator, openmm.CompoundIntegrator):
            integrator_id = integrator.getCurrentIntegrator()
            integrator = integrator.getIntegrator(integrator_id)
        try:
            return integrator.getTemperature() == self.temperature
        except AttributeError:
            return True

    def _set_integrator_temperature(self, integrator):
        """Set heat bath temperature of the integrator.

        Returns
        -------
        has_changed : bool
            True if the integrator temperature has changed.

        """
        has_changed = False
        try:
            if integrator.getTemperature() != self._temperature:
                integrator.setTemperature(self.temperature)
                has_changed = True
        except AttributeError:
            pass
        return has_changed

    # -------------------------------------------------------------------------
    # Internal-usage: barostat handling
    # -------------------------------------------------------------------------

    _SUPPORTED_BAROSTATS = {'MonteCarloBarostat'}

    @property
    def _barostat(self):
        """Shortcut for self._find_barostat(self._system)."""
        return self._find_barostat(self._system)

    @classmethod
    def _find_barostat(cls, system):
        """Shortcut for system.getForce(cls._find_barostat_index(system)).

        Returns
        -------
        barostat : OpenMM Force object
            The barostat in system, or None if no barostat is found.

        Raises
        ------
        ThermodynamicsError
            If the system contains unsupported barostats.

        """
        barostat_id = cls._find_barostat_index(system)
        if barostat_id is None:
            return None
        barostat = system.getForce(barostat_id)
        if barostat.__class__.__name__ not in cls._SUPPORTED_BAROSTATS:
            raise ThermodynamicsError(ThermodynamicsError.UNSUPPORTED_BAROSTAT,
                                      barostat.__class__.__name__)
        return barostat

    @classmethod
    def _find_barostat_index(cls, system):
        """Return the index of the first barostat found in the system.

        Returns
        -------
        barostat_id : int
            The index of the barostat force in self._system or None if
            no barostat is found.

        Raises
        ------
        ThermodynamicsError
            If the system contains multiple barostats.

        """
        barostat_ids = [i for i, force in enumerate(system.getForces())
                        if 'Barostat' in force.__class__.__name__]
        if len(barostat_ids) == 0:
            return None
        if len(barostat_ids) > 1:
            raise ThermodynamicsError(ThermodynamicsError.MULTIPLE_BAROSTATS)
        return barostat_ids[0]

    def _is_barostat_consistent(self, barostat):
        """Check the barostat's temperature and pressure."""
        try:
            barostat_temperature = barostat.getDefaultTemperature()
        except AttributeError:  # versions previous to OpenMM 7.1
            barostat_temperature = barostat.getTemperature()
        barostat_pressure = barostat.getDefaultPressure()
        is_consistent = barostat_temperature == self.temperature
        is_consistent = is_consistent and barostat_pressure == self.pressure
        return is_consistent

    def _set_system_pressure(self, system, pressure):
        """Add or configure the system barostat to the given pressure.

        The barostat temperature is set to self._temperature.

        Returns
        -------
        has_changed : bool
            True if the system has changed, False if it was already
            configured.

        """
        has_changed = False

        if pressure is None: # If new pressure is None, remove barostat.
            barostat_id = self._find_barostat_index(system)
            if barostat_id is not None:
                system.removeForce(barostat_id)
                has_changed = True

        elif not system.usesPeriodicBoundaryConditions():
            raise ThermodynamicsError(ThermodynamicsError.BAROSTATED_NONPERIODIC)

        else:  # Add/configure barostat
            barostat = self._find_barostat(system)
            if barostat is None:  # Add barostat
                barostat = openmm.MonteCarloBarostat(pressure, self._temperature)
                system.addForce(barostat)
                has_changed = True
            elif barostat.getDefaultPressure() != pressure:  # Set existing barostat
                barostat.setDefaultPressure(pressure)
                has_changed = True

        return has_changed

    def _set_barostat_temperature(self, barostat):
        """Shortcut to ensure OpenMM backwards compatibility.

        Returns
        -------
        has_changed : bool
            True if the barostat has changed, False if it was already
            configured with the correct temperature.

        """
        has_changed = False
        # TODO remove this when we OpenMM 7.0 drop support
        try:
            if barostat.getDefaultTemperature() != self._temperature:
                barostat.setDefaultTemperature(self._temperature)
                has_changed = True
        except AttributeError:  # versions previous to OpenMM 7.1
            if barostat.getTemperature() != self._temperature:
                barostat.setTemperature(self._temperature)
                has_changed = True
        return has_changed


# =============================================================================
# SAMPLER STATE
# =============================================================================

class SamplerState(object):
    """The state of a Context that change with integration."""

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def __init__(self, positions, velocities=None, box_vectors=None,
                 potential_energy=None, kinetic_energy=None):
        self.positions = positions
        self._velocities = None
        self.velocities = velocities  # Property checks consistency.
        self.box_vectors = box_vectors
        self.potential_energy = potential_energy
        self.kinetic_energy = kinetic_energy

    @staticmethod
    def from_context(context):
        sampler_state = SamplerState([])
        sampler_state._set_context_state(context, check_consistency=False)
        return sampler_state

    @property
    def velocities(self):
        return self._velocities

    @velocities.setter
    def velocities(self, value):
        if value is not None and len(self.positions) != len(value):
            raise SamplerStateError(SamplerStateError.INCONSISTENT_VELOCITIES)
        self._velocities = value

    @property
    def total_energy(self):
        return self.potential_energy + self.kinetic_energy

    @property
    def volume(self):
        return _box_vectors_volume(self.box_vectors)

    def is_context_compatible(self, context):
        openmm_state = context.getState(getPositions=True)
        return len(self.positions) == len(openmm_state.getPositions())

    def update_from_context(self, context):
        """Update the state with the context.

        The context must be compatible. Use SamplerState.from_context()
        if you want to build a new sampler state from an incompatible.

        Raises
        ------
        SamplerStateError
            If the given context is not compatible.

        """
        self._set_context_state(context, check_consistency=True)

    def apply_to_context(self, context):
        """Set the context state.

        If velocities and box vectors have not been specified, they are
        not set.

        """
        context.setPositions(self.positions)
        if self._velocities is not None:
            context.setVelocities(self._velocities)
        if self.box_vectors is not None:
            context.setPeriodicBoxVectors(*self.box_vectors)

    # -------------------------------------------------------------------------
    # Internal-usage
    # -------------------------------------------------------------------------

    def _set_context_state(self, context, check_consistency):
        openmm_state = context.getState(getPositions=True, getVelocities=True,
                                        getEnergy=True)
        if check_consistency:
            # We assign first the property velocities that perform a
            # consistency check with the current positions and raise
            # an error if the number of elements is different.
            self.velocities = openmm_state.getVelocities(asNumpy=True)
        else:
            self._velocities = openmm_state.getVelocities(asNumpy=True)
        self.positions = openmm_state.getPositions(asNumpy=True)
        self.box_vectors = openmm_state.getPeriodicBoxVectors()
        self.potential_energy = openmm_state.getPotentialEnergy()
        self.kinetic_energy = openmm_state.getKineticEnergy()


# =============================================================================
# COMPOUND THERMODYNAMIC STATE
# =============================================================================

class IComposableState(utils.SubhookedABCMeta):
    """A state composable through CompoundThermodynamicState."""

    @abc.abstractmethod
    def set_system_state(self, system):
        """This method is called everytime an attribute of the
        composable state is called, and on init to update the
        system with the stored new state. The system will be
        used to create_context()."""
        pass

    @abc.abstractmethod
    def check_system_consistency(self, system):
        """Raise error if the system is not consistent with the state.

        This is called when the system of the ThermodynamicState is set.

        """
        pass

    @abc.abstractmethod
    def apply_to_context(self, context):
        """If the changes are only in system, this can be avoided."""
        pass

    @classmethod
    @abc.abstractmethod
    def turn_to_standard_system(cls, system):
        """ThermodynamicState relies on this method to create
        a standard system hash used to check compatibility.

        Raise ValueError if it cannot be standardized."""
        pass


class CompoundThermodynamicState(ThermodynamicState):
    """Thermodynamic state composed by multiple states.

    Allows to extend a ThermodynamicState through composition rather
    than inheritance.

    It is the user's responsibility to check that IComposableStates are
    compatible to each other (i.e. that they do not depend on and modify
    the same properties of the system). If this is not the case
    consider merging them into a single IComposableStates. If an
    IComposableState needs to access properties of ThermodynamicState
    (e.g. temperature, pressure) consider extending through normal
    inheritance. CompoundThermodynamicState is compatible also with
    subclasses of ThermodynamicState.

    The class dynamically inherits from the given thermodynamic state.
    It does not support thermodynamic state objects which make use of
    __slots__.

    It is not necessary to explicitly inherit from IComposableState to
    be compatible. All attributes will still be accessible unless they
    are hidden by ThermodynamicState or by a previous IComposableState.

    IComposableState objects must be independent of each other, and
    they don't normally have access to each other states.

    """

    def __init__(self, thermodynamic_state, composable_states):
        # Check that composable states expose the correct interface.
        # for composable_state in composable_states:
        #     assert isinstance(composable_state, IComposableState)

        # Dynamically inherit from thermodynamic_state class.
        composable_bases = [s.__class__ for s in composable_states]
        self.__class__ = type(self.__class__.__name__,
                              (self.__class__, thermodynamic_state.__class__),
                              {'_composable_bases': composable_bases})
        self.__dict__ = thermodynamic_state.__dict__

        self._composable_states = composable_states
        for s in self._composable_states:
            s.set_system_state(self._system)

    @property
    def system(self):
        return super(CompoundThermodynamicState, self).system

    @system.setter
    def system(self, value):
        super(CompoundThermodynamicState, self.__class__).system.fset(self, value)
        for s in self._composable_states:
            s.check_system_consistency(self._system)

    @classmethod
    def turn_to_standard_system(cls, system):
        super(CompoundThermodynamicState, cls).turn_to_standard_system(system)
        for composable_cls in cls._composable_bases:
            composable_cls.turn_to_standard_system(system)

    def apply_to_context(self, context):
        super(CompoundThermodynamicState, self).apply_to_context(context)
        for s in self._composable_states:
            s.apply_to_context(context)

    def __getattr__(self, name):
        # Called only if the attribute couldn't be found in __dict__.
        # In this case we fall back to composable state, in the order.
        for s in self._composable_states:
            try:
                return getattr(s, name)
            except AttributeError:
                pass
        # Attribute not found, fall back to normal behavior.
        return super(CompoundThermodynamicState, self).__getattribute__(name)

    def __setattr__(self, name, value):
        # Add new attribute to CompoundThermodynamicState.
        if '_composable_states' not in self.__dict__:
            super(CompoundThermodynamicState, self).__setattr__(name, value)
        # Update existing ThermodynamicState attribute (check ancestors).
        elif any(name in C.__dict__ for C in self.__class__.__mro__):
            super(CompoundThermodynamicState, self).__setattr__(name, value)
        else:  # Update composable states attributes.
            for s in self._composable_states:
                if any(name in C.__dict__ for C in s.__class__.__mro__):
                    s.__setattr__(name, value)
                    s.set_system_state(self._system)
                    break
        # Monkey patching.
        super(CompoundThermodynamicState, self).__setattr__(name, value)
