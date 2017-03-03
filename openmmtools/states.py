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

from openmmtools import utils, integrators


# =============================================================================
# MODULE FUNCTIONS
# =============================================================================

def _box_vectors_volume(box_vectors):
    """Return the volume of the box vectors.

    Support also triclinic boxes.

    Parameters
    ----------
    box_vectors : simtk.unit.Quantity
        Vectors defining the box.

    Returns
    -------

    volume : simtk.unit.Quantity
        The box volume in units of length^3.

    Examples
    --------

    Compute the volume of a Lennard-Jones fluid at 100 K and 1 atm.

    >>> from openmmtools import testsystems
    >>> system = testsystems.LennardJonesFluid(nparticles=100).system
    >>> v = _box_vectors_volume(system.getDefaultPeriodicBoxVectors())

    """
    a, b, c = box_vectors
    box_matrix = np.array([a/a.unit, b/a.unit, c/a.unit])
    return np.linalg.det(box_matrix) * a.unit**3


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

class ThermodynamicsError(Exception):
    """Custom ThermodynamicState error.

    The exception defines error codes as class constants. Currently
    defined constants are MULTIPLE_BAROSTATS, UNSUPPORTED_BAROSTAT,
    INCONSISTENT_BAROSTAT, BAROSTATED_NONPERIODIC, and
    INCONSISTENT_INTEGRATOR.

    Parameters
    ----------
    code : ThermodynamicsError.Code
        The error code.

    Attributes
    ----------
    code : ThermodynamicsError.Code
        The code associated to this error.

    Examples
    --------
    >>> raise ThermodynamicsError(ThermodynamicsError.MULTIPLE_BAROSTATS)
    Traceback (most recent call last):
    ...
    ThermodynamicsError: System has multiple barostats.

    """

    # TODO substitute this with enum when we drop Python 2.7 support
    (MULTIPLE_THERMOSTATS,
     NO_THERMOSTAT,
     NONE_TEMPERATURE,
     INCONSISTENT_THERMOSTAT,
     MULTIPLE_BAROSTATS,
     NO_BAROSTAT,
     UNSUPPORTED_BAROSTAT,
     INCONSISTENT_BAROSTAT,
     BAROSTATED_NONPERIODIC,
     INCONSISTENT_INTEGRATOR,
     INCOMPATIBLE_SAMPLER_STATE,
     INCOMPATIBLE_ENSEMBLE) = range(12)

    error_messages = {
        MULTIPLE_THERMOSTATS: "System has multiple thermostats.",
        NO_THERMOSTAT: "System does not have a thermostat specifying the temperature.",
        NONE_TEMPERATURE: "Cannot set temperature of the thermodynamic state to None.",
        INCONSISTENT_THERMOSTAT: "System thermostat is inconsistent with thermodynamic state.",
        MULTIPLE_BAROSTATS: "System has multiple barostats.",
        UNSUPPORTED_BAROSTAT: "Found unsupported barostat {} in system.",
        NO_BAROSTAT: "System does not have a barostat specifying the pressure.",
        INCONSISTENT_BAROSTAT: "System barostat is inconsistent with thermodynamic state.",
        BAROSTATED_NONPERIODIC: "Non-periodic systems cannot have a barostat.",
        INCONSISTENT_INTEGRATOR: "Integrator is coupled to a heat bath at a different temperature.",
        INCOMPATIBLE_SAMPLER_STATE: "The sampler state has a different number of particles.",
        INCOMPATIBLE_ENSEMBLE: "Cannot apply to a context in a different thermodynamic ensemble."
    }

    def __init__(self, code, *args):
        error_message = self.error_messages[code].format(*args)
        super(ThermodynamicsError, self).__init__(error_message)
        self.code = code


class SamplerStateError(Exception):
    """Custom SamplerState error.

    The exception defines error codes as class constants. The only
    currently defined constant is INCONSISTENT_VELOCITIES.

    Parameters
    ----------
    code : SamplerStateError.Code
        The error code.

    Attributes
    ----------
    code : SamplerStateError.Code
        The code associated to this error.

    Examples
    --------
    >>> raise SamplerStateError(SamplerStateError.INCONSISTENT_VELOCITIES)
    Traceback (most recent call last):
    ...
    SamplerStateError: Velocities have different length than positions.

    """

    # TODO substitute this with enum when we drop Python 2.7 support
    (INCONSISTENT_VELOCITIES,
     INCONSISTENT_POSITIONS) = range(2)

    error_messages = {
        INCONSISTENT_VELOCITIES: "Velocities have different length than positions.",
        INCONSISTENT_POSITIONS: "Specified positions with inconsistent number of particles."
    }

    def __init__(self, code, *args):
        error_message = self.error_messages[code].format(*args)
        super(SamplerStateError, self).__init__(error_message)
        self.code = code


# =============================================================================
# THERMODYNAMIC STATE
# =============================================================================


class ThermodynamicState(object):
    """Thermodynamic state of a system.

    Represent the portion of the state of a Context that does not
    change with integration. Its main objectives are to wrap an
    OpenMM system object to easily maintain a consistent thermodynamic
    state. It can be used to create new OpenMM Contexts, or to convert
    an existing Context to this particular thermodynamic state.

    Only NVT and NPT ensembles are supported. The temperature must
    be specified in the constructor, either implicitly via a thermostat
    force in the system, or explicitly through the temperature
    parameter, which overrides an eventual thermostat indication.

    Parameters
    ----------
    system : simtk.openmm.System
        An OpenMM system in a particular thermodynamic state.
    temperature : simtk.unit.Quantity, optional
        The temperature for the system at constant temperature. If
        a MonteCarloBarostat is associated to the system, its
        temperature will be set to this. If None, the temperature
        is inferred from the system thermostat.
    pressure : simtk.unit.Quantity, optional
        The pressure for the system at constant pressure. If this
        is specified, a MonteCarloBarostat is added to the system,
        or just set to this pressure in case it already exists. If
        None, the pressure is inferred from the system barostat, and
        NVT ensemble is assumed if there is no barostat.

    Attributes
    ----------
    system
    temperature
    pressure
    volume
    n_particles

    Notes
    -----
    This state object cannot describe states obeying non-Boltzamnn
    statistics, such as Tsallis statistics.

    Examples
    --------
    Specify an NVT state for a water box at 298 K.

    >>> from openmmtools import testsystems
    >>> temperature = 298.0*unit.kelvin
    >>> waterbox = testsystems.WaterBox(box_edge=10*unit.angstroms,
    ...                                 cutoff=4*unit.angstroms).system
    >>> state = ThermodynamicState(system=waterbox, temperature=temperature)

    In an NVT ensemble volume is constant and pressure is None.

    >>> state.volume
    Quantity(value=1.0, unit=nanometer**3)
    >>> state.pressure is None
    True

    Convert this to an NPT state at 298 K and 1 atm pressure. This
    operation automatically adds a MonteCarloBarostat to the system.

    >>> pressure = 1.0*unit.atmosphere
    >>> state.pressure = pressure
    >>> state.pressure
    Quantity(value=1.01325, unit=bar)
    >>> state.volume is None
    True

    You cannot set a non-periodic system at constant pressure

    >>> nonperiodic_system = testsystems.TolueneVacuum().system
    >>> state = ThermodynamicState(nonperiodic_system, temperature=300*unit.kelvin,
    ...                            pressure=1.0*unit.atmosphere)
    Traceback (most recent call last):
    ...
    ThermodynamicsError: Non-periodic systems cannot have a barostat.

    When temperature and/or pressure are not specified (i.e. they are
    None) ThermodynamicState tries to infer them from a thermostat or
    a barostat.

    >>> state = ThermodynamicState(system=waterbox)
    Traceback (most recent call last):
    ...
    ThermodynamicsError: System does not have a thermostat specifying the temperature.
    >>> thermostat = openmm.AndersenThermostat(200.0*unit.kelvin, 1.0/unit.picosecond)
    >>> force_id = waterbox.addForce(thermostat)
    >>> state = ThermodynamicState(system=waterbox)
    >>> state.pressure is None
    True
    >>> state.temperature
    Quantity(value=200.0, unit=kelvin)
    >>> barostat = openmm.MonteCarloBarostat(1.0*unit.atmosphere, 200.0*unit.kelvin)
    >>> force_id = waterbox.addForce(barostat)
    >>> state = ThermodynamicState(system=waterbox)
    >>> state.pressure
    Quantity(value=1.01325, unit=bar)
    >>> state.temperature
    Quantity(value=200.0, unit=kelvin)

    """

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def __init__(self, system, temperature=None, pressure=None):
        self._initialize(system, temperature, pressure)

    @property
    def system(self):
        """The system in this thermodynamic state.

        The returned system is a copy and can be modified without
        altering the internal state of ThermodynamicState. In order
        to ensure a consistent thermodynamic state, the system has
        a Thermostat force. You can use get_system() to obtained a
        copy of the system without the thermostat. The method
        create_context then takes care of removing the thermostat
        when an integrator with a coupled heat bath is used (e.g.
        LangevinIntegrator).

        It can be set only to a system which is consistent with the
        current thermodynamic state. Use set_system() if you want to
        correct the thermodynamic state of the system automatically
        before assignment.

        See Also
        --------
        ThermodynamicState.get_system
        ThermodynamicState.set_system
        ThermodynamicState.create_context

        Examples
        --------

        """
        return self.get_system()

    @system.setter
    def system(self, value):
        self.set_system(value)

    def set_system(self, system, fix_state=False):
        """Manipulate and set the system.

        With default arguments, this is equivalent to using the system
        property, which raises an exception if the thermostat and the
        barostat are not configured according to the thermodynamic state.
        With this method it is possible to adjust temperature and
        pressure of the system to make the assignment possible, without
        manually configuring thermostat and barostat.

        Parameters
        ----------
        system : simtk.openmm.System
            The system to set.
        fix_state : bool, optional
            If True, a thermostat is added to the system (if not already
            present) and set to the correct temperature. If this state is
            in NPT ensemble, a barostat is added or configured if it
            exist already. If False, this simply check that thermostat
            and barostat are correctly configured without modifying them.
            Default is False.

        Raises
        ------
        ThermodynamicsError
            If the system after the requested manipulation is still in
            an incompatible state.

        Examples
        --------
        The constructor adds a thermostat and a barostat to configure
        the system in an NPT ensemble.

        >>> from openmmtools import testsystems
        >>> alanine = testsystems.AlanineDipeptideExplicit()
        >>> state = ThermodynamicState(alanine.system, temperature=300*unit.kelvin,
        ...                            pressure=1.0*unit.atmosphere)

        If we try to set a system not in NPT ensemble, an error occur.

        >>> state.system = alanine.system
        Traceback (most recent call last):
        ...
        ThermodynamicsError: System does not have a thermostat specifying the temperature.

        We can fix both thermostat and barostat while setting the system.
        >>> state.set_system(alanine.system, fix_state=True)

        """
        system = copy.deepcopy(system)
        if fix_state:
            self._set_system_thermostat(system, self.temperature)
            barostat = self._set_system_pressure(system, self.pressure)
            if barostat is not None:
                self._set_barostat_temperature(barostat, self.temperature)
        else:
            self._check_system_consistency(system)
        self._system = system
        self._cached_standard_system_hash = None  # Invalidate cache.

    def get_system(self, remove_thermostat=False, remove_barostat=False):
        """Manipulate and return the system.

        With default arguments, this is equivalent as the system property.
        By setting the arguments it is possible to obtain a modified copy
        of the system without the thermostat or the barostat.

        Parameters
        ----------
        remove_thermostat : bool
            If True, the system thermostat is removed.
        remove_barostat : bool
            If True, the system barostat is removed.

        Returns
        -------
        system : simtk.openmm.System
            The system of this ThermodynamicState.

        Examples
        --------
        The constructor adds a thermostat and a barostat to configure
        the system in an NPT ensemble.

        >>> from openmmtools import testsystems
        >>> alanine = testsystems.AlanineDipeptideExplicit()
        >>> state = ThermodynamicState(alanine.system, temperature=300*unit.kelvin,
        ...                            pressure=1.0*unit.atmosphere)

        The system property returns a copy of the system with the
        added thermostat and barostat.

        >>> system = state.system
        >>> [force.__class__.__name__ for force in system.getForces()
        ...  if 'Thermostat' in force.__class__.__name__]
        ['AndersenThermostat']

        We can remove them while getting the arguments with

        >>> system = state.get_system(remove_thermostat=True, remove_barostat=True)
        >>> [force.__class__.__name__ for force in system.getForces()
        ...  if 'Thermostat' in force.__class__.__name__]
        []

        """
        system = copy.deepcopy(self._system)
        if remove_thermostat:
            self._remove_thermostat(system)
        if remove_barostat:
            self._pop_barostat(system)
        return system

    @property
    def temperature(self):
        """Constant temperature of the thermodynamic state."""
        return self._thermostat.getDefaultTemperature()

    @temperature.setter
    def temperature(self, value):
        if value is None:
            raise ThermodynamicsError(ThermodynamicsError.NONE_TEMPERATURE)
        self._set_system_thermostat(self._system, value)
        barostat = self._barostat
        if barostat is not None:
            self._set_barostat_temperature(barostat, value)

    @property
    def pressure(self):
        """Constant pressure of the thermodynamic state.

        If the pressure is allowed to fluctuate, this is None. Setting
        this will automatically add/configure a barostat to the system.
        If it is set to None, the barostat will be removed.

        """
        barostat = self._barostat
        if barostat is None:
            return None
        return barostat.getDefaultPressure()

    @pressure.setter
    def pressure(self, value):
        # Invalidate cache if the ensemble changes.
        if (value is None) != (self._barostat is None):
            self._cached_standard_system_hash = None
        self._set_system_pressure(self._system, value)

    @property
    def barostat(self):
        """The barostat associated to the system.

        Note that this is only a copy of the barostat, and you will need
        to set back the ThermodynamicState.barostat property for the changes
        to take place internally. If the pressure is allowed to fluctuate,
        this is None. Normally, you should only need to access the pressure
        and temperature properties, but this allows you to modify other parameters
        of the MonteCarloBarostat (e.g. frequency) after initialization. Setting
        this to None will place the system in an NVT ensemble.

        """
        return copy.deepcopy(self._barostat)

    @barostat.setter
    def barostat(self, value):
        if value is None:
            # Reset the standard system hash only if we actually switch to NVT.
            if self._pop_barostat(self._system) is not None:
                self._cached_standard_system_hash = None
        else:
            old_barostat = self._pop_barostat(self._system)
            new_barostat = copy.deepcopy(value)
            self._system.addForce(new_barostat)
            try:
                self._check_internal_consistency()
            except Exception as e:
                # Restore old barostat to leave state consistent.
                self.barostat = old_barostat
                raise e
            self._cached_standard_system_hash = None

    @property
    def volume(self):
        """Constant volume of the thermodynamic state.

        Read-only. If the volume is allowed to fluctuate, or if the
        system is not in a periodic box this is None.

        """
        if self.pressure is not None:  # Volume fluctuates.
            return None
        if not self._system.usesPeriodicBoundaryConditions():
            return None
        box_vectors = self._system.getDefaultPeriodicBoxVectors()
        return _box_vectors_volume(box_vectors)

    @property
    def n_particles(self):
        """Number of particles (read-only)."""
        return self._system.getNumParticles()

    def reduced_potential(self, context_state):
        """Reduced potential in this thermodynamic state.

        Parameters
        ----------
        context_state : SamplerState or simtk.openmm.Context
            Carry the configurational properties of the system.

        Returns
        -------
        u : float
            The unit-less reduced potential, which can be considered
            to have units of kT.

        Notes
        -----
        The reduced potential is defined as in Ref. [1]

        u = \beta [U(x) + p V(x) + \mu N(x)]

        where the thermodynamic parameters are

        \beta = 1/(kB T) is the inverse temperature
        p is the pressure
        \mu is the chemical potential

        and the configurational properties are

        x the atomic positions
        U(x) is the potential energy
        V(x) is the instantaneous box volume
        N(x) the numbers of various particle species (e.g. protons of
             titratible groups)

        References
        ----------
        [1] Shirts MR and Chodera JD. Statistically optimal analysis of
        equilibrium states. J Chem Phys 129:124105, 2008.

        Examples
        --------

        Compute the reduced potential of a water box at 298 K and 1 atm.

        >>> from openmmtools import testsystems
        >>> waterbox = testsystems.WaterBox(box_edge=20.0*unit.angstroms)
        >>> system, positions = waterbox.system, waterbox.positions
        >>> state = ThermodynamicState(system=waterbox.system,
        ...                            temperature=298.0*unit.kelvin,
        ...                            pressure=1.0*unit.atmosphere)
        >>> integrator = openmm.VerletIntegrator(1.0*unit.femtosecond)
        >>> context = state.create_context(integrator)
        >>> context.setPositions(waterbox.positions)
        >>> sampler_state = SamplerState.from_context(context)
        >>> u = state.reduced_potential(sampler_state)

        If the sampler state is incompatible, an error is raised

        >>> incompatible_sampler_state = sampler_state[:-1]
        >>> state.reduced_potential(incompatible_sampler_state)
        Traceback (most recent call last):
        ...
        ThermodynamicsError: The sampler state has a different number of particles.

        In case a cached SamplerState containing the potential energy
        and the volume of the context is not available, the method
        accepts a Context object and compute them with Context.getState().

        >>> u = state.reduced_potential(context)

        """
        # Read Context/SamplerState n_particles, energy and volume.
        if isinstance(context_state, openmm.Context):
            n_particles = context_state.getSystem().getNumParticles()
            openmm_state = context_state.getState(getEnergy=True)
            potential_energy = openmm_state.getPotentialEnergy()
            volume = openmm_state.getPeriodicBoxVolume()
        else:
            n_particles = context_state.n_particles
            potential_energy = context_state.potential_energy
            volume = context_state.volume

        # Check compatibility.
        if n_particles != self.n_particles:
            raise ThermodynamicsError(ThermodynamicsError.INCOMPATIBLE_SAMPLER_STATE)

        beta = 1.0 / (unit.BOLTZMANN_CONSTANT_kB * self.temperature)
        reduced_potential = potential_energy
        reduced_potential = reduced_potential / unit.AVOGADRO_CONSTANT_NA
        pressure = self.pressure
        if pressure is not None:
            reduced_potential += pressure * volume
        return beta * reduced_potential

    def is_state_compatible(self, thermodynamic_state):
        """Check compatibility between ThermodynamicStates.

        The state is compatible if Contexts created by thermodynamic_state
        can be set to this ThermodynamicState through apply_to_context.
        The property is symmetric and transitive.

        This is faster than checking compatibility of a Context object
        through is_context_compatible since, and it should be preferred
        when possible.

        Parameters
        ----------
        thermodynamic_state : ThermodynamicState
            The thermodynamic state to test.

        Returns
        -------
        is_compatible : bool
            True if the context created by thermodynamic_state can be
            converted to this state through apply_to_context().

        See Also
        --------
        ThermodynamicState.apply_to_context
        ThermodynamicState.is_context_compatible

        Examples
        --------
        States in the same ensemble (NVT or NPT) are compatible.

        >>> from simtk import unit
        >>> from openmmtools import testsystems
        >>> alanine = testsystems.AlanineDipeptideExplicit()
        >>> state1 = ThermodynamicState(alanine.system, 273*unit.kelvin)
        >>> state2 = ThermodynamicState(alanine.system, 310*unit.kelvin)
        >>> state1.is_state_compatible(state2)
        True

        States in different ensembles are not compatible.

        >>> state1.pressure = 1.0*unit.atmosphere
        >>> state1.is_state_compatible(state2)
        False

        States that store different systems (that differ by more than
        barostat and thermostat pressure and temperature) are also not
        compatible.

        >>> alanine_implicit = testsystems.AlanineDipeptideImplicit().system
        >>> state_implicit = ThermodynamicState(alanine_implicit, 310*unit.kelvin)
        >>> state2.is_state_compatible(state_implicit)
        False

        """
        state_system_hash = thermodynamic_state._standard_system_hash
        return self._standard_system_hash == state_system_hash

    def is_context_compatible(self, context):
        """Check compatibility of the given context.

        This is equivalent to is_state_compatible but slower, and it should
        be used only when the state the created the context is unknown. The
        context is compatible if it can be set to this ThermodynamicState
        through apply_to_context().

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

        The context contains a copy of the system. If the integrator
        is coupled to a heat bath (e.g. LangevinIntegrator), the system
        in the context will not have a thermostat, and vice versa if
        the integrator is not thermostated the system in the context will
        have a thermostat.

        An integrator is considered thermostated if it exposes a method
        getTemperature(). A CompoundIntegrator is considered coupled to
        a heat bath if at least one of its integrators is. An exception
        is raised if the integrator is thermostated at a temperature
        different from the thermodynamic state's.

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
            If the integrator has a temperature different from this
            ThermodynamicState.

        Examples
        --------
        Creating a context with a thermostated integrator means that
        the context system will not have a thermostat.

        >>> from simtk import openmm, unit
        >>> from openmmtools import testsystems
        >>> toluene = testsystems.TolueneVacuum()
        >>> state = ThermodynamicState(toluene.system, 300*unit.kelvin)
        >>> integrator = openmm.VerletIntegrator(1.0*unit.femtosecond)
        >>> context = state.create_context(integrator)
        >>> system = context.getSystem()
        >>> [force.__class__.__name__ for force in system.getForces()
        ...  if 'Thermostat' in force.__class__.__name__]
        ['AndersenThermostat']

        The thermostat is removed if we choose an integrator coupled
        to a heat bath.

        >>> integrator = openmm.LangevinIntegrator(300*unit.kelvin, 5.0/unit.picosecond,
        ...                                        2.0*unit.femtosecond)
        >>> context = state.create_context(integrator)
        >>> system = context.getSystem()
        >>> [force.__class__.__name__ for force in system.getForces()
        ...  if 'Thermostat' in force.__class__.__name__]
        []

        """
        # Check that integrator is consistent and if it is thermostated.
        # With CompoundIntegrator, at least one must be thermostated.
        is_thermostated = self._is_integrator_thermostated(integrator)

        # If integrator is coupled to heat bath, remove system thermostat.
        system = copy.deepcopy(self._system)
        if is_thermostated:
            self._remove_thermostat(system)

        # Create platform.
        if platform is None:
            return openmm.Context(system, integrator)
        else:
            return openmm.Context(system, integrator, platform)

    def apply_to_context(self, context):
        """Apply this ThermodynamicState to the context.

        The method apply_to_context does *not* check for the compatibility
        of the context. The user is responsible for this. Depending on the
        system size, is_context_compatible can be an expensive operation,
        so is_state_compatible should be preferred when possible.

        Parameters
        ----------
        context : simtk.openmm.Context
           The OpenMM Context to be set to this ThermodynamicState.

        Raises
        ------
        ThermodynamicsError
            If the context is in a different thermodynamic ensemble w.r.t.
            this state. This is just a quick check which does not substitute
            is_state_compatible or is_context_compatible.

        See Also
        --------
        ThermodynamicState.is_state_compatible
        ThermodynamicState.is_context_compatible

        Examples
        --------
        The method doesn't verify compatibility with the context, it is
        the user's responsibility to do so, possibly with is_state_compatible
        rather than is_context_compatible which is slower.

        >>> from simtk import openmm, unit
        >>> from openmmtools import testsystems
        >>> toluene = testsystems.TolueneVacuum()
        >>> state1 = ThermodynamicState(toluene.system, 273.0*unit.kelvin)
        >>> state2 = ThermodynamicState(toluene.system, 310.0*unit.kelvin)
        >>> integrator = openmm.VerletIntegrator(1.0*unit.femtosecond)
        >>> context = state1.create_context(integrator)
        >>> if state2.is_state_compatible(state1):
        ...     state2.apply_to_context(context)
        >>> context.getParameter(openmm.AndersenThermostat.Temperature())
        310.0

        """
        system = context.getSystem()

        # Apply pressure and temperature to barostat.
        barostat = self._find_barostat(system)
        if barostat is not None:
            if self._barostat is None:
                # The context is NPT but this is NVT.
                raise ThermodynamicsError(ThermodynamicsError.INCOMPATIBLE_ENSEMBLE)

            has_changed = self._set_barostat_pressure(barostat, self.pressure)
            if has_changed:
                context.setParameter(barostat.Pressure(), self.pressure)
            has_changed = self._set_barostat_temperature(barostat, self.temperature)
            if has_changed:
                # TODO remove try except when drop openmm7.0 support
                try:
                    context.setParameter(barostat.Temperature(), self.temperature)
                except AttributeError:  # OpenMM < 7.1
                    openmm_state = context.getState(getPositions=True, getVelocities=True,
                                                    getParameters=True)
                    context.reinitialize()
                    context.setState(openmm_state)
        elif self._barostat is not None:
            # The context is NVT but this is NPT.
            raise ThermodynamicsError(ThermodynamicsError.INCOMPATIBLE_ENSEMBLE)

        # Apply temperature to thermostat or integrator.
        thermostat = self._find_thermostat(system)
        if thermostat is not None:
            if not utils.is_quantity_close(thermostat.getDefaultTemperature(),
                                           self.temperature):
                thermostat.setDefaultTemperature(self.temperature)
                context.setParameter(thermostat.Temperature(), self.temperature)
        else:
            integrator = context.getIntegrator()
            self._set_integrator_temperature(integrator)

    def __getstate__(self):
        """Return a dictionary representation of the state."""
        # We just need to serialize the system since everything
        # else is inferred from thermostat and barostat state.
        serialized_system = openmm.XmlSerializer.serialize(self._system)
        return dict(system=serialized_system)

    def __setstate__(self, serialization):
        """Set the state from a dictionary representation."""
        deserialized_system = openmm.XmlSerializer.deserialize(serialization['system'])
        self._initialize(deserialized_system)

    # -------------------------------------------------------------------------
    # Internal-usage: initialization
    # -------------------------------------------------------------------------

    def _initialize(self, system, temperature=None, pressure=None):
        """Initialize the thermodynamic state."""
        # The standard system hash is cached and computed on-demand.
        self._cached_standard_system_hash = None

        # Do not modify original system.
        self._system = copy.deepcopy(system)

        # If temperature is None, the user must specify a thermostat.
        if temperature is None:
            if self._thermostat is None:
                raise ThermodynamicsError(ThermodynamicsError.NO_THERMOSTAT)
            # Read temperature from thermostat and pass it to barostat.
            temperature = self.temperature

        # Set thermostat and barostat temperature.
        self.temperature = temperature

        # Set barostat pressure.
        if pressure is not None:
            self.pressure = pressure

        self._check_internal_consistency()

    # -------------------------------------------------------------------------
    # Internal-usage: system handling
    # -------------------------------------------------------------------------

    # Standard values are not standard in a physical sense, they are
    # just consistent between ThermodynamicStates to make comparison
    # of standard system hashes possible. We set this to round floats
    # and use OpenMM units to avoid funniness due to precision errors
    # caused by periodic binary representation/unit conversion.
    _STANDARD_PRESSURE = 1.0*unit.bar
    _STANDARD_TEMPERATURE = 273.0*unit.kelvin

    _NONPERIODIC_NONBONDED_METHODS = {openmm.NonbondedForce.NoCutoff,
                                      openmm.NonbondedForce.CutoffNonPeriodic}

    def _check_internal_consistency(self):
        """Shortcut self._check_system_consistency(self._system)."""
        self._check_system_consistency(self._system)

    def _check_system_consistency(self, system):
        """Check system consistency with this ThermodynamicState.

        Raise an error if the system is inconsistent. Currently checks
        that there's 1 and only 1 thermostat at the correct temperature,
        that there's only 1 barostat (or none in case this is in NVT),
        that the barostat is supported, has the correct temperature and
        pressure, and that it is not associated to a non-periodic system.

        Parameters
        ----------
        system : simtk.openmm.System
            The system to test.

        Raises
        ------
        ThermodynamicsError
            If the system is inconsistent with this state.

        """
        TE = ThermodynamicsError  # shortcut

        # This raises MULTIPLE_THERMOSTATS
        thermostat = self._find_thermostat(system)
        # When system is self._system, we check the presence of a
        # thermostat before the barostat to avoid crashes when
        # checking the barostat temperature.
        if thermostat is None:
            raise TE(TE.NO_THERMOSTAT)
        elif not utils.is_quantity_close(thermostat.getDefaultTemperature(),
                                         self.temperature):
            raise TE(TE.INCONSISTENT_THERMOSTAT)

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
        elif self._barostat is not None:
            raise TE(TE.NO_BAROSTAT)

    @classmethod
    def _standardize_system(cls, system):
        """Return a copy of the system in a standard representation.

        This effectively defines which ThermodynamicStates are compatible
        between each other. Compatible ThermodynamicStates have the same
        standard systems, and is_state_compatible will return True if
        the (cached) serialization of the standard systems are identical.

        Here, the standard system has the barostat pressure/temperature
        set to _STANDARD_PRESSURE/TEMPERATURE (if a barostat exist), and
        the thermostat removed (if it is present). Removing the thermostat
        means that systems that will enforce a temperature through an
        integrator coupled to a heat bath will be compatible as well. The
        method apply_to_context then sets the parameters in the Context.

        Effectively this means that only same systems in the same ensemble
        (NPT or NVT) are compatible between each other.

        Parameters
        ----------
        system : simtk.openmm.System
            The system to standardize.

        See Also
        --------
        ThermodynamicState.apply_to_context
        ThermodynamicState.is_state_compatible
        ThermodynamicState.is_context_compatible

        """
        cls._remove_thermostat(system)
        barostat = cls._find_barostat(system)
        if barostat is not None:
            barostat.setDefaultPressure(cls._STANDARD_PRESSURE)
            cls._set_barostat_temperature(barostat, cls._STANDARD_TEMPERATURE)


    @classmethod
    def _get_standard_system_hash(cls, system):
        """Return the serialization hash of the standard system."""
        standard_system = copy.deepcopy(system)
        cls._standardize_system(standard_system)
        system_serialization = openmm.XmlSerializer.serialize(standard_system)
        return system_serialization.__hash__()

    @property
    def _standard_system_hash(self):
        """Shortcut for _get_standard_system_hash(self._system).

        This property is marked for internal usage since we may change
        this system in the future, but ContextCache makes use of this
        property too.

        See Also
        --------
        cache.ContextCache._generate_context_id

        """
        if self._cached_standard_system_hash is None:
            self._cached_standard_system_hash = self._get_standard_system_hash(self._system)
        return self._cached_standard_system_hash

    # -------------------------------------------------------------------------
    # Internal-usage: integrator handling
    # -------------------------------------------------------------------------

    @staticmethod
    def _loop_over_integrators(integrator):
        """Unify manipulation of normal, compound and thermostated integrators."""
        if isinstance(integrator, openmm.CompoundIntegrator):
            for integrator_id in range(integrator.getNumIntegrators()):
                _integrator = integrator.getIntegrator(integrator_id)
                integrators.ThermostatedIntegrator.restore_interface(_integrator)
                yield _integrator
        else:
            integrators.ThermostatedIntegrator.restore_interface(integrator)
            yield integrator

    def _is_integrator_thermostated(self, integrator):
        """True if integrator is coupled to a heat bath.

        If integrator is a CompoundIntegrator, it returns true if at least
        one of its integrators is coupled to a heat bath.

        Raises
        ------
        ThermodynamicsError
            If integrator is couple to a heat bath at a different
            temperature than this thermodynamic state.

        """
        # Loop over integrators to handle CompoundIntegrators.
        is_thermostated = False
        for _integrator in self._loop_over_integrators(integrator):
            try:
                temperature = _integrator.getTemperature()
            except AttributeError:
                pass
            else:
                # Raise exception if the heat bath is at the wrong temperature.
                if not utils.is_quantity_close(temperature, self.temperature):
                    err_code = ThermodynamicsError.INCONSISTENT_INTEGRATOR
                    raise ThermodynamicsError(err_code)
                is_thermostated = True
                # We still need to loop over every integrator to make sure
                # that the temperature is consistent for all of them.
        return is_thermostated

    def _set_integrator_temperature(self, integrator):
        """Set heat bath temperature of the integrator.

        If integrator is a CompoundIntegrator, it sets the temperature
        of every sub-integrator.

        Returns
        -------
        has_changed : bool
            True if the integrator temperature has changed.

        """
        def set_temp(_integrator):
            try:
                if not utils.is_quantity_close(_integrator.getTemperature(),
                                               self.temperature):
                    _integrator.setTemperature(self.temperature)
                    return True
            except AttributeError:
                pass
            return False

        # Loop over integrators to handle CompoundIntegrators.
        has_changed = False
        for _integrator in self._loop_over_integrators(integrator):
            has_changed = has_changed or set_temp(_integrator)
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
        """Return the first barostat found in the system.

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
    def _pop_barostat(cls, system):
        """Remove the system barostat.

        Returns
        -------
        The removed barostat if it was found, None otherwise.

        """
        barostat_id = cls._find_barostat_index(system)
        if barostat_id is not None:
            barostat = system.getForce(barostat_id)
            system.removeForce(barostat_id)
            return barostat
        return None

    @staticmethod
    def _find_barostat_index(system):
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
        is_consistent = utils.is_quantity_close(barostat_temperature, self.temperature)
        is_consistent = is_consistent and utils.is_quantity_close(barostat_pressure,
                                                                  self.pressure)
        return is_consistent

    def _set_system_pressure(self, system, pressure):
        """Add or configure the system barostat to the given pressure.

        If a new barostat is added, its temperature is set to
        self.temperature.

        Parameters
        ----------
        system : simtk.openmm.System
            The system's barostat will be added/configured.
        pressure : simtk.unit.Quantity or None
            The pressure with units compatible to bars. If None, the
            barostat of the system is removed.

        Returns
        -------
        barostat : OpenMM Force object or None
            The current barostat of the system. None if the barostat
            was removed.

        Raises
        ------
        ThermodynamicsError
            If pressure needs to be set for a non-periodic system.

        """
        if pressure is None:  # If new pressure is None, remove barostat.
            self._pop_barostat(system)
            return None

        if not system.usesPeriodicBoundaryConditions():
            raise ThermodynamicsError(ThermodynamicsError.BAROSTATED_NONPERIODIC)

        barostat = self._find_barostat(system)
        if barostat is None:  # Add barostat
            barostat = openmm.MonteCarloBarostat(pressure, self.temperature)
            system.addForce(barostat)
        else:  # Set existing barostat
            self._set_barostat_pressure(barostat, pressure)
        return barostat

    @staticmethod
    def _set_barostat_pressure(barostat, pressure):
        """Set barostat pressure.

        Returns
        -------
        has_changed : bool
            True if the barostat has changed, False if it was already
            configured with the correct pressure.

        """
        if not utils.is_quantity_close(barostat.getDefaultPressure(), pressure):
            barostat.setDefaultPressure(pressure)
            return True
        return False

    @staticmethod
    def _set_barostat_temperature(barostat, temperature):
        """Set barostat temperature.

        Returns
        -------
        has_changed : bool
            True if the barostat has changed, False if it was already
            configured with the correct temperature.

        """
        has_changed = False
        # TODO remove this when we OpenMM 7.0 drop support
        try:
            if not utils.is_quantity_close(barostat.getDefaultTemperature(),
                                           temperature):
                barostat.setDefaultTemperature(temperature)
                has_changed = True
        except AttributeError:  # versions previous to OpenMM 7.1
            if not utils.is_quantity_close(barostat.getTemperature(), temperature):
                barostat.setTemperature(temperature)
                has_changed = True
        return has_changed

    # -------------------------------------------------------------------------
    # Internal-usage: thermostat handling
    # -------------------------------------------------------------------------

    @property
    def _thermostat(self):
        """Shortcut for self._find_thermostat(self._system)."""
        return self._find_thermostat(self._system)

    @classmethod
    def _find_thermostat(cls, system):
        """Return the first thermostat in the system.

        Returns
        -------
        thermostat : OpenMM Force object or None
            The thermostat in system, or None if no thermostat is found.

        """
        thermostat_id = cls._find_thermostat_index(system)
        if thermostat_id is not None:
            return system.getForce(thermostat_id)
        return None

    @classmethod
    def _remove_thermostat(cls, system):
        """Remove the system thermostat.

        Returns
        -------
        True if the thermostat was found and removed, False otherwise.

        """
        thermostat_id = cls._find_thermostat_index(system)
        if thermostat_id is not None:
            system.removeForce(thermostat_id)
            return True
        return False

    @staticmethod
    def _find_thermostat_index(system):
        """Return the index of the first thermostat in the system."""
        thermostat_ids = [i for i, force in enumerate(system.getForces())
                          if 'Thermostat' in force.__class__.__name__]
        if len(thermostat_ids) == 0:
            return None
        if len(thermostat_ids) > 1:
            raise ThermodynamicsError(ThermodynamicsError.MULTIPLE_THERMOSTATS)
        return thermostat_ids[0]

    def _is_thermostat_consistent(self, thermostat):
        """Check thermostat temperature."""
        return utils.is_quantity_close(thermostat.getDefaultTemperature(), self.temperature)

    @classmethod
    def _set_system_thermostat(cls, system, temperature):
        """Configure the system thermostat.

        If temperature is None and the system has a thermostat, it is
        removed. Otherwise the thermostat temperature is set, or a new
        AndersenThermostat is added if it doesn't exist.

        Parameters
        ----------
        system : simtk.openmm.System
            The system to modify.
        temperature : simtk.unit.Quantity or None
            The temperature for the thermostat, or None to remove it.

        Returns
        -------
        has_changed : bool
            True if the thermostat has changed, False if it was already
            configured with the correct temperature.

        """
        has_changed = False
        if temperature is None:  # Remove thermostat.
            has_changed = cls._remove_thermostat(system)
        else:  # Add/configure existing thermostat.
            thermostat = cls._find_thermostat(system)
            if thermostat is None:
                thermostat = openmm.AndersenThermostat(temperature, 1.0/unit.picosecond)
                system.addForce(thermostat)
                has_changed = True
            elif not utils.is_quantity_close(thermostat.getDefaultTemperature(),
                                             temperature):
                thermostat.setDefaultTemperature(temperature)
                has_changed = True
        return has_changed


# =============================================================================
# SAMPLER STATE
# =============================================================================

class SamplerState(object):
    """State carrying the configurational properties of a system.

    Represent the portion of the state of a Context that changes with
    integration. When initialized through the normal constructor, the
    object is only partially defined as the energy attributes are None
    until the SamplerState is updated with update_from_context. The
    state can still be applied to a newly created context to set its
    positions, velocities and box vectors. To initialize all attributes,
    use the alternative constructor from_context.

    Parameters
    ----------
    positions : Nx3 simtk.unit.Quantity
        Position vectors for N particles (length units).
    velocities : Nx3 simtk.unit.Quantity, optional
        Velocity vectors for N particles (velocity units).
    box_vectors : 3x3 simtk.unit.Quantity
        Current box vectors (length units).

    Attributes
    ----------
    positions
    velocities
    box_vectors : 3x3 simtk.unit.Quantity.
        Current box vectors (length units).
    potential_energy : simtk.unit.Quantity or None
        Potential energy of this configuration.
    kinetic_energy : simtk.unit.Quantity
        Kinetic energy of this configuration.
    total_energy
    volume
    n_particles

    Examples
    --------

    >>> from openmmtools import testsystems
    >>> toluene_test = testsystems.TolueneVacuum()
    >>> sampler_state = SamplerState(toluene_test.positions)

    At this point only the positions are defined

    >>> sampler_state.velocities is None
    True
    >>> sampler_state.total_energy is None
    True

    but it can still be used to set up a context

    >>> temperature = 300.0*unit.kelvin
    >>> thermodynamic_state = ThermodynamicState(toluene_test.system, temperature)
    >>> integrator = openmm.VerletIntegrator(1.0*unit.femtosecond)
    >>> context = thermodynamic_state.create_context(integrator)
    >>> sampler_state.apply_to_context(context)  # Set initial positions.

    A SamplerState cannot be updated by an incompatible context
    which here is defined as having the same number of particles

    >>> hostguest_test = testsystems.HostGuestVacuum()
    >>> incompatible_state = ThermodynamicState(hostguest_test.system, temperature)
    >>> integrator2 = openmm.VerletIntegrator(1.0*unit.femtosecond)
    >>> incompatible_context = incompatible_state.create_context(integrator2)
    >>> incompatible_context.setPositions(hostguest_test.positions)
    >>> sampler_state.is_context_compatible(incompatible_context)
    False
    >>> sampler_state.update_from_context(incompatible_context)
    Traceback (most recent call last):
    ...
    SamplerStateError: Specified positions with inconsistent number of particles.

    Create a new SamplerState instead

    >>> sampler_state2 = SamplerState.from_context(context)
    >>> sampler_state2.potential_energy is not None
    True

    It is possible to slice a sampler state to obtain positions and
    particles of a subset of atoms

    >>> sliced_sampler_state = sampler_state[:10]
    >>> sliced_sampler_state.n_particles
    10

    """

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def __init__(self, positions, velocities=None, box_vectors=None):
        self._initialize(positions, velocities, box_vectors)

    @staticmethod
    def from_context(context):
        """Alternative constructor.

        Read all the configurational properties from a Context object.
        This guarantees that all attributes (including energy attributes)
        are initialized.

        Parameters
        ----------
        context : simtk.openmm.Context
            The context to read.

        Returns
        -------
        sampler_state : SamplerState
            A new SamplerState object.

        """
        sampler_state = SamplerState([])
        sampler_state._read_context_state(context, check_consistency=False)
        return sampler_state

    @property
    def positions(self):
        """Particle positions.

        An Nx3 simtk.unit.Quantity object, where N is the number of
        particles.

        Raises
        ------
        SamplerStateError
            If set to an array with a number of particles different
            than n_particles.

        """
        return self._positions

    @positions.setter
    def positions(self, value):
        if value is None or len(value) != self.n_particles:
            raise SamplerStateError(SamplerStateError.INCONSISTENT_POSITIONS)
        self._positions = value
        self._cached_positions_in_md_units = None  # Invalidate cache.

    @property
    def velocities(self):
        """Particle velocities.

        An Nx3 simtk.unit.Quantity object, where N is the number of
        particles.

        Raises
        ------
        SamplerStateError
            If set to an array with a number of particles different
            than n_particles.

        """
        return self._velocities

    @velocities.setter
    def velocities(self, value):
        if value is not None and self.n_particles != len(value):
            raise SamplerStateError(SamplerStateError.INCONSISTENT_VELOCITIES)
        self._velocities = value
        self._cached_velocities_in_md_units = None  # Invalidate cache.

    @property
    def total_energy(self):
        """The sum of potential and kinetic energy (read-only)."""
        if self.potential_energy is None or self.kinetic_energy is None:
            return None
        return self.potential_energy + self.kinetic_energy

    @property
    def volume(self):
        """The volume of the box (read-only)"""
        return _box_vectors_volume(self.box_vectors)

    @property
    def n_particles(self):
        """Number of particles (read-only)."""
        return len(self.positions)

    def is_context_compatible(self, context):
        """Check compatibility of the given context.

        The context is compatible if this SamplerState can be applied
        through apply_to_context.

        Parameters
        ----------
        context : simtk.openmm.Context
            The context to test.

        Returns
        -------
        is_compatible : bool
            True if this SamplerState can be applied to context.

        See Also
        --------
        SamplerState.apply_to_context

        """
        is_compatible = self.n_particles == context.getSystem().getNumParticles()
        return is_compatible

    def update_from_context(self, context):
        """Read the state from the given context.

        The context must be compatible. Use SamplerState.from_context
        if you want to build a new sampler state from an incompatible.

        Parameters
        ----------
        context : simtk.openmm.Context
            The context to read.

        Raises
        ------
        SamplerStateError
            If the given context is not compatible.

        """
        self._read_context_state(context, check_consistency=True)

    def apply_to_context(self, context):
        """Set the context state.

        If velocities and box vectors have not been specified in the
        constructor, they are not set.

        Parameters
        ----------
        context : simtk.openmm.Context
            The context to set.

        """
        context.setPositions(self._positions_in_md_units)
        if self._velocities is not None:
            context.setVelocities(self._velocities_in_md_units)
        if self.box_vectors is not None:
            context.setPeriodicBoxVectors(*self.box_vectors)

    def has_nan(self):
        """Check that energies and positions are finite.

        Returns
        -------
        True if the potential energy or any of the generalized coordinates
        are nan.

        """
        if self.potential_energy is not None and np.isnan(self.potential_energy):
            return True
        if np.any(np.isnan(self._positions_in_md_units)):
            return True
        return False

    def __getitem__(self, item):
        sampler_state = SamplerState([])
        if isinstance(item, slice):
            sampler_state._positions = self._positions[item]
            if self._velocities is not None:
                sampler_state._velocities = self._velocities[item]
        else:
            pos_value = self._positions[item].value_in_unit(self._positions.unit)
            sampler_state._positions = unit.Quantity(np.array([pos_value]),
                                                     self._positions.unit)
            if self._velocities is not None:
                vel_value = self._velocities[item].value_in_unit(self._velocities.unit)
                sampler_state._velocities = unit.Quantity(np.array([vel_value]),
                                                          self._velocities.unit)
        sampler_state.box_vectors = self.box_vectors
        sampler_state.potential_energy = self.potential_energy
        sampler_state.kinetic_energy = self.kinetic_energy
        return sampler_state

    def __getstate__(self):
        """Return a dictionary representation of the state."""
        serialization = dict(
            positions=self.positions, velocities=self.velocities,
            box_vectors=self.box_vectors, potential_energy=self.potential_energy,
            kinetic_energy=self.kinetic_energy
        )
        return serialization

    def __setstate__(self, serialization):
        """Set the state from a dictionary representation."""
        self._initialize(**serialization)

    # -------------------------------------------------------------------------
    # Internal-usage
    # -------------------------------------------------------------------------

    def _initialize(self, positions, velocities, box_vectors,
                    potential_energy=None, kinetic_energy=None):
        """Initialize the sampler state."""
        self._positions = positions
        self._cached_positions_in_md_units = None
        self._velocities = None
        self._cached_velocities_in_md_units = None
        self.velocities = velocities  # Checks consistency and units.
        self.box_vectors = box_vectors
        self.potential_energy = potential_energy
        self.kinetic_energy = kinetic_energy

    @property
    def _positions_in_md_units(self):
        """Positions in md units system.

        Handles a unitless cache that can reduce the time setting context
        positions by more than half. The cache needs to be invalidated
        when positions are changed.

        """
        if self._cached_positions_in_md_units is None:
            temp_pos = self._positions.value_in_unit_system(unit.md_unit_system)
            self._cached_positions_in_md_units = temp_pos
        return self._cached_positions_in_md_units

    @property
    def _velocities_in_md_units(self):
        """Velocities in md units system.

        Handles a unitless cache that can reduce the time setting context
        velocities by more than half. The cache needs to be invalidated
        when velocities are changed.

        """
        if self._cached_velocities_in_md_units is None:
            temp_vel = self._velocities.value_in_unit_system(unit.md_unit_system)
            self._cached_velocities_in_md_units = temp_vel
        return self._cached_velocities_in_md_units

    def _read_context_state(self, context, check_consistency):
        """Read the Context state.

        Parameters
        ----------
        context : simtk.openmm.Context
            The context to read.
        check_consistency : bool
            If True, raise an error if the context system have a
            different number of particles than the current state.

        Raises
        ------
        SamplerStateError
            If the the context system have a different number of
            particles than the current state.

        """
        openmm_state = context.getState(getPositions=True, getVelocities=True,
                                        getEnergy=True)

        # We assign positions first, since the velocities
        # property will check its length for consistency
        if check_consistency:
            self.positions = openmm_state.getPositions(asNumpy=True)
        else:
            # The positions in md units cache is updated below.
            self._positions = openmm_state.getPositions(asNumpy=True)

        self.velocities = openmm_state.getVelocities(asNumpy=True)
        self.box_vectors = openmm_state.getPeriodicBoxVectors(asNumpy=True)
        self.potential_energy = openmm_state.getPotentialEnergy()
        self.kinetic_energy = openmm_state.getKineticEnergy()

        # We need to set the cached positions and velocities at the end
        # because every call to the properties invalidate the cache. I
        # know this is ugly but it saves us A LOT of time in unit stripping.
        self._cached_positions_in_md_units = openmm_state._coordList
        self._cached_velocities_in_md_units = openmm_state._velList


# =============================================================================
# COMPOUND THERMODYNAMIC STATE
# =============================================================================

class ComposableStateError(Exception):
    """Error raised by a ComposableState."""
    pass


class IComposableState(utils.SubhookedABCMeta):
    """A state composable through CompoundThermodynamicState.

    Define the interface that needs to be implemented to extend a
    ThermodynamicState through CompoundThermodynamicState.

    See Also
    --------
    CompoundThermodynamicState

    """

    @abc.abstractmethod
    def apply_to_system(self, system):
        """Change the system properties to be consistent with this state.

        This method is called on CompoundThermodynamicState init to update
        the system stored in the main ThermodynamicState, and every time
        an attribute/property of the composable state is set or a setter
        method (i.e. a method that starts with 'set_') is called.

        This is the system that will be used during context creation, so
        it is important that it is up-to-date.

        Parameters
        ----------
        system : simtk.openmm.System
            The system to modify.

        Raises
        ------
        CompatibleStateError
            If the system is not compatible with the state.

        """
        pass

    @abc.abstractmethod
    def check_system_consistency(self, system):
        """Check if the system is consistent with the state.

        It raises a ComposableStateError if the system is not consistent
        with the state. This is called when the ThermodynamicState's
        system is set.

        Parameters
        ----------
        system : simtk.openmm.System
            The system to test.

        Raises
        ------
        ComposableStateError
            If the system is not consistent with this state.

        """
        pass

    @abc.abstractmethod
    def apply_to_context(self, context):
        """Apply changes to the context to be consistent with the state.

        Parameters
        ----------
        context : simtk.openmm.Context
            The context to set.

        Raises
        ------
        ComposableStateError
            If the context is not compatible with the state.

        """
        pass

    @classmethod
    @abc.abstractmethod
    def _standardize_system(cls, system):
        """Standardize the given system.

        ThermodynamicState relies on this method to create a standard
        system that defines compatibility with another state or context.
        The definition of a standard system is tied to the implementation
        of apply_to_context. For example, if apply_to_context sets a
        global parameter of the context, _standardize_system should
        set the default value of the parameter in the system to a
        standard value.

        Parameters
        ----------
        system : simtk.openmm.System
            The system to standardize.

        Raises
        ------
        CompatibleStateError
            If the system is not compatible with the state.

        """
        pass


class CompoundThermodynamicState(ThermodynamicState):
    """Thermodynamic state composed by multiple states.

    Allows to extend a ThermodynamicState through composition rather
    than inheritance.

    The class dynamically inherits from the ThermodynamicState object
    given in the constructor, and it preserves direct access to all its
    methods and attributes. It is compatible also with subclasses of
    ThermodynamicState, but it does not support objects which make use
    of __slots__.

    It is the user's responsibility to check that IComposableStates are
    compatible to each other (i.e. that they do not depend on and/or
    modify the same properties of the system). If this is not the case,
    consider merging them into a single IComposableStates. If an
    IComposableState needs to access properties of ThermodynamicState
    (e.g. temperature, pressure) consider extending it through normal
    inheritance.

    It is not necessary to explicitly inherit from IComposableState for
    compatibility as long as all abstract methods are implemented. All
    its attributes and methods will still be directly accessible unless
    they are masked by the main ThermodynamicState or by a IComposableState
    that appeared before in the constructor argument composable_states.

    After construction, changing the original thermodynamic_state or
    any of the composable_states changes the state of the compound state.

    Parameters
    ----------
    thermodynamic_state : ThermodynamicState
        The main ThermodynamicState which holds the OpenMM system.
    composable_states : list of IComposableState
        Each element represent a portion of the overall thermodynamic
        state.

    Examples
    --------
    Create an alchemically modified system.

    >>> from openmmtools import testsystems, alchemy
    >>> factory = alchemy.AlchemicalFactory(consistent_exceptions=False)
    >>> alanine_vacuum = testsystems.AlanineDipeptideVacuum().system
    >>> alchemical_region = alchemy.AlchemicalRegion(alchemical_atoms=range(22))
    >>> alanine_alchemical_system = factory.create_alchemical_system(reference_system=alanine_vacuum,
    ...                                                              alchemical_regions=alchemical_region)
    >>> alchemical_state = alchemy.AlchemicalState.from_system(alanine_alchemical_system)

    AlchemicalState implement the IComposableState interface, so it can be
    used with CompoundThermodynamicState. All the alchemical parameters are
    accessible through the compound state.

    >>> from simtk import openmm, unit
    >>> thermodynamic_state = ThermodynamicState(system=alanine_alchemical_system,
    ...                                                 temperature=300*unit.kelvin)
    >>> compound_state = CompoundThermodynamicState(thermodynamic_state=thermodynamic_state,
    ...                                                    composable_states=[alchemical_state])
    >>> compound_state.lambda_sterics
    1.0
    >>> compound_state.lambda_electrostatics
    1.0

    You can control the parameters in the OpenMM Context in this state by
    setting the state attributes.

    >>> compound_state.lambda_sterics = 0.5
    >>> integrator = openmm.VerletIntegrator(1.0*unit.femtosecond)
    >>> context = compound_state.create_context(integrator)
    >>> context.getParameter('lambda_sterics')
    0.5
    >>> compound_state.lambda_sterics = 1.0
    >>> compound_state.apply_to_context(context)
    >>> context.getParameter('lambda_sterics')
    1.0

    """
    def __init__(self, thermodynamic_state, composable_states):
        self._initialize(thermodynamic_state, composable_states)

    def set_system(self, system, fix_state=False):
        """Allow to set the system and fix its thermodynamic state.

        With default arguments, this is equivalent to assign the
        system property, which raise an error if the system is in
        a different thermodynamic state.

        Parameters
        ----------
        system : simtk.openmm.System
            The system to set.
        fix_state : bool, optional
            The thermodynamic state of the state will be fixed by
            all the composable states. Default is False.

        See Also
        --------
        ThermodynamicState.set_system

        """
        for s in self._composable_states:
            if fix_state is True:
                s.apply_to_system(system)
            else:
                s.check_system_consistency(system)
        super(CompoundThermodynamicState, self).set_system(system, fix_state)

    def is_context_compatible(self, context):
        """Check compatibility of the given context.

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
        ThermodynamicState.is_context_compatible

        """
        # We override ThermodynamicState.is_context_compatible to
        # handle the case in which one of the composable states
        # raises ComposableStateError when standardizing the context system.
        try:
            return super(CompoundThermodynamicState, self).is_context_compatible(context)
        except ComposableStateError:
            return False

    def apply_to_context(self, context):
        """Apply this compound thermodynamic state to the context.

        See Also
        --------
        ThermodynamicState.apply_to_context

        """
        super(CompoundThermodynamicState, self).apply_to_context(context)
        for s in self._composable_states:
            s.apply_to_context(context)

    def __getattr__(self, name):
        def setter_decorator(func, composable_state):
            def _setter_decorator(*args, **kwargs):
                func(*args, **kwargs)
                composable_state.apply_to_system(self._system)
            return _setter_decorator

        # Called only if the attribute couldn't be found in __dict__.
        # In this case we fall back to composable state, in the given order.
        for s in self._composable_states:
            try:
                attr = getattr(s, name)
            except AttributeError:
                pass
            else:
                if name.startswith('set_'):
                    # Decorate the setter so that apply_to_system is called
                    # after the attribute is modified.
                    attr = setter_decorator(attr, s)
                return attr

        # Attribute not found, fall back to normal behavior.
        return super(CompoundThermodynamicState, self).__getattribute__(name)

    def __setattr__(self, name, value):
        # Add new attribute to CompoundThermodynamicState.
        if '_composable_states' not in self.__dict__:
            super(CompoundThermodynamicState, self).__setattr__(name, value)

        # Update existing ThermodynamicState attribute (check ancestors).
        elif any(name in C.__dict__ for C in self.__class__.__mro__):
            super(CompoundThermodynamicState, self).__setattr__(name, value)

        # Update composable states attributes (check ancestors).
        else:
            for s in self._composable_states:
                if any(name in C.__dict__ for C in s.__class__.__mro__):
                    s.__setattr__(name, value)
                    s.apply_to_system(self._system)
                    return

            # No attribute found. This is monkey patching.
            super(CompoundThermodynamicState, self).__setattr__(name, value)

    def __getstate__(self):
        """Return a dictionary representation of the state."""
        # Create original ThermodynamicState to serialize
        thermodynamic_state = object.__new__(self.__class__.__bases__[1])
        thermodynamic_state.__dict__ = copy.deepcopy(self.__dict__)
        serialization = dict(
            thermodynamic_state=utils.serialize(thermodynamic_state),
            composable_states=[utils.serialize(composable_state)
                               for composable_state in self._composable_states]
        )
        return serialization

    def __setstate__(self, serialization):
        """Set the state from a dictionary representation."""
        thermodynamic_state = utils.deserialize(serialization['thermodynamic_state'])
        composable_states = [utils.deserialize(composable_state)
                             for composable_state in serialization['composable_states']]
        self._initialize(thermodynamic_state, composable_states)

    # -------------------------------------------------------------------------
    # Internal-usage
    # -------------------------------------------------------------------------

    def _initialize(self, thermodynamic_state, composable_states):
        """Initialize the sampler state."""
        # Check that composable states expose the correct interface.
        for composable_state in composable_states:
            assert isinstance(composable_state, IComposableState)

        # Dynamically inherit from thermodynamic_state class and
        # store the types of composable_states to be able to call
        # class methods.
        composable_bases = [s.__class__ for s in composable_states]
        self.__class__ = type(self.__class__.__name__,
                              (self.__class__, thermodynamic_state.__class__),
                              {'_composable_bases': composable_bases})
        self.__dict__ = thermodynamic_state.__dict__

        # Set the stored system to the given states. Setting
        # self._composable_states signals __setattr__ to start
        # searching in composable states as well, so this must
        # be the last new attribute set in the constructor.
        self._composable_states = composable_states
        for s in self._composable_states:
            s.apply_to_system(self._system)

    @classmethod
    def _standardize_system(cls, system):
        """Standardize the system.

        Override ThermodynamicState._standardize_system to standardize
        the system also with respect to all other composable states.

        Raises
        ------
        ComposableStateError
            If it is impossible to standardize the system.

        See Also
        --------
        ThermodynamicState._standardize_system

        """
        super(CompoundThermodynamicState, cls)._standardize_system(system)
        for composable_cls in cls._composable_bases:
            composable_cls._standardize_system(system)


if __name__ == '__main__':
    import doctest
    # doctest.testmod()
    doctest.run_docstring_examples(CompoundThermodynamicState, globals())
