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
import sys
import copy
import zlib
import inspect
import weakref
import collections

import numpy as np
from simtk import openmm, unit

from openmmtools import utils, integrators, forces, constants


# =============================================================================
# MODULE FUNCTIONS
# =============================================================================

def create_thermodynamic_state_protocol(system, protocol, constants=None,
                                        composable_states=None):
    """An optimized utility function to create a list of thermodynamic states.

    The method takes advantage of the fact that copying a thermodynamic state
    does not require a copy of the OpenMM ``System`` object and that setting
    parameters that are controlled by the ``(Compound)ThermodynamicState``
    is effectively instantaneous.

    Parameters
    ----------
    reference_state : ThermodynamicState or simtk.openmm.System
        ``ThermodynamicState`` or The OpenMM ``System``. If a ``System`` the
        constants must specify the temperature.
    protocol : dict: str -> list
        A dictionary associating the thermodynamic parameters to a list of
        values. All the lists must have the same length.
    constants : dict: str -> list
        A dictionary associating a thermodnamic parameter to a value that
        must remain constant along the protocol.
    composable_states : IComposableState or list, optional
        If specified, the function returns a list of ``CompoundThermodynamicState``
        instead of simple ``ThermodynamicState`` objects.

    Returns
    -------
    states : list of ``ThermodynamicState`` or ``CompoundThermodynamicState``
        The sequence of thermodynamic states for the given protocol.

    Examples
    --------

    >>> from simtk import unit
    >>> from openmmtools import testsystems
    >>> system = testsystems.AlanineDipeptideExplicit().system
    >>> protocol = {'temperature': [300, 310, 330]*unit.kelvin,
    ...             'pressure': [1.0, 1.1, 1.2]*unit.atmosphere}
    >>> states = create_thermodynamic_state_protocol(system, protocol)
    >>> len(states)
    3

    """
    # Check that all elements of the protocol have the same length.
    if len(protocol) == 0:
        raise ValueError('No protocol has been specified.')
    values_lengths = [len(values) for values in protocol.values()]
    if len(set(values_lengths)) != 1:
        raise ValueError('The protocol parameter values have different '
                         'lengths!\n{}'.format(protocol))
    protocol_length = values_lengths[0]

    # Handle default value.
    if constants is None:
        constants = {}

    # Check that the user didn't specify the same parameter as both
    # a constant and a protocol variable.
    if len(set(constants).intersection(set(protocol))) != 0:
        raise ValueError('Some parameters have been specified both '
                         'in constants and protocol.')

    # Augument protocol to include the constants values as well.
    for constant_parameter, value in constants.items():
        protocol[constant_parameter] = [value for _ in range(protocol_length)]

    # Create the reference ThermodynamicState.
    if isinstance(system, openmm.System):
        # Make sure the temperature is defined somewhere.
        try:
            temperature = constants['temperature']
        except KeyError:
            try:
                temperature = protocol['temperature'][0]
            except KeyError:
                raise ValueError('If a System is passed the list of '
                                 'constants must specify the temperature.')
        thermo_state = ThermodynamicState(system, temperature=temperature)
    else:
        thermo_state = system

    # Check if we need to create a reference CompoundThermodynamicState.
    # Cast a single ComposableState into a list.
    if isinstance(composable_states, IComposableState):
        composable_states = [composable_states]
    if composable_states is not None:
        thermo_state = CompoundThermodynamicState(thermo_state, composable_states)

    # Create all the states. Copying a state is much faster than
    # initializing one because we don't have to copy System object.
    states = [copy.deepcopy(thermo_state) for _ in range(protocol_length)]

    # Assign protocol parameters.
    protocol_keys, protocol_values = zip(*protocol.items())
    for state_idx, state_values in enumerate(zip(*protocol_values)):
        state = states[state_idx]
        for lambda_key, lambda_value in zip(protocol_keys, state_values):
            if hasattr(state, lambda_key):
                setattr(state, lambda_key, lambda_value)
            else:
                raise AttributeError('{} object does not have protocol attribute '
                                     '{}'.format(type(state), lambda_key))

    return states


def reduced_potential_at_states(sampler_state, thermodynamic_states, context_cache):
    """Compute the reduced potential of a single configuration at multiple thermodynamic states.

    Parameters
    ----------
    sampler_state : SamplerState
        The state holding the coordinates used to compute the potential.
    thermodynamic_states : list of ``ThermodynamicState``
        The list of thermodynamic states at which to compute the potential.
    context_cache : cache.ContextCache
        The context cache to use to request ``Context`` objects.

    Returns
    -------
    reduced_potentials : np.ndarray of float
        ``reduced_potentials[i]`` is the unit-less reduced potentials
        (i.e., in kT units) of state ``thermodynamic_states[i]``.

    """
    reduced_potentials = np.zeros(len(thermodynamic_states))

    # Group thermodynamic states by compatibility.
    compatible_groups, original_indices = group_by_compatibility(thermodynamic_states)

    # Compute the reduced potentials of all the compatible states.
    for compatible_group, state_indices in zip(compatible_groups, original_indices):
        # Get the context, any Integrator works.
        context, integrator = context_cache.get_context(compatible_group[0])

        # Update positions and box vectors. We don't need
        # to set Context velocities for the potential.
        sampler_state.apply_to_context(context, ignore_velocities=True)

        # Compute and update the reduced potentials.
        compatible_energies = ThermodynamicState.reduced_potential_at_states(
            context, compatible_group)
        for energy_idx, state_idx in enumerate(state_indices):
            reduced_potentials[state_idx] = compatible_energies[energy_idx]


def group_by_compatibility(thermodynamic_states):
    """Utility function to split the thermodynamic states by compatibility.

    Parameters
    ----------
    thermodynamic_states : list of ThermodynamicState
        The thermodynamic state to group by compatibility.

    Returns
    -------
    compatible_groups : list of list of ThermodynamicState
        The states grouped by compatibility.
    original_indices: list of list of int
        The indices of the ThermodynamicStates in theoriginal list.

    """
    compatible_groups = []
    original_indices = []
    for state_idx, state in enumerate(thermodynamic_states):
        # Search for compatible group.
        found_compatible = False
        for group, indices in zip(compatible_groups, original_indices):
            if state.is_state_compatible(group[0]):
                found_compatible = True
                group.append(state)
                indices.append(state_idx)

        # Create new one.
        if not found_compatible:
            compatible_groups.append([state])
            original_indices.append([state_idx])
    return compatible_groups, original_indices


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
    openmmtools.states.ThermodynamicsError: System has multiple barostats.

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
    openmmtools.states.SamplerStateError: Velocities have different length than positions.

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
    Quantity(value=1.0, unit=atmosphere)
    >>> state.volume is None
    True

    You cannot set a non-periodic system at constant pressure

    >>> nonperiodic_system = testsystems.TolueneVacuum().system
    >>> state = ThermodynamicState(nonperiodic_system, temperature=300*unit.kelvin,
    ...                            pressure=1.0*unit.atmosphere)
    Traceback (most recent call last):
    ...
    openmmtools.states.ThermodynamicsError: Non-periodic systems cannot have a barostat.

    When temperature and/or pressure are not specified (i.e. they are
    None) ThermodynamicState tries to infer them from a thermostat or
    a barostat.

    >>> state = ThermodynamicState(system=waterbox)
    Traceback (most recent call last):
    ...
    openmmtools.states.ThermodynamicsError: System does not have a thermostat specifying the temperature.
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
        a Thermostat force. You can use `get_system()` to obtain a
        copy of the system without the thermostat. The method
        `create_context()` then takes care of removing the thermostat
        when an integrator with a coupled heat bath is used (e.g.
        `LangevinIntegrator`).

        It can be set only to a system which is consistent with the
        current thermodynamic state. Use `set_system()` if you want to
        correct the thermodynamic state of the system automatically
        before assignment.

        See Also
        --------
        ThermodynamicState.get_system
        ThermodynamicState.set_system
        ThermodynamicState.create_context

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
        openmmtools.states.ThermodynamicsError: System does not have a thermostat specifying the temperature.

        We can fix both thermostat and barostat while setting the system.
        >>> state.set_system(alanine.system, fix_state=True)

        """
        # Copy the system to avoid modifications during standardization.
        system = copy.deepcopy(system)
        self._unsafe_set_system(system, fix_state)

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
        system = copy.deepcopy(self._standard_system)

        # Remove or configure standard pressure barostat.
        if remove_barostat:
            self._pop_barostat(system)
        else:  # Set pressure of standard barostat.
            self._set_system_pressure(system, self.pressure)

        # Set temperature of standard thermostat and barostat.
        if not (remove_barostat and remove_thermostat):
            self._set_system_temperature(system, self.temperature)

        # Remove or configure standard temperature thermostat.
        if remove_thermostat:
            self._remove_thermostat(system)

        return system

    @property
    def temperature(self):
        """Constant temperature of the thermodynamic state."""
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        if value is None:
            raise ThermodynamicsError(ThermodynamicsError.NONE_TEMPERATURE)
        self._temperature = value

    @property
    def kT(self):
        """Thermal energy per mole."""
        return constants.kB * self.temperature

    @property
    def beta(self):
        """Thermodynamic beta in units of mole/energy."""
        return 1.0 / self.kT

    @property
    def pressure(self):
        """Constant pressure of the thermodynamic state.

        If the pressure is allowed to fluctuate, this is None. Setting
        this will automatically add/configure a barostat to the system.
        If it is set to None, the barostat will be removed.

        """
        return self._pressure

    @pressure.setter
    def pressure(self, new_pressure):
        old_pressure = self._pressure
        self._pressure = new_pressure

        # If we change ensemble, we need to modify the standard system.
        if (new_pressure is None) != (old_pressure is None):
            # The barostat will be removed/added since fix_state is True.
            try:
                self.set_system(self._standard_system, fix_state=True)
            except ThermodynamicsError:
                # Restore old pressure to keep object consistent.
                self._pressure = old_pressure
                raise

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
        # Retrieve the barostat with standard temperature/pressure, then
        # set temperature and pressure to the thermodynamic state values.
        barostat = copy.deepcopy(self._find_barostat(self._standard_system))
        if barostat is not None:  # NPT ensemble.
            self._set_barostat_pressure(barostat, self.pressure)
            self._set_barostat_temperature(barostat, self.temperature)
        return barostat

    @barostat.setter
    def barostat(self, new_barostat):
        # If None, just remove the barostat from the standard system.
        if new_barostat is None:
            self.pressure = None
            return

        # Remember old pressure in case something goes wrong.
        old_pressure = self.pressure

        # Build the system with the new barostat.
        system = self.get_system(remove_barostat=True)
        system.addForce(copy.deepcopy(new_barostat))

        # Update the internally stored standard system, and restore the old
        # pressure if something goes wrong (e.g. the system is not periodic).
        try:
            self._pressure = new_barostat.getDefaultPressure()
            self._unsafe_set_system(system, fix_state=False)
        except ThermodynamicsError:
            self._pressure = old_pressure
            raise

    @property
    def default_box_vectors(self):
        """The default box vectors of the System (read-only)."""
        return self._standard_system.getDefaultPeriodicBoxVectors()

    @property
    def volume(self):
        """Constant volume of the thermodynamic state (read-only).

        If the volume is allowed to fluctuate, or if the system is
        not in a periodic box this is None.

        """
        return self.get_volume()

    def get_volume(self, ignore_ensemble=False):
        """Volume of the periodic box (read-only).

        Parameters
        ----------
        ignore_ensemble : bool, optional
            If True, the volume of the periodic box vectors is returned
            even if the volume fluctuates.

        Returns
        -------
        volume : simtk.unit.Quantity
            The volume of the periodic box (units of length^3) or
            None if the system is not periodic or allowed to fluctuate.

        """
        # Check if volume fluctuates
        if self.pressure is not None and not ignore_ensemble:
            return None
        if not self._standard_system.usesPeriodicBoundaryConditions():
            return None
        return _box_vectors_volume(self.default_box_vectors)

    @property
    def n_particles(self):
        """Number of particles (read-only)."""
        return self._standard_system.getNumParticles()

    @property
    def is_periodic(self):
        """True if the system is in a periodic box (read-only)."""
        return self._standard_system.usesPeriodicBoundaryConditions()

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
        openmmtools.states.ThermodynamicsError: The sampler state has a different number of particles.

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

        return self._compute_reduced_potential(potential_energy, self.temperature,
                                               volume, self.pressure)

    @classmethod
    def reduced_potential_at_states(cls, context, thermodynamic_states):
        """Efficiently compute the reduced potential for a list of compatible states.

        The user is responsible to ensure that the given context is compatible
        with the thermodynamic states.

        Parameters
        ----------
        context : openmm.Context
            The OpenMM `Context` object with box vectors and positions set.
        thermodynamic_states : list of ThermodynamicState
            The list of thermodynamic states at which to compute the reduced
            potential.

        Returns
        -------
        reduced_potentials : list of float
            The unit-less reduced potentials, which can be considered
            to have units of kT.

        Raises
        ------
        ValueError
            If the thermodynamic states are not compatible to each other.

        """
        # Isolate first thermodynamic state.
        if len(thermodynamic_states) == 1:
            thermodynamic_states[0].apply_to_context(context)
            return [thermodynamic_states[0].reduced_potential(context)]

        # Check that the states are compatible.
        for state_idx, state in enumerate(thermodynamic_states[:-1]):
            if not state.is_state_compatible(thermodynamic_states[state_idx + 1]):
                raise ValueError('State {} is not compatible.')

        # In NPT, we'll need also the volume.
        is_npt = thermodynamic_states[0].pressure is not None
        volume = None

        energy_by_force_group = {force.getForceGroup(): 0.0*unit.kilocalories_per_mole
                                 for force in context.getSystem().getForces()}

        # Create new cache for memoization.
        memo = {}

        # Go through thermodynamic states and compute only the energy of the
        # force groups that changed. Compute all the groups the first pass.
        force_groups_to_compute = set(energy_by_force_group)
        reduced_potentials = [0.0 for _ in range(len(thermodynamic_states))]
        for state_idx, state in enumerate(thermodynamic_states):
            if state_idx == 0:
                state.apply_to_context(context)
            else:
                state._apply_to_context_in_state(context, thermodynamic_states[state_idx - 1])

            # Compute the energy of all the groups to update.
            for force_group_idx in force_groups_to_compute:
                openmm_state = context.getState(getEnergy=True, groups=2**force_group_idx)
                energy_by_force_group[force_group_idx] = openmm_state.getPotentialEnergy()

            # Compute volume if this is the first time we obtain a state.
            if is_npt and volume is None:
                volume = openmm_state.getPeriodicBoxVolume()

            # Compute the new total reduced potential.
            potential_energy = unit.sum(list(energy_by_force_group.values()))
            reduced_potential = cls._compute_reduced_potential(potential_energy, state.temperature,
                                                               volume, state.pressure)
            reduced_potentials[state_idx] = reduced_potential

            # Update groups to compute for next states.
            if state_idx < len(thermodynamic_states) - 1:
                next_state = thermodynamic_states[state_idx + 1]
                force_groups_to_compute = next_state._find_force_groups_to_update(context, state, memo)

        return reduced_potentials

    def is_state_compatible(self, thermodynamic_state):
        """Check compatibility between ThermodynamicStates.

        The state is compatible if Contexts created by thermodynamic_state
        can be set to this ThermodynamicState through apply_to_context.
        The property is symmetric and transitive.

        This is faster than checking compatibility of a Context object
        through is_context_compatible, and it should be preferred when
        possible.

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
        # Avoid modifying the context system during standardization.
        context_system = copy.deepcopy(context.getSystem())
        context_integrator = context.getIntegrator()

        # If the temperature is controlled by the integrator, the compatibility
        # is independent on the parameters of the thermostat, so we add one
        # identical to self._standard_system. We don't care if the integrator's
        # temperature != self.temperature, so we set check_consistency=False.
        if self._is_integrator_thermostated(context_integrator, check_consistency=False):
            thermostat = self._find_thermostat(self._standard_system)
            context_system.addForce(copy.deepcopy(thermostat))

        # Compute and compare standard system hash.
        self._standardize_system(context_system)
        context_system_hash = self._compute_standard_system_hash(context_system)
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
        When passing an integrator that does not expose getter and setter
        for the temperature, the context will be created with a thermostat.

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

        >>> del context  # Delete previous context to free memory.
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

        # Get a copy of the system. If integrator is coupled
        # to heat bath, remove the system thermostat.
        system = self.get_system(remove_thermostat=is_thermostated)

        # Create context.
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
        self._set_context_barostat(context, update_pressure=True, update_temperature=True)
        self._set_context_thermostat(context)

    # -------------------------------------------------------------------------
    # Magic methods
    # -------------------------------------------------------------------------

    def __copy__(self):
        """Overwrite normal implementation to share standard system."""
        cls = self.__class__
        new_state = cls.__new__(cls)
        new_state.__dict__.update({k: v for k, v in self.__dict__.items()
                                   if k != '_standard_system'})
        new_state.__dict__['_standard_system'] = self._standard_system
        return new_state

    def __deepcopy__(self, memo):
        """Overwrite normal implementation to share standard system."""
        cls = self.__class__
        new_state = cls.__new__(cls)
        memo[id(self)] = new_state
        for k, v in self.__dict__.items():
            if k != '_standard_system':
                new_state.__dict__[k] = copy.deepcopy(v, memo)
        new_state.__dict__['_standard_system'] = self._standard_system
        return new_state

    _ENCODING = 'utf-8'

    def __getstate__(self, skip_system=False):
        """Return a dictionary representation of the state.

        Zlib compresses the serialized system after its created. Many
        alchemical systems have very long serializations so this method
        helps reduce space in memory and on disk. The compression forces
        the encoding for compatibility between separate Python installs
        (utf-8 by default).

        Parameters
        ----------
        skip_system: bool, Default: False
            Choose whether or not to get the serialized system as the part
            of the return. If False, then the serialized system is computed
            and included in the serialization. If True, then ``None`` is
            returned for the ``'standard_system'`` field of the serialization.

        """
        serialized_system = None
        if not skip_system:
            serialized_system = openmm.XmlSerializer.serialize(self._standard_system)
            serialized_system = zlib.compress(serialized_system.encode(self._ENCODING))
        return dict(standard_system=serialized_system, temperature=self.temperature,
                    pressure=self.pressure)

    def __setstate__(self, serialization):
        """Set the state from a dictionary representation."""
        self._temperature = serialization['temperature']
        self._pressure = serialization['pressure']

        serialized_system = serialization['standard_system']
        # Decompress system, if need be
        try:
            serialized_system = zlib.decompress(serialized_system)
            # Py2 returns the string, Py3 returns a byte string to decode, but if we
            # decode the string in Py2 we get a unicode object that OpenMM can't parse.
            if sys.version_info > (3, 0):
                serialized_system = serialized_system.decode(self._ENCODING)
        except (TypeError, zlib.error):  # Py3/2 throws different error types
            # Catch the "serialization is not compressed" error, do nothing to string.
            # Preserves backwards compatibility
            pass

        self._standard_system_hash = serialized_system.__hash__()

        # Check first if we have already the system in the cache.
        try:
            self._standard_system = self._standard_system_cache[self._standard_system_hash]
        except KeyError:
            system = openmm.XmlSerializer.deserialize(serialized_system)
            self._standard_system_cache[self._standard_system_hash] = system
            self._standard_system = system

    # -------------------------------------------------------------------------
    # Internal-usage: initialization
    # -------------------------------------------------------------------------

    def _initialize(self, system, temperature=None, pressure=None):
        """Initialize the thermodynamic state."""
        # Avoid modifying the original system when setting temperature and pressure.
        system = copy.deepcopy(system)

        # If pressure is None, we try to infer the pressure from the barostat.
        barostat = self._find_barostat(system)
        if pressure is None and barostat is not None:
            self._pressure = barostat.getDefaultPressure()
        else:
            self._pressure = pressure  # Pressure here can also be None.

        # If temperature is None, we infer the temperature from a thermostat.
        if temperature is None:
            thermostat = self._find_thermostat(system)
            if thermostat is None:
                raise ThermodynamicsError(ThermodynamicsError.NO_THERMOSTAT)
            self._temperature = thermostat.getDefaultTemperature()
        else:
            self._temperature = temperature

        # Fix system temperature/pressure if requested.
        if temperature is not None:
            self._set_system_temperature(system, temperature)
        if pressure is not None:
            self._set_system_pressure(system, pressure)

        # We can use the unsafe set_system since the system has been copied.
        self._unsafe_set_system(system, fix_state=False)

    # -------------------------------------------------------------------------
    # Internal-usage: system handling
    # -------------------------------------------------------------------------

    # Standard values are not standard in a physical sense, they are
    # just consistent between ThermodynamicStates to make comparison
    # of standard system hashes possible. We set this to round floats
    # and use OpenMM units to avoid funniness due to precision errors
    # caused by unit conversion.
    _STANDARD_PRESSURE = 1.0*unit.bar
    _STANDARD_TEMPERATURE = 273.0*unit.kelvin

    _NONPERIODIC_NONBONDED_METHODS = {openmm.NonbondedForce.NoCutoff,
                                      openmm.NonbondedForce.CutoffNonPeriodic}

    # Shared cache of standard systems to minimize memory consumption
    # when simulating a lot of thermodynamic states. The cache holds
    # only weak references so ThermodynamicState objects must keep the
    # system as an internal variable.
    _standard_system_cache = weakref.WeakValueDictionary()

    def _unsafe_set_system(self, system, fix_state):
        """This implements self.set_system but modifies the passed system."""
        # Configure temperature and pressure.
        if fix_state:
            # We just need to add/remove the barostat according to the ensemble.
            # Temperature and pressure of thermostat and barostat will be set
            # to their standard value afterwards.
            self._set_system_pressure(system, self.pressure)
        else:
            # If the flag is deactivated, we check that temperature
            # and pressure of the system are correct.
            self._check_system_consistency(system)

        # Update standard system.
        self._standardize_system(system)
        self._update_standard_system(system)

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

        # This line raises MULTIPLE_BAROSTATS and UNSUPPORTED_BAROSTAT.
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
        elif self.pressure is not None:
            raise TE(TE.NO_BAROSTAT)

    def _standardize_system(self, system):
        """Return a copy of the system in a standard representation.

        This effectively defines which ThermodynamicStates are compatible
        between each other. Compatible ThermodynamicStates have the same
        standard systems, and is_state_compatible will return True if
        the (cached) serialization of the standard systems are identical.

        If no thermostat is present, an AndersenThermostat is added. The
        presence of absence of a barostat determine whether this system is
        in NPT or NVT ensemble. Pressure and temperature of barostat (if
        any) and thermostat are set to _STANDARD_PRESSURE/TEMPERATURE.
        If present, the barostat force is pushed at the end so that the
        order of the two forces won't matter.

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
        # This adds a thermostat if it doesn't exist already. This way
        # the comparison between system using thermostat with different
        # parameters (e.g. collision frequency) will fail as expected.
        self._set_system_temperature(system, self._STANDARD_TEMPERATURE)

        # We need to be sure that thermostat and barostat always are
        # in the same order, as the hash depends on the Forces order.
        # Here we push the barostat at the end.
        barostat = self._pop_barostat(system)
        if barostat is not None:
            barostat.setDefaultPressure(self._STANDARD_PRESSURE)
            system.addForce(barostat)

    def _compute_standard_system_hash(self, standard_system):
        """Compute the standard system hash."""
        system_serialization = openmm.XmlSerializer.serialize(standard_system)
        return system_serialization.__hash__()

    def _update_standard_system(self, standard_system):
        """Update the standard system, its hash and the standard system cache."""
        self._standard_system_hash = self._compute_standard_system_hash(standard_system)
        try:
            self._standard_system = self._standard_system_cache[self._standard_system_hash]
        except KeyError:
            self._standard_system_cache[self._standard_system_hash] = standard_system
            self._standard_system = standard_system

    # -------------------------------------------------------------------------
    # Internal-usage: context handling
    # -------------------------------------------------------------------------

    def _set_context_barostat(self, context, update_pressure, update_temperature):
        """Set the barostat parameters in the Context."""
        barostat = self._find_barostat(context.getSystem())

        # Check if we are in the same ensemble.
        if (barostat is None) != (self._pressure is None):
            raise ThermodynamicsError(ThermodynamicsError.INCOMPATIBLE_ENSEMBLE)

        # No need to set the barostat if we are in NVT.
        if self._pressure is None:
            return

        # Apply pressure and temperature to barostat.
        if update_pressure:
            self._set_barostat_pressure(barostat, self.pressure)
            context.setParameter(barostat.Pressure(), self.pressure)
        if update_temperature:
            self._set_barostat_temperature(barostat, self.temperature)
            # TODO remove try except when drop openmm7.0 support
            try:
                context.setParameter(barostat.Temperature(), self.temperature)
            except AttributeError:  # OpenMM < 7.1
                openmm_state = context.getState(getPositions=True, getVelocities=True,
                                                getParameters=True)
                context.reinitialize()
                context.setState(openmm_state)

    def _set_context_thermostat(self, context):
        """Set the thermostat parameters in the Context."""
        # First try to set the integrator (most common case).
        # If this fails retrieve the Andersen thermostat.
        is_thermostated = self._set_integrator_temperature(context.getIntegrator())
        if not is_thermostated:
            thermostat = self._find_thermostat(context.getSystem())
            thermostat.setDefaultTemperature(self.temperature)
            context.setParameter(thermostat.Temperature(), self.temperature)

    def _apply_to_context_in_state(self, context, thermodynamic_state):
        """Apply this ThermodynamicState to the context.

        When we know the thermodynamic state of the context, this is much faster
        then apply_to_context(). The given thermodynamic state is assumed to be
        compatible.

        Parameters
        ----------
        context : simtk.openmm.Context
            The OpenMM Context to be set to this ThermodynamicState.
        thermodynamic_state : ThermodynamicState
            The ThermodynamicState of this context.

        """
        update_pressure = self.pressure != thermodynamic_state.pressure
        update_temperature = self.temperature != thermodynamic_state.temperature

        if update_pressure or update_temperature:
            self._set_context_barostat(context, update_pressure, update_temperature)
        if update_temperature:
            self._set_context_thermostat(context)

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

    def _is_integrator_thermostated(self, integrator, check_consistency=True):
        """True if integrator is coupled to a heat bath.

        If integrator is a CompoundIntegrator, it returns true if at least
        one of its integrators is coupled to a heat bath.

        Raises
        ------
        ThermodynamicsError
            If check_consistency is True and the integrator is
            coupled to a heat bath at a different temperature
            than this thermodynamic state.

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
                if (check_consistency and
                        not utils.is_quantity_close(temperature, self.temperature)):
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
        is_thermostated : bool
            True if the integrator is thermostated.

        """
        def set_temp(_integrator):
            try:
                _integrator.setTemperature(self.temperature)
                return True
            except AttributeError:
                return False

        # Loop over integrators to handle CompoundIntegrators.
        is_thermostated = False
        for _integrator in self._loop_over_integrators(integrator):
            is_thermostated = is_thermostated or set_temp(_integrator)
        return is_thermostated

    # -------------------------------------------------------------------------
    # Internal-usage: barostat handling
    # -------------------------------------------------------------------------

    _SUPPORTED_BAROSTATS = {'MonteCarloBarostat'}

    @classmethod
    def _find_barostat(cls, system, get_index=False):
        """Return the first barostat found in the system.

        Returns
        -------
        force_idx : int or None, optional
            The force index of the barostat.
        barostat : OpenMM Force object
            The barostat in system, or None if no barostat is found.

        Raises
        ------
        ThermodynamicsError
            If the system contains unsupported barostats.

        """
        try:
            force_idx, barostat = forces.find_forces(system, '.*Barostat.*', only_one=True)
        except forces.MultipleForcesError:
            raise ThermodynamicsError(ThermodynamicsError.MULTIPLE_BAROSTATS)
        except forces.NoForceFoundError:
            force_idx, barostat = None, None
        else:
            if barostat.__class__.__name__ not in cls._SUPPORTED_BAROSTATS:
                raise ThermodynamicsError(ThermodynamicsError.UNSUPPORTED_BAROSTAT,
                                          barostat.__class__.__name__)
        if get_index:
            return force_idx, barostat
        return barostat

    @classmethod
    def _pop_barostat(cls, system):
        """Remove the system barostat.

        Returns
        -------
        The removed barostat if it was found, None otherwise.

        """
        barostat_idx, barostat = cls._find_barostat(system, get_index=True)
        if barostat_idx is not None:
            # We need to copy the barostat since we don't own
            # its memory (i.e. we can't add it back to the system).
            barostat = copy.deepcopy(barostat)
            system.removeForce(barostat_idx)
            return barostat
        return None

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

        Raises
        ------
        ThermodynamicsError
            If pressure needs to be set for a non-periodic system.

        """
        if pressure is None:  # If new pressure is None, remove barostat.
            self._pop_barostat(system)
            return

        if not system.usesPeriodicBoundaryConditions():
            raise ThermodynamicsError(ThermodynamicsError.BAROSTATED_NONPERIODIC)

        barostat = self._find_barostat(system)
        if barostat is None:  # Add barostat
            barostat = openmm.MonteCarloBarostat(pressure, self.temperature)
            system.addForce(barostat)
        else:  # Set existing barostat
            self._set_barostat_pressure(barostat, pressure)

    @staticmethod
    def _set_barostat_pressure(barostat, pressure):
        """Set barostat pressure."""
        barostat.setDefaultPressure(pressure)

    @staticmethod
    def _set_barostat_temperature(barostat, temperature):
        """Set barostat temperature."""
        barostat.setDefaultTemperature(temperature)

    # -------------------------------------------------------------------------
    # Internal-usage: thermostat handling
    # -------------------------------------------------------------------------

    @classmethod
    def _find_thermostat(cls, system, get_index=False):
        """Return the first thermostat in the system.

        Returns
        -------
        force_idx : int or None, optional
            The force index of the thermostat.
        thermostat : OpenMM Force object or None
            The thermostat in system, or None if no thermostat is found.

        """
        try:
            force_idx, thermostat = forces.find_forces(system, '.*Thermostat.*', only_one=True)
        except forces.MultipleForcesError:
            raise ThermodynamicsError(ThermodynamicsError.MULTIPLE_THERMOSTATS)
        except forces.NoForceFoundError:
            force_idx, thermostat = None, None
        if get_index:
            return force_idx, thermostat
        return thermostat

    @classmethod
    def _remove_thermostat(cls, system):
        """Remove the system thermostat."""
        thermostat_idx, thermostat = cls._find_thermostat(system, get_index=True)
        if thermostat_idx is not None:
            system.removeForce(thermostat_idx)

    @classmethod
    def _set_system_temperature(cls, system, temperature):
        """Configure thermostat and barostat to the given temperature.

        The thermostat temperature is set, or a new AndersenThermostat
        is added if it doesn't exist.

        Parameters
        ----------
        system : simtk.openmm.System
            The system to modify.
        temperature : simtk.unit.Quantity
            The temperature for the thermostat.

        """
        thermostat = cls._find_thermostat(system)
        if thermostat is None:
            thermostat = openmm.AndersenThermostat(temperature, 1.0/unit.picosecond)
            system.addForce(thermostat)
        else:
            thermostat.setDefaultTemperature(temperature)

        barostat = cls._find_barostat(system)
        if barostat is not None:
            cls._set_barostat_temperature(barostat, temperature)

    # -------------------------------------------------------------------------
    # Internal-usage: initialization
    # -------------------------------------------------------------------------

    @staticmethod
    def _compute_reduced_potential(potential_energy, temperature, volume, pressure):
        """Convert potential energy into reduced potential."""
        beta = 1.0 / (unit.BOLTZMANN_CONSTANT_kB * temperature)
        reduced_potential = potential_energy / unit.AVOGADRO_CONSTANT_NA
        if pressure is not None:
            reduced_potential += pressure * volume
        return beta * reduced_potential

    def _find_force_groups_to_update(self, context, thermodynamic_state, memo):
        """Find the force groups to be recomputed when moving to the given state.

        With the current implementation of ThermodynamicState, no force group has
        to be recomputed as only temperature and pressure change between compatible
        states, but this method becomes essential in CompoundThermodynamicState.
        """
        return set()


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
    potential_energy
    kinetic_energy
    total_energy
    volume
    n_particles
    collective_variables

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
    openmmtools.states.SamplerStateError: Specified positions with inconsistent number of particles.

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
        # Allocate variables, they get set in _initialize
        self._positions = None
        self._velocities = None
        self._box_vectors = None
        self._collective_variables = None
        self._kinetic_energy = None
        self._potential_energy = None
        args = []
        for input in [positions, velocities, box_vectors]:
            if isinstance(input, unit.Quantity) and not isinstance(input._value, np.ndarray):
                args.append(np.array(input/input.unit)*input.unit)
            else:
               args.append(copy.deepcopy(input))
        self._initialize(*args)

    @classmethod
    def from_context(cls, context_state, ignore_collective_variables=False):
        """Alternative constructor.

        Read all the configurational properties from a Context object or
        an OpenMM State object. This guarantees that all attributes
        (including energy attributes) are initialized.


        Parameters
        ----------
        context_state : simtk.openmm.Context or simtk.openmm.State
            The object to read. If a State object, it must contain information
            about positions, velocities and energy.
        ignore_collective_variables : bool, optional
            If True, the collective variables are not updated from the
            Context, and will be invalidated. If a State is passed in,
            this raises an error if False, otherwise, it would be ambiguous
            between a State tied to a System with collective variables, and one without.

        Returns
        -------
        sampler_state : SamplerState
            A new SamplerState object.

        """
        sampler_state = cls([])
        sampler_state._read_context_state(context_state, check_consistency=False,
                                          ignore_positions=False,
                                          ignore_velocities=False,
                                          ignore_collective_variables=ignore_collective_variables)
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
        self._set_positions(value, from_context=False, check_consistency=True)

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
        self._set_velocities(value, from_context=False)

    @property
    def box_vectors(self):
        """Box vectors.

        An 3x3 simtk.unit.Quantity object.

        """
        return self._box_vectors

    @box_vectors.setter
    def box_vectors(self, value):
        # Make sure this is a Quantity. System.getDefaultPeriodicBoxVectors
        # returns a list of Quantity objects instead for example.
        if value is not None and not isinstance(value, unit.Quantity):
            value = unit.Quantity(value)
        self._box_vectors = value

    # Derived properties

    @property
    def potential_energy(self):
        """simtk.unit.Quantity or None: Potential energy of this configuration."""
        if self._are_positions_valid:
            return None
        return self._potential_energy

    @potential_energy.setter
    def potential_energy(self, new_value):
        if new_value is not None:
            raise AttributeError("Cannot set potential energy as it is a function of Context")
        self._potential_energy = None

    @property
    def kinetic_energy(self):
        """simtk.unit.Quantity or None: Kinetic energy of this configuration."""
        if self.velocities is None or self.velocities.has_changed:
            return None
        return self._kinetic_energy

    @kinetic_energy.setter
    def kinetic_energy(self, new_value):
        if new_value is not None:
            raise AttributeError("Cannot set kinetic energy as it is a function of Context")
        self._kinetic_energy = None

    @property
    def collective_variables(self):
        """dict or None: Collective variables for this configuration if present in Context"""
        if self._are_positions_valid:
            return None
        return self._collective_variables

    @collective_variables.setter
    def collective_variables(self, new_value):
        if new_value is not None:
            raise AttributeError("Cannot set collective variables as it is a function of Context")
        self._collective_variables = new_value

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

    def update_from_context(self, context_state, ignore_positions=False, ignore_velocities=False,
                            ignore_collective_variables=False):
        """Read the state from the given Context or State object.

        The context must be compatible. Use SamplerState.from_context
        if you want to build a new sampler state from an incompatible.

        Parameters
        ----------
        context_state : simtk.openmm.Context or simtk.openmm.State
            The object to read. If a State, it must contain information
            on positions, velocities and energies. Collective
            variables can only be updated from a Context, NOT a State
            at the moment.
        ignore_positions : bool, optional
            If True, the positions (and potential energy) are not updated from the
            Context. This can cause the SamplerState to no longer be consistent between
            its variables, so the defaults err on the side of updating everything,
            if possible. Only use if you know what you are doing.
        ignore_velocities : bool, optional
            If True, the velocities (and kinetic energy) are not updated from the
            Context. This can cause the SamplerState to no longer be consistent between
            its variables, so the defaults err on the side of updating everything,
            if possible. Only use if you know what you are doing.
        ignore_collective_variables : bool, optional
            If True, the collective variables are not updated from the
            Context. If a State is passed in,
            this raises an error if False, otherwise, it would be ambiguous
            between a State tied to a System with collective variables, and one without.

        Raises
        ------
        SamplerStateError
            If the given context is not compatible, or if a State is given without
            setting ignore_collective_variables

        """
        self._read_context_state(context_state, check_consistency=True,
                                 ignore_positions=ignore_positions,
                                 ignore_velocities=ignore_velocities,
                                 ignore_collective_variables=ignore_collective_variables)

    def apply_to_context(self, context, ignore_velocities=False):
        """Set the context state.

        If velocities and box vectors have not been specified in the
        constructor, they are not set.

        Parameters
        ----------
        context : simtk.openmm.Context
            The context to set.
        ignore_velocities : bool, optional
            If True, velocities are not set in the Context even if they
            are defined. This can be useful if you only need to use the
            Context only to compute energies.


        """
        # NOTE: Box vectors MUST be updated before positions are set.
        if self.box_vectors is not None:
            context.setPeriodicBoxVectors(*self.box_vectors)
        context.setPositions(self._unitless_positions)
        if self._velocities is not None and not ignore_velocities:
            context.setVelocities(self._unitless_velocities)

    def has_nan(self):
        """Check that energies and positions are finite.

        Returns
        -------
        True if the potential energy or any of the generalized coordinates
        are nan.

        """
        if (self.potential_energy is not None and
                np.isnan(self.potential_energy.value_in_unit(self.potential_energy.unit))):
            return True
        if np.any(np.isnan(self._positions)):
            return True
        return False

    def __getitem__(self, item):
        sampler_state = self.__class__([])

        # Handle single index.
        if np.issubdtype(type(item), np.integer):
            # Here we don't need to copy since we instantiate a new array.
            pos_value = self._positions[item].value_in_unit(self._positions.unit)
            new_positions = unit.Quantity(np.array([pos_value]), self._positions.unit)
            sampler_state._set_positions(new_positions, from_context=False, check_consistency=False)
            if self._velocities is not None:
                vel_value = self._velocities[item].value_in_unit(self._velocities.unit)
                new_velocities = unit.Quantity(np.array([vel_value]), self._velocities.unit)
                sampler_state._set_velocities(new_velocities, from_context=False)
        else:  # Assume slice or sequence.
            # Copy original values to avoid side effects.
            sampler_state._set_positions(copy.deepcopy(self._positions[item]),
                                         from_context=False, check_consistency=False)
            if self._velocities is not None:
                sampler_state._set_velocities(copy.deepcopy(self._velocities[item].copy()),
                                              from_context=False)

        # Copy box vectors.
        sampler_state.box_vectors = copy.deepcopy(self.box_vectors)

        # Energies/CV's for only a subset of atoms is undefined.
        sampler_state._potential_energy = None
        sampler_state._kinetic_energy = None
        sampler_state._collective_variables = None
        return sampler_state

    def __getstate__(self, ignore_velocities=False):
        """Return a dictionary representation of the state.

        Parameters
        ----------
        ignore_velocities : bool, optional
            If True, velocities are not serialized. This can be useful for
            example to save bandwidth when sending a ``SamplerState`` over
            the network and velocities are not required (default is False).
        """
        velocities = None if ignore_velocities else self.velocities
        serialization = dict(
            positions=self.positions, velocities=velocities,
            box_vectors=self.box_vectors, potential_energy=self.potential_energy,
            kinetic_energy=self.kinetic_energy,
            collective_variables=self.collective_variables
        )
        return serialization

    def __setstate__(self, serialization, ignore_velocities=False):
        """Set the state from a dictionary representation.

        Parameters
        ----------
        ignore_velocities : bool, optional
            If True and the ``SamplerState`` has already velocities
            defined, this does not overwrite the velocities.
        """
        if ignore_velocities and '_velocities' in self.__dict__:
            serialization['velocities'] = self.velocities
        self._initialize(**serialization)

    # -------------------------------------------------------------------------
    # Internal-usage
    # -------------------------------------------------------------------------

    def _initialize(self, positions, velocities, box_vectors,
                    potential_energy=None, kinetic_energy=None, collective_variables=None):
        """Initialize the sampler state."""
        self._set_positions(positions, from_context=False, check_consistency=False)
        self.velocities = velocities  # Checks consistency and units.
        self.box_vectors = box_vectors  # Make sure box vectors is Quantity.
        self._potential_energy = potential_energy
        self._kinetic_energy = kinetic_energy
        self._collective_variables = collective_variables

    def _set_positions(self, new_positions, from_context, check_consistency):
        """Set the positions without checking for consistency."""
        if check_consistency and (new_positions is None or len(new_positions) != self.n_particles):
            raise SamplerStateError(SamplerStateError.INCONSISTENT_POSITIONS)

        if from_context:
            self._unitless_positions_cache = new_positions._value
            assert new_positions.unit == unit.nanometer
        else:
            self._unitless_positions_cache = None

        self._positions = utils.TrackedQuantity(new_positions)

        # The potential energy changes with different positions.
        self._potential_energy = None

        # The CVs change with different positions too
        self._collective_variables = None

    def _set_velocities(self, new_velocities, from_context):
        """Set the velocities."""
        if from_context:
            self._unitless_velocities_cache = new_velocities._value
            assert new_velocities.unit == unit.nanometer/unit.picoseconds
        else:
            if new_velocities is not None and self.n_particles != len(new_velocities):
                raise SamplerStateError(SamplerStateError.INCONSISTENT_VELOCITIES)
            self._unitless_velocities_cache = None

        if new_velocities is not None:
            new_velocities = utils.TrackedQuantity(new_velocities)
        self._velocities = new_velocities

        # The kinetic energy changes with different positions.
        self._kinetic_energy = None

    @property
    def _unitless_positions(self):
        """Keeps a cache of unitless positions."""
        if self._unitless_positions_cache is None or self._positions.has_changed:
            self._unitless_positions_cache = self.positions.value_in_unit_system(unit.md_unit_system)
        if self._positions.has_changed:
            self._positions.has_changed = False
            self._potential_energy = None
        return self._unitless_positions_cache

    @property
    def _unitless_velocities(self):
        """Keeps a cache of unitless velocities."""
        if self._velocities is None:
            return None
        if self._unitless_velocities_cache is None or self._velocities.has_changed:
            self._unitless_velocities_cache = self._velocities.value_in_unit_system(unit.md_unit_system)
        if self._velocities.has_changed:
            self._velocities.has_changed = False
            self._kinetic_energy = None
        return self._unitless_velocities_cache

    def _read_context_state(self, context_state, check_consistency,
                            ignore_positions,
                            ignore_velocities,
                            ignore_collective_variables):
        """Read the Context state.

        Parameters
        ----------
        context_state : simtk.openmm.Context or simtk.openmm.State
            The object to read.
        check_consistency : bool
            If True, raise an error if the context system have a
            different number of particles than the current state.
        ignore_positions : bool
            If True, the positions and potential energy are not updated from the
            Context.
        ignore_velocities : bool
            If True, the velocities and kinetic energy are not updated from the
            Context.
        ignore_collective_variables : bool
            If True, the collective variables are not updated from the
            Context. If a State is passed in,
            this raises an error if False, otherwise, it would be ambiguous
            between a State tied to a System with collective variables, and one without.

        Raises
        ------
        SamplerStateError
            If the the context system have a different number of
            particles than the current state.

        """
        if isinstance(context_state, openmm.Context):
            system = context_state.getSystem()
            openmm_state = context_state.getState(getPositions=not ignore_positions,
                                                  getVelocities=not ignore_velocities,
                                                  getEnergy=not (ignore_velocities and ignore_positions),
                                                  enforcePeriodicBox=system.usesPeriodicBoundaryConditions())
        else:
            if not ignore_collective_variables:
                raise SamplerStateError("State objects must have ignore_collective_variables=True because they "
                                        "don't track CV's and would be ambiguous between a System with no "
                                        "collective variables.")
            openmm_state = context_state

        # We assign positions first, since the velocities
        # property will check its length for consistency.
        # Potential energy and kinetic energy must be updated
        # after positions and velocities or they'll be reset.
        if not ignore_positions:
            positions = openmm_state.getPositions(asNumpy=True)
            self._set_positions(positions, from_context=True, check_consistency=check_consistency)
            self._potential_energy = openmm_state.getPotentialEnergy()
        if not ignore_velocities:
            velocities = openmm_state.getVelocities(asNumpy=True)
            self._set_velocities(velocities, from_context=True)
            self._kinetic_energy = openmm_state.getKineticEnergy()
        self.box_vectors = openmm_state.getPeriodicBoxVectors(asNumpy=True)
        if not ignore_collective_variables:
            self._read_collective_variables(context_state)

    def _read_collective_variables(self, context_state):
        """
        Update the collective variables from the context object

        Parameters
        ----------
        context_state : simtk.openmm.Context
            The object to read. This only works with Context's for now,
            but in the future, this may support OpenMM State objects as well.
        """
        # Allows direct key assignment without initializing each key:dict pair
        collective_variables = collections.defaultdict(dict)
        system = context_state.getSystem()
        for force_index, force in enumerate(system.getForces()):
            try:
                cv_values = force.getCollectiveVariableValues(context_state)
                for cv_index in range(force.getNumCollectiveVariables()):
                    cv_name = force.getCollectiveVariableName(cv_index)
                    collective_variables[cv_name][force_index] = cv_values[cv_index]
            except AttributeError:
                pass
        # Trap no variables found (empty dict), return None
        # Cast defaultdict back to dict
        self._collective_variables = dict(collective_variables) if collective_variables else None

    @property
    def _are_positions_valid(self):
        """Helper function to reduce this check duplication in multiple properties"""
        return self.positions is None or self.positions.has_changed


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
        """Set the system to be in this state.

        This method is called in three situations:
        1) On initialization, before standardizing the system.
        2) When a new system is set and the argument ``fix_state`` is
           set to ``True``.
        3) When the system is retrieved to convert the standard system
           into a system in the correct thermodynamic state for the
           simulation.

        Parameters
        ----------
        system : simtk.openmm.System
            The system to modify.

        Raises
        ------
        ComposableStateError
            If the system is not compatible with the state.

        """
        pass

    @abc.abstractmethod
    def check_system_consistency(self, system):
        """Check if the system is in this state.

        It raises a ComposableStateError if the system is not in
        this state. This is called when the ThermodynamicState's
        system is set with the ``fix_state`` argument set to False.

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
        """Set the context to be in this state.

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

    @abc.abstractmethod
    def _standardize_system(self, system):
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
        ComposableStateError
            If the system is not compatible with the state.

        """
        pass

    @abc.abstractmethod
    def _on_setattr(self, standard_system, attribute_name, old_composable_state):
        """Check if standard system needs to be updated after a state attribute is set.

        This callback function is called after an attribute is set (i.e.
        after __setattr__ is called on this state) or if an attribute whose
        name starts with "set_" is requested (i.e. if a setter is retrieved
        from this state through __getattr__).

        Parameters
        ----------
        standard_system : simtk.openmm.System
            The standard system before setting the attribute.
        attribute_name : str
            The name of the attribute that has just been set or retrieved.
        old_composable_state : IComposableState
            A copy of the composable state before the attribute was set.

        Returns
        -------
        need_changes : bool
            True if the standard system has to be updated, False if no change
            occurred.

        Raises
        ------
        ComposableStateError
            If the attribute change put the system in an inconsistent state.

        """
        pass

    @abc.abstractmethod
    def _find_force_groups_to_update(self, context, current_context_state, memo):
        """Find the force groups whose energy must be recomputed after applying self.

        This is used to compute efficiently the potential energy of the
        same configuration in multiple thermodynamic states to minimize
        the number of force evaluations.

        Parameters
        ----------
        context : Context
            The context, currently in `current_context_state`, that will
            be moved to this state.
        current_context_state : ThermodynamicState
            The full thermodynamic state of the given context. This is
            guaranteed to be compatible with self.
        memo : dict
            A dictionary that can be used by the state for memoization
            to speed up consecutive calls on the same context.

        Returns
        -------
        force_groups_to_update : set of int
            The indices of the force groups whose energy must be computed
            again after applying this state, assuming the context to be in
            `current_context_state`.
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
    >>> factory = alchemy.AbsoluteAlchemicalFactory(consistent_exceptions=False)
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
        # Check that composable states expose the correct interface.
        for composable_state in composable_states:
            assert isinstance(composable_state, IComposableState)

        # Copy internal attributes of thermodynamic state.
        thermodynamic_state = copy.deepcopy(thermodynamic_state)
        self.__dict__ = thermodynamic_state.__dict__

        # Setting self._composable_states signals __setattr__ to start
        # searching in composable states as well, so this must be the
        # last new attribute set in the constructor.
        composable_states = copy.deepcopy(composable_states)
        self._composable_states = composable_states

        # This call causes the thermodynamic state standard system
        # to be standardized also w.r.t. all the composable states.
        self.set_system(self._standard_system, fix_state=True)

    def get_system(self, **kwargs):
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

        """
        system = super(CompoundThermodynamicState, self).get_system(**kwargs)

        # The system returned by ThermodynamicState has standard parameters,
        # so we need to set them to the actual value of the composable states.
        for s in self._composable_states:
            s.apply_to_system(system)
        return system

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
        system = copy.deepcopy(system)
        for s in self._composable_states:
            if fix_state:
                s.apply_to_system(system)
            else:
                s.check_system_consistency(system)
        super(CompoundThermodynamicState, self)._unsafe_set_system(system, fix_state)

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
        def setter_decorator(funcs, composable_states):
            def _setter_decorator(*args, **kwargs):
                for func, composable_state in zip(funcs, composable_states):
                    old_state = copy.deepcopy(composable_state)
                    func(*args, **kwargs)
                    self._on_setattr_callback(composable_state, name, old_state)
            return _setter_decorator

        # Called only if the attribute couldn't be found in __dict__.
        # In this case we fall back to composable state, in the given order.
        attrs = []
        composable_states = []
        for s in self._composable_states:
            try:
                attr = getattr(s, name)
            except AttributeError:
                pass
            else:
                attrs.append(attr)
                composable_states.append(s)

        if len(attrs) > 0:
            # If this is a setter, we need to set the attribute in all states
            # and ensure that the callback is called in each of them.
            if name.startswith('set_'):
                # Decorate the setter so that _on_setattr is called after the
                # attribute is modified. This also reduces the calls to multiple
                # setter to a single function.
                attr = setter_decorator(attrs, composable_states)
            else:
                if len(attrs) > 1 and not all(np.isclose(attrs[0], a) for a in attrs[1:]):
                    raise RuntimeError('The composable states of {} expose the same '
                                       'attribute with different values: {}'.format(
                        self.__class__.__name__, set(attrs)))
                attr = attrs[0]
            return attr

        # Attribute not found, fall back to normal behavior.
        return super(CompoundThermodynamicState, self).__getattribute__(name)

    def __setattr__(self, name, value):
        # Add new attribute to CompoundThermodynamicState.
        if '_composable_states' not in self.__dict__:
            super(CompoundThermodynamicState, self).__setattr__(name, value)

        # Update existing ThermodynamicState attribute (check ancestors).
        # We can't use hasattr here because it calls __getattr__, which
        # search in all composable states as well. This means that this
        # will catch only properties and methods.
        elif any(name in C.__dict__ for C in self.__class__.__mro__):
            super(CompoundThermodynamicState, self).__setattr__(name, value)

        # Update composable states attributes. This catches also normal
        # attributes besides properties and methods.
        else:
            old_state = None
            for s in self._composable_states:
                try:
                    getattr(s, name)
                except AttributeError:
                    pass
                else:
                    old_state = copy.deepcopy(s)
                    s.__setattr__(name, value)
                    self._on_setattr_callback(s, name, old_state)

            # No attribute found. This is monkey patching.
            if old_state is None:
                super(CompoundThermodynamicState, self).__setattr__(name, value)

    def __getstate__(self, **kwargs):
        """Return a dictionary representation of the state."""
        # Create original ThermodynamicState to serialize.
        thermodynamic_state = object.__new__(self.__class__.__bases__[0])
        thermodynamic_state.__dict__ = self.__dict__
        # Set the instance _standardize_system method to CompoundState._standardize_system
        # so that the composable states standardization will be called during serialization.
        thermodynamic_state._standardize_system = self._standardize_system
        serialized_thermodynamic_state = utils.serialize(thermodynamic_state, **kwargs)

        # Serialize composable states.
        serialized_composable_states = [utils.serialize(state)
                                        for state in self._composable_states]

        return dict(thermodynamic_state=serialized_thermodynamic_state,
                    composable_states=serialized_composable_states)

    def __setstate__(self, serialization):
        """Set the state from a dictionary representation."""
        serialized_thermodynamic_state = serialization['thermodynamic_state']
        serialized_composable_states = serialization['composable_states']
        thermodynamic_state = utils.deserialize(serialized_thermodynamic_state)
        self.__dict__ = thermodynamic_state.__dict__
        self._composable_states = [utils.deserialize(state)
                                   for state in serialized_composable_states]

    # -------------------------------------------------------------------------
    # Internal-usage
    # -------------------------------------------------------------------------

    def _standardize_system(self, system):
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
        super(CompoundThermodynamicState, self)._standardize_system(system)
        for composable_state in self._composable_states:
            composable_state._standardize_system(system)

    def _on_setattr_callback(self, composable_state, attribute_name, old_composable_state):
        """Updates the standard system (and hash) after __setattr__."""
        try:
            change_standard_system = composable_state._on_setattr(self._standard_system, attribute_name, old_composable_state)
        except TypeError:
            change_standard_system = composable_state._on_setattr(self._standard_system, attribute_name)
            # TODO Drop support for the old signature and remove deprecation warning from 0.17 on.
            import warnings
            old_signature = '_on_setattr(self, standard_system, attribute_name)'
            new_signature = old_signature[:-1] + ', old_composable_state)'
            warnings.warn('The signature IComposableState.{} has been deprecated, '
                          'and future versions of openmmtools will support only the '
                          'new one: {}.'.format(old_signature, new_signature))
        if change_standard_system:
            new_standard_system = copy.deepcopy(self._standard_system)
            composable_state.apply_to_system(new_standard_system)
            composable_state._standardize_system(new_standard_system)
            self._update_standard_system(new_standard_system)

    def _apply_to_context_in_state(self, context, thermodynamic_state):
        super(CompoundThermodynamicState, self)._apply_to_context_in_state(context, thermodynamic_state)
        for s in self._composable_states:
            s.apply_to_context(context)

    def _find_force_groups_to_update(self, context, current_context_state, memo):
        """Find the force groups to be recomputed when moving to the given state.

        Override ThermodynamicState._find_force_groups_to_update to find
        groups to update for changes of composable states.
        """
        # Initialize memo: create new cache for each composable state.
        if len(memo) == 0:
            memo.update({i: {} for i in range(len(self._composable_states))})
        # Find force group to update for parent class.
        force_groups = super(CompoundThermodynamicState, self)._find_force_groups_to_update(
            context, current_context_state, memo)
        # Find force group to update for composable states.
        for composable_state_idx, composable_state in enumerate(self._composable_states):
            force_groups.update(composable_state._find_force_groups_to_update(
                context, current_context_state, memo[composable_state_idx]))
        return force_groups


# =============================================================================
# GLOBAL PARAMETER STATE
# =============================================================================

class GlobalParameterError(ComposableStateError):
    """Exception raised by ``GlobalParameterState``."""
    pass


class GlobalParameterFunction(object):
    """A function of global parameters.

    All the functions supported by ``openmmtools.utils.math_eval``
    are supported.

    Parameters
    ----------
    expression : str
        A mathematical expression involving global parameters.

    See Also
    --------
    openmmtools.utils.math_eval

    Examples
    --------
    >>> class MyComposableState(GlobalParameterState):
    ...     gamma = GlobalParameterState.GlobalParameter('gamma', standard_value=1.0)
    ...     lambda_angles = GlobalParameterState.GlobalParameter('lambda_angles', standard_value=1.0)
    ...
    >>> composable_state = MyComposableState(gamma=1.0, lambda_angles=0.5)
    >>> composable_state.set_function_variable('lambda', 0.5)
    >>> composable_state.set_function_variable('lambda2', 1.0)
    >>> composable_state.gamma = GlobalParameterFunction('lambda**2')
    >>> composable_state.gamma
    0.25
    >>> composable_state.lambda_angles = GlobalParameterFunction('(lambda + lambda2) / 2')
    >>> composable_state.lambda_angles
    0.75
    >>> composable_state.set_function_variable('lambda2', 0.5)
    >>> composable_state.lambda_angles
    0.5

    """
    def __init__(self, expression):
        self._expression = expression

    def __call__(self, variables):
        return utils.math_eval(self._expression, variables)


class GlobalParameterState(object):
    """A composable state controlling one or more OpenMM ``Force``'s global parameters.

    This is a partially abstract class that provides facilities to implement
    composable states that control one or more global parameters defined in
    OpenMM ``Force`` objects. Global parameters are implemented  through the
    use of the ``GlobalParameterState.GlobalParameter`` descriptor.

    A ``Force`` object can use one or more global parameters that are
    controlled by the same state. Conversely, multiple ``Force``s can use
    the same global parameter (i.e. with the same name) controlled by the
    state object.

    It is possible to enslave the global parameters to one or more arbitrary
    variables entering a mathematical expression through the use of
    ``GlobalParameterFunction``. Global parameters that are associated to a
    global parameter function are validated on get rather than on set.

    Parameters
    ----------
    parameters_name_suffix : str, optional
        If specified, the state will control a modified version of the global
        parameters with the name ``parameter_name + '_' + parameters_name_suffix``.
        When this is the case, the normal parameters are not accessible.
    **kwargs
        The value of the parameters controlled by this state. Parameters
        that are not passed here are left undefined.

    Notes
    -----
    The class automatically implement the static constructor ``from_system``
    that reads and create a state object from an OpenMM ``System``. The function
    calls ``__init__`` and passes the parameter name suffix string as the
    first positional argument, so it is possible to overwrite ``__init__``
    and rename ``parameters_name_suffix`` as long as it is the first parameter
    of the constructor.

    See Also
    --------
    GlobalParameterFunction

    Examples
    --------

    Assume we have a ``System`` with a custom force object whose energy
    function is controlled by two global variables called ``lambda_bonds``
    and ``gamma``.

    >>> from simtk import openmm, unit
    >>> # Define a diatomic molecule.
    >>> system = openmm.System()
    >>> particle_idx = system.addParticle(40.0*unit.amu)
    >>> particle_idx = system.addParticle(40.0*unit.amu)
    >>> custom_force = openmm.CustomBondForce('lambda_bonds^gamma*60000*(r-0.15)^2;')
    >>> parameter_idx = custom_force.addGlobalParameter('lambda_bonds', 1.0)  # Default value is 1.0.
    >>> parameter_idx = custom_force.addGlobalParameter('gamma', 1.0)  # Default value is 1.0.
    >>> bond_idx = custom_force.addBond(0, 1, [])
    >>> force_index = system.addForce(custom_force)
    >>> # Create a thermodynamic state object controlling the temperature of the system.
    >>> thermodynamic_state = ThermodynamicState(system, temperature=300.0*unit.kelvin)

    Define a composable state controlling the two global parameters ``gamma``
    and ``lambda_bonds``, both with standard state value 0.0. Parameters that
    are not specified in the constructor are left undefined.

    >>> class MyComposableState(GlobalParameterState):
    ...     gamma = GlobalParameterState.GlobalParameter('gamma', standard_value=1.0)
    ...     lambda_bonds = GlobalParameterState.GlobalParameter('lambda_bonds', standard_value=1.0)
    ...
    >>> my_composable_state = MyComposableState(gamma=1.0)
    >>> my_composable_state.gamma
    1.0
    >>> my_composable_state.lambda_bonds is None
    True

    There is a second static constructor you can use to read the state
    of an OpenMM ``System`` from the default values of its force parameters.

    >>> my_composable_state = MyComposableState.from_system(system)
    >>> my_composable_state.lambda_bonds
    1.0
    >>> my_composable_state.gamma
    1.0

    Optionally, you can limit the values that a global parameter can assume.
    In this case, ``lambda_bonds`` is forced to be between 0.0 and 1.0.

    >>> class MyComposableState(GlobalParameterState):
    ...     gamma = GlobalParameterState.GlobalParameter('gamma', standard_value=0.0)
    ...     lambda_bonds = GlobalParameterState.GlobalParameter('lambda_bonds', standard_value=0.0)
    ...     @lambda_bonds.validator
    ...     def lambda_bonds(self, instance, new_value):
    ...         if new_value is not None and not (0.0 <= new_value <= 1.0):
    ...             raise ValueError('lambda_bonds must be between 0.0 and 1.0')
    ...         return new_value
    ...
    >>> my_composable_state = MyComposableState(gamma=1.0)
    >>> my_composable_state.lambda_bonds = 2.0
    Traceback (most recent call last):
    ...
    ValueError: lambda_bonds must be between 0.0 and 1.0

    You can then add it to a ``CompoundThermodynamicState`` to manipulate
    OpenMM ``System`` and ``Context`` objects.

    >>> my_composable_state.lambda_bonds = 1.0
    >>> compound_state = CompoundThermodynamicState(thermodynamic_state, composable_states=[my_composable_state])
    >>> state_system = compound_state.get_system()
    >>> state_system.getForce(0).getGlobalParameterDefaultValue(0)  # lambda_bonds global parameter.
    1.0
    >>> compound_state.lambda_bonds = 0.0
    >>> state_system = compound_state.get_system()
    >>> state_system.getForce(0).getGlobalParameterDefaultValue(0)  # lambda_bonds global parameter.
    0.0

    >>> context = compound_state.create_context(openmm.VerletIntegrator(1.0*unit.femtoseconds))
    >>> context.getParameter('lambda_bonds')
    0.0
    >>> compound_state.lambda_bonds = 1.0
    >>> compound_state.apply_to_context(context)
    >>> context.getParameter('lambda_bonds')
    1.0

    You can express enslave global parameters to a mathematical expression
    involving arbitrary variables.

    >>> compound_state.set_function_variable('lambda', 1.0)
    >>> compound_state.lambda_bonds = GlobalParameterFunction('2*(lambda - 0.5) * step(lambda - 0.5)')
    >>> for l in [0.5, 0.75, 1.0]:
    ...     compound_state.set_function_variable('lambda', l)
    ...     print(compound_state.lambda_bonds)
    0.0
    0.5
    1.0

    If you need to control similar forces with the same state object,
    this parent class provides a suffix mechanism to control different
    global variables with the same state object. This allows to reuse
    the same logic to control multiple objects

    >>> # Add a second custom force using similar global parameters.
    >>> custom_force = openmm.CustomBondForce('lambda_bonds_mysuffix*20000*(r-0.15)^2;')
    >>> parameter_idx = custom_force.addGlobalParameter('lambda_bonds_mysuffix', 1.0)  # Default value is 1.0.
    >>> bond_idx = custom_force.addBond(0, 1, [])
    >>> force_idx = system.addForce(custom_force)
    >>> # Create a state controlling the modified global parameter.
    >>> my_composable_state = MyComposableState(parameters_name_suffix='mysuffix', lambda_bonds=0.0)
    >>> my_composable_state.lambda_bonds_mysuffix = 1.0
    >>> my_composable_state.gamma_mysuffix is None
    True
    >>> my_composable_state.apply_to_system(system)
    >>> # The unmodified parameter becomes unaccessible.
    >>> my_composable_state.apply_to_system(system)
    >>> my_composable_state.lambda_bonds
    Traceback (most recent call last):
    ...
    AttributeError: This state does not control lambda_bonds but lambda_bonds_mysuffix.

    Note also in the example above that the forces don't need to define
    all the global parameters controlled by the state. The state object
    will perform some check to ensure that you won't try to set an undefined
    parameter.

    >>> my_composable_state.gamma_mysuffix = 2
    >>> my_composable_state.apply_to_system(system)
    Traceback (most recent call last):
    ...
    openmmtools.states.GlobalParameterError: Could not find global parameter gamma_mysuffix in the system.

    """

    # This constant can be overwritten by inheriting classes to
    # raise a custom exception class when an error is encountered.
    _GLOBAL_PARAMETER_ERROR = GlobalParameterError

    def __init__(self, parameters_name_suffix=None, **kwargs):
        self._initialize(parameters_name_suffix=parameters_name_suffix, **kwargs)

    @classmethod
    def from_system(cls, system, parameters_name_suffix=None):
        """Static constructor reading the state from an OpenMM System.

        Parameters
        ----------
        system : simtk.openmm.System
            An OpenMM ``System`` object defining a non-empty subset
            of global parameters controlled by this state.
        parameters_name_suffix : str, optional
            If specified, the state will search for a modified
            version of the global parameters with the name
            ``parameter_name + '_' + parameters_name_suffix``.

        Returns
        -------
        The GlobalParameterState object representing the state of the system.

        Raises
        ------
        GlobalParameterStateError
            If the same parameter has different values in the system, or
            if the system has no lambda parameters.

        """
        state_parameters = {}
        for force, parameter_name, parameter_id in cls._get_system_controlled_parameters(
                system, parameters_name_suffix):

            if parameter_id >= force.getNumGlobalParameters():
                raise GlobalParameterStateError(f'Attempted to access system parameter {parameter_name} (id {parameter_id}) that does not exist in {force.__class__.__name__}')

            parameter_value = force.getGlobalParameterDefaultValue(parameter_id)

            # Check that we haven't already found
            # the parameter with a different value.
            if parameter_name in state_parameters:
                if state_parameters[parameter_name] != parameter_value:
                    err_msg = ('Parameter {} has been found twice (Force {}) with two values: '
                               '{} and {}').format(parameter_name, force.__class__.__name__,
                                                   parameter_value, state_parameters[parameter_name])
                    raise cls._GLOBAL_PARAMETER_ERROR(err_msg)
            else:
                state_parameters[parameter_name] = parameter_value

        # Check that the system can be controlled by this state..
        if len(state_parameters) == 0:
            err_msg = 'System has no global parameters controlled by this state.'
            raise cls._GLOBAL_PARAMETER_ERROR(err_msg)

        # Create and return the GlobalParameterState. The constructor of
        # GlobalParameterState takes the parameters without the suffix so
        # we left them undefined in the constructor and assign the attributes.
        state = cls(parameters_name_suffix)
        for parameter_name, parameter_value in state_parameters.items():
            setattr(state, parameter_name, parameter_value)
        return state

    # -------------------------------------------------------------------------
    # Function variables
    # -------------------------------------------------------------------------

    def get_function_variable(self, variable_name):
        """Return the value of the function variable.

        Function variables are _not_ global parameters but rather variables
        entering mathematical expressions specified with ``GlobalParameterFunction``,
        which can be use to enslave global parameter to arbitrary variables.

        Parameters
        ----------
        variable_name : str
            The name of the function variable.

        Returns
        -------
        variable_value : float
            The value of the function variable.

        """
        try:
            variable_value = self._function_variables[variable_name]
        except KeyError:
            err_msg = 'Unknown function variable {}'.format(variable_name)
            raise self._GLOBAL_PARAMETER_ERROR(err_msg)
        return variable_value

    def set_function_variable(self, variable_name, new_value):
        """Set the value of the function variable.

        Function variables are _not_ global parameters but rather variables
        entering mathematical expressions specified with ``GlobalParameterFunction``,
        which can be use to enslave global parameter to arbitrary variables.

        Parameters
        ----------
        variable_name : str
            The name of the function variable.
        new_value : float
            The new value for the variable.

        """
        forbidden_variable_names = set(self._parameters)
        if variable_name in forbidden_variable_names:
            err_msg = ('Cannot have an function variable with the same name '
                       'of the predefined global parameter {}.'.format(variable_name))
            raise self._GLOBAL_PARAMETER_ERROR(err_msg)
        # Check that the new value is a scalar,
        if not (np.isreal(new_value) and np.isscalar(new_value)):
            err_msg = 'Only integers and floats can be assigned to a function variable.'
            raise self._GLOBAL_PARAMETER_ERROR(err_msg)
        self._function_variables[variable_name] = new_value

    # -------------------------------------------------------------------------
    # Operators
    # -------------------------------------------------------------------------

    def __eq__(self, other):
        """Equality operator.

        Two GlobalParameterState are equal if they control the same global
        parameters and they all have the same values. This way the operator
        preserves the commutative property.

        """
        # Check if other is a global parameter state.
        if not isinstance(other, GlobalParameterState):
            return False
        # Check that they define the same parameters.
        if not set(self._parameters) == set(other._parameters):
            return False
        # Check that all values are the same
        is_equal = True
        for parameter_name in self._parameters:
            self_value = getattr(self, parameter_name)
            other_value = getattr(other, parameter_name)
            is_equal = is_equal and self_value == other_value
        return is_equal

    def __ne__(self, other):
        # TODO: we can safely remove this when dropping support for Python 2
        return not self == other

    def __str__(self):
        return str(self._parameters)

    # -------------------------------------------------------------------------
    # Global parameters descriptor class.
    # -------------------------------------------------------------------------

    # The global parameter descriptor makes it easy for the user to
    # create their own state classes. The set of controlled parameters is
    # dynamically discovered by _get_controlled_parameters() by checking
    # which descriptors are GlobalParameter objects.

    class GlobalParameter(object):
        """Descriptor for a global parameter.

        Parameters
        ----------
        parameter_name : str
            The name of the global parameter.
        standard_value : float
            The value of the global parameter in the standard state. This
            is used to define the concept of compatible states (i.e. whether
            a ``System`` or ``Context`` can be transformed from a state
            to another).
        validator : callable, optional
            A function to call before setting a new value with signature
            ``validator(self, instance, new_value) -> validated_value``.
            It is also possible to define this through the ``validator``
            decorator.

        """
        def __init__(self, parameter_name, standard_value, validator=None):
            self.parameter_name = parameter_name
            self.standard_value = standard_value
            self.validator_func = validator

        def __get__(self, instance, owner_class=None):
            self._check_controlled(instance)
            return instance._get_global_parameter_value(self.parameter_name, self)

        def __set__(self, instance, new_value):
            self._check_controlled(instance)
            instance._set_global_parameter_value(self.parameter_name, new_value, self)

        def validator(self, validator):
            return self.__class__(self.parameter_name, self.standard_value, validator)

        def _check_controlled(self, instance):
            """Raise GlobalParameterError if the parameter is not controlled by the state.

            If the state uses a parameter name suffix, the normal parameter
            name is not accessible.
            """
            if instance._parameters_name_suffix is not None:
                suffixed_parameter_name = self.parameter_name + '_' + instance._parameters_name_suffix
                err_msg = 'This state does not control {} but {}.'.format(
                    self.parameter_name, suffixed_parameter_name)
                raise AttributeError(err_msg)

    # -------------------------------------------------------------------------
    # Internal usage: IComposableState interface
    # -------------------------------------------------------------------------

    def apply_to_system(self, system):
        """Set the state of the system to this.

        Parameters
        ----------
        system : simtk.openmm.System
            The system to modify.

        Raises
        ------
        GlobalParameterError
            If the system does not defined some of the global parameters
            controlled by this state.

        """
        parameters_applied = set()
        for force, parameter_name, parameter_id in self._get_system_controlled_parameters(
                system, self._parameters_name_suffix):
            parameter_value = getattr(self, parameter_name)
            if parameter_value is None:
                err_msg = 'The system parameter {} is not defined in this state.'
                raise self._GLOBAL_PARAMETER_ERROR(err_msg.format(parameter_name))
            else:

                if parameter_id >= force.getNumGlobalParameters():
                    raise GlobalParameterStateError(f'Attempted to access system parameter {parameter_name} (id {parameter_id}) that does not exist in {force.__class__.__name__}')

                parameters_applied.add(parameter_name)
                force.setGlobalParameterDefaultValue(parameter_id, parameter_value)

        # Check that we set all the defined parameters.
        for parameter_name in self._get_controlled_parameters(self._parameters_name_suffix):
            if (self._parameters[parameter_name] is not None and
                    parameter_name not in parameters_applied):
                err_msg = 'Could not find global parameter {} in the system.'
                raise self._GLOBAL_PARAMETER_ERROR(err_msg.format(parameter_name))

    def check_system_consistency(self, system):
        """Check if the system is in this state.

        It raises a GlobalParameterError if the system is not consistent
        with this state.

        Parameters
        ----------
        system : simtk.openmm.System
            The system to test.

        Raises
        ------
        GlobalParameterError
            If the system is not consistent with this state.

        """
        system_state = self.__class__.from_system(system, self._parameters_name_suffix)

        # Check if parameters are all the same.
        if self != system_state:
            err_msg = ('Consistency check failed:\n'
                       '\tSystem parameters {}\n'
                       '\t{} parameters {}')
            class_name = self.__class__.__name__
            raise self._GLOBAL_PARAMETER_ERROR(err_msg.format(system_state, class_name, self))

    def apply_to_context(self, context):
        """Put the Context into this state.

        Parameters
        ----------
        context : simtk.openmm.Context
            The context to set.

        Raises
        ------
        GlobalParameterError
            If the context does not have the required global parameters.

        """
        context_parameters = context.getParameters()

        # Set the global parameters in Context.
        for parameter_name in self._parameters:
            parameter_value = getattr(self, parameter_name)
            if parameter_value is None:
                # Check that Context does not have this parameter.
                if parameter_name in context_parameters:
                    err_msg = 'Context has parameter {} which is undefined in this state.'
                    raise self._GLOBAL_PARAMETER_ERROR(err_msg.format(parameter_name))
                continue

            try:
                context.setParameter(parameter_name, parameter_value)
            except Exception:
                err_msg = 'Could not find parameter {} in context'
                raise self._GLOBAL_PARAMETER_ERROR(err_msg.format(parameter_name))


    def _standardize_system(self, system):
        """Standardize the given system.

        Set all global parameters of the system their standard value.

        Parameters
        ----------
        system : simtk.openmm.System
            The system to standardize.

        Raises
        ------
        GlobalParameterError
            If the system is not consistent with this state.

        """
        # Collect all the global parameters' standard values.
        standard_values = {}
        controlled_parameters = self._get_controlled_parameters(self._parameters_name_suffix)
        for parameter_name, parameter_descriptor in controlled_parameters.items():
            standard_values[parameter_name] = parameter_descriptor.standard_value

        # Create a standard state.
        standard_state = copy.deepcopy(self)
        for parameter_name, standard_value in standard_values.items():
            # Skip undefined parameters.
            if getattr(standard_state, parameter_name) is not None:
                setattr(standard_state, parameter_name, standard_value)

        # Standardize the system.
        standard_state.apply_to_system(system)

    def _on_setattr(self, standard_system, attribute_name, old_global_parameter_state):
        """Check if the standard system needs changes after a state attribute is set.

        Parameters
        ----------
        standard_system : simtk.openmm.System
            The standard system before setting the attribute.
        attribute_name : str
            The name of the attribute that has just been set or retrieved.
        old_global_parameter_state : GlobalParameterState
            A copy of the composable state before the attribute was set.

        Returns
        -------
        need_changes : bool
            True if the standard system has to be updated, False if no change
            occurred.

        """
        # There are no attributes that can be set that can alter the standard system,
        # but if a parameter goes from defined to undefined, we should raise an error.
        old_attribute_value = getattr(old_global_parameter_state, attribute_name)
        new_attribute_value = getattr(self, attribute_name)
        if (old_attribute_value is None) != (new_attribute_value is None):
            err_msg = 'Cannot set the parameter {} in the system from {} to {}'.format(
                attribute_name, old_attribute_value, new_attribute_value)
            # Set back old value to maintain a consistent state in case the exception
            # is catched. If this attribute was associated to a GlobalParameterFunction,
            # we need to retrieve the original function object before setting.
            old_attribute_value = old_global_parameter_state._get_global_parameter_value(
                attribute_name, resolve_function=None)
            setattr(self, attribute_name, old_attribute_value)
            raise self._GLOBAL_PARAMETER_ERROR(err_msg)
        return False

    def _find_force_groups_to_update(self, context, current_context_state, memo):
        """Find the force groups whose energy must be recomputed after applying self.

        Parameters
        ----------
        context : Context
            The context, currently in `current_context_state`, that will
            be moved to this state.
        current_context_state : ThermodynamicState
            The full thermodynamic state of the given context. This is
            guaranteed to be compatible with self.
        memo : dict
            A dictionary that can be used by the state for memoization
            to speed up consecutive calls on the same context.

        Returns
        -------
        force_groups_to_update : set of int
            The indices of the force groups whose energy must be computed
            again after applying this state, assuming the context to be in
            `current_context_state`.
        """
        # Cache information about system force groups.
        # We create a dictionary "memo" mapping parameter_name -> list of force groups to update.
        if len(memo) == 0:
            system = context.getSystem()
            system_parameters = self._get_system_controlled_parameters(system, self._parameters_name_suffix)
            for force, parameter_name, _ in system_parameters:
                # Keep track of valid parameters only.
                if self._parameters[parameter_name] is not None:
                    try:
                        memo[parameter_name].append(force.getForceGroup())
                    except KeyError:
                        memo[parameter_name] = [force.getForceGroup()]

        # Find lambda parameters that will change.
        force_groups_to_update = set()
        for parameter_name, force_groups in memo.items():
            self_parameter_value = getattr(self, parameter_name)
            current_parameter_value = getattr(current_context_state, parameter_name)
            if self_parameter_value != current_parameter_value:
                force_groups_to_update.update(force_groups)
        return force_groups_to_update

    # -------------------------------------------------------------------------
    # Internal usage: Attributes handling
    # -------------------------------------------------------------------------

    @classmethod
    def _get_controlled_parameters(cls, parameters_name_suffix=None):
        """Return a set of the global parameters controlled by the state class.

        This is constructed dynamically by introspection gathering all the
        descriptors that belong to the GlobalParameter class.

        Parameters
        ----------
        parameters_name_suffix : str, optional
            If specified, this returns the set of parameter names with the
            name suffix.

        Returns
        -------
        controlled_parameters : dict of str -> GlobalParameter
            A map from the name of the controlled parameter to the
            GlobalParameter descriptor handling it.

        """
        if parameters_name_suffix is None:
            suffix = ''
        else:
            suffix = '_' + parameters_name_suffix
        # TODO just use inspect.getmembers when dropping Python 2 which automatically resolves the MRO.
        # controlled_parameters = {name + suffix: descriptor for name, descriptor in inspect.getmembers(cls)
        #                          if isinstance(descriptor, cls.GlobalParameter)}
        controlled_parameters = {name + suffix: descriptor for c in inspect.getmro(cls)
                                 for name, descriptor in c.__dict__.items()
                                 if isinstance(descriptor, cls.GlobalParameter)}
        return controlled_parameters

    def _validate_global_parameter(self, parameter_name, parameter_value, descriptor=None):
        """Return the validated parameter value using the descriptor validator.

        Parameters
        ----------
        parameter_name : str
            Parameter name (with eventual suffix) to validate.
        parameter_value : float
            Parameter value to validate. If a GlobalParameterFunction is associated
            to the parameter, this must be evaluated before calling this.
        descriptor : GlobalParameterState.GlobalParameter, optional
            If None, the functions automatically looks for the descriptor associated
            to this parameter and calls its validator (if any). This search is
            skipped if this argument is provided.

        Returns
        -------
        validated_value : float
            The validated value.

        Raises
        ------
        KeyError
            If parameter_name is not controlled by this state.
        """
        if descriptor is None:
            # Get the descriptors of all controlled parameters.
            controlled_parameters = self._get_controlled_parameters(self._parameters_name_suffix)
            # Call validator, before setting the parameter. This raises KeyError.
            descriptor = controlled_parameters[parameter_name]
        if descriptor.validator_func is not None:
            parameter_value = descriptor.validator_func(descriptor, self, parameter_value)
        return parameter_value

    def _get_global_parameter_value(self, parameter_name, descriptor=None, resolve_function=True):
        """Retrieve the current value of a global parameter.

        Parameters
        ----------
        parameter_name : str
            Parameter name (with eventual suffix) to validate.
        descriptor : GlobalParameterState.GlobalParameter, optional
            If None, and the parameter is associated to a GlobalParameterFunction,
            the functions automatically looks for the descriptor associated to this
            parameter and calls its validator (if any). This search is skipped if
            this argument is provided. Default is None.
        resolve_function : bool, optional
            If False and the parameter is associated to a GlobalParameterFunction,
            the function is not evaluated (and its result is not validated), and
            the GlobalParameterFunction object is returned instead. Default is True.

        Returns
        -------
        parameter_value : float
            The parameter value.

        Raises
        ------
        KeyError
            If parameter_name is not controlled by this state.
        """
        parameter_value = self._parameters[parameter_name]
        if resolve_function and isinstance(parameter_value, GlobalParameterFunction):
            parameter_value = parameter_value(self._function_variables)
            # If the value is generated through a mathematical expression,
            # we validate the value after the expression is evaluated rather
            # than on setting.
            parameter_value = self._validate_global_parameter(parameter_name, parameter_value, descriptor)
        return parameter_value

    def _set_global_parameter_value(self, parameter_name, new_value, descriptor=None):
        """Set the value of a global parameter.

        Parameters
        ----------
        parameter_name : str
            Parameter name (with eventual suffix) to validate.
        new_value : float or GlobalParameterFunction
            The new parameter value.
        descriptor : GlobalParameterState.GlobalParameter, optional
            If None, and the parameter is not a GlobalParameterFunction, the functions
            automatically looks for the descriptor associated to this parameter and
            calls its validator (if any). This search is skipped if this argument is
            provided.

        Raises
        ------
        KeyError
            If parameter_name is not controlled by this state.

        """
        # Check if the parameter is defined and raise KeyError otherwise.
        if parameter_name not in self._parameters:
            raise KeyError(parameter_name)
        # If the value is generated through a mathematical expression,
        # we validate the value after the expression is evaluated rather
        # than on setting.
        if not isinstance(new_value, GlobalParameterFunction):
            new_value = self._validate_global_parameter(parameter_name, new_value, descriptor)
        self._parameters[parameter_name] = new_value

    def __getattr__(self, key):
        """Handles global parameters with a suffix."""
        # __getattr__ is called only if the item is not found in the
        # usual ways, so we don't need to handle GlobalParameter here.
        try:
            parameter_value = self._get_global_parameter_value(key)
        except KeyError:
            # Parameter not found, fall back to normal behavior.
            parameter_value = super(GlobalParameterState, self).__getattribute__(key)
        return parameter_value

    def __setattr__(self, key, value):
        """Handles global parameters with a suffix."""
        # Check if the object has been initialized and we can
        # start resolving dynamically suffixed parameters.
        if '_parameters_name_suffix' in self.__dict__ and self._parameters_name_suffix is not None:
            try:
                self._set_global_parameter_value(key, value)
            except KeyError:
                pass
            else:
                return
        # This is not a "suffixed" parameter. Fallback to normal behavior.
        super(GlobalParameterState, self).__setattr__(key, value)

    @classmethod
    def _get_system_controlled_parameters(cls, system, parameters_name_suffix):
        """Yields the controlled global parameters defined in the System.

        Yields
        ------
        A tuple force, parameter_name, parameter_index for each supported
        lambda parameter.

        """
        searched_parameters = cls._get_controlled_parameters(parameters_name_suffix)

        # Retrieve all the forces with global supported parameters.
        for force_index in range(system.getNumForces()):
            force = system.getForce(force_index)
            try:
                n_global_parameters = force.getNumGlobalParameters()
            except AttributeError:
                continue
            for parameter_id in range(n_global_parameters):
                parameter_name = force.getGlobalParameterName(parameter_id)
                if parameter_name in searched_parameters:
                    yield force, parameter_name, parameter_id

    def __getstate__(self):
        """Return a dictionary representation of the state."""
        serialization = dict(
            parameters={},
            function_variables=self._function_variables.copy(),
            parameters_name_suffix=self._parameters_name_suffix
        )
        # Copy parameters. We serialize the parameters with their original name
        # (i.e., without suffix) because we'll pass them to _initialize().
        if self._parameters_name_suffix is None:
            suffix = ''
        else:
            suffix = '_' + self._parameters_name_suffix
        for parameter_name in self._get_controlled_parameters():
            parameter_value = self._parameters[parameter_name + suffix]
            # Convert global parameter function into string expressions.
            if isinstance(parameter_value, GlobalParameterFunction):
                parameter_value = parameter_value._expression
            serialization['parameters'][parameter_name] = parameter_value
        return serialization

    def __setstate__(self, serialization):
        """Set the state from a dictionary representation."""
        parameters = serialization['parameters']
        # parameters_name_suffix is optional for backwards compatibility since openmmtools 0.16.0.
        parameters_name_suffix = serialization.get('parameters_name_suffix', None)
        # Global parameter functions has been added in openmmtools 0.17.0.
        function_variables = serialization.get('function_variables', {})

        # Temporarily store global parameter functions.
        global_parameter_functions = {}
        for parameter_name, value in parameters.items():
            if isinstance(value, str):
                global_parameter_functions[parameter_name] = value
                parameters[parameter_name] = None

        # Initialize parameters and add all function variables.
        self._initialize(parameters_name_suffix=parameters_name_suffix, **parameters)
        for variable_name, value in function_variables.items():
            self.set_function_variable(variable_name, value)

        # Add global parameter functions.
        if parameters_name_suffix is not None:
            parameters_name_suffix = '_' + parameters_name_suffix
        else:
            parameters_name_suffix = ''
        for parameter_name, expression in global_parameter_functions.items():
            setattr(self, parameter_name + parameters_name_suffix, GlobalParameterFunction(expression))

    # -------------------------------------------------------------------------
    # Internal usage: Initialization
    # -------------------------------------------------------------------------

    def _initialize(self, parameters_name_suffix=None, **kwargs):
        """Initialize the state.

        It takes the global parameters and their values as keywords arguments.
        Controlled parameters that are not passed are left undefined (i.e. are
        set to None).
        """
        self._function_variables = {}
        # Get controlled parameters from introspection.
        controlled_parameters = set(self._get_controlled_parameters())

        # Check for unknown parameters
        unknown_parameters = set(kwargs) - controlled_parameters
        if len(unknown_parameters) > 0:
            err_msg = "Unknown parameters {}".format(unknown_parameters)
            raise self._GLOBAL_PARAMETER_ERROR(err_msg)

        # Append suffix to parameters before storing them internally.
        if parameters_name_suffix is not None:
            kwargs = {key + '_' + parameters_name_suffix: value for key, value in kwargs.items()}
            controlled_parameters = {key + '_' + parameters_name_suffix for key in controlled_parameters}

        # Default value for all parameters is None.
        self._parameters = dict.fromkeys(controlled_parameters, None)

        # This signals to __setattr__ that we can start resolving dynamically
        # suffixed parameters so it should be the last direct assignment.
        self._parameters_name_suffix = parameters_name_suffix

        # Update parameters with constructor arguments.
        for parameter_name, value in kwargs.items():
            setattr(self, parameter_name, value)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
    # doctest.run_docstring_examples(CompoundThermodynamicState, globals())
