#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Provide cache classes to handle creation of OpenMM Context objects.

"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import re
import copy
import collections

from simtk import openmm, unit

from openmmtools import integrators


# =============================================================================
# GENERAL LRU CACHE
# =============================================================================

class LRUCache(object):
    """A simple LRU cache with a dictionary-like interface that supports maximum capacity and expiration.

    It can be configured to have a maximum number of elements (capacity)
    and an element expiration (time_to_live) measured in number of accesses
    to the cache. Both read and write operations count as an access, but
    only successful reads (i.e. those not raising KeyError) increment the
    counter.

    Parameters
    ----------
    capacity : int, optional
        Maximum number of elements in the cache. When set to None, the
        cache has infinite capacity (default is None).
    time_to_live : int, optional
        If an element is not accessed after time_to_live read/write
        operations, the element is removed. When set to None, elements do
        not have an expiration (default is None).

    Attributes
    ----------
    capacity
    time_to_live

    Examples
    --------
    >>> cache = LRUCache(capacity=2, time_to_live=3)

    When the capacity is exceeded, the least recently used element is
    removed.

    >>> cache['1'] = 1
    >>> cache['2'] = 2
    >>> elem = cache['1']  # read '1', now '2' is the least recently used
    >>> cache['3'] = 3
    >>> len(cache)
    2
    >>> '2' in cache
    False

    After time_to_live read/write operations an element is deleted if
    it is not used.

    >>> elem = cache['3']  # '3' is used, counter is reset
    >>> elem = cache['1']  # access 1
    >>> elem = cache['1']  # access 2
    >>> elem = cache['1']  # access 3
    >>> len(cache)
    1
    >>> '3' in cache
    False

    """

    def __init__(self, capacity=None, time_to_live=None):
        self._data = collections.OrderedDict()
        self._capacity = capacity
        self._ttl = time_to_live
        self._n_access = 0

    @property
    def capacity(self):
        """Maximum number of elements that can be cached.

        If None, the capacity is unlimited.

        """
        return self._capacity

    @capacity.setter
    def capacity(self, new_capacity):
        # Remove excess elements
        while len(self._data) > new_capacity:
            self._data.popitem(last=False)
        self._capacity = new_capacity

    @property
    def time_to_live(self):
        """Number of read/write operations before an cached element expires.

        If None, elements have no expiration.

        """
        return self._ttl

    @time_to_live.setter
    def time_to_live(self, new_time_to_live):
        # Update entries only if we are changing the ttl.
        if new_time_to_live == self._ttl:
            return

        # Update expiration of cache entries.
        for entry in self._data.values():
            # If there was no time to live before, just let entries
            # expire in new_time_to_live accesses
            if self._ttl is None:
                entry.expiration = self._n_access + new_time_to_live
            # If we don't want expiration anymore, delete the field.
            # This way we save memory in case there are a lot of entries.
            elif new_time_to_live is None:
                del entry.expiration
            # Otherwise just add/subtract the difference.
            else:
                entry.expiration += new_time_to_live - self._ttl

        # Purge cache only if there is a time to live.
        if new_time_to_live is not None:
            self._remove_expired()
        self._ttl = new_time_to_live

    def empty(self):
        """Purge the cache."""
        self._data = collections.OrderedDict()

    def __getitem__(self, key):
        # When we access data, push element at the
        # end to make it the most recently used.
        entry = self._data.pop(key)
        self._data[key] = entry

        # We increment the number of accesses only on successful reads.
        self._n_access += 1

        # Update expiration and cleanup expired values.
        if self._ttl is not None:
            entry.expiration = self._n_access + self._ttl
            self._remove_expired()
        return entry.value

    def __setitem__(self, key, value):
        self._n_access += 1

        # When we access data, push element at the
        # end to make it the most recently used.
        try:
            self._data.pop(key)
        except KeyError:
            # Remove first item if we hit maximum capacity.
            if self._capacity is not None and len(self._data) >= self._capacity:
                self._data.popitem(last=False)

        # Determine expiration and clean up expired.
        if self._ttl is None:
            ttl = None
        else:
            ttl = self._ttl + self._n_access
            self._remove_expired()
        self._data[key] = _CacheEntry(value, ttl)

    def __len__(self):
        return len(self._data)

    def __contains__(self, item):
        return item in self._data

    def __iter__(self):
        return self._data.__iter__()

    def _remove_expired(self):
        """Remove all expired cache entries.

        Assumes that entries were created with an expiration attribute.

        """
        keys_to_remove = set()
        for key, entry in self._data.items():
            if entry.expiration <= self._n_access:
                keys_to_remove.add(key)
            else:
                # Later entries have been accessed later
                # and they surely haven't expired yet.
                break
        for key in keys_to_remove:
            del self._data[key]


# =============================================================================
# GENERAL CONTEXT CACHE
# =============================================================================

class ContextCache(object):
    """LRU cache hosting the minimum amount of incompatible Contexts.

    Two Contexts are compatible if they are in a compatible ThermodynamicState,
    and have compatible integrators. In general, two integrators are compatible
    if they have the same serialized state, but ContextCache can decide to store
    a single Context to optimize memory when two integrators differ by only few
    parameters that can be set after the Context is initialized. These parameters
    include all the global variables defined by a ``CustomIntegrator``.

    You can force ``ContextCache`` to consider an integrator global variable incompatible
    by adding it to the blacklist ``ContextCache.INCOMPATIBLE_INTEGRATOR_ATTRIBUTES``.
    Similarly, you can add other attributes that should be considered compatible
    through the whitelist ``ContextCache.COMPATIBLE_INTEGRATOR_ATTRIBUTES``. If
    an attribute in that dictionary is not found in the integrator, the cache
    will search for a corresponding getter and setter.

    Parameters
    ----------
    platform : simtk.openmm.Platform, optional
        The OpenMM platform to use to create Contexts. If None, OpenMM
        tries to select the fastest one available (default is None).
    **kwargs
        Parameters to pass to the underlying LRUCache constructor such
        as capacity and time_to_live.

    Attributes
    ----------
    platform
    capacity
    time_to_live

    Warnings
    --------
    Python instance attributes are not copied when ``ContextCache.get_context()``
    is called. You can force this by setting adding them to the whitelist
    ``ContextCache.COMPATIBLE_INTEGRATOR_ATTRIBUTES``, but if modifying your
    Python attributes won't modify the OpenMM serialization, this will likely cause
    problems so this is discouraged unless you know exactly what you are doing.

    Examples
    --------
    >>> from simtk import unit
    >>> from openmmtools import testsystems
    >>> from openmmtools.states import ThermodynamicState
    >>> alanine = testsystems.AlanineDipeptideExplicit()
    >>> thermodynamic_state = ThermodynamicState(alanine.system, 310*unit.kelvin)
    >>> time_step = 1.0*unit.femtosecond

    Two compatible thermodynamic states generate only a single cached Context.
    ContextCache can also (in few explicitly supported cases) recycle the same
    Context even if the integrators differ by some parameters.

    >>> context_cache = ContextCache()
    >>> context1, integrator1 = context_cache.get_context(thermodynamic_state,
    ...                                                   openmm.VerletIntegrator(time_step))
    >>> thermodynamic_state.temperature = 300*unit.kelvin
    >>> time_step2 = 2.0*unit.femtosecond
    >>> context2, integrator2 = context_cache.get_context(thermodynamic_state,
    ...                                                   openmm.VerletIntegrator(time_step2))
    >>> id(context1) == id(context2)
    True
    >>> len(context_cache)
    1

    When we switch to NPT the states are not compatible and so neither the
    Contexts are.

    >>> integrator2 = openmm.VerletIntegrator(2.0*unit.femtosecond)
    >>> thermodynamic_state_npt = copy.deepcopy(thermodynamic_state)
    >>> thermodynamic_state_npt.pressure = 1.0*unit.atmosphere
    >>> context3, integrator3 = context_cache.get_context(thermodynamic_state_npt,
    ...                                                   openmm.VerletIntegrator(time_step))
    >>> id(context1) == id(context3)
    False
    >>> len(context_cache)
    2

    You can set a capacity and a time to live for contexts like in a normal
    LRUCache.

    >>> context_cache = ContextCache(capacity=1, time_to_live=5)
    >>> context2, integrator2 = context_cache.get_context(thermodynamic_state,
    ...                                                   openmm.VerletIntegrator(time_step))
    >>> context3, integrator3 = context_cache.get_context(thermodynamic_state_npt,
    ...                                                   openmm.VerletIntegrator(time_step))
    >>> len(context_cache)
    1

    See Also
    --------
    LRUCache
    states.ThermodynamicState.is_state_compatible

    """

    def __init__(self, platform=None, **kwargs):
        self._platform = platform
        self._lru = LRUCache(**kwargs)

    def __len__(self):
        return len(self._lru)

    @property
    def platform(self):
        """The OpenMM platform to use to create Contexts.

        If None, OpenMM tries to select the fastest one available. This
        can be set only if the cache is empty.

        """
        return self._platform

    @platform.setter
    def platform(self, new_platform):
        if len(self._lru) > 0:
            raise RuntimeError('Cannot change platform of a non-empty ContextCache')
        self._platform = new_platform

    @property
    def capacity(self):
        """The maximum number of Context cached.

        If None, the capacity is unlimited.

        """
        return self._lru.capacity

    @capacity.setter
    def capacity(self, new_capacity):
        self._lru.capacity = new_capacity

    @property
    def time_to_live(self):
        """The Contexts expiration date in number of accesses to the LRUCache.

        If None, Contexts do not expire.

        """
        return self._lru.time_to_live

    @time_to_live.setter
    def time_to_live(self, new_time_to_live):
        self._lru.time_to_live = new_time_to_live

    def empty(self):
        """Clear up cache and remove all Contexts."""
        self._lru.empty()

    def get_context(self, thermodynamic_state, integrator=None):
        """Return a context in the given thermodynamic state.

        In general, the Context must be considered newly initialized. This
        means that positions and velocities must be set afterwards.

        If the integrator is not provided, this will search the cache for
        any Context in the given ThermodynamicState, regardless of its
        integrator. In this case, the method guarantees that two consecutive
        calls with the same thermodynamic state will retrieve the same context.

        This creates a new Context if no compatible one has been cached.
        If a compatible Context exists, the ThermodynamicState is applied
        to it, and the Context integrator state is changed to match the
        one passed as an argument. As a consequence, the returned integrator
        is guaranteed to be in the same state as the one provided, but it
        can be a different instance. This is to minimize the number of
        Contexts objects cached that use the same or very similar integrator.

        Parameters
        ----------
        thermodynamic_state : states.ThermodynamicState
            The thermodynamic state of the system.
        integrator : simtk.openmm.Integrator, optional
            The integrator for the context (default is None).

        Returns
        -------
        context : simtk.openmm.Context
            The context in the given thermodynamic system.
        context_integrator : simtk.openmm.Integrator
            The integrator to be used to propagate the Context. Can be
            a difference instance from the one passed as an argument.

        Warnings
        --------
        Python instance attributes are not copied when ``get_context()``
        is called. You can force this by setting adding them to the whitelist
        ``ContextCache.COMPATIBLE_INTEGRATOR_ATTRIBUTES``, but if modifying the
        attributes won't modify the OpenMM serialization, this will likely cause
        problems so this is discouraged unless you know exactly what you're doing.

        """
        context = None

        # If the user requires a specific integrator, look for one that matches.
        if integrator is None:
            thermodynamic_state_id = self._generate_state_id(thermodynamic_state)
            matching_context_ids = [context_id for context_id in self._lru
                                    if context_id[0] == thermodynamic_state_id]
            if len(matching_context_ids) == 0:
                # We have to create a new Context.
                integrator = self._get_default_integrator(thermodynamic_state.temperature)
            elif len(matching_context_ids) == 1:
                # Only one match.
                context = self._lru[matching_context_ids[0]]
            else:
                # Multiple matches, prefer the non-default Integrator.
                # Always pick the least recently used to make two consective
                # calls retrieving the same integrator.
                for context_id in reversed(matching_context_ids):
                    if context_id[1] != self._default_integrator_id():
                        context = self._lru[context_id]
                        break

        if context is None:
            # Determine the Context id matching the pair state-integrator.
            context_id = self._generate_context_id(thermodynamic_state, integrator)

            # Search for previously cached compatible Contexts or create new one.
            try:
                context = self._lru[context_id]
            except KeyError:
                context = thermodynamic_state.create_context(integrator, self._platform)
            self._lru[context_id] = context
        context_integrator = context.getIntegrator()

        # Update state of system and integrator of the cached context.
        # We don't have to copy the state of the integrator if the user
        # didn't ask for a specific one.
        if integrator is not None:
            self._copy_integrator_state(integrator, context_integrator)
        thermodynamic_state.apply_to_context(context)
        return context, context_integrator

    def __getstate__(self):
        if self.platform is not None:
            platform_serialization = self.platform.getName()
        else:
            platform_serialization = None
        return dict(platform=platform_serialization, capacity=self.capacity,
                    time_to_live=self.time_to_live)

    def __setstate__(self, serialization):
        if serialization['platform'] is None:
            self._platform = None
        else:
            self._platform = openmm.Platform.getPlatformByName(serialization['platform'])
        self._lru = LRUCache(serialization['capacity'], serialization['time_to_live'])

    # -------------------------------------------------------------------------
    # Internal usage
    # -------------------------------------------------------------------------

    # Each element is the name of the integrator attribute used before
    # get/set, and its standard value used to check for compatibility.
    COMPATIBLE_INTEGRATOR_ATTRIBUTES = {
        'StepSize': 0.001,
        'ConstraintTolerance': 1e-05,
        'Temperature': 273,
        'Friction': 5,
        'RandomNumberSeed': 0,
    }

    INCOMPATIBLE_INTEGRATOR_ATTRIBUTES = {
        '_restorable__class_hash',
    }

    @classmethod
    def _check_integrator_compatibility_configuration(cls):
        """Verify that the user didn't specify the same attributes as both compatible and incompatible."""
        shared_attributes = set(cls.COMPATIBLE_INTEGRATOR_ATTRIBUTES)
        shared_attributes = shared_attributes.intersection(cls.INCOMPATIBLE_INTEGRATOR_ATTRIBUTES)
        if len(shared_attributes) != 0:
            raise RuntimeError('These integrator attributes have been specified both as '
                               'compatible and incompatible: {}'.format(shared_attributes))

    @classmethod
    def _set_integrator_compatible_variables(cls, integrator, reference_value):
        """Set all the global variables to the specified reference.

        If the argument reference_value is another integrator, the global
        variables will be copied. If integrator is not a CustomIntegrator,
        the function has no effect.

        The function doesn't copy the global variables that are included in
        the blacklist INCOMPATIBLE_INTEGRATOR_ATTRIBUTES.
        """
        # Check if the integrator has no global variables.
        try:
            n_global_variables = integrator.getNumGlobalVariables()
        except AttributeError:
            return
        # Check if we'll have to copy the values from a reference integrator.
        is_reference_integrator = isinstance(reference_value, integrator.__class__)

        for global_variable_idx in range(n_global_variables):
            # Do not set variables that should be incompatible.
            global_variable_name = integrator.getGlobalVariableName(global_variable_idx)
            if global_variable_name in cls.INCOMPATIBLE_INTEGRATOR_ATTRIBUTES:
                continue
            # Either copy the value from the reference integrator or just set it.
            if is_reference_integrator:
                value = reference_value.getGlobalVariable(global_variable_idx)
            else:
                value = reference_value
            integrator.setGlobalVariable(global_variable_idx, value)


    @classmethod
    def _copy_integrator_state(cls, copied_integrator, integrator):
        """Copy the compatible parameters of copied_integrator to integrator.

        Simply using __getstate__ and __setstate__ doesn't work because
        __setstate__ set also the bound Context.

        We can assume the two integrators are of the same class since
        get_context() found that they match the has.

        """
        # Check that there are no contrasting settings for the attribute compatibility.
        cls._check_integrator_compatibility_configuration()

        # Restore temperature getter/setter before copying attributes.
        integrators.ThermostatedIntegrator.restore_interface(integrator)
        integrators.ThermostatedIntegrator.restore_interface(copied_integrator)
        assert integrator.__class__ == copied_integrator.__class__

        # Copy all compatible global variables.
        cls._set_integrator_compatible_variables(integrator, copied_integrator)

        # Copy other compatible attributes through getters/setters.
        for attribute in cls.COMPATIBLE_INTEGRATOR_ATTRIBUTES:
            try:  # getter/setter
                value = getattr(copied_integrator, 'get' + attribute)()
            except AttributeError:
                pass
            else:  # getter/setter
                getattr(integrator, 'set' + attribute)(value)

    @classmethod
    def _standardize_integrator(cls, integrator):
        """Return a standard copy of the integrator.

        This is used to determine if the same context can be used with
        different integrators that differ by only compatible parameters.

        """
        # Check that there are no contrasting settings for the attribute compatibility.
        cls._check_integrator_compatibility_configuration()

        standard_integrator = copy.deepcopy(integrator)
        integrators.ThermostatedIntegrator.restore_interface(standard_integrator)

        # Set all compatible global variables to 0, except those in the blacklist.
        cls._set_integrator_compatible_variables(standard_integrator, 0.0)

        # Copy other compatible attributes through getters/setters overwriting
        # eventual global variables with a different standard value.
        for attribute, std_value in cls.COMPATIBLE_INTEGRATOR_ATTRIBUTES.items():
            try:  # setter
                getattr(standard_integrator, 'set' + attribute)(std_value)
            except AttributeError:
                # Try to set CustomIntegrator global variable
                try:
                    standard_integrator.setGlobalVariableByName(attribute, std_value)
                except Exception:
                    pass
        return standard_integrator

    @staticmethod
    def _generate_state_id(thermodynamic_state):
        """Return a unique key for the ThermodynamicState."""
        # We take advantage of the cached _standard_system_hash property
        # to generate a compatible hash for the thermodynamic state.
        return thermodynamic_state._standard_system_hash

    @classmethod
    def _generate_integrator_id(cls, integrator):
        """Return a unique key for the given Integrator."""
        standard_integrator = cls._standardize_integrator(integrator)
        xml_serialization = openmm.XmlSerializer.serialize(standard_integrator)
        # Ignore per-DOF variables for the purpose of hashing.
        if isinstance(integrator, openmm.CustomIntegrator):
            tag_iter = re.finditer(r'PerDofVariables>', xml_serialization)
            try:
                open_tag_index = next(tag_iter).start() - 1
            except StopIteration:  # No DOF variables.
                pass
            else:
                close_tag_index = next(tag_iter).end() + 1
                xml_serialization = xml_serialization[:open_tag_index] + xml_serialization[close_tag_index:]
        return xml_serialization.__hash__()

    @classmethod
    def _generate_context_id(cls, thermodynamic_state, integrator):
        """Return a unique key for a context in the given state.

        We return a tuple containing the ThermodynamicState hash and the
        the serialization of the Integrator. Keeping the two separated
        makes it possible to search for Contexts in a given state regardless
        of the integrator.

        """
        state_id = cls._generate_state_id(thermodynamic_state)
        integrator_id = cls._generate_integrator_id(integrator)
        return state_id, integrator_id

    @staticmethod
    def _get_default_integrator(temperature):
        """Return a new instance of the default integrator."""
        # Use a likely-to-be-used Integrator.
        return integrators.GeodesicBAOABIntegrator(temperature=temperature)

    @classmethod
    def _default_integrator_id(cls):
        """Return the unique key of the default integrator."""
        if cls._cached_default_integrator_id is None:
            default_integrator = cls._get_default_integrator(300*unit.kelvin)
            default_integrator_id = cls._generate_integrator_id(default_integrator)
            cls._cached_default_integrator_id = default_integrator_id
        return cls._cached_default_integrator_id
    _cached_default_integrator_id = None


# =============================================================================
# DUMMY CONTEXT CACHE
# =============================================================================

class DummyContextCache(object):
    """A dummy ContextCache which always create a new Context.

    Parameters
    ----------
    platform : simtk.openmm.Platform, optional
        The OpenMM platform to use. If None, OpenMM tries to select
        the fastest one available (default is None).

    Attributes
    ----------
    platform : simtk.openmm.Platform
        The OpenMM platform to use. If None, OpenMM tries to select
        the fastest one available.

    """
    def __init__(self, platform=None):
        self.platform = platform

    def get_context(self, thermodynamic_state, integrator):
        """Create a new context in the given thermodynamic state."""
        context = thermodynamic_state.create_context(integrator, self.platform)
        return context, integrator

    def __getstate__(self):
        if self.platform is not None:
            platform_serialization = self.platform.getName()
        else:
            platform_serialization = None
        return dict(platform=platform_serialization)

    def __setstate__(self, serialization):
        if serialization['platform'] is None:
            self.platform = None
        else:
            self.platform = openmm.Platform.getPlatformByName(serialization['platform'])


# =============================================================================
# GLOBAL CONTEXT CACHE
# =============================================================================

global_context_cache = ContextCache(capacity=None, time_to_live=None)
"""A shared ContextCache that minimizes Context object creating when using MCMCMove."""

# =============================================================================
# CACHE ENTRY (MODULE INTERNAL USAGE)
# =============================================================================

class _CacheEntry(object):
    """A cache entry holding an optional expiration attribute."""
    def __init__(self, value, expiration=None):
        self.value = value

        # We create the self.expiration attribute only if requested
        # to save memory in case the cache stores a lot of entries.
        if expiration is not None:
            self.expiration = expiration


if __name__ == '__main__':
    import doctest
    doctest.testmod()
