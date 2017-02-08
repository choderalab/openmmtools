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

import copy
import collections

from simtk import openmm

from openmmtools import utils


# =============================================================================
# GENERAL LRU CACHE
# =============================================================================

class LRUCache(object):
    """A simple LRU cache.

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
    they are associated to the same platform, and have compatible integrators.
    In general, two integrators are compatible if they have the same serialized
    state, but ContextCache can decide to store a single Context to optimize
    memory when two integrators differ by only few parameters that can be set
    after the Context is initialized.

    Parameters
    ----------
    **kwargs
        Parameters to pass to the underlying LRUCache constructor such
        as capacity and time_to_live.

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

    def __init__(self, **kwargs):
        self._lru = LRUCache(**kwargs)

    def __len__(self):
        return len(self._lru)

    def get_context(self, thermodynamic_state, integrator, platform=None):
        """Return a context in the given thermodynamic state.

        In general, the Context must be considered newly initialized. This
        means that positions and velocities must be set afterwards.

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
        integrator : simtk.openmm.Integrator
            The integrator for the context.
        platform : simtk.openmm.Platform, optional
            The OpenMM platform to use. If None, OpenMM tries to select
            the fastest one available (default is None).

        Returns
        -------
        context : simtk.openmm.Context
            The context in the given thermodynamic system.
        context_integrator : simtk.openmm.Integrator
            The integrator to be used to propagate the Context. Can be
            a difference instance from the one passed as an argument.

        """
        context_id = self._generate_context_id(thermodynamic_state, integrator,
                                               platform)
        try:
            context = self._lru[context_id]
        except KeyError:
            context = thermodynamic_state.create_context(integrator, platform)
            self._lru[context_id] = context
        context_integrator = context.getIntegrator()

        # Update state of system and integrator of the cached context.
        self._copy_integrator_state(integrator, context_integrator)
        thermodynamic_state.apply_to_context(context)
        return context, context_integrator

    # -------------------------------------------------------------------------
    # Internal usage
    # -------------------------------------------------------------------------

    # Each element is the name of the integrator attribute used before
    # get/set, and its standard value used to check for compatibility.
    _COMPATIBLE_INTEGRATOR_ATTRIBUTES = {
        'StepSize': 0.001,
        'ConstraintTolerance': 1e-05,
        'Temperature': 273,
        'Friction': 5,
        'RandomNumberSeed': 0
    }

    @classmethod
    def _copy_integrator_state(cls, copied_integrator, integrator):
        """Copy the supported attributes of copied_integrator to integrator.

        Simply using __getstate__ and __setstate__ doesn't work because
        __setstate__ set also the bound Context.

        """
        assert type(integrator) == type(copied_integrator)
        for attribute in cls._COMPATIBLE_INTEGRATOR_ATTRIBUTES:
            try:
                value = getattr(copied_integrator, 'get' + attribute)()
            except AttributeError:
                pass
            else:
                getattr(integrator, 'set' + attribute)(value)

    @classmethod
    def _standardize_integrator(cls, integrator):
        """Return a standard copy of the integrator.

        This is used to determine if the same context can be used with
        different integrators that differ by only few supported parameters.

        """
        standard_integrator = copy.deepcopy(integrator)
        for attribute, std_value in cls._COMPATIBLE_INTEGRATOR_ATTRIBUTES.items():
            try:
                getattr(standard_integrator, 'set' + attribute)(std_value)
            except AttributeError:
                pass
        return standard_integrator

    @classmethod
    def _generate_context_id(cls, thermodynamic_state, integrator, platform):
        """Return the unique string key of the context for this state."""
        if platform is None:
            platform = utils.get_fastest_platform()

        # We take advantage of the cached _standard_system_hash property
        # to generate a compatible hash for the thermodynamic state.
        state_id = str(thermodynamic_state._standard_system_hash)
        standard_integrator = cls._standardize_integrator(integrator)
        integrator_id = openmm.XmlSerializer.serialize(standard_integrator)
        platform_id = platform.getName()

        return state_id + integrator_id + platform_id


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
