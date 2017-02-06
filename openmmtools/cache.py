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
    """A simple LRU cache."""

    def __init__(self, capacity=None, time_to_live=None):
        self._data = collections.OrderedDict()
        self._capacity = capacity
        self._ttl = time_to_live
        self._n_access = 0

    def __getitem__(self, key):
        self._n_access += 1

        # When we access data, push element at the
        # end to make it the most recently used.
        entry = self._data.pop(key)
        self._data[key] = entry

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
    """Cache the minimum amount of incompatible Contexts."""

    def __init__(self, **kwargs):
        self._lru = LRUCache(**kwargs)

    def __len__(self):
        return len(self._lru)

    def get_context(self, thermodynamic_state, integrator, platform=None):
        """Return a context in the given thermodynamic state.

        Creates a new context if no compatible one has been cached.

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
