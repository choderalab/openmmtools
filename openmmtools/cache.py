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

import collections


# =============================================================================
# GENERAL LRU CACHE
# =============================================================================

class LRUCache(object):
    """A simple LRU cache."""

    def __init__(self, capacity, time_to_live=None):
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
            if len(self._data) >= self._capacity:
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
