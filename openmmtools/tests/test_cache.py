#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test Context cache classes in cache.py.

"""

# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

from openmmtools.cache import *


# =============================================================================
# TEST LRU CACHE
# =============================================================================

def test_lru_cache_cache_entry_unpacking():
    """Values in LRUCache are unpacked from CacheEntry."""
    cache = LRUCache(capacity=5)
    cache['first'] = 1
    assert cache['first'] == 1

    # When we don't require a time-to-leave, there's
    # no expiration attribute in the cache entry.
    assert not hasattr(cache._data['first'], 'expiration')


def test_lru_cache_maximum_capacity():
    """Maximum size of LRUCache is handled correctly."""
    cache = LRUCache(capacity=2)
    cache['first'] = 1
    cache['second'] = 2
    assert len(cache) == 2
    cache['third'] = 3
    assert len(cache) == 2
    assert 'first' not in cache

    # Test infinite capacity
    cache = LRUCache()
    for i in range(100):
        cache[str(i)] = i
    assert len(cache) == 100


def test_lru_cache_eliminate_least_recently_used():
    """LRUCache deletes LRU element when size exceeds capacity."""
    cache = LRUCache(capacity=3)
    cache['first'] = 1
    cache['second'] = 2

    # We access 'first' through setting, so that it becomes the LRU.
    cache['first'] = 1
    cache['third'] = 3
    cache['fourth'] = 4  # Here size exceed capacity.
    assert len(cache) == 3
    assert 'second' not in cache

    # We access 'first' through getting now.
    cache['first']
    cache['fifth'] = 5  # Size exceed capacity.
    assert len(cache) == 3
    assert 'third' not in cache


def test_lru_cache_access_to_live():
    """LRUCache deletes element after specified number of accesses."""
    def almost_expire_first():
        cache['first'] = 1  # Update expiration.
        for _ in range(ttl - 1):
            cache['second']
            assert 'first' in cache

    ttl = 3
    cache = LRUCache(capacity=2, time_to_live=ttl)
    cache['first'] = 1
    cache['second'] = 2  # First access.
    assert cache._data['first'].expiration == ttl + 1
    cache['first']  # Expiration gets updated.
    assert cache._data['first'].expiration == ttl + 3

    # At the ttl-th read access, 'first' gets deleted.
    almost_expire_first()
    cache['second']
    assert 'second' in cache
    assert 'first' not in cache

    # The same happen at the ttl-th write access.
    almost_expire_first()
    cache['second'] = 2
    assert 'second' in cache
    assert 'first' not in cache

    # If we touch at the last minute 'first', it remains in memory.
    almost_expire_first()
    cache['first']
    assert 'second' in cache
    assert 'first' in cache
