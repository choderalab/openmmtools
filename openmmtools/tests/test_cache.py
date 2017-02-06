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

import itertools

from simtk import unit, openmm

from openmmtools import testsystems, states

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


# =============================================================================
# TEST CONTEXT CACHE
# =============================================================================

class TestContextCache(object):
    """Test ContextCache class."""

    @classmethod
    def setup_class(cls):
        """Create the thermodynamic states used in the test suite."""
        water_test = testsystems.WaterBox(box_edge=2.0*unit.nanometer)
        cls.water_300k = states.ThermodynamicState(water_test.system, 300*unit.kelvin)
        cls.water_310k = states.ThermodynamicState(water_test.system, 310*unit.kelvin)
        cls.water_310k_1atm = states.ThermodynamicState(water_test.system, 310*unit.kelvin,
                                                        1*unit.atmosphere)

        cls.verlet_2fm = openmm.VerletIntegrator(2.0*unit.femtosecond)
        cls.verlet_3fm = openmm.VerletIntegrator(3.0*unit.femtosecond)
        cls.langevin_2fm = openmm.LangevinIntegrator(310*unit.kelvin, 5.0/unit.picosecond,
                                                     2.0*unit.femtosecond)

    def test_generate_compatible_context_key(self):
        """Context._generate_context_id creates same id for compatible contexts."""
        compatible_states = [self.water_300k, self.water_310k]
        compatible_integrators = [self.verlet_2fm, self.verlet_3fm]
        compatible_platforms = [None, utils.get_fastest_platform()]

        all_keys = set()
        for state, integrator, platform in itertools.product(compatible_states,
                                                             compatible_integrators,
                                                             compatible_platforms):
            all_keys.add(ContextCache._generate_context_id(state, integrator, platform))
        assert len(all_keys) == 1

    def test_generate_incompatible_context_key(self):
        """Context._generate_context_id creates different ids for incompatible contexts."""
        incompatible_states = [self.water_310k, self.water_310k_1atm]
        incompatible_integrators = [self.verlet_2fm, self.langevin_2fm]
        incompatible_platforms = [openmm.Platform.getPlatform(i)
                                  for i in range(openmm.Platform.getNumPlatforms())]

        all_keys = set()
        for state, integrator, platform in itertools.product(incompatible_states,
                                                             incompatible_integrators,
                                                             incompatible_platforms):
            all_keys.add(ContextCache._generate_context_id(state, integrator, platform))
        assert len(all_keys) == 4 * len(incompatible_platforms)
