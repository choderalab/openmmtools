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

import nose
from simtk import unit

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


def test_lru_cache_capacity_property():
    """When capacity is reduced, LRUCache delete elements."""
    capacity = 4
    cache = LRUCache(capacity=capacity)
    for i in range(capacity):
        cache[str(i)] = 1
    cache.capacity = 1
    assert len(cache) == 1
    assert cache.capacity == 1
    assert str(capacity-1) in cache


def test_lru_cache_time_to_live_property():
    """Decreasing the time to live updates the expiration of elements."""
    cache = LRUCache(time_to_live=50)
    for i in range(4):
        cache[str(i)] = i
    assert len(cache) == 4
    cache.time_to_live = 1
    assert len(cache) == 1
    assert cache.time_to_live == 1
    assert '3' in cache


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

        cls.verlet_2fs = openmm.VerletIntegrator(2.0*unit.femtosecond)
        cls.verlet_3fs = openmm.VerletIntegrator(3.0*unit.femtosecond)
        cls.langevin_2fs_310k = openmm.LangevinIntegrator(310*unit.kelvin, 5.0/unit.picosecond,
                                                          2.0*unit.femtosecond)

        cls.compatible_states = [cls.water_300k, cls.water_310k]
        cls.compatible_integrators = [cls.verlet_2fs, cls.verlet_3fs]

        cls.incompatible_states = [cls.water_310k, cls.water_310k_1atm]
        cls.incompatible_integrators = [cls.verlet_2fs, cls.langevin_2fs_310k]

    @classmethod
    def cache_incompatible_contexts(cls, cache):
        """Return the number of contexts created."""
        context_ids = set()
        for state, integrator in itertools.product(cls.incompatible_states,
                                                   cls.incompatible_integrators):
            # Avoid binding same integrator to multiple contexts
            integrator = copy.deepcopy(integrator)
            context, context_integrator = cache.get_context(state, integrator)
            context_ids.add(id(context))
        return len(context_ids)

    def test_copy_integrator_state(self):
        """ContextCache._copy_integrator_state correctly copies state."""
        langevin1 = copy.deepcopy(self.langevin_2fs_310k)
        langevin2 = openmm.LangevinIntegrator(300*unit.kelvin, 8.0/unit.picosecond,
                                              3.0*unit.femtosecond)
        assert langevin1.__getstate__() != langevin2.__getstate__()
        ContextCache._copy_integrator_state(langevin1, langevin2)
        assert langevin1.__getstate__() == langevin2.__getstate__()

    def test_generate_compatible_context_key(self):
        """Context._generate_context_id creates same id for compatible contexts."""
        all_ids = set()
        for state, integrator in itertools.product(self.compatible_states,
                                                   self.compatible_integrators):
            all_ids.add(ContextCache._generate_context_id(state, integrator))
        assert len(all_ids) == 1

    def test_generate_incompatible_context_key(self):
        """Context._generate_context_id creates different ids for incompatible contexts."""
        all_ids = set()
        for state, integrator in itertools.product(self.incompatible_states,
                                                   self.incompatible_integrators):
            all_ids.add(ContextCache._generate_context_id(state, integrator))
        assert len(all_ids) == 4

    def test_get_compatible_context(self):
        """ContextCache.get_context method do not recreate a compatible context."""
        cache = ContextCache()
        context_ids = set()
        for state, integrator in itertools.product(self.compatible_states,
                                                   self.compatible_integrators):
            # Avoid binding same integrator to multiple contexts
            integrator = copy.deepcopy(integrator)
            context, context_integrator = cache.get_context(state, integrator)
            context_ids.add(id(context))
            assert integrator.__getstate__() == context_integrator.__getstate__()
            assert integrator.__getstate__() == context.getIntegrator().__getstate__()
            assert context.getSystem().__getstate__() == state.system.__getstate__()
        assert len(cache) == 1
        assert len(context_ids) == 1

    def test_get_incompatible_context(self):
        """ContextCache.get_context method create handles incompatible contexts."""
        cache = ContextCache()
        n_contexts = self.cache_incompatible_contexts(cache)
        assert len(cache) == 4
        assert n_contexts == 4

    def test_get_context_any_integrator(self):
        """ContextCache.get_context first search the cache when integrator is unspecified."""
        cache = ContextCache()
        state1, state2 = self.incompatible_states[:2]

        # First we create a Context in state1.
        cache.get_context(state1, copy.deepcopy(self.verlet_2fs))
        assert len(cache) == 1

        # When we don't specify the integrator, it first looks for cached Contexts.
        context, integrator = cache.get_context(state1)
        assert len(cache) == 1
        assert state1.is_context_compatible(context)
        assert isinstance(integrator, openmm.VerletIntegrator)

        # With an incompatible state, a new Context is created.
        cache.get_context(state2)
        assert len(cache) == 2

    def test_cache_capacity_ttl(self):
        """Check that the cache capacity and time_to_live work as expected."""
        cache = ContextCache(capacity=3)
        n_contexts = self.cache_incompatible_contexts(cache)
        assert len(cache) == 3
        assert n_contexts == 4

        cache = ContextCache(time_to_live=3)
        n_contexts = self.cache_incompatible_contexts(cache)
        assert len(cache) == 3
        assert n_contexts == 4

    def test_platform_property(self):
        """Platform change at runtime is only possible when cache is empty."""
        platforms = [openmm.Platform.getPlatformByName(name) for name in ['Reference', 'CPU']]
        cache = ContextCache(platform=platforms[0])
        cache.platform = platforms[1]
        integrator = copy.deepcopy(self.compatible_integrators[0])
        cache.get_context(self.compatible_states[0], integrator)
        with nose.tools.assert_raises(RuntimeError):
            cache.platform = platforms[0]
