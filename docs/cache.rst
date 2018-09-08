.. _cache:

Cache
=====

The module :mod:`openmmtools.cache` implements a shared LRU cache for OpenMM `Context <http://docs.openmm.org/7.1.0/api-python/generated/simtk.openmm.openmm.Context.html>`_ objects that tries to minimize the number of objects in memory at the same time.

Cache objects
-------------

.. currentmodule:: openmmtools.cache
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    LRUCache
    ContextCache
    global_context_cache
