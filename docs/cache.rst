.. _cache::

Cache
=====

The module :mod:`openmmtools.cache` implements a shared LRU cache for OpenMM `Context <http://docs.openmm.org/7.1.0/api-python/generated/simtk.openmm.openmm.Context.html>`_ objects that tries to minimize the number of objects in memory at the same time.

 - :class:`LRUCache`: A simple LRU cache with a dictionary-like interface. It supports a maximum capacity and expiration.
 - :class:`ContextCache`: A LRU cache for OpenMM `Context <http://docs.openmm.org/7.1.0/api-python/generated/simtk.openmm.openmm.Context.html>`_ objects.
 - ``global_context_cache``: A shared :class:`ContextCache` that minimizes the number of `Context <http://docs.openmm.org/7.1.0/api-python/generated/simtk.openmm.openmm.Context.html>`_ creations when employing :class:`MCMCMove`s.

Cache objects
-------------

.. currentmodule:: openmmtools.cache
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    LRUCache
    ContextCache
