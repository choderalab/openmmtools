.. _distributed:

.. warning::

   These classes are experimental and their API is subject to change.

Distributed computing tools
===========================

MPI
---

A wrapper for executing methods remotely via MPI.

.. currentmodule:: openmmtools.distributed.mpi
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    get_mpicomm
    run_single_node
    on_single_node
    distribute
    delay_termination
    delayed_termination
