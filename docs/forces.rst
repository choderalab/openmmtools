.. _forces::

Forces
======

The module :mod:`openmmtools.forces` implements custom forces that are not natively found in OpenMM.

 - :class:`UnshiftedReactionFieldForce`: A `CustomNonbondedForce <http://docs.openmm.org/7.1.0/api-python/generated/simtk.openmm.openmm.CustomNonbondedForce.html>`_ implementing a reaction field variant with `c_rf` term set to zero and a switching function.
 - :func:`find_nonbonded_force`: Find the first ``NonbondedForce`` in an OpenMM ``System``.
 - :func:`iterate_nonbonded_forces`: Iterate over all the ``NonbondedForce``s in an OpenMM ``System``.

Cache objects
-------------

.. currentmodule:: openmmtools.forces
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    UnshiftedReactionFieldForce
    find_nonbonded_force
    iterate_nonbonded_forces
