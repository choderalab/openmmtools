.. _alchemy:

Alchemical factories
====================

:mod:`openmmtools.alchemy` contains factories for generating `alchemically-modified <http://alchemistry.org>`_ versions of OpenMM `System <http://docs.openmm.org/7.1.0/api-python/generated/simtk.openmm.openmm.System.html>`_ objects for use in alchemical free energy calculations.

Absolute alchemical factories
-----------------------------

Absolute alchical factories modify the ``System`` object to allow part of the system to be alchemically annihilated or decoupled.
This is useful for computing free energies of transfer, solvation, or binding for small molecules.

.. currentmodule:: openmmtools.alchemy
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    AlchemicalFunction
    AlchemicalState
    AbsoluteAlchemicalFactory

Relative alchemical factories
-----------------------------

Coming soon!
