.. _forces:

Forces
======

The module :mod:`openmmtools.forces` implements custom forces that are not natively found in OpenMM.

|

Restraint Force Classes
-----------------------

.. currentmodule:: openmmtools.forces
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    RadiallySymmetricRestraintForce
    RadiallySymmetricCentroidRestraintForce
    RadiallySymmetricBondRestraintForce
    HarmonicRestraintForceMixIn
    HarmonicRestraintForce
    HarmonicRestraintBondForce
    FlatBottomRestraintForceMixIn
    FlatBottomRestraintForce
    FlatBottomRestraintBondForce

|

Useful Custom Forces
--------------------

.. currentmodule:: openmmtools.forces
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    UnshiftedReactionFieldForce

|

Utility functions
-----------------

.. currentmodule:: openmmtools.forces
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    iterate_forces
    find_forces

|

Exceptions
----------

.. currentmodule:: openmmtools.forces
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    MultipleForcesError
    NoForceFoundError
