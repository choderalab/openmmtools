.. _forcefactories::

Cache
=====

The module :mod:`openmmtools.forcefactories` implements utility methods and factories to configure system forces.

 - :func:`replace_reaction_field`: Configure a system to model the electrostatics with an :class:`UnshiftedReactionField` force.
 - :func:`restrain_atoms`: Apply a soft harmonic restraint to the given atoms.

Cache objects
-------------

.. currentmodule:: openmmtools.forcefactories
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    replace_reaction_field
    restrain_atoms
