"""
Alchemical factory for free energy calculations package that operates directly on OpenMM
System objects.

DESCRIPTION

This package contains enumerative factories for generating alchemically-modified System objects
usable for the calculation of free energy differences of hydration or ligand binding.

Provided classes include:

- :class:`openmmtools.alchemy.alchemy.AlchemicalFunction`
- :class:`openmmtools.alchemy.alchemy.AlchemicalState`
- :class:`openmmtools.alchemy.alchemy.AlchemicalRegion`
- :class:`openmmtools.alchemy.alchemy.AbsoluteAlchemicalFactory`
- :class:`openmmtools.alchemy.alchemy.AlchemicalStateError`

"""

# Automatically importing everything from the lower level alchemy module to avoid API breakage
from .alchemy import AlchemicalState, AlchemicalFunction, AlchemicalStateError, AlchemicalRegion, \
    AbsoluteAlchemicalFactory
