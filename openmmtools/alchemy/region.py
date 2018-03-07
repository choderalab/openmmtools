#!/usr/bin/python

"""
Alchemical state storage objects

"""

# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import copy
import logging
import collections

import numpy as np
from simtk import openmm, unit

from openmmtools import states, forcefactories, utils
from openmmtools.constants import ONE_4PI_EPS0

logger = logging.getLogger(__name__)

# =============================================================================
# ALCHEMICAL REGION
# =============================================================================

_ALCHEMICAL_REGION_ARGS = collections.OrderedDict([
    ('alchemical_atoms', None),
    ('alchemical_bonds', None),
    ('alchemical_angles', None),
    ('alchemical_torsions', None),
    ('annihilate_electrostatics', True),
    ('annihilate_sterics', False),
    ('softcore_alpha', 0.5), ('softcore_a', 1), ('softcore_b', 1), ('softcore_c', 6),
    ('softcore_beta', 0.0), ('softcore_d', 1), ('softcore_e', 1), ('softcore_f', 2)
])


# The class is just a way to document the namedtuple.
class AlchemicalRegion(collections.namedtuple('AlchemicalRegion', _ALCHEMICAL_REGION_ARGS.keys())):
    """Alchemical region.

    This is a namedtuple used to tell the AbsoluteAlchemicalFactory which
    region of the system to alchemically modify and how.

    Parameters
    ----------
    alchemical_atoms : list of int, optional
        List of atoms to be designated for which the nonbonded forces (both
        sterics and electrostatics components) have to be alchemically
        modified (default is None).
    alchemical_bonds : bool or list of int, optional
        If a list of bond indices are specified, these HarmonicBondForce
        entries are softened with 'lambda_bonds'. If set to True, this list
        is auto-generated to include all bonds involving any alchemical
        atoms (default is None).
    alchemical_angles : bool or list of int, optional
        If a list of angle indices are specified, these HarmonicAngleForce
        entries are softened with 'lambda_angles'. If set to True, this
        list is auto-generated to include all angles involving any alchemical
        atoms (default is None).
    alchemical_torsions : bool or list of int, optional
        If a list of torsion indices are specified, these PeriodicTorsionForce
        entries are softened with 'lambda_torsions'. If set to True, this list
        is auto-generated to include al proper torsions involving any alchemical
        atoms. Improper torsions are not softened (default is None).
    annihilate_electrostatics : bool, optional
        If True, electrostatics should be annihilated, rather than decoupled
        (default is True).
    annihilate_sterics : bool, optional
        If True, sterics (Lennard-Jones or Halgren potential) will be annihilated,
        rather than decoupled (default is False).
    softcore_alpha : float, optional
        Alchemical softcore parameter for Lennard-Jones (default is 0.5).
    softcore_a, softcore_b, softcore_c : float, optional
        Parameters modifying softcore Lennard-Jones form. Introduced in
        Eq. 13 of Ref. [1] (default is 1).
    softcore_beta : float, optional
        Alchemical softcore parameter for electrostatics. Set this to zero
        to recover standard electrostatic scaling (default is 0.0).
    softcore_d, softcore_e, softcore_f : float, optional
        Parameters modifying softcore electrostatics form (default is 1).

    Notes
    -----
    The parameters softcore_e and softcore_f determine the effective distance
    between point charges according to

    r_eff = sigma*((softcore_beta*(lambda_electrostatics-1)^softcore_e + (r/sigma)^softcore_f))^(1/softcore_f)

    References
    ----------
    [1] Pham TT and Shirts MR. Identifying low variance pathways for free
    energy calculations of molecular transformations in solution phase.
    JCP 135:034114, 2011. http://dx.doi.org/10.1063/1.3607597

    """
AlchemicalRegion.__new__.__defaults__ = tuple(_ALCHEMICAL_REGION_ARGS.values())
