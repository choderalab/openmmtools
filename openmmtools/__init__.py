#!/usr/local/bin/env python

"""
Various Python utilities for OpenMM.

"""

# Import modules.
from openmmtools import testsystems, integrators

from .version import get_versions
__version__ = get_versions()['version']
del get_versions
