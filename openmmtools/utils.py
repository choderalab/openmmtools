#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
General utility functions for the repo.

"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import abc

import numpy as np
from simtk import openmm, unit


# =============================================================================
# QUANTITY UTILITIES
# =============================================================================

def is_quantity_close(quantity1, quantity2):
    """Check if the quantities are equal up to floating-point precision errors.

    Parameters
    ----------
    quantity1 : simtk.unit.Quantity
        The first quantity to compare.
    quantity2 : simtk.unit.Quantity
        The second quantity to compare.

    Returns
    -------
    True if the quantities are equal up to approximately 10 digits.

    Raises
    ------
    TypeError
        If the two quantities are of incompatible units.

    """
    if not quantity1.unit.is_compatible(quantity2.unit):
        raise TypeError('Cannot compare incompatible quantities {} and {}'.format(
            quantity1, quantity2))

    value1 = quantity1.value_in_unit_system(unit.md_unit_system)
    value2 = quantity2.value_in_unit_system(unit.md_unit_system)

    # np.isclose is not symmetric, so we make it so.
    if value2 >= value1:
        return np.isclose(value1, value2, rtol=1e-10, atol=0.0)
    else:
        return np.isclose(value2, value1, rtol=1e-10, atol=0.0)


# =============================================================================
# OPENMM PLATFORM UTILITIES
# =============================================================================

def get_available_platforms():
    """Return a list of the available OpenMM Platforms."""
    return [openmm.Platform.getPlatform(i)
            for i in range(openmm.Platform.getNumPlatforms())]


def get_fastest_platform():
    """Return the fastest available platform.

    This relies on the hardcoded speed values in Platform.getSpeed().

    Returns
    -------
    platform : simtk.openmm.Platform
       The fastest available platform.

    """
    platforms = get_available_platforms()
    fastest_platform = max(platforms, key=lambda x: x.getSpeed())
    return fastest_platform


# =============================================================================
# METACLASS UTILITIES
# =============================================================================

# TODO Remove this when we drop Python 2 support.
def with_metaclass(metaclass, *bases):
    """Create a base class with a metaclass.

    Imported from six (MIT license): https://pypi.python.org/pypi/six.
    Provide a Python2/3 compatible way to create a metaclass.

    """
    # This requires a bit of explanation: the basic idea is to make a dummy
    # metaclass for one level of class instantiation that replaces itself with
    # the actual metaclass.
    class Metaclass(metaclass):
        def __new__(cls, name, this_bases, d):
            return metaclass(name, bases, d)
    return type.__new__(Metaclass, 'temporary_class', (), {})


class SubhookedABCMeta(with_metaclass(abc.ABCMeta)):
    """Abstract class with an implementation of __subclasshook__.

    The __subclasshook__ method checks that the instance implement the
    abstract properties and methods defined by the abstract class. This
    allow classes to implement an abstraction without explicitly
    subclassing it.

    Examples
    --------
    >>> class MyInterface(SubhookedABCMeta):
    ...     @abc.abstractmethod
    ...     def my_method(self): pass
    >>> class Implementation(object):
    ...     def my_method(self): return True
    >>> isinstance(Implementation(), MyInterface)
    True

    """
    @classmethod
    def __subclasshook__(cls, subclass):
        for abstract_method in cls.__abstractmethods__:
            if not any(abstract_method in C.__dict__ for C in subclass.__mro__):
                return False
        return True


if __name__ == '__main__':
    import doctest
    doctest.testmod()
