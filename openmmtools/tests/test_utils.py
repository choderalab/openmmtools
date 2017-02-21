#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test utility functions in utils.py.

"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import nose

from openmmtools.utils import *


# =============================================================================
# TEST STRING PARSING UTILITIES
# =============================================================================

def test_math_eval():
    """Test math_eval method."""
    test_cases = [('1 + 3', None, 4),
                  ('x + y', {'x': 1.5, 'y': 2}, 3.5),
                  ('(x + y) / z * 4', {'x': 1, 'y': 2, 'z': 3}, 4.0),
                  ('-((x + y) / z * 4)**2', {'x': 1, 'y': 2, 'z': 3}, -16.0)]
    for expression, variables, result in test_cases:
        assert math_eval(expression, variables) == result


# =============================================================================
# TEST QUANTITY UTILITIES
# =============================================================================

def test_is_quantity_close():
    """Test is_quantity_close method."""
    # (quantity1, quantity2, test_result)
    test_cases = [(300.0*unit.kelvin, 300.000000004*unit.kelvin, True),
                  (300.0*unit.kelvin, 300.00000004*unit.kelvin, False),
                  (1.01325*unit.bar, 1.01325000006*unit.bar, True),
                  (1.01325*unit.bar, 1.0132500006*unit.bar, False)]
    for quantity1, quantity2, test_result in test_cases:
        assert is_quantity_close(quantity1, quantity2) is test_result

    # Passing quantities with different units raise an exception.
    with nose.tools.assert_raises(TypeError):
        is_quantity_close(300*unit.kelvin, 1*unit.atmosphere)


# =============================================================================
# TEST METACLASS UTILITIES
# =============================================================================

def test_subhooked_abcmeta():
    """Test class SubhookedABCMeta."""
    # Define an interface
    class IInterface(SubhookedABCMeta):
        @abc.abstractmethod
        def my_method(self): pass

        @abc.abstractproperty
        def my_property(self): pass

        @staticmethod
        @abc.abstractmethod
        def my_static_method(): pass

    # Define object implementing the interface with duck typing
    class InterfaceImplementation(object):
        def my_method(self): pass

        def my_property(self): pass

        @staticmethod
        def my_static_method(): pass

    implementation_instance = InterfaceImplementation()
    assert isinstance(implementation_instance, IInterface)

    # Define incomplete implementation
    class WrongInterfaceImplementation(object):
        def my_method(self): pass

    implementation_instance = WrongInterfaceImplementation()
    assert not isinstance(implementation_instance, IInterface)
