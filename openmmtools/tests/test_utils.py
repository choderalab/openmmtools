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

from openmmtools.utils import _RESERVED_WORDS_PATTERNS
from openmmtools.utils import *


# =============================================================================
# TEST STRING MATHEMATICAL EXPRESSION PARSING UTILITIES
# =============================================================================

def test_sanitize_expression():
    """Test that reserved keywords are substituted correctly."""
    prefix = '_sanitized__'

    # Generate a bunch of test cases for each supported reserved keyword.
    test_cases = {}
    for word in _RESERVED_WORDS_PATTERNS:
        s_word = prefix + word  # sanitized word
        test_cases[word] = [(word, s_word),
                            ('(' + word + ')', '(' + s_word + ')'),  # parenthesis
                            ('( ' + word + ' )', '( ' + s_word + ' )'),  # parenthesis w/ spaces
                            (word + '_suffix', word + '_suffix'),  # w/ suffix
                            ('prefix_' + word, 'prefix_' + word),  # w/ prefix
                            ('2+' + word + '-' + word + '_suffix/(' + word + ' - 3)',  # expression
                             '2+' + s_word + '-' + word + '_suffix/(' + s_word + ' - 3)')]

    # Run test cases.
    for word in _RESERVED_WORDS_PATTERNS:
        variables = {word: 5.0}
        for expression, result in test_cases[word]:
            sanitized_expression, sanitized_variables = sanitize_expression(expression, variables)
            assert sanitized_expression == result, '{}, {}'.format(sanitized_expression, result)
            assert word not in sanitized_variables
            assert sanitized_variables[prefix + word] == 5.0


def test_math_eval():
    """Test math_eval method."""
    test_cases = [('1 + 3', None, 4),
                  ('x + y', {'x': 1.5, 'y': 2}, 3.5),
                  ('(x + lambda) / z * 4', {'x': 1, 'lambda': 2, 'z': 3}, 4.0),
                  ('-((x + y) / z * 4)**2', {'x': 1, 'y': 2, 'z': 3}, -16.0),
                  ('ceil(0.8) + acos(x) + step(0.5 - x) + step(0.5)', {'x': 1}, 2),
                  ('step_hm(x)', {'x': 0}, 0.5)]
    for expression, variables, result in test_cases:
        evaluated_expression = math_eval(expression, variables)
        assert evaluated_expression == result, '{}, {}, {}'.format(
            expression, evaluated_expression, result)


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
