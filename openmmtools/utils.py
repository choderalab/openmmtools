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

import ast
import abc
import time
import math
import logging
import operator

import numpy as np
from simtk import openmm, unit

logger = logging.getLogger(__name__)


# =============================================================================
# BENCHMARKING UTILITIES
# =============================================================================

class Timer(object):
    """A class with stopwatch-style timing functions.

    Examples
    --------
    >>> timer = Timer()
    >>> timer.start('my benchmark')
    >>> for i in range(10):
    ...     pass
    >>> timer.stop('my benchmark')
    >>> timer.start('second benchmark')
    >>> for i in range(10):
    ...     for j in range(10):
    ...         pass
    >>> timer.start('second benchmark')
    >>> timer.report_timing()

    """

    def __init__(self):
        self.reset_timing_statistics()

    def reset_timing_statistics(self):
        """Reset the timing statistics."""
        self._t0 = {}
        self._t1 = {}
        self._elapsed = {}

    def start(self, benchmark_id):
        """Start a timer with given benchmark_id."""
        self._t0[benchmark_id] = time.time()

    def stop(self, benchmark_id):
        if benchmark_id in self._t0:
            self._t1[benchmark_id] = time.time()
            self._elapsed[benchmark_id] = self._t1[benchmark_id] - self._t0[benchmark_id]
        else:
            logger.info("Can't stop timing for {}".format(benchmark_id))

    def report_timing(self, clear=True):
        """Log all the timings at the debug level.

        Parameters
        ----------
        clear : bool
            If True, the stored timings are deleted after being reported.

        """
        logger.debug('Saved timings:')

        for benchmark_id, elapsed_time in self._elapsed.items():
            logger.debug('{:.24}: {:8.3f}s'.format(benchmark_id, elapsed_time))

        if clear is True:
            self.reset_timing_statistics()


# =============================================================================
# STRING PARSING UTILITIES
# =============================================================================

def math_eval(expression, variables=None):
    """Evaluate a mathematical expression with variables.

    All the functions in the standard module math are available together with
    - step(x) : Heaviside step function (1.0 for x=0)
    - step_hm(x) : Heaviside step function with half-maximum convention.
    - sign(x) : sign function (0.0 for x=0.0)

    Parameters
    ----------
    expression : str
        The mathematical expression as a string.
    variables : dict of str: float, optional
        The variables in the expression, if any (default is None).

    Returns
    -------
    float
        The result of the evaluated expression.

    Examples
    --------
    >>> expr = '-((x + ceil(y)) / z * 4 + step(-0.2))**2'
    >>> vars = {'x': 1, 'y': 1.9, 'z': 3}
    >>> math_eval(expr, vars)
    -16.0

    """
    # Supported operators.
    operators = {ast.Add: operator.add, ast.Sub: operator.sub,
                 ast.Mult: operator.mul, ast.Div: operator.truediv,
                 ast.Pow: operator.pow, ast.USub: operator.neg}

    # Supported functions, not defined in math.
    functions = {'step': lambda x: 1 * (x >= 0),
                 'step_hm': lambda x: 0.5 * (np.sign(x) + 1),
                 'sign': lambda x: np.sign(x)}

    def _math_eval(node):
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.UnaryOp):
            return operators[type(node.op)](_math_eval(node.operand))
        elif isinstance(node, ast.BinOp):
            return operators[type(node.op)](_math_eval(node.left),
                                            _math_eval(node.right))
        elif isinstance(node, ast.Name):
            try:
                return variables[node.id]
            except KeyError:
                raise ValueError('Variable {} was not provided'.format(node.id))
        elif isinstance(node, ast.Call):
            args = [_math_eval(arg) for arg in node.args]
            try:
                return getattr(math, node.func.id)(*args)
            except AttributeError:
                try:
                    return functions[node.func.id](*args)
                except KeyError:
                    raise ValueError('Function {} is not supported'.format(node.func.id))
        else:
            raise TypeError('Cannot parse expression: {}'.format(expression))

    if variables is None:
        variables = {}

    return _math_eval(ast.parse(expression, mode='eval').body)


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
