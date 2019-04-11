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

import re
import ast
import abc
import time
import math
import copy
import shutil
import inspect
import logging
import operator
import tempfile
import functools
import importlib
import contextlib
import zlib

import numpy as np
from simtk import openmm, unit

logger = logging.getLogger(__name__)


# =============================================================================
# MISCELLANEOUS
# =============================================================================

@contextlib.contextmanager
def temporary_directory():
    """Context for safe creation of temporary directories."""
    tmp_dir = tempfile.mkdtemp()
    try:
        yield tmp_dir
    finally:
        shutil.rmtree(tmp_dir)


# =============================================================================
# BENCHMARKING UTILITIES
# =============================================================================

@contextlib.contextmanager
def time_it(task_name):
    """Context manager to log execution time of a block of code.

    Parameters
    ----------
    task_name : str
        The name of the task that will be reported.

    """
    timer = Timer()
    timer.start(task_name)
    yield timer  # Resume program
    timer.stop(task_name)
    timer.report_timing()


def with_timer(task_name):
    """Decorator that logs the execution time of a function.

    Parameters
    ----------
    task_name : str
        The name of the task that will be reported.

    """
    def _with_timer(func):
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            with time_it(task_name):
                return func(*args, **kwargs)
        return _wrapper
    return _with_timer


class Timer(object):
    """A class with stopwatch-style timing functions.

    Examples
    --------
    >>> timer = Timer()
    >>> timer.start('my benchmark')
    >>> for i in range(10):
    ...     pass
    >>> elapsed_time = timer.stop('my benchmark')
    >>> timer.start('second benchmark')
    >>> for i in range(10):
    ...     for j in range(10):
    ...         pass
    >>> elsapsed_time = timer.stop('second benchmark')
    >>> timer.report_timing()

    """

    def __init__(self):
        self.reset_timing_statistics()

    def reset_timing_statistics(self, benchmark_id=None):
        """Reset the timing statistics.

        Parameters
        ----------
        benchmark_id : str, optional
            If specified, only the timings associated to this benchmark
            id will be reset, otherwise all timing information are.

        """
        if benchmark_id is None:
            self._t0 = {}
            self._t1 = {}
            self._completed = {}
        else:
            self._t0.pop(benchmark_id, None)
            self._t1.pop(benchmark_id, None)
            self._completed.pop(benchmark_id, None)

    def start(self, benchmark_id):
        """Start a timer with given benchmark_id."""
        self._t0[benchmark_id] = time.time()

    def stop(self, benchmark_id):
        try:
            t0 = self._t0[benchmark_id]
        except KeyError:
            logger.warning("Can't stop timing for {}".format(benchmark_id))
        else:
            self._t1[benchmark_id] = time.time()
            elapsed_time = self._t1[benchmark_id] - t0
            self._completed[benchmark_id] = elapsed_time
            return elapsed_time

    def partial(self, benchmark_id):
        """Return the elapsed time of the given benchmark so far."""
        try:
            t0 = self._t0[benchmark_id]
        except KeyError:
            logger.warning("Couldn't return partial timing for {}".format(benchmark_id))
        else:
            return time.time() - t0

    def report_timing(self, clear=True):
        """Log all the timings at the debug level.

        Parameters
        ----------
        clear : bool
            If True, the stored timings are deleted after being reported.

        Returns
        -------
        elapsed_times : dict
            The dictionary benchmark_id : elapsed time for all benchmarks.

        """
        for benchmark_id, elapsed_time in self._completed.items():
            logger.debug('{} took {:8.3f}s'.format(benchmark_id, elapsed_time))

        if clear is True:
            self.reset_timing_statistics()


# =============================================================================
# STRING MATHEMATICAL EXPRESSION PARSING UTILITIES
# =============================================================================

# Dict reserved_keyword: compiled_regex_pattern. This is used by
_RESERVED_WORDS_PATTERNS = {
    'lambda': re.compile(r'(?<![a-zA-Z0-9_])lambda(?![a-zA-Z0-9_])')
}


def sanitize_expression(expression, variables):
    """Sanitize variables with an illegal Python name.

    Transform variable names in the string expression that are illegal in
    Python so that the expression can be evaluated in pure Python. Currently
    this just handle variables called with the reserved word 'lambda'.

    Parameters
    ----------
    expression : str
        The mathematical expression as a string.
    variables : dict of str: float
        The variables in the expression.

    Returns
    -------
    sanitized_expression : str
        The same mathematical expression that can be executed in Python.
    sanitized_variables : dict of str: float
        The updated variable names with their values.

    """
    sanitized_variables = None
    sanitized_expression = expression

    # Substitute all reserved words in expression and variables.
    for word, pattern in _RESERVED_WORDS_PATTERNS.items():
        if word in variables:  # Don't make unneeded substitutions.
            if sanitized_variables is None:
                sanitized_variables = copy.deepcopy(variables)
            sanitized_word = '_sanitized__' + word
            sanitized_expression = pattern.sub(sanitized_word, sanitized_expression)
            variable_value = sanitized_variables.pop(word)
            sanitized_variables[sanitized_word] = variable_value

    # If no substitutions are made return same variables.
    if sanitized_variables is None:
        sanitized_variables = variables

    return sanitized_expression, sanitized_variables


def math_eval(expression, variables=None, functions=None):
    """Evaluate a mathematical expression with variables.

    All the functions in the standard module math are available together with
    - step(x) : Heaviside step function (1.0 for x=0)
    - step_hm(x) : Heaviside step function with half-maximum convention.
    - sign(x) : sign function (0.0 for x=0.0)

    Available operators are ``+``, ``-``, ``*``, ``/``, ``**``, ``-x`` (negative),
    ``&``, ``and``, ``|``, and ``or``

    **The operators ``and`` and ``or`` operate BITWISE and behave the same as ``&`` and ``|`` respectively as this
    function is not designed to handle logical operations.** If you provide sets, they must be as variables.

    Parameters
    ----------
    expression : str
        The mathematical expression as a string.
    variables : dict of str: float, optional
        The variables in the expression, if any (default is None).
    functions : dict of str: callable function, optional
        Additional functions to teach the math eval statement how to handle.
        Built-in functions are 'step', 'step_hm', and 'sign'

    Returns
    -------
    result
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
                 ast.Pow: operator.pow, ast.USub: operator.neg,
                 ast.BitAnd: operator.and_, ast.And: operator.and_,
                 ast.BitOr: operator.or_, ast.Or: operator.or_
                 }

    # Supported functions, not defined in math.
    extra_functions = {'step': lambda x: 1 * (x >= 0),
                       'step_hm': lambda x: 0.5 * (np.sign(x) + 1),
                       'sign': lambda x: np.sign(x)}

    # Allow overwrite of extra_functions.
    if functions is not None:
        extra_functions.update(functions)
    functions = extra_functions

    def _math_eval(node):
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.UnaryOp):
            return operators[type(node.op)](_math_eval(node.operand))
        elif isinstance(node, ast.BinOp):
            return operators[type(node.op)](_math_eval(node.left),
                                            _math_eval(node.right))
        elif isinstance(node, ast.BoolOp):
            # Parse ternary operator
            if len(node.values) > 2:
                # Left-to-right precedence.
                left_value = copy.deepcopy(node)
                left_value.values.pop(-1)
            else:
                left_value = node.values[0]
            return operators[type(node.op)](_math_eval(left_value), _math_eval(node.values[-1]))
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

    # Sanitized reserved words.
    expression, variables = sanitize_expression(expression, variables)

    return _math_eval(ast.parse(expression, mode='eval').body)


# =============================================================================
# QUANTITY UTILITIES
# =============================================================================

def _changes_state(func):
    """Decorator to signal changes in TrackedQuantity."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        self.has_changed = True
        return func(self, *args, **kwargs)
    return wrapper


class TrackedQuantity(unit.Quantity):
    """A quantity that keeps track of whether it has been changed."""

    def __init__(self, *args, **kwargs):
        super(TrackedQuantity, self).__init__(*args, **kwargs)
        self.has_changed = False

    def __getitem__(self, item):
        if isinstance(item, slice) and isinstance(self._value, np.ndarray):
            return TrackedQuantityView(self, super(TrackedQuantity, self).__getitem__(item))
        # No need to track a copy.
        return super(TrackedQuantity, self).__getitem__(item)

    __setitem__ = _changes_state(unit.Quantity.__setitem__)
    __delitem__ = _changes_state(unit.Quantity.__delitem__)
    append = _changes_state(unit.Quantity.append)
    extend = _changes_state(unit.Quantity.extend)
    insert = _changes_state(unit.Quantity.insert)
    remove = _changes_state(unit.Quantity.remove)
    pop = _changes_state(unit.Quantity.pop)


class TrackedQuantityView(unit.Quantity):
    """Keeps truck of a numpy view for TrackedQuantity."""

    def __init__(self, tracked_quantity, *args, **kwargs):
        super(TrackedQuantityView, self).__init__(*args, **kwargs)
        self._tracked_quantity = tracked_quantity  # Parent.

    def __getitem__(self, item):
        if isinstance(item, slice):
            return TrackedQuantityView(self._tracked_quantity,
                                       super(TrackedQuantityView, self).__getitem__(item))
        # No need to track a copy.
        return super(TrackedQuantityView, self).__getitem__(item)

    def __setitem__(self, key, value):
        super(TrackedQuantityView, self).__setitem__(key, value)
        self._tracked_quantity.has_changed = True



# List of simtk.unit methods that are actually units and functions instead of base classes
# Pre-computed to reduce run-time cost
# Get the built-in units
_VALID_UNITS = {method: getattr(unit, method) for method in dir(unit) if type(getattr(unit, method)) is unit.Unit}
# Get the built in unit functions and make sure they are not just types
_VALID_UNIT_FUNCTIONS = {method: getattr(unit, method) for method in dir(unit)
                         if callable(getattr(unit, method)) and type(getattr(unit, method)) is not type}


def is_quantity_close(quantity1, quantity2, rtol=1e-10, atol=0.0):
    """Check if the quantities are equal up to floating-point precision errors.

    Parameters
    ----------
    quantity1 : simtk.unit.Quantity
        The first quantity to compare.
    quantity2 : simtk.unit.Quantity
        The second quantity to compare.
    rtol : float, optional
        Relative tolerance (default is 1e-10).
    atol : float, optional
        Absolute tolerance (default is 0.0).

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
    if abs(value2) >= abs(value1):
        return np.isclose(value1, value2, rtol=rtol, atol=atol)
    else:
        return np.isclose(value2, value1, rtol=rtol, atol=atol)


def quantity_from_string(expression):
    """Special call to the math_eval function designed to handle simtk.unit Quantity strings

    All the functions in the standard module math are available together with
    most of the methods inside the simtk.unit module.

    Parameters
    ----------
    expression : str
        The mathematical expression to rebuild a Quantityas a string.

    Returns
    -------
    Quantity
        The result of the evaluated expression.

    Examples
    --------
    >>> expr = '4 * kilojoules / mole'
    >>> quantity_from_string(expr)
    Quantity(value=4.000000000000002, unit=kilojoule/mole)

    """

    # Supported functions, not defined in math.
    functions = _VALID_UNIT_FUNCTIONS

    # Define the units from simtk.unit as the variables
    variables = _VALID_UNITS

    # Eliminate nested quotes and excess whitespace
    expression = expression.strip('\'" ')

    # Handle a special case of the unit when it is just "inverse unit", e.g. Hz == /second
    if expression[0] == '/':
        expression = '(' + expression[1:] + ')**(-1)'

    return math_eval(expression, variables=variables, functions=functions)


def typename(atype):
    """Convert a type object into a fully qualified typename.

    Parameters
    ----------
    atype : type
        The type to convert

    Returns
    -------
    typename : str
        The string typename.

    For example,

    >>> typename(type(1))
    'int'

    >>> import numpy
    >>> x = numpy.array([1,2,3], numpy.float32)
    >>> typename(type(x))
    'numpy.ndarray'

    """
    if not isinstance(atype, type):
        raise Exception('Argument is not a type')

    modulename = atype.__module__
    typename = atype.__name__

    if modulename not in ['__builtin__', 'builtins']:
        typename = modulename + '.' + typename

    return typename


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
# SERIALIZATION UTILITIES
# =============================================================================

_SERIALIZED_MANGLED_PREFIX = '_serialized__'


def serialize(instance, **kwargs):
    """Serialize an object.

    The object must expose a __getstate__ method that returns a
    dictionary representation of its state. This will be passed to
    __setstate__ for deserialization. The function automatically
    handle the resolution of the correct class.

    Parameters
    ----------
    instance : object
        An instance of a new style class.

    kwargs : Keyword arguments which are passed onto the __getstate__ function.
        If you implement your own class with a __getstate__ method, have it accept **kwargs and then manipulate
            them inside the __getstate__ method itself.
        These are primarily optimization settings and will not normally be publicly documented because they can
        fundamentally change how the "state" of an object is returned.

    Returns
    -------
    serialization : dict
        A dictionary representation of the object that can be
        stored in several formats (e.g. JSON, YAML, HDF5) and
        reconstructed into the original object with deserialize().

    """
    module_name = instance.__module__
    class_name = instance.__class__.__name__
    try:
        serialization = instance.__getstate__(**kwargs)
    except AttributeError:
        raise ValueError('Cannot serialize class {} without a __getstate__ method'.format(class_name))
    serialization[_SERIALIZED_MANGLED_PREFIX + 'module_name'] = module_name
    serialization[_SERIALIZED_MANGLED_PREFIX + 'class_name'] = class_name
    return serialization


def deserialize(serialization):
    """Deserialize an object.

    The original class must expose a __setstate__ that takes the
    dictionary representation of its state generated by its
    __getstate__.

    Parameters
    ----------
    serialization : dict
        A dictionary generated by serialize().

    Returns
    -------
    instance : object
        An instance in the state given by serialization.

    """
    names = []
    for key in ['module_name', 'class_name']:
        try:
            names.append(serialization.pop(_SERIALIZED_MANGLED_PREFIX + key))
        except KeyError:
            raise ValueError('Cannot find {} in the serialization. Was the original object '
                             'serialized with openmmtools.utils.serialize()?'.format(key))
    module_name, class_name = names  # unpack
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    instance = cls.__new__(cls)
    try:
        instance.__setstate__(serialization)
    except AttributeError:
        raise ValueError('Cannot deserialize class {} without a __setstate__ method'.format(class_name))
    return instance


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


def find_all_subclasses(parent_cls, discard_abstract=False, include_parent=True):
    """Return a set of all the classes inheriting from ``parent_cls``.

    The functions handle multiple inheritance and discard the same classes.

    Parameters
    ----------
    parent_cls : type
        The parent class.
    discard_abstract : bool, optional
        If True, abstract classes are not returned (default is False).
    include_parent : bool, optional
        If True, the parent class will be included, unless it is abstract
        and ``discard_abstract`` is ``True``.

    Returns
    -------
    subclasses : set of type
        The set of all the classes inheriting from ``parent_cls``.

    """
    subclasses = set()
    for subcls in parent_cls.__subclasses__():
        if not (discard_abstract and inspect.isabstract(subcls)):
            subclasses.add(subcls)
        subclasses.update(find_all_subclasses(subcls, discard_abstract))

    if include_parent and not inspect.isabstract(parent_cls):
        subclasses.add(parent_cls)
    return subclasses


def find_subclass(parent_cls, subcls_name):
    """Return the class called ``subcls_name`` inheriting from ``parent_cls``.

    Parameters
    ----------
    parent_cls : type
        The parent class.
    subcls_name : str
        The name of the class inheriting from ``parent_cls``.

    Returns
    -------
    subcls : type
        The class inheriting from ``parent_cls`` called ``subcls_name``.

    Raises
    ------
    ValueError
        If there is no class or there are multiple classes called ``subcls_name``
        that inherit from ``parent_cls``.
    """
    subclasses = []
    for subcls in find_all_subclasses(parent_cls):
        if subcls.__name__ == subcls_name:
            subclasses.append(subcls)
    if len(subclasses) == 0:
        raise ValueError('Could not find class {} inheriting from {}'
                         ''.format(subcls_name, parent_cls))
    if len(subclasses) > 1:
        raise ValueError('Found multiple classes inheriting from {}: {}'
                         ''.format(parent_cls, subclasses))
    return subclasses[0]


# =============================================================================
# RESTORABLE OPENMM OBJECT
# =============================================================================

class RestorableOpenMMObjectError(Exception):
    """Raised when the object has a restorable hash but no matching class can be found."""
    pass


class RestorableOpenMMObject(object):
    """Base class for restorable custom integrators and forces.

    Normally, a custom OpenMM object loses its specific class (and all its
    methods) when it is copied or deserialized from its XML representation.
    Class interfaces inheriting from this can be restored through the method
    ``restore_interface()``. Also, this class extend the copying functions
    to copy also Python attributes.

    The class automatically adds a global parameter or variable in custom
    forces and integrators respectively on __init__ to keep track of the
    original class.

    """

    _cached_hash_subclasses = {}

    def __init__(self, *args, **kwargs):
        super(RestorableOpenMMObject, self).__init__(*args, **kwargs)
        self._add_global_parameter(self, self._hash_parameter_name,
                                   self._compute_class_hash(self.__class__))

    @classmethod
    def is_restorable(cls, openmm_object):
        """Check if the custom integrator or force has a restorable interface.

        Parameters
        ----------
        openmm_object : object
            The custom integrator or force to check.

        Returns
        -------
        True if the object has a restorable interface, False otherwise.

        """
        try:
            hash_parameter_name = cls._get_hash_parameter_name(openmm_object)
            cls._get_global_parameter(openmm_object, hash_parameter_name)
        except Exception:
            return False
        return True

    @classmethod
    def restore_interface(cls, openmm_object):
        """Restore the original interface of an OpenMM custom force or integrator.

        The function restore the methods of the original class that
        inherited from ``RestorableOpenMMObject``. Return False if the
        interface could not be restored.

        Parameters
        ----------
        openmm_object : object
            The object to restore.

        Returns
        -------
        True if the original class interface could be restored, False otherwise.

        """
        try:
            hash_parameter_name = cls._get_hash_parameter_name(openmm_object)
            object_hash = cls._get_global_parameter(openmm_object, hash_parameter_name)
        except Exception:
            return False

        # Reload the hash table for all subclasses if there's no matching class.
        if object_hash not in cls._cached_hash_subclasses:
            all_subclasses = find_all_subclasses(parent_cls=cls, discard_abstract=True,
                                                 include_parent=True)
            cls._cached_hash_subclasses = {cls._compute_class_hash(subcls): subcls
                                           for subcls in all_subclasses}

        # Retrieve object class.
        try:
            object_class = cls._cached_hash_subclasses[object_hash]
        except KeyError:
            raise RestorableOpenMMObjectError('Could not find a class matching '
                                              'the hash {}'.format(object_hash))

        # Restore class interface.
        openmm_object.__class__ = object_class
        return True

    # -------------------------------------------------------------------------
    # Global parameters.
    # -------------------------------------------------------------------------

    @property
    def _hash_parameter_name(self):
        """The hash parameter name of this restorable object."""
        return self._get_hash_parameter_name(self)

    @classmethod
    def _get_hash_parameter_name(cls, openmm_object):
        """Return the name of the openmm_object global variable containing the hash.

        As of OpenMM 7.2, it is impossible to create a context with an integrator
        having a global variable with the same name of a custom force.
        """
        if cls._is_force(openmm_object):
            return '_restorable_force__class_hash'
        else:
            # Use _restorable__class_hash with integrators for backwards compatibility.
            return '_restorable__class_hash'

    @classmethod
    def _add_global_parameter(cls, openmm_object, parameter_name, parameter_value):
        """Add a new global parameter/variable to the OpenMM custom force/integrator.

        Parameters
        ----------
        openmm_object : object
            The OpenMM custom integrator/force to which add the parameter.
        parameter_name : str
            The name of the global parameter.
        parameter_value : float
            The value of the global parameter.

        """
        if cls._is_force(openmm_object):
            openmm_object.addGlobalParameter(parameter_name, parameter_value)
        else:
            openmm_object.addGlobalVariable(parameter_name, parameter_value)

    @classmethod
    def _get_global_parameter(cls, openmm_object, parameter_name):
        """Get a global parameter/variable from the OpenMM custom force/integrator.

        Parameters
        ----------
        openmm_object : object
            The OpenMM integrator/force to which add the parameter.
        parameter_name : str
            The name of the global parameter.

        Returns
        -------
        parameter_value : float
            The value of the global parameter.

        """
        if cls._is_force(openmm_object):
            return cls._get_force_parameter_by_name(openmm_object, parameter_name)
        else:
            return openmm_object.getGlobalVariableByName(parameter_name)

    @classmethod
    def _get_force_parameter_by_name(cls, force, parameter_name):
        """Get a force global parameter default value from its name."""
        for parameter_idx in range(force.getNumGlobalParameters()):
            if force.getGlobalParameterName(parameter_idx) == parameter_name:
                return force.getGlobalParameterDefaultValue(parameter_idx)
        raise KeyError('No parameter called {} in force {}'.format(parameter_name, force))

    # -------------------------------------------------------------------------
    # Copy and serialization utilities
    # -------------------------------------------------------------------------

    @classmethod
    def deserialize_xml(cls, xml_serialization):
        """Shortcut to deserialize the XML representation and the restore interface.

        Parameters
        ----------
        xml_serialization : str
            The XML representation of the OpenMM custom force/integrator.

        Returns
        -------
        openmm_object
            The deserialized OpenMM force/integrator with the original interface
            restored (if restorable).

        """
        openmm_object = openmm.XmlSerializer.deserialize(xml_serialization)
        cls.restore_interface(openmm_object)
        return openmm_object

    def __deepcopy__(self, memo):
        """Overwrite implementation to copy class and attributes."""
        return self.__copy__()

    def __copy__(self):
        """Overwrite implementation to copy class and attributes."""
        copied_self = super(RestorableOpenMMObject, self).__copy__()

        # Assign correct class instead of OpenMM class.
        copied_self.__class__ = self.__class__

        # Copy attributes. SWIG objects have only 1 attribute (this),
        # everything else is part of the implementation.
        attributes_self = {k: v for k, v in self.__dict__.items() if k != 'this'}
        copied_self.__dict__.update(copy.deepcopy(attributes_self))

        return copied_self

    # -------------------------------------------------------------------------
    # Internal-usage
    # -------------------------------------------------------------------------

    @staticmethod
    def _is_force(openmm_object):
        """Return True if openmm_object is a force object, False if it is an integrator."""
        if isinstance(openmm_object, openmm.Force):
            return True
        elif isinstance(openmm_object, openmm.CustomIntegrator):
            return False
        else:
            raise TypeError('Object of type {} is not supported.'.format(type(openmm_object)))

    @staticmethod
    def _compute_class_hash(openmm_class):
        """Return a numeric hash for the OpenMM class.

        The hash will become part of the OpenMM object serialization,
        so it is important for it consistent across processes in case
        the integrator is sent to a remote worker. The hash() built-in
        function is seeded by the PYTHONHASHSEED environmental variable,
        so we can't use it here.

        We also need to convert to float because some digits may be
        lost in the conversion.
        """
        return float(zlib.adler32(openmm_class.__name__.encode()))


if __name__ == '__main__':
    import doctest
    doctest.testmod()
