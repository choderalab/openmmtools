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
import logging
import operator
import tempfile
import functools
import importlib
import contextlib

import numpy as np
from simtk import openmm, unit

logger = logging.getLogger(__name__)


# =============================================================================
# MISCELLANEOUS
# =============================================================================

def wraps_py2(wrapped, *args):
    """Wrap a function and add the __wrapped__ attribute.
    In Python 2, functools.wraps does not add the __wrapped__ attribute, and it
    becomes impossible to retrieve the signature of the wrapped method.
    """
    def decorator(wrapper):
        functools.update_wrapper(wrapper, wrapped, *args)
        wrapper.__wrapped__ = wrapped
        return wrapper
    return decorator


def unwrap_py2(func):
    """Unwrap a wrapped function.
    The function inspect.unwrap has been implemented only in Python 3.4. With
    Python 2, this works only for functions wrapped by wraps_py2().
    """
    unwrapped_func = func
    try:
        while True:
            unwrapped_func = unwrapped_func.__wrapped__
    except AttributeError:
        return unwrapped_func

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

    # Sanitized reserved words.
    expression, variables = sanitize_expression(expression, variables)

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

#========================================================================================
# Logging functions
#========================================================================================

def is_terminal_verbose():
    """Check whether the logging on the terminal is configured to be verbose.

    This is useful in case one wants to occasionally print something that is not really
    relevant to yank's log (e.g. external library verbose, citations, etc.).

    Returns
    is_verbose : bool
        True if the terminal is configured to be verbose, False otherwise.
    """

    # If logging.root has no handlers this will ensure that False is returned
    is_verbose = False

    for handler in logging.root.handlers:
        # logging.FileHandler is a subclass of logging.StreamHandler so
        # isinstance and issubclass do not work in this case
        if type(handler) is logging.StreamHandler and handler.level <= logging.DEBUG:
            is_verbose = True
            break

    return is_verbose

def config_root_logger(verbose, log_file_path=None):
    """Setup the the root logger's configuration.

     The log messages are printed in the terminal and saved in the file specified
     by log_file_path (if not None) and printed. Note that logging use sys.stdout
     to print logging.INFO messages, and stderr for the others. The root logger's
     configuration is inherited by the loggers created by logging.getLogger(name).

     Different formats are used to display messages on the terminal and on the log
     file. For example, in the log file every entry has a timestamp which does not
     appear in the terminal. Moreover, the log file always shows the module that
     generate the message, while in the terminal this happens only for messages
     of level WARNING and higher.

    Parameters
    ----------
    verbose : bool
        Control the verbosity of the messages printed in the terminal. The logger
        displays messages of level logging.INFO and higher when verbose=False.
        Otherwise those of level logging.DEBUG and higher are printed.
    log_file_path : str, optional, default = None
        If not None, this is the path where all the logger's messages of level
        logging.DEBUG or higher are saved.

    """

    class TerminalFormatter(logging.Formatter):
        """
        Simplified format for INFO and DEBUG level log messages.

        This allows to keep the logging.info() and debug() format separated from
        the other levels where more information may be needed. For example, for
        warning and error messages it is convenient to know also the module that
        generates them.
        """

        # This is the cleanest way I found to make the code compatible with both
        # Python 2 and Python 3
        simple_fmt = logging.Formatter('%(asctime)-15s: %(message)s')
        default_fmt = logging.Formatter('%(asctime)-15s: %(levelname)s - %(name)s - %(message)s')

        def format(self, record):
            if record.levelno <= logging.INFO:
                return self.simple_fmt.format(record)
            else:
                return self.default_fmt.format(record)

    # Check if root logger is already configured
    n_handlers = len(logging.root.handlers)
    if n_handlers > 0:
        root_logger = logging.root
        for i in range(n_handlers):
            root_logger.removeHandler(root_logger.handlers[0])

    # If this is a worker node, don't save any log file
    from openmmtools.distributed import mpi
    mpicomm = mpi.get_mpicomm()
    if mpicomm:
        rank = mpicomm.rank
    else:
        rank = 0

    # Create different log files for each MPI process
    if rank != 0 and log_file_path is not None:
        basepath, ext = os.path.splitext(log_file_path)
        log_file_path = '{}_{}{}'.format(basepath, rank, ext)

    # Add handler for stdout and stderr messages
    terminal_handler = logging.StreamHandler()
    terminal_handler.setFormatter(TerminalFormatter())
    if rank != 0:
        terminal_handler.setLevel(logging.WARNING)
    elif verbose:
        terminal_handler.setLevel(logging.DEBUG)
    else:
        terminal_handler.setLevel(logging.INFO)
    logging.root.addHandler(terminal_handler)

    # Add file handler to root logger
    if log_file_path is not None:
        file_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(file_format))
        logging.root.addHandler(file_handler)

    # Do not handle logging.DEBUG at all if unnecessary
    if log_file_path is not None:
        logging.root.setLevel(logging.DEBUG)
    else:
        logging.root.setLevel(terminal_handler.level)

#=============================================================================================
# Python 2/3 compatability
#=============================================================================================

"""
Generate same behavior for dict.item in both versions of Python
Avoids external dependancies on future.utils or six

"""
try:
    dict.iteritems
except AttributeError:
    # Python 3
    def listvalues(d):
        return list(d.values())
    def listitems(d):
        return list(d.items())
    def dictiter(d):
        return d.items()

else:
    # Python 2
    def listvalues(d):
        return d.values()
    def listitems(d):
        return d.items()
    def dictiter(d):
        return d.iteritems()
        
#========================================================================================
# MAIN
#========================================================================================

if __name__ == '__main__':
    import doctest
    doctest.testmod()
