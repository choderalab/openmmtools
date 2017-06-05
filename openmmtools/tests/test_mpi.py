#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test MPI utility functions from openmm.distributed.mpi

"""

# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import numpy as np
from simtk import unit

from openmmtools.distributed.mpi import *


# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

NODE_RANK = 1  # The node rank passed to rank or send_results_to.


# =============================================================================
# UTILITY FUNCTION
# =============================================================================

def assert_is_equal(a, b):
    err_msg = '{} != {}'.format(a, b)
    try:
        assert a == b, err_msg
    except ValueError:
        # This is a list or tuple of numpy arrays.
        try:
            for element_a, element_b in zip(a, b):
                assert_is_equal(element_a, element_b)
        except AssertionError:
            raise AssertionError(err_msg)


# =============================================================================
# TEST CASES
# =============================================================================

def square(x):
    return x**2


def multiply(a, b):
    return a * b


@on_single_node(rank=NODE_RANK, broadcast_result=True)
def multiply_decorated_broadcast(a, b):
    return multiply(a, b)


@on_single_node(rank=NODE_RANK, broadcast_result=False)
def multiply_decorated_nobroadcast(a, b):
    return multiply(a, b)


class MyClass(object):

    def __init__(self, par):
        self.par = par

    @staticmethod
    def square_static(x):
        return x**2

    @staticmethod
    def multiply_static(a, b):
        return a * b

    @staticmethod
    @on_single_node(rank=NODE_RANK, broadcast_result=True)
    def multiply_decorated_broadcast_static(a, b):
        return a * b

    @classmethod
    @on_single_node(rank=NODE_RANK, broadcast_result=False)
    def multiply_decorated_nobroadcast_static(cls, a, b):
        return cls.multiply_static(a, b)

    def multiply_by_par(self, a):
        return self.par * a

    @on_single_node(rank=NODE_RANK, broadcast_result=True)
    def multiply_by_par_decorated_broadcast(self, a):
        return self.multiply_by_par(a)

    @on_single_node(rank=NODE_RANK, broadcast_result=False)
    def multiply_by_par_decorated_nobroadcast(self, a):
        return self.multiply_by_par(a)


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_run_single_node():
    """Test run_single_node function."""
    mpicomm = get_mpicomm()
    my_instance = MyClass(3.0)

    # Test_case: (function, args, kwargs)
    test_cases = [
        (multiply, (3, 4), {}),
        (multiply, (), {'a': 5, 'b': 4}),
        (multiply, (2, 'teststring',), {}),
        (multiply, (5, [3, 4],), {}),
        (multiply, (4, np.array([3, 4, 5.0]),), {}),
        (multiply, (3.0, unit.Quantity(np.array([3, 4, 5.0]), unit=unit.angstrom),), {}),
        (square, (5,), {}),
        (square, (), {'x': 2}),
        (MyClass.multiply_static, (3, 4), {}),
        (MyClass.multiply_static, (), {'a': 5, 'b': 4}),
        (MyClass.square_static, (5,), {}),
        (MyClass.square_static, (), {'x': 2}),
        (my_instance.multiply_by_par, (4,), {}),
        (my_instance.multiply_by_par, (), {'a': 2}),
    ]

    for task, args, kwargs in test_cases:
        expected_result = task(*args, **kwargs)
        for broadcast_result in [True, False]:
            result = run_single_node(NODE_RANK, task, *args,
                                     broadcast_result=broadcast_result, **kwargs)
            if not broadcast_result and mpicomm is not None and mpicomm.rank != NODE_RANK:
                assert result is None
            else:
                assert_is_equal(result, expected_result)


def test_on_single_node():
    """Test on_single_node decorator."""
    mpicomm = get_mpicomm()
    my_instance = MyClass(3.0)

    # Test case: (function, args, kwargs, broadcast_result, expected_result)
    test_cases = [
        (multiply_decorated_broadcast, (3, 4), {}, True, 12),
        (multiply_decorated_nobroadcast, (), {'a': 5, 'b': 5}, False, 25),
        (MyClass.multiply_decorated_broadcast_static, (3, 4), {}, True, 12),
        (MyClass.multiply_decorated_nobroadcast_static, (), {'a': 5, 'b': 5}, False, 25),
        (my_instance.multiply_by_par_decorated_broadcast, (4,), {}, True, 4 * my_instance.par),
        (my_instance.multiply_by_par_decorated_broadcast, (), {'a': 5}, True, 5 * my_instance.par),
    ]

    for task, args, kwargs, broadcast_result, expected_result in test_cases:
        result = task(*args, **kwargs)
        if not broadcast_result and mpicomm is not None and mpicomm.rank != NODE_RANK:
            assert result is None
        else:
            assert_is_equal(result, expected_result)


def test_distribute():
    """Test distribute function."""
    mpicomm = get_mpicomm()
    my_instance = MyClass(4)

    # Testcase: (function, distributed_args)
    test_cases = [
        (square, [1, 2, 3]),
        (MyClass.square_static, [1, 2, 3, 4]),
        (my_instance.multiply_by_par, [1, 2, 3, 4, 5]),
        (my_instance.multiply_by_par, ['a', 'b', 'c', 'd', 'e', 'f']),
        (my_instance.multiply_by_par, [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
        (my_instance.multiply_by_par, [np.array([1, 2]), np.array([3, 4]), np.array([5, 6]), np.array([7, 8])]),
        (my_instance.multiply_by_par, [unit.Quantity(np.array([1, 2]), unit=unit.angstrom),
                                       unit.Quantity(np.array([3, 4]), unit=unit.angstrom),
                                       unit.Quantity(np.array([5, 6]), unit=unit.angstrom)]),
    ]

    for task, distributed_args in test_cases:
        all_indices = list(range(len(distributed_args)))
        all_expected_results = [task(x) for x in distributed_args]

        # Determining full and partial results.
        if mpicomm is not None:
            partial_job_indices = list(range(mpicomm.rank, len(distributed_args), mpicomm.size))
        else:
            partial_job_indices = all_indices
        partial_expected_results = [all_expected_results[i] for i in partial_job_indices]

        result = distribute(task, distributed_args, send_results_to='all')
        assert_is_equal(result, all_expected_results)

        result = distribute(task, distributed_args, send_results_to=NODE_RANK)
        if mpicomm is not None and mpicomm.rank != NODE_RANK:
            assert_is_equal(result, (partial_expected_results, partial_job_indices))
        else:
            assert_is_equal(result, (all_expected_results, all_indices))

        result = distribute(task, distributed_args, send_results_to=None)
        assert_is_equal(result, (partial_expected_results, partial_job_indices))
