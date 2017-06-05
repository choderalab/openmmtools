#!/usr/bin/env python

# ==============================================================================
# MODULE DOCSTRING
# ==============================================================================

"""Utilities to run on MPI.

Provide functions and decorators that simplify running the same code on
multiple nodes. One benefit is that serial and parallel code is exactly
the same.

Global variables
----------------
disable_mpi : bool
    Set this to True to force running serially.

Routines
--------
get_mpicomm
    Automatically detect and configure MPI execution and return an
    MPI communicator.
run_single_node
    Run a task on a single node.
on_single_node
    Decorator version of run_single_node.
distribute
    Map a task on a sequence of arguments on all the nodes.
delay_termination
    A context manager to delay the response to termination signals.
delayed_termination
    A decorator version of delay_termination.

"""


# ==============================================================================
# GLOBAL IMPORTS
# ==============================================================================

import os
import sys
import signal
import logging
from contextlib import contextmanager

# TODO drop this when we drop Python 2 support
from openmmtools.utils import wraps_py2

logger = logging.getLogger(__name__)

# ==============================================================================
# GLOBAL VARIABLES
# ==============================================================================

disable_mpi = False


# ==============================================================================
# MAIN FUNCTIONS
# ==============================================================================

def get_mpicomm():
    """Retrieve the MPI communicator for this execution.

    The function automatically detects if the program runs on MPI by checking
    specific environment variables set by various MPI implementations. On
    first execution, it modifies sys.excepthook and register a handler for
    SIGINT, SIGTERM, SIGABRT to call Abort() to correctly terminate all
    processes.

    Returns
    -------
    mpicomm : mpi4py communicator or None
        The communicator for this node, None if the program doesn't run
        with MPI.

    """
    # If MPI execution is forcefully disabled, return None.
    if disable_mpi:
        return None

    # If we have already initialized MPI, return the cached MPI communicator.
    if get_mpicomm._is_initialized:
        return get_mpicomm._mpicomm

    # Check for environment variables set by mpirun. Variables are from
    # http://docs.roguewave.com/threadspotter/2012.1/linux/manual_html/apas03.html
    variables = ['PMI_RANK', 'OMPI_COMM_WORLD_RANK', 'OMPI_MCA_ns_nds_vpid',
                 'PMI_ID', 'SLURM_PROCID', 'LAMRANK', 'MPI_RANKID',
                 'MP_CHILD', 'MP_RANK', 'MPIRUN_RANK']
    use_mpi = False
    for var in variables:
        if var in os.environ:
            use_mpi = True
            break

    # Return None if we are not running on MPI.
    if not use_mpi:
        logger.debug('Cannot find MPI environment. MPI disabled.')
        get_mpicomm._mpicomm = None
        get_mpicomm._is_initialized = True
        return get_mpicomm._mpicomm

    # Initialize MPI
    from mpi4py import MPI
    mpicomm = MPI.COMM_WORLD

    # Override sys.excepthook to abort MPI on exception
    def mpi_excepthook(type, value, traceback):
        sys.__excepthook__(type, value, traceback)
        sys.stdout.flush()
        sys.stderr.flush()
        if mpicomm.size > 1:
            mpicomm.Abort(1)
    # Use our eception handler
    sys.excepthook = mpi_excepthook

    # Catch sigterm signals
    def handle_signal(signal, frame):
        if mpicomm.size > 1:
            mpicomm.Abort(1)
    for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGABRT]:
        signal.signal(sig, handle_signal)

    # Cache and return the MPI communicator.
    get_mpicomm._is_initialized = True
    get_mpicomm._mpicomm = mpicomm

    # Report initialization
    logger.debug("MPI initialized on node {}/{}".format(mpicomm.rank+1, mpicomm.size))

    return mpicomm

get_mpicomm._is_initialized = False  # Static variable


def run_single_node(rank, task, *args, **kwargs):
    """Run task on a single node.

    If MPI is not activated, this simply runs locally.

    Parameters
    ----------
    task : callable
        The task to run on node rank.
    rank : int
        The rank of the MPI communicator that must execute the task.
    broadcast_result : bool, optional
        If True, the result is broadcasted to all nodes. If False,
        only the node executing the task will receive the return
        value of the task, and all other nodes will receive None
        (default is False).
    sync_nodes : bool, optional
        If True, the nodes will be synchronized at the end of the
        execution (i.e. the task will be blocking) even if the
        result is not broadcasted  (default is False).

    Other Parameters
    ----------------
    *args
        The ordered arguments to pass to task.
    **kwargs
        The keyword arguments to pass to task.

    Returns
    -------
    result
        The return value of the task. This will be None on all nodes
        that is not the rank unless broadcast_result is set to True.

    Examples
    --------
    >>> def add(a, b):
    ...     return a + b
    >>> # Run 3+4 on node 0.
    >>> run_single_node(0, task=add, a=3, b=4, broadcast_result=True)
    7

    """
    broadcast_result = kwargs.pop('broadcast_result', False)
    sync_nodes = kwargs.pop('sync_nodes', False)
    result = None
    mpicomm = get_mpicomm()

    if mpicomm is not None:
        node_name = 'Node {}/{}'.format(mpicomm.rank+1, mpicomm.size)
    else:
        node_name = 'Single node'

    # Execute the task only on the specified node.
    if mpicomm is None or mpicomm.rank == rank:
        logger.debug('{}: executing {}'.format(node_name, task))
        result = task(*args, **kwargs)

    # Broadcast the result if required.
    if mpicomm is not None:
        if broadcast_result is True:
            logger.debug('{}: waiting for broadcast of {}'.format(node_name, task))
            result = mpicomm.bcast(result, root=rank)
        elif sync_nodes is True:
            logger.debug('{}: waiting for barrier after {}'.format(node_name, task))
            mpicomm.barrier()

    # Return result.
    return result


def on_single_node(rank, broadcast_result=False, sync_nodes=False):
    """A decorator version of run_single_node.

    Decorates a function to be always executed with run_single_node.

    Parameters
    ----------
    rank : int
        The rank of the MPI communicator that must execute the task.
    broadcast_result : bool, optional
        If True the result is broadcasted to all nodes. If False,
        only the node executing the function will receive its return
        value, and all other nodes will receive None (default is False).
    sync_nodes : bool, optional
        If True, the nodes will be synchronized at the end of the
        execution (i.e. the task will be blocking) even if the
        result is not broadcasted (default is False).

    See Also
    --------
    run_single_node

    Examples
    --------
    >>> @on_single_node(rank=0, broadcast_result=True)
    ... def add(a, b):
    ...     return a + b
    >>> add(3, 4)
    7

    """
    def _on_single_node(task):
        @wraps_py2(task)
        def _wrapper(*args, **kwargs):
            kwargs['broadcast_result'] = broadcast_result
            kwargs['sync_nodes'] = sync_nodes
            return run_single_node(rank, task, *args, **kwargs)
        return _wrapper
    return _on_single_node


def distribute(task, distributed_args, *other_args, **kwargs):
    """Map the task on a sequence of arguments to be executed on different nodes.

    If MPI is not activated, this simply runs serially on this node. The
    algorithm guarantees that each node will be assigned to the same job_id
    (i.e. the index of the argument in distributed_args) every time.

    Parameters
    ----------
    task : callable
        The task to be distributed among nodes. The task will be called as
        task(distributed_args[job_id], *other_args, **kwargs), so the parameter
        to be distributed must the the first one.
    distributed_args : iterable
        The sequence of the parameters to distribute among nodes.
    send_results_to : int or 'all', optional
        If the string 'all', the result will be sent to all nodes. If an
        int, the result will be send only to the node with rank send_results_to.
        The return value of distribute depends on the value of this parameter
        (default is None).
    sync_nodes : bool, optional
        If True, the nodes will be synchronized at the end of the
        execution (i.e. the task will be blocking) even if the
        result is not shared (default is False).

    Other Parameters
    ----------------
    *other_args
        Other parameters to pass to task beside the assigned distributed
        parameters.
    **kwargs
        Keyword arguments to pass to task beside the assigned distributed
        parameters.

    Returns
    -------
    all_results : list
        All the return values for all the arguments if the results where sent
        to the node, or only the return values of the arguments processed by
        this node otherwise.
    arg_indices : list of int, optional
        This is returned as part of a tuple (all_results, job_indices) only
        if send_results_to is set to an int or None. In this case all_results[i]
        is the return value of task(all_args[arg_indices[i]]).

    Examples
    --------
    >>> def square(x):
    ...     return x**2
    >>> distribute(square, [1, 2, 3, 4], send_results_to='all')
    [1, 4, 9, 16]

    When send_results_to is not set to `all`, the return value include also
    the indices of the arguments associated to the result.

    >>> distribute(square, [1, 2, 3, 4], send_results_to=0)
    ([1, 4, 9, 16], [0, 1, 2, 3])

    """
    send_results_to = kwargs.pop('send_results_to', None)
    sync_nodes = kwargs.pop('sync_nodes', False)
    mpicomm = get_mpicomm()
    n_jobs = len(distributed_args)

    # If MPI is not activated, just run serially.
    if mpicomm is None:
        logger.debug('Running {} serially.'.format(task.__name__))
        all_results = [task(job_args, *other_args, **kwargs) for job_args in distributed_args]
        if send_results_to == 'all':
            return all_results
        else:
            return all_results, list(range(n_jobs))

    node_job_ids = range(mpicomm.rank, n_jobs, mpicomm.size)

    # Compute all the results assigned to this node.
    results = []
    node_name = 'Node {}/{}'.format(mpicomm.rank+1, mpicomm.size)
    for job_id in node_job_ids:
        distributed_arg = distributed_args[job_id]
        logger.debug('{}: execute {}({})'.format(node_name, task.__name__, distributed_arg))
        results.append(task(distributed_arg, *other_args, **kwargs))

    # Share result as specified.
    if send_results_to == 'all':
        logger.debug('{}: allgather results of {}'.format(node_name, task.__name__))
        all_results = mpicomm.allgather(results)
    elif isinstance(send_results_to, int):
        logger.debug('{}: gather results of {}'.format(node_name, task.__name__))
        all_results = mpicomm.gather(results, root=send_results_to)

        # If this is not the receiving node, we can safely return.
        if mpicomm.rank != send_results_to:
            return results, list(node_job_ids)
    else:
        assert send_results_to is None  # Safety check.
        if sync_nodes is True:
            logger.debug('{}: waiting for barrier after {}'.format(node_name, task.__name__))
            mpicomm.barrier()
        return results, list(node_job_ids)

    # all_results is a list of list of results. The internal lists of
    # results are ordered by rank. We need to reorder the results as a
    # flat list or results ordered by job_id.

    # job_indices[job_id] is the tuple of indices (rank, i). The result
    # of job_id is stored in all_results[rank][i].
    job_indices = []
    max_jobs_per_node = max([len(r) for r in all_results])
    for i in range(max_jobs_per_node):
        for rank in range(mpicomm.size):
            # Not all nodes have executed max_jobs_per_node tasks.
            if len(all_results[rank]) > i:
                job_indices.append((rank, i))

    # Reorder the results.
    all_results = [all_results[rank][i] for rank, i in job_indices]

    # Return result.
    if send_results_to == 'all':
        return all_results
    else:
        return all_results, list(range(n_jobs))


@contextmanager
def delay_termination():
    """Context manager to delay handling of termination signals.

    This allows to avoid interrupting tasks such as writing to the file
    system, which could result in the corruption of the file.

    """
    signals_to_catch = [signal.SIGINT, signal.SIGTERM, signal.SIGABRT]
    old_handlers = {signum: signal.getsignal(signum) for signum in signals_to_catch}
    signals_received = {signum: None for signum in signals_to_catch}

    def delay_handler(signum, frame):
        signals_received[signum] = (signum, frame)

    # Set handlers fot delay
    for signum in signals_to_catch:
        signal.signal(signum, delay_handler)

    yield  # Resume program

    # Restore old handlers
    for signum, handler in old_handlers.items():
        signal.signal(signum, handler)

    # Fire delayed signals
    for signum, s in signals_received.items():
        if s is not None:
            old_handlers[signum](*s)


def delayed_termination(func):
    """Decorator that runs the function with delay_termination()."""
    @wraps_py2(func)
    def _delayed_termination(*args, **kwargs):
        with delay_termination():
            return func(*args, **kwargs)
    return _delayed_termination


# ==============================================================================
# MAIN AND TESTS
# ==============================================================================

if __name__ == "__main__":
    import doctest
    doctest.testmod()
