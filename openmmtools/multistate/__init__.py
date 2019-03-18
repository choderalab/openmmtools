#!/usr/local/bin/env python

# ==============================================================================
# MODULE DOCSTRING
# ==============================================================================

"""
MultiState
==========

Multistate Sampling simulation algorithms, specific variants, and analyzers

This module provides a general facility for running multiple thermodynamic state multistate simulations, both general
as well as derived classes for special cases such as parallel tempering (in which
the states differ only in temperature).

The classes also provide

Provided classes include:

- :class:`openmmtools.multistate.MultiStateSampler`
    Base class for general, multi-thermodynamic state parallel multistate
- :class:`openmmtools.multistate.ReplicaExchangeSampler`
    Derived class from MultiStateSampler which allows sampled thermodynamic states
    to swap based on Hamiltonian Replica Exchange
- :class:`openmmtools.multistate.ParallelTemperingSampler`
    Convenience subclass of ReplicaExchange for parallel tempering simulations
    (one System object, many temperatures).
- :class:`openmmtools.multistate.SAMSSampler`
    Single-replica sampler which samples through multiple thermodynamic states
    on the fly.
- :class:`openmmtools.multistate.MultiStateReporter`
    Replica Exchange reporter class to store all variables and data

Analyzers
---------
The MultiState module also provides analysis modules to analyze simulations and compute observables from data generated
under any of the MultiStateSampler's

Extending and Subclassing
-------------------------
Subclassing a sampler and analyzer is done by importing and extending any of the following:

    * The base ``MultiStateSampler`` from ``multistatesampler``
    * The base ``MultiStateReporter`` from ``multistatereporter``
    * The base ``MultiStateAnalyzer`` or ``PhaseAnalyzer`` and base `ObservablesRegistry`` from ``multistateanalyzer``

COPYRIGHT
---------

Current version by Andrea Rizzi <andrea.rizzi@choderalab.org>, Levi N. Naden <levi.naden@choderalab.org> and
John D. Chodera <john.chodera@choderalab.org> while at Memorial Sloan Kettering Cancer Center.

Original version by John D. Chodera <jchodera@gmail.com> while at the University of
California Berkeley.

LICENSE
-------

This code is licensed under the latest available version of the MIT License.

"""

from openmmtools.multistate.multistatesampler import MultiStateSampler
from openmmtools.multistate.multistatereporter import MultiStateReporter
from openmmtools.multistate.replicaexchange import ReplicaExchangeSampler, ReplicaExchangeAnalyzer
from openmmtools.multistate.paralleltempering import ParallelTemperingSampler, ParallelTemperingAnalyzer
from openmmtools.multistate.sams import SAMSSampler, SAMSAnalyzer
from openmmtools.multistate.multistateanalyzer import *
from openmmtools.multistate.utils import *
