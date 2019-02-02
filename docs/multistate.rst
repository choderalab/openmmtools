.. _multistate:

Sampling multiple thermodynamic states
======================================

``openmmtools`` provides several schemes for sampling from multiple thermodynamic states within a single calculation:

* ``MultistateSampler``: Independent simulations at distinct thermodynamic states
* ``ReplicaExchangeSampler``: Replica exchange among thermodynamic states (also called Hamiltonian exchange if only the Hamiltonian is changing)
* ``SAMSSampler``: Self-adjusted mixture sampling (also known as optimally-adjusted mixture sampling)

While the thermodynamic states sampled usually differ only in the alchemical parameters, other thermodynamic parameters (such as temperature) can be modulated as well at intermediate alchemical states.
This may be useful in, for example, experimenting with ways to reduce correlation times.

In all of these schemes, one or more **replicas** is simulated.
Each iteration includes the following phases:
 * Allow replicas to switch thermodynamic states (optional)
 * Allow replicas to sample a new configuration using Markov chain Monte Carlo (MCMC)
 * Each replica computes the potential energy of the current configuration in multiple thermodynamic states
 * Data is written to disk

Below, we describe some of the aspects of these samplers.

``MultiStateSampler``: Independent simulations at multiple thermodynamic states
-------------------------------------------------------------------------------

The ``MultiStateSampler`` allows independent simulations from multiple thermodynamic states to be sampled.
In this case, the MCMC scheme is used to propagate each replica by sampling from a fixed thermodynamic state.

.. math::

   s_{k,n+1} = s_{k, n} \\
   x_{k,n+1} \sim p(x | s_{k, n+1})

An inclusive "neighborhood" of thermodynamic states around this specified state can be used to define which thermodynamic states the reduced potential should be computed for after each iteration.
If all thermodynamic states are included in this neighborhood (the default), the MBAR scheme :cite:`Shirts2008statistically` can be used to optimally estimate free energies and uncertainties.
If a restricted neighborhood is used (in order to reduce the amount of time spent in the energy evaluation stage), a variant of the L-WHAM (local weighted histogram analysis method) :cite:`kumar1992weighted` is used to extract an estimate from all available information.

.. currentmodule:: openmmtools.multistate
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    MultiStateSampler
    MultiStateSamplerAnalyzer

``ReplicaExchangeSampler``: Replica exchange among thermodynamic states
-----------------------------------------------------------------------

The ``ReplicaExchangeSampler`` implements a Hamiltonian replica exchange scheme with Gibbs sampling :cite:`Chodera2011` to sample multiple thermodynamic states in a manner that improves mixing of the overall Markov chain.
By allowing replicas to execute a random walk in thermodynamic state space, correlation times may be reduced when sampling certain thermodynamic states (such as those with alchemically-softened potentials or elevated temperatures).

In the basic version of this scheme, a proposed swap of configurations between two alchemical states, *i* and *j*, made by comparing the energy of each configuration in each replica and swapping with a basic Metropolis criteria of

.. math::
    P_{\text{accept}}(i, x_i, j, x_j) &= \text{min}\begin{cases}
                               1, \frac{ e^{-\left[u_i(x_j) + u_j(x_i)\right]}}{e^{-\left[u_i(x_i) + u_j(x_j)\right]}}
                               \end{cases} \\
        &= \text{min}\begin{cases}
          1, \exp\left[\Delta u_{ji}(x_i) + \Delta u_{ij}(x_j)\right]
          \end{cases}

where :math:`x` is the configuration of the subscripted states :math:`i` or :math:`j`, and :math:`u` is the reduced potential energy.
While this scheme is typically carried out on neighboring states only, we also implement a much more efficient form of Gibbs sampling in which many swaps are attempted to generate an approximately uncorrelated sample of the state permutation over all :math:`K` :cite:`Chodera2011`.
This speeds up mixing and reduces the total number of samples needed to produce uncorrelated samples.

.. currentmodule:: openmmtools.multistate
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    ReplicaExchangeSampler
    ReplicaExchangeAnalyzer

``SAMSSampler``: Self-adjusted mixture sampling
-----------------------------------------------

The ``SAMSSampler`` implements self-adjusted mixture sampling (SAMS; also known as optimally adjusted mixture sampling) :cite:`Tan2017:SAMS`.
This combines one or more replicas that sample from an expanded ensemble with an asymptotically optimal Wang-Landau-like weight update scheme.

.. math::

   s_{k,n+1} = p(s | x_{k,n}) \\
   x_{k,n+1} \sim p(x | s_{k, n+1})

SAMS state update schemes
^^^^^^^^^^^^^^^^^^^^^^^^^

Several state update schemes are available:

* ``global-jump`` (default): The sampler can jump to any thermodynamic state (RECOMMENDED)
* ``restricted-range-jump``: The sampler can jump to any thermodynamic state within the specified local neighborhood (EXPERIMENTAL; DISABLED)
* ``local-jump``: Only proposals within the specified neighborhood are considered, but rejection rates may be high (EXPERIMENTAL; DISABLED)

SAMS Locality
^^^^^^^^^^^^^

The local neighborhood is specified by the ``locality`` parameter.
If this is a positive integer, the neighborhood will be defined by state indices ``[k - locality, k + locality]``.
Reducing locality will restrict the range of states for which reduced potentials are evaluated, which can speed up the energy evaluation stage of each iteration at the cost of restricting the amount of information available for free energy estimation.
By default, the ``locality`` is global, such that energies at all thermodynamic states are computed; this allows the use of MBAR in data analysis.

SAMS weight adaptation algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SAMS provides two ways of accumulating log weights each iteration:

* ``optimal`` accumulates weight only in the currently visited state ``s``
* ``rao-blackwellized`` accumulates fractional weight in all states within the energy evaluation neighborhood

SAMS initial weight adaptation stage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Because the asymptotically-optimal weight adaptation scheme works best only when the log weights are close to optimal, a heuristic initial stage is used to more rapidly adapt the log weights before the asymptotically optimal scheme is used.
The behavior of this first stage can be controlled by setting two parameters:

* ``gamma0`` controls the initial rate of weight adaptation. By default, this is 1.0, but can be set larger (e.g., 10.0) if the free energy differences between states are much larger.
* ``flatness_threshold`` controls the number of (fractional) visits to each thermodynamic state that must be accumulated before the asymptotically optimal weight adaptation scheme is used.

.. currentmodule:: openmmtools.multistate
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    SAMSSampler
    SAMSAnalyzer

Parallel tempering
------------------

.. currentmodule:: openmmtools.multistate
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    ParallelTemperingSampler
    ParallelTemperingAnalyzer

Multistate Reporters
--------------------

.. currentmodule:: openmmtools.multistate
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    MultiStateReporter

Analysis of multiple thermodynamic transformations
--------------------------------------------------

.. currentmodule:: openmmtools.multistate
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    MultiPhaseAnalyzer

Miscellaneous support classes
-----------------------------

.. currentmodule:: openmmtools.multistate.multistateanalyzer
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    ObservablesRegistry
    CachedProperty
    InsufficientData
    PhaseAnalyzer
