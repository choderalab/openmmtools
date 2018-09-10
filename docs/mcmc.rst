.. _mcmc:

Markov chain Monte Carlo (MCMC)
===============================

``openmmtools`` provides an extensible Markov chain Monte Carlo simulation framework.

This module provides a framework for equilibrium sampling from a given thermodynamic state of a biomolecule using a Markov chain Monte Carlo scheme.

It currently offer supports for

 - Langevin dynamics (assumed to be free of integration error; use at your own risk]),
 - hybrid Monte Carlo,
 - generalized hybrid Monte Carlo, and
 - Monte Carlo barostat moves,

which can be combined through the ``SequenceMove`` and ``WeightedMove`` classes.

By default, the ``MCMCMoves`` use the fastest OpenMM platform available and a shared global ``ContextCache`` that minimizes the number of OpenMM ``Context`` objects that must be maintained at once.
The examples below show how to configure these aspects.

.. NOTE::
   To use the ``ContextCache`` on the CUDA platform, the NVIDIA driver must be set to ``shared`` mode to allow the process to create multiple GPU contexts.

Using the MCMC framework requires importing :class:`ThermodynamicState` and :class:`SamplerState` from :mod:`openmmtools.states`:

::

    from simtk import unit
    from openmmtools import testsystems, cache
    from openmmtools.states import ThermodynamicState, SamplerState

Create the initial state (thermodynamic and microscopic) for an alanine
dipeptide system in vacuum.

::

    test = testsystems.AlanineDipeptideVacuum()
    thermodynamic_state = ThermodynamicState(system=test.system, temperature=298*unit.kelvin)
    sampler_state = SamplerState(positions=test.positions)

Create an MCMC move to perform at every iteration of the simulation, and
initialize a sampler instance.

::

    ghmc_move = GHMCMove(timestep=1.0*unit.femtosecond, n_steps=50)
    langevin_move = LangevinDynamicsMove(n_steps=10)
    sampler = MCMCSampler(thermodynamic_state, sampler_state, move=ghmc_move)

You can combine them to form a sequence of moves

::

    sequence_move = SequenceMove([ghmc_move, langevin_move])
    sampler = MCMCSampler(thermodynamic_state, sampler_state, move=sequence_move)

or create a move that selects one of them at random with given probability
at each iteration.

::

    weighted_move = WeightedMove([(ghmc_move, 0.5), (langevin_move, 0.5)])
    sampler = MCMCSampler(thermodynamic_state, sampler_state, move=weighted_move)

By default the ``MCMCMove`` use a global ContextCache that creates ``Context`` on the
fastest available OpenMM platform. You can configure the default platform to use
before starting the simulation

::

    reference_platform = openmm.Platform.getPlatformByName('Reference')
    cache.global_context_cache.platform = reference_platform
    cache.global_context_cache.time_to_live = 10  # number of read/write operations

Minimize and run the simulation for few iterations.

::

    sampler.minimize()
    sampler.run(n_iterations=2)

If you don't want to use a global cache, you can create local ones.

::

    local_cache1 = cache.ContextCache(capacity=5, time_to_live=50)
    local_cache2 = cache.ContextCache(platform=reference_platform, capacity=1)
    sequence_move = SequenceMove([HMCMove(), LangevinDynamicsMove()], context_cache=local_cache1)
    ghmc_move = GHMCMove(context_cache=local_cache2)

If you don't want to cache ``Context`` at all but create one every time, you can use
the ``DummyCache``.

::

    dummy_cache = cache.DummyContextCache(platform=reference_platform)
    ghmc_move = GHMCMove(context_cache=dummy_cache)

This book by Jun Liu is an excellent overview of Markov chain Monte Carlo:

Jun S. Liu. Monte Carlo Strategies in Scientific Computing. Springer, 2008.

MCMC samplers
-------------

An MCMC sampler driver is provided that can either utilize a programmed sequence of moves or draw from a weighted set of moves.

.. currentmodule:: openmmtools.mcmc
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    MCMCSampler
    SequenceMove
    WeightedMove

MCMC move types
---------------

A number of MCMC component move types that can be arranged into groups or subclassed are provided.

.. currentmodule:: openmmtools.mcmc
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    MCMCMove
    BaseIntegratorMove
    MetropolizedMove
    IntegratorMove
    LangevinDynamicsMove
    LangevinSplittingDynamicsMove
    GHMCMove
    HMCMove
    MonteCarloBarostatMove
    MCDisplacementMove
    MCRotationMove
