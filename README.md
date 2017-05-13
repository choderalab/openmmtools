[![Linux Build Status](https://travis-ci.org/choderalab/openmmtools.png?branch=master)](https://travis-ci.org/choderalab/openmmtools)
[![Windows Build status](https://ci.appveyor.com/api/projects/status/70knpvcgvmah2qin?svg=true)](https://ci.appveyor.com/project/jchodera/openmmtools)
[![Anaconda Badge](https://anaconda.org/omnia/openmmtools/badges/version.svg)](https://anaconda.org/omnia/openmmtools)
[![Downloads Badge](https://anaconda.org/omnia/openmmtools/badges/downloads.svg)](https://anaconda.org/omnia/openmmtools/files)
[![Documentation Status](https://readthedocs.org/projects/openmmtools/badge/?version=latest)](http://openmmtools.readthedocs.io/en/latest/?badge=latest)

# Various Python tools for OpenMM

## Integrators

This repository contains a number of additional integrators for OpenMM in `openmmtools.integrators`, including
* `MTSIntegrator` - a multiple timestep integrator
* `DummyIntegrator` - a "dummy" integrator that does not update positions
* `GradientDescentMinimizationIntegrator` - a simple gradient descent minimizer (without line search)
* `VelocityVerletIntegrator` - a velocity Verlet integrator
* `AndersenVelocityVerletIntegrator` - a velocity Verlet integrator with Andersen thermostat using per-particle collisions
* `MetropolisMonteCarloIntegrator` - a Metropolis Monte Carlo integrator that uses Gaussian displacement trials
* `HMCIntegrator` - a hybrid Monte Carlo (HMC) integrator
* `GHMCIntegrator` - a generalized hybrid Monte Carlo (GHMC) integrator
* `VVVRIntegrator` - a velocity Verlet with velocity randomization (VVVR) integrator

## Test system suite

The `openmmtools.testsystems` module contains a large suite of test systems---including many with simple exactly-computable properties---that can be used to test molecular simulation algorithms

## Markov chain Monte Carlo proposal schemes and move compositions
An implementation of an `MCMCMove` encodes how to propagate an OpenMM `System` to generate a new sample. Different `MCMCMove`s can be combined for more advanced schemes.
- `LangevinDynamicsMove`: Langevin dynamics segment as a (pseudo) Monte Carlo move (WARNING: Does not preserve the true target distribution.).
- `HMCMove`: Assigns velocities from the Maxwell-Boltzmann distribution and propagate through velocity Verlet steps.
- `GHMCMove`: Generalized hybrid Monte Carlo Markov chain Monte Carlo.
- `MonteCarloBarostatMove`: Attempts to update the box volume using Metropolis-Hastings Monte Carlo updates.
- `SequenceMove`: Combine multiple `MCMCMove`s to apply in sequence at each iteration.
- `WeightedMove`: At each iteration, select one `MCMCMove`s to apply from a set with given probability.

## States

The module `openmmtools.states` contains classes to maintain a consistent state of the simulation.
- `ThermodynamicState`: Represent and manipulate the thermodynamic state of OpenMM `System`s and `Context`s.
- `SamplerState`: Represent and cache the state of the simulation that changes when the `System` is integrated.
- `CompoundThermodynamicState`: Extend the `ThermodynamicState` to handle parameters other than temperature and pressure through the implementations of the `IComposableState` abstract class.

## Cache

The module `openmmtools.cache` implements a shared LRU cache for `Context` objects that tries to minimize the number of `Context` in memory at the same time.
- `LRUCache`: A simple LRU cache with a dictionary-like interface. It supports a maximum capacity and expiration.
- `ContextCache`: A LRU cache for OpenMM `Context` objects.
- `global_context_cache`: A shared `ContextCache` that minimizes the number of `Context` creations when employing `MCMCMove`s.

## OpenMM testing scripts

`scripts/` contains a script that may be useful in testing your OpenMM installation is installed:

* `test-openmm-platforms` will test the various platforms available to OpenMM to ensure that all systems in `openmmtools.testsystems` give consistent potential energies.
If differences in energies in excess of `ENERGY_TOLERANCE` (default: 0.06 kcal/mol) are detected, these systems will be serialized to XML for further debugging.

This is installed onto the command line when the repository is installed.
