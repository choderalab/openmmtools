[![Linux Build Status](https://travis-ci.org/choderalab/openmmtools.png?branch=master)](https://travis-ci.org/choderalab/openmmtools)
[![Windows Build status](https://ci.appveyor.com/api/projects/status/70knpvcgvmah2qin?svg=true)](https://ci.appveyor.com/project/jchodera/openmmtools)
[![Binstar Badge](https://binstar.org/omnia/openmmtools/badges/version.svg)](https://binstar.org/omnia/openmmtools)
<!--- [![PyPI Version](https://badge.fury.io/py/openmmtools.png)](https://pypi.python.org/pypi/openmmtools) -->
<!--- [![Downloads](https://pypip.in/d/mdtraj/badge.png)](https://pypi.python.org/pypi/openmmtools) -->

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

The `openmmtools.testsystems` module contains a large suite of test systems---including many with simple exactly-computable properties---that can be used to test molecular simulation algorith,s

## OpenMM testing scripts

`scripts/` contains a script that may be useful in testing your OpenMM installation is installed:

* `test-openmm-platforms` will test the various platforms available to OpenMM to ensure that all systems in `openmmtools.testsystems` give consistent potential energies.
If differences in energies in excess of `ENERGY_TOLERANCE` (default: 0.06 kcal/mol) are detected, these systems will be serialized to XML for further debugging.

This is installed onto the command line when the repository is installed.
