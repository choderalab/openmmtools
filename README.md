[![Linux Build Status](https://travis-ci.org/choderalab/openmmtools.png?branch=master)](https://travis-ci.org/choderalab/openmmtools)
<!--- [![Windows Build status](https://ci.appveyor.com/api/projects/status/80ov9tdffg5jkr7i/branch/master)](https://ci.appveyor.com/project/rmcgibbo/mdtraj-813/branch/master) -->
<!--- [![PyPI Version](https://badge.fury.io/py/openmmtools.png)](https://pypi.python.org/pypi/openmmtools) -->
<!--- [![Binstar Badge](https://binstar.org/omnia/openmmtools/badges/version.svg)](https://binstar.org/omnia/openmmtools) -->
<!--- [![Downloads](https://pypip.in/d/mdtraj/badge.png)](https://pypi.python.org/pypi/openmmtools) -->

# Various Python tools for OpenMM

## `openmmtools.testsystems`

This repository contains a suite of molecular systems that can be used
for the testing of various molecular mechanics related software.  The
idea is that this repository will host a number classes that generate
OpenMM objects for simulating the desired systems.

These classes will also contain the member functions that calculate known
analytical properties of these systems, enabling the proper testing.

## `scripts`

A script that may be useful in testing your OpenMM installation is installed:

* `test-openmm-platforms` will test the various platforms available to OpenMM to ensure that all systems in `openmmtools.testsystems` give consistent potential energies.
If differences in energies in excess of `ENERGY_TOLERANCE` (default: 0.06 kcal/mol) are detected, these systems will be serialized to XML for further debugging.
