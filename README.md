[![Linux Build Status](https://travis-ci.org/choderalab/openmm-testsystems.png?branch=master)](https://travis-ci.org/choderalab/openmm-testsystems)
<!--- [![Windows Build status](https://ci.appveyor.com/api/projects/status/80ov9tdffg5jkr7i/branch/master)](https://ci.appveyor.com/project/rmcgibbo/mdtraj-813/branch/master) -->
<!--- [![PyPI Version](https://badge.fury.io/py/openmm-testsystems.png)](https://pypi.python.org/pypi/openmm-testsystems) -->
<!--- [![Binstar Badge](https://binstar.org/omnia/openmm-testsystems/badges/version.svg)](https://binstar.org/omnia/openmm-testsystems) -->
<!--- [![Downloads](https://pypip.in/d/mdtraj/badge.png)](https://pypi.python.org/pypi/openmm-testsystems) -->

# OpenMM Test Systems

Test systems for OpenMM

# Description

This repository contains a suite of molecular systems that can be used
for the testing of various molecular mechanics related software.  The
idea is that this repository will host a number classes that generate
OpenMM objects for simulating the desired systems.

These classes will also contain the member functions that calculate known
analytical properties of these systems, enabling the proper testing.

Note: setup.py does not currently work.  Also, the doctests must be run
from the testsystems directory.  This will be fixed.
