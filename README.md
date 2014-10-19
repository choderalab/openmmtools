OpenMM Test Systems
===========

This repository contains a suite of molecular systems that can be used
for the testing of various molecular mechanics related software.  The
idea is that this repository will host a number classes that generate
OpenMM objects for simulating the desired systems.  

These classes will also contain the member functions that calculate known
analytical properties of these systems, enabling the proper testing.

Note: setup.py does not currently work.  Also, the doctests must be run
from the testsystems directory.  This will be fixed.
