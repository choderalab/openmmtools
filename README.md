[![Linux Build Status](https://travis-ci.org/choderalab/openmmtools.png?branch=master)](https://travis-ci.org/choderalab/openmmtools)
[![Windows Build status](https://ci.appveyor.com/api/projects/status/70knpvcgvmah2qin?svg=true)](https://ci.appveyor.com/project/jchodera/openmmtools)
[![Anaconda Badge](https://anaconda.org/omnia/openmmtools/badges/version.svg)](https://anaconda.org/omnia/openmmtools)
[![Downloads Badge](https://anaconda.org/omnia/openmmtools/badges/downloads.svg)](https://anaconda.org/omnia/openmmtools/files)
[![ReadTheDocs Badge](https://readthedocs.org/projects/openmmtools/badge/?version=latest)](http://openmmtools.readthedocs.io/en/latest/?badge=latest)
[![Zenodo DOI Badge](https://zenodo.org/badge/25416166.svg)](https://zenodo.org/badge/latestdoi/25416166)

## OpenMMTools

A batteries-included toolkit for the GPU-accelerated OpenMM molecular simulation engine.

``openmmtools`` is a Python library layer that sits on top of `OpenMM <http://openmm.org>`_ to provide access to a variety of useful tools for building full-featured molecular simulation packages.

Features include:

 - high-quality Langevin integrators, including [g-BAOAB](http://rspa.royalsocietypublishing.org/content/472/2189/20160138), [VVVR](http://pubs.acs.org/doi/abs/10.1021/jp411770f), and other splittings
 - integrators that support nonequilibrium switching for free energy calculations or [nonequilibrium candidate Monte Carlo (NCMC)](http://dx.doi.org/10.1073/pnas.1106094108)
 - an extensible Markov chain Monte Carlo framework for mixing Monte Carlo and molecular dynamics-based methods
 - enhanced sampling methods, including replica-exchange (REMD) and self-adjusted mixture sampling (SAMS)
 - factories for generating [alchemically-modified](http://alchemistry.org) systems for absolute and relative free energy calculations
 - a suite of test systems for benchmarking, validation, and debugging

See the [documentation](http://openmmtools.readthedocs.io) at [ReadTheDocs](http://openmmtools.readthedocs.io).

#### License

OpenMMTools is distributed under the MIT License.
