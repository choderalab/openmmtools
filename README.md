[![Linux Build Status](https://travis-ci.org/choderalab/openmmtools.png?branch=master)](https://travis-ci.org/choderalab/openmmtools/branches)
[![Windows Build status](https://ci.appveyor.com/api/projects/status/70knpvcgvmah2qin?svg=true)](https://ci.appveyor.com/project/jchodera/openmmtools)
[![Anaconda Badge](https://anaconda.org/omnia/openmmtools/badges/version.svg)](https://anaconda.org/omnia/openmmtools)
[![Downloads Badge](https://anaconda.org/omnia/openmmtools/badges/downloads.svg)](https://anaconda.org/omnia/openmmtools/files)
[![ReadTheDocs Badge](https://readthedocs.org/projects/openmmtools/badge/?version=master)](https://openmmtools.readthedocs.io/en/master/)
[![Zenodo DOI Badge](https://zenodo.org/badge/25416166.svg)](https://zenodo.org/badge/latestdoi/25416166)

## OpenMMTools

A batteries-included toolkit for the GPU-accelerated OpenMM molecular simulation engine.

`openmmtools` is a Python library layer that sits on top of [OpenMM](http://openmm.org) to provide access to a variety of useful tools for building full-featured molecular simulation packages.

Features include:

 - high-quality [Langevin integrators](https://openmmtools.readthedocs.io/en/stable/integrators.html#langevin-integrators), including [g-BAOAB](http://rspa.royalsocietypublishing.org/content/472/2189/20160138), [VVVR](http://pubs.acs.org/doi/abs/10.1021/jp411770f), and other splittings
 - [nonequilibrium integrators](https://openmmtools.readthedocs.io/en/stable/integrators.html#nonequilibrium-integrators) for free energy calculations or [nonequilibrium candidate Monte Carlo (NCMC)](http://dx.doi.org/10.1073/pnas.1106094108)
 - an extensible [Markov chain Monte Carlo (MCMC) framework](https://openmmtools.readthedocs.io/en/stable/mcmc.html) for molecular simulations
 - enhanced sampling methods, including replica-exchange (REMD) and self-adjusted mixture sampling (SAMS)
 - [alchemical factories](https://openmmtools.readthedocs.io/en/stable/alchemy.html) for generating [alchemically-modified](http://alchemistry.org) systems for absolute and relative free energy calculations
 - a suite of [test systems](https://openmmtools.readthedocs.io/en/stable/testsystems.html) for benchmarking, validation, and debugging
 - user-friendly storage interface layer

See the [full documentation](http://openmmtools.readthedocs.io) at [ReadTheDocs](http://openmmtools.readthedocs.io).

#### License

OpenMMTools is distributed under the [MIT License](https://opensource.org/licenses/MIT).

#### Contributors

A complete list of contributors can be found [here](https://github.com/choderalab/openmmtools/graphs/contributors).

Major contributors include:

* Andrea Rizzi `<andrea.rizzi@choderalab.org>` (WCMC)
* John D. Chodera `<john.chodera@choderalab.org>` (MSKCC)
* Levi N. Naden `<levi.naden@choderalab.org>` (MSKCC)
* Patrick Grinaway `<patrick.grinaway@choderalab.org>` (MSKCC)
* Kyle A. Beauchamp `<kyle.beauchamp@choderalab.org>` (MSKCC)
* Josh Fass `<josh.fass@choderalab.org>` (MSKCC)
* Bas Rustenburg `<bas.rustenburg@choderalab.org>` (MSKCC)
* Gregory Ross `<greg.ross@choderalab.org>` (MSKCC)
* David W.H. Swenson `<dwhs@hyperblazer.net>`
* Hannah Bruce Macdonald `<hannah.brucemacdonald>` (MSKCC)
