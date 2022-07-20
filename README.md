[![GH Actions Status](https://github.com/choderalab/openmmtools/workflows/CI/badge.svg)](https://github.com/choderalab/openmmtools/actions?query=branch%3Amain)
[![Anaconda Badge](https://anaconda.org/omnia/openmmtools/badges/version.svg)](https://anaconda.org/omnia/openmmtools)
[![Downloads Badge](https://anaconda.org/omnia/openmmtools/badges/downloads.svg)](https://anaconda.org/omnia/openmmtools/files)
[![Documentation Status](https://readthedocs.org/projects/openmmtools/badge/?version=latest)](https://openmmtools.readthedocs.io/en/latest/?badge=latest)
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

* [Andrea Rizzi](https://github.com/andrrizzi) 
* [John D. Chodera](https://github.com/jchodera)
* [Levi N. Naden](https://github.com/Lnaden)
* [Patrick Grinaway](https://github.com/pgrinaway)
* [Kyle A. Beauchamp](https://github.com/kyleabeauchamp)
* [Josh Fass](https://github.com/maxentile)
* [Bas Rustenburg](https://github.com/bas-rustenburg)
* [Gregory Ross](https://github.com/gregoryross)
* [David W.H. Swenson](https://github.com/dwhswenson)
* [Hannah Bruce Macdonald](https://github.com/hannahbrucemacdonald)
* [Iván Pulido](https://github.com/ijpulidos)
* [Ivy Zhang](https://github.com/zhang-ivy)
* [Dominic Rufa](https://github.com/dominicrufa)
* [Mike Henry](https://github.com/mikemhenry)