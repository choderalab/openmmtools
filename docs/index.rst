.. openmmtools documentation master file, created by
   sphinx-quickstart on Sat May 13 13:35:37 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. caution::

    This is module is undergoing heavy development. None of the API calls are final.
    This software is provided without any guarantees of correctness, you will likely encounter bugs.

    If you are interested in this code, please wait for the official release to use it.
    In the mean time, to stay informed of development progress you are encouraged to:

    * Follow `this feed`_ for updates on releases.
    * Check out the `github repository`_ .

.. _this feed: https://github.com/choderalab/openmmtools/releases.atom
.. _github repository: https://github.com/choderalab/openmmtools

OpenMMTools
===========

A batteries-included toolkit for the GPU-accelerated OpenMM molecular simulation engine.

``openmmtools`` is a Python library layer that sits on top of `OpenMM <http://openmm.org>`_ to provide access to a variety
of useful tools for building full-featured molecular simulation packages.

Features include:

 - high-quality Langevin integrators, including `g-BAOAB <http://rspa.royalsocietypublishing.org/content/472/2189/20160138>`_, `VVVR <http://pubs.acs.org/doi/abs/10.1021/jp411770f>`_, and other splittings
 - integrators that support nonequilibrium switching for free energy calculations or `nonequilibrium candidate Monte Carlo (NCMC) <http://dx.doi.org/10.1073/pnas.1106094108>`_
 - an extensible Markov chain Monte Carlo framework for mixing Monte Carlo and molecular dynamics-based methods
 - factories for generating `alchemically-modified <http://alchemistry.org>`_ systems for absolute and relative free energy calculations
 - a suite of test systems for benchmarking, validation, and debugging

You can go through the :ref:`getting started tutorial <gettingstarted>` for an overview of the library or the
:ref:`developer's guide <devtutorial>` for information on how to extend the existing features.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   installation
   gettingstarted
   Developer's guide <devtutorial>
   releasehistory

Modules
-------

.. toctree::
  :maxdepth: 2

  testsystems
  integrators
  states
  cache
  mcmc
  multistate
  alchemy
  forces
  forcefactories
  storage
  utils
  scripts

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
