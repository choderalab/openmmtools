.. _references:

**********
References
**********

Here are a list of references for the various components and algorithms used in ``openmmtools``.

OpenMM GPU-accelerated molecular mechanics library
""""""""""""""""""""""""""""""""""""""""""""""""""

  Friedrichs MS, Eastman P, Vaidyanathan V, Houston M, LeGrand S, Beberg AL, Ensign DL, Bruns CM, and Pande VS. Accelerating molecular dynamic simulations on graphics processing units.
  J. Comput. Chem. 30:864, 2009.
  http://dx.doi.org/10.1002/jcc.21209

  Eastman P and Pande VS. OpenMM: A hardware-independent framework for molecular simulations.
  Comput. Sci. Eng. 12:34, 2010.
  http://dx.doi.org/10.1109/MCSE.2010.27

  Eastman P and Pande VS. Efficient nonbonded interactions for molecular dynamics on a graphics processing unit.
  J. Comput. Chem. 31:1268, 2010.
  http://dx.doi.org/10.1002/jcc.21413

  Eastman P and Pande VS. Constant constraint matrix approximation: A robust, parallelizable constraint method for molecular simulations.
  J. Chem. Theor. Comput. 6:434, 2010.
  http://dx.doi.org/10.1021/ct900463w

  Eastman P, Friedrichs M, Chodera JD, Radmer RJ, Bruns CM, Ku JP, Beauchamp KA, Lane TJ, Wang LP, Shukla D, Tye T, Houston M, Stich T, Klein C, Shirts M, and Pande VS.  OpenMM 4: A Reusable, Extensible,
  Hardware Independent Library for High Performance Molecular Simulation. J. Chem. Theor. Comput. 2012.
  http://dx.doi.org/10.1021/ct300857j

Replica-exchange with Gibbs sampling
""""""""""""""""""""""""""""""""""""

  Chodera JD and Shirts MR. Replica exchange and expanded ensemble simulations as Gibbs sampling: Simple improvements for enhanced mixing.
  J. Chem. Phys. 135:19410, 2011.
  http://dx.doi.org/10.1063/1.3660669

MBAR for estimation of free energies from simulation data
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

  Shirts MR and Chodera JD. Statistically optimal analysis of samples from multiple equilibrium states.
  J. Chem. Phys. 129:124105, 2008.
  http://dx.doi.org/10.1063/1.2978177

Long-range dispersion corrections for explicit solvent free energy calculations
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

  Shirts MR, Mobley DL, Chodera JD, and Pande VS. Accurate and efficient corrections or missing dispersion interactions in molecular simulations.
  J. Phys. Chem. 111:13052, 2007.
  http://dx.doi.org/10.1021/jp0735987


Bibliography
############

.. The :all: directive searches subfolders for uses of :cite: for correct reference
   However, this has the effect of dropping all citations in the .bib file in here and
   the compiler complains about unused citations.
   As such, unused articles in the .bib file are simply commented so as not to delete them if needed in the future.

.. bibliography:: references.bib
   :style: unsrt
   :all:
