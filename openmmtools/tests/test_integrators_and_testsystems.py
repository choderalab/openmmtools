#!/usr/local/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Test combinations of custom integrators and testsystems to make sure there are no namespace collisions.

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import re
import numpy
from functools import partial

from simtk import unit
from simtk import openmm

#=============================================================================================
# CONSTANTS
#=============================================================================================

kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA

#=============================================================================================
# UTILITY SUBROUTINES
#=============================================================================================

def check_combination(integrator, test, platform=None):
   """
   Check combination of integrator and testsystem.

   Parameters
   ----------
   integrator : simtk.openmm.Integrator
      The integrator to test.
   test : testsystem
      The testsystem to test.

   """

   # Create Context and initialize positions.
   if platform:
      context = openmm.Context(test.system, integrator, platform)
   else:
      context = openmm.Context(test.system, integrator)

   # Clean up.
   del context

   return

#=============================================================================================
# TESTS
#=============================================================================================

def test_integrators_and_testsystems():
   """
   Test combinations of integrators and testsystems to ensure there are no global context parameters.

   """
   from openmmtools import integrators, testsystems

   # Create lists of integrator and testsystem names.
   integrator_names = [ methodname for methodname in dir(integrators) if re.match('.*Integrator$', methodname) ]

   def all_subclasses(cls):
       """Return list of all subclasses and subsubclasses for a given class."""
       return cls.__subclasses__() + [s for s in cls.__subclasses__()]
   testsystem_classes = all_subclasses(testsystems.TestSystem)
   testsystem_names = [ cls.__name__ for cls in testsystem_classes ]

   # Use Reference platform.
   platform = openmm.Platform.getPlatformByName('Reference')

   for testsystem_name in testsystem_names:
      # Create testsystem.
      testsystem = getattr(testsystems, testsystem_name)()
      for integrator_name in integrator_names:
         # Create integrator.
         integrator = getattr(integrators, integrator_name)()

         # Create test.
         f = partial(check_combination, integrator, testsystem, platform)
         f.description = "Checking combination of %s and %s" % (integrator_name, testsystem_name)
         yield f

