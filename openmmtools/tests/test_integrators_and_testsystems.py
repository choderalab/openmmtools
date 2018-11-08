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
import inspect
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

    # Get all the CustomIntegrators in the integrators module.
    is_integrator = lambda x: (inspect.isclass(x) and
                               issubclass(x, openmm.CustomIntegrator) and
                               x != integrators.ThermostatedIntegrator)
    custom_integrators = inspect.getmembers(integrators, predicate=is_integrator)

    def all_subclasses(cls):
        """Return list of all subclasses and subsubclasses for a given class."""
        return cls.__subclasses__() + [s for s in cls.__subclasses__()]
    testsystem_classes = all_subclasses(testsystems.TestSystem)
    testsystem_names = [ cls.__name__ for cls in testsystem_classes ]

    # Use Reference platform.
    platform = openmm.Platform.getPlatformByName('Reference')

    for testsystem_name in testsystem_names:
        # Create testsystem.
        try:
            testsystem = getattr(testsystems, testsystem_name)()
        except ImportError as e:
            print(e)
            print("Skipping %s due to missing dependency" % testsystem_name)
            continue

        for integrator_name, integrator_class in custom_integrators:
            if integrator_name == "NoseHooverChainVelocityVerletIntegrator":
                # The system should be passed to the NoseHooverChainVelocityVerletIntegrator
                # to extract the correct number of degrees of freedom from the system.
                integrator = integrator_class(testsystem.system)
            else:
                integrator = integrator_class()

            # because it is being initialized without a system. That's OK.

            # Create test.
            f = partial(check_combination, integrator, testsystem, platform)
            f.description = "Checking combination of %s and %s" % (integrator_name, testsystem_name)
            yield f
