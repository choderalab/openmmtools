"""
Tests for expanded ensemble and SAMS samplers.

"""

__author__ = 'John D. Chodera'

################################################################################
# IMPORTS
################################################################################

from simtk import openmm, unit
from simtk.openmm import app
import os, os.path
import sys, math
import numpy as np
import logging
from functools import partial

from . import testsystems

################################################################################
# TEST SAMPLERS
################################################################################

def test_testsystems():
    np.set_printoptions(linewidth=130, precision=3)
    niterations = 2
    import sams
    # TODO: Automatically discover subclasses of SAMSTestSystem that are not abstract base classes
    for testsystem_name in ['AlanineDipeptideVacuumSimulatedTempering', 'AlanineDipeptideExplicitSimulatedTempering', 'AlanineDipeptideVacuumAlchemical', 'AlanineDipeptideExplicitAlchemical', 'WaterBoxAlchemical', 'HostGuestAlchemical']:
        testsystem = getattr(testsystems, testsystem_name)
        test = testsystem()
        # Reduce number of steps for testing
        test.mcmc_sampler.nsteps = 2
        f = partial(test.sams_sampler.run, niterations)
        f.description = 'Testing ' + test.description + ' SAMS simulation'
        yield f

def test_sampler_options():
    """
    Test sampler options on a single test system.

    """
    test = testsystems.WaterBoxAlchemical()
    testsystem_name = test.__class__.__name__
    niterations = 5 # number of iterations to run
    test.mcmc_sampler.nsteps = 50

    # Test SAMSSampler samplers.
    for update_scheme in test.exen_sampler.supported_update_schemes:
        test.exen_sampler.update_scheme = update_scheme
        for update_method in test.sams_sampler.supported_update_methods:
            test.sams_sampler.update_method = update_method
            f = partial(test.sams_sampler.run, niterations)
            f.description = "Testing SAMS sampler with %s using expanded ensemble update scheme '%s' and SAMS update method '%s'" % (testsystem_name, update_scheme, update_method)
            yield f


if __name__=="__main__":
    test_sampler_options()
