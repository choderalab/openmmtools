import numpy as np

from simtk import unit
from simtk import openmm

import os, os.path
import tempfile
import logging

from openmmtools import testsystems

def test_doctest():
    """Performing doctests.
    """
    import doctest
    doctest.testmod(testsystems)

def test_get_data_filename():
    """Testing retrieval of data files shipped with distro.
    """
    relative_path = 'data/alanine-dipeptide-gbsa/alanine-dipeptide.prmtop'
    filename = testsystems.get_data_filename(relative_path)
    if not os.path.exists(filename):
        raise Exception("Could not locate data files. Expected %s" % relative_path)

def test_subrandom_particle_positions():
    """Testing deterministic subrandom particle position assignment.
    """
    # Test halton sequence.
    x = testsystems.halton_sequence(2,100)

    # Test subrandom positions.
    nparticles = 216
    box_vectors = openmm.System().getDefaultPeriodicBoxVectors()
    positions = testsystems.subrandom_particle_positions(nparticles, box_vectors)

def test_properties_all_testsystems():
    """Testing computation of analytic properties for all systems.
    """
    testsystem_classes = testsystems.TestSystem.__subclasses__()
    logging.info("Testing analytical property computation:")
    for testsystem_class in testsystem_classes:
        class_name = testsystem_class.__name__
        logging.info(class_name)
        print class_name # DEBUG
        testsystem = testsystem_class()
        property_list = testsystem.analytical_properties
        state = testsystems.ThermodynamicState(temperature=300.0*unit.kelvin, pressure=1.0*unit.atmosphere)
        if len(property_list) > 0:
            for property_name in property_list:
                method = getattr(testsystem, 'get_' + property_name)
                logging.info("%32s . %32s : %32s" % (class_name, property_name, str(method(state))))

fast_testsystems = ["HarmonicOscillator", "PowerOscillator", "Diatom", "ConstraintCoupledHarmonicOscillator", "HarmonicOscillatorArray", "SodiumChlorideCrystal", "LennardJonesCluster", "LennardJonesFluid", "IdealGas", "AlanineDipeptideVacuum"]

def test_energy_all_testsystems(skip_slow_tests=False):
    """Testing computation of potential energy for all systems.
    """
    testsystem_classes = testsystems.TestSystem.__subclasses__()

    for testsystem_class in testsystem_classes:
        class_name = testsystem_class.__name__
        if skip_slow_tests and not (class_name in fast_testsystems):
            logging.info("Skipping potential energy test for testsystem %s." % class_name)
            continue
        logging.info("Testing potential energy test for testsystem %s" % class_name)
        print class_name # DEBUG

        # Create system.
        testsystem = testsystem_class()

        # Create a Context.
        timestep = 1.0 * unit.femtoseconds
        integrator = openmm.VerletIntegrator(timestep)
        context = openmm.Context(testsystem.system, integrator)
        context.setPositions(testsystem.positions)

        # Compute potential energy to make sure it is finite.
        openmm_state = context.getState(getEnergy=True)
        potential_energy = openmm_state.getPotentialEnergy()

        # Check if finite.
        if np.isnan(potential_energy / unit.kilocalories_per_mole):
            raise Exception("Energy of test system %s is NaN." % class_name)

        # Clean up
        del context, integrator
