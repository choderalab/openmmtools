import numpy as np

try:
    import openmm
    from openmm import unit
except ImportError:  # OpenMM < 7.6
    from simtk import unit
    from simtk import openmm

import os, os.path
import logging
import pytest

from openmmtools import testsystems


def _equiv_topology(top_1, top_2):
    """Compare topologies using string reps of atoms and bonds"""
    for b1, b2 in zip(top_1.bonds(), top_2.bonds()):
        if str(b1) != str(b2):
            return False

    for a1, a2 in zip(top_1.atoms(), top_2.atoms()):
        if str(a1) != str(a2):
            return False

    return True


def get_all_subclasses(cls):
    """
    Return all subclasses of a specified class.

    Parameters
    ----------
    cls : class
       The class for which all subclasses are to be returned.

    Returns
    -------
    all_subclasses : list of class
       List of all subclasses of `cls`.

    """

    all_subclasses = []

    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))

    return all_subclasses


def test_get_data_filename():
    """Testing retrieval of data files shipped with distro."""
    relative_path = "data/alanine-dipeptide-gbsa/alanine-dipeptide.prmtop"
    filename = testsystems.get_data_filename(relative_path)
    if not os.path.exists(filename):
        raise Exception("Could not locate data files. Expected %s" % relative_path)


def test_subrandom_particle_positions():
    """Testing deterministic subrandom particle position assignment."""
    # Test halton sequence.
    x = testsystems.halton_sequence(2, 100)

    # Test Sobol.
    from openmmtools import sobol

    x = sobol.i4_sobol_generate(3, 100, 1)

    # Test subrandom positions.
    nparticles = 216
    box_vectors = openmm.System().getDefaultPeriodicBoxVectors()
    positions = testsystems.subrandom_particle_positions(nparticles, box_vectors)


def check_properties(testsystem):
    class_name = testsystem.__class__.__name__
    property_list = testsystem.analytical_properties
    state = testsystems.ThermodynamicState(
        temperature=300.0 * unit.kelvin, pressure=1.0 * unit.atmosphere
    )
    if len(property_list) > 0:
        for property_name in property_list:
            method = getattr(testsystem, "get_" + property_name)
            logging.info(
                "%32s . %32s : %32s" % (class_name, property_name, str(method(state)))
            )
    return


def test_properties_all_testsystems():
    """Testing computation of analytic properties for all systems."""
    testsystem_classes = get_all_subclasses(testsystems.TestSystem)
    logging.info("Testing analytical property computation:")
    for testsystem_class in testsystem_classes:
        class_name = testsystem_class.__name__
        try:
            testsystem = testsystem_class()
        except ImportError as e:
            print(e)
            print("Skipping %s due to missing dependency" % class_name)
            continue
        check_properties(testsystem)


fast_testsystems = [
    "HarmonicOscillator",
    "PowerOscillator",
    "Diatom",
    "DiatomicFluid",
    "UnconstrainedDiatomicFluid",
    "ConstrainedDiatomicFluid",
    "DipolarFluid",
    "UnconstrainedDipolarFluid",
    "ConstrainedDipolarFluid",
    "ConstraintCoupledHarmonicOscillator",
    "HarmonicOscillatorArray",
    "SodiumChlorideCrystal",
    "LennardJonesCluster",
    "LennardJonesFluid",
    "LennardJonesGrid",
    "CustomLennardJonesFluidMixture",
    "WCAFluid",
    "DoubleWellDimer_WCAFluid",
    "DoubleWellChain_WCAFluid",
    "IdealGas",
    "WaterBox",
    "FlexibleWaterBox",
    "FourSiteWaterBox",
    "FiveSiteWaterBox",
    "DischargedWaterBox",
    "DischargedWaterBoxHsites",
    "AlchemicalWaterBox",
    "AlanineDipeptideVacuum",
    "AlanineDipeptideImplicit",
    "MethanolBox",
    "MolecularIdealGas",
    "CustomGBForceSystem",
    "AlchemicalLennardJonesCluster",
    "LennardJonesPair",
    "TolueneVacuum",
    "TolueneImplicit",
    "TolueneImplicitHCT",
    "TolueneImplicitOBC1",
    "TolueneImplicitOBC2",
    "TolueneImplicitGBn",
    "TolueneImplicitGBn2",
    "HostGuestVacuum",
    "HostGuestImplicit",
    "HostGuestImplicitHCT",
    "HostGuestImplicitOBC1",
]


def check_potential_energy(system, positions):
    """
    Compute potential energy for system and positions and ensure that it is finite.

    Parameters
    ----------
    system : openmm.System
       The system
    positions : openmm.unit.Quantity
       The positions

    """

    # Create a Context.
    timestep = 1.0 * unit.femtoseconds
    integrator = openmm.VerletIntegrator(timestep)
    context = openmm.Context(system, integrator)
    context.setPositions(positions)

    # Compute potential energy to make sure it is finite.
    openmm_state = context.getState(getEnergy=True)
    potential_energy = openmm_state.getPotentialEnergy()

    # Check if finite.
    if np.isnan(potential_energy / unit.kilocalories_per_mole):
        raise Exception("Energy is NaN.")

    # Clean up
    del context, integrator


def test_energy_all_testsystems(skip_slow_tests=True):
    """Testing computation of potential energy for all systems."""
    testsystem_classes = get_all_subclasses(testsystems.TestSystem)
    for testsystem_class in testsystem_classes:
        class_name = testsystem_class.__name__
        if skip_slow_tests and not (class_name in fast_testsystems):
            logging.info(
                "Skipping potential energy test for testsystem %s." % class_name
            )
            continue

        # Create test.
        try:
            testsystem = testsystem_class()
        except ImportError as e:
            print(e)
            print("Skipping %s due to missing dependency" % class_name)
            continue
        check_potential_energy(testsystem.system, testsystem.positions)


def check_topology(system, topology):
    """Check the topology object contains the correct number of atoms."""

    # Get number of particles from topology.
    nparticles_topology = 0
    for atom in topology.atoms():
        nparticles_topology += 1

    # Get number of particles from system.
    nparticles_system = system.getNumParticles()

    assert nparticles_topology == nparticles_system


def test_topology_all_testsystems():
    """Testing topology contains correct number of atoms for all systems."""
    testsystem_classes = get_all_subclasses(testsystems.TestSystem)

    for testsystem_class in testsystem_classes:
        class_name = testsystem_class.__name__

        # Create test.
        try:
            testsystem = testsystem_class()
        except ImportError as e:
            print(e)
            print("Skipping %s due to missing dependency" % class_name)
            continue
        check_topology(testsystem.system, testsystem.topology)


def test_dw_systems_as_wca():
    # check that the double-well systems are equivalent to WCA fluid in
    # certain limits
    dimers = testsystems.DoubleWellDimer_WCAFluid(ndimers=0)
    chain_1 = testsystems.DoubleWellChain_WCAFluid(nchained=1)
    chain_0 = testsystems.DoubleWellChain_WCAFluid(nchained=0)
    wca = testsystems.WCAFluid()
    assert _equiv_topology(dimers.topology, wca.topology)
    assert _equiv_topology(chain_1.topology, wca.topology)
    assert _equiv_topology(chain_0.topology, wca.topology)


def test_dw_systems_1_dimer():
    # check that the double-well systems are equivalent when there's only
    # one dimer pair
    dimers = testsystems.DoubleWellDimer_WCAFluid(ndimers=1)
    chain = testsystems.DoubleWellChain_WCAFluid(nchained=2)
    assert _equiv_topology(dimers.topology, chain.topology)


def test_double_well_dimer_errors():
    with pytest.raises(ValueError):
        testsystems.DoubleWellDimer_WCAFluid(ndimers=-1)
    with pytest.raises(ValueError):
        testsystems.DoubleWellDimer_WCAFluid(ndimers=6, nparticles=10)


def test_double_well_chain_errors():
    with pytest.raises(ValueError):
        testsystems.DoubleWellChain_WCAFluid(nchained=-1)
    with pytest.raises(ValueError):
        testsystems.DoubleWellChain_WCAFluid(nchained=11, nparticles=10)
