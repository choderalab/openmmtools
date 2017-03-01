#!/usr/bin/python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Tests for alchemical factory in `alchemy.py`.

"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

from __future__ import print_function

import os
import sys
import pickle
import itertools
from functools import partial

import nose
import scipy
from nose.plugins.attrib import attr

from openmmtools import testsystems
from openmmtools.constants import kB
from openmmtools.alchemy import *

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

temperature = 300.0 * unit.kelvin  # reference temperature
# MAX_DELTA = 0.01 * kB * temperature # maximum allowable deviation
MAX_DELTA = 1.0 * kB * temperature  # maximum allowable deviation
GLOBAL_ENERGY_UNIT = unit.kilojoules_per_mole  # controls printed units
GLOBAL_ALCHEMY_PLATFORM = None  # This is used in every energy calculation.


# =============================================================================
# TESTING UTILITIES
# =============================================================================

def compute_energy(system, positions, platform=None, force_group=-1):
    """Compute energy of the system in the given positions.

    Parameters
    ----------
    platform : simtk.openmm.Platform or None, optional
        If None, the global GLOBAL_ALCHEMY_PLATFORM will be used.
    force_group : int flag or set of int, optional
        Passed to the groups argument of Context.getState().

    """
    timestep = 1.0 * unit.femtoseconds
    integrator = openmm.VerletIntegrator(timestep)
    if platform is None:
        platform = GLOBAL_ALCHEMY_PLATFORM
    if platform is not None:
        context = openmm.Context(system, integrator, platform)
    else:
        context = openmm.Context(system, integrator)
    context.setPositions(positions)
    state = context.getState(getEnergy=True, groups=force_group)
    potential = state.getPotentialEnergy()
    del context, integrator, state
    return potential


def compute_energy_force(system, positions, force_name):
    """Compute the energy of the force with the given name."""
    system = copy.deepcopy(system)  # Copy to avoid modifications
    force_name_index = 1
    found_force = False

    # Separate force group of force_name from all others.
    for force in system.getForces():
        if force.__class__.__name__ == force_name:
            force.setForceGroup(force_name_index)
            found_force = True
        else:
            force.setForceGroup(0)

    if not found_force:
        return None

    force_energy = compute_energy(system, positions, force_group=2**force_name_index)
    del system
    return force_energy


def assert_almost_equal(energy1, energy2, err_msg):
    delta = energy1 - energy2
    err_msg += ' interactions do not match! Reference {}, alchemical {},' \
                ' difference {}'.format(energy1, energy2, delta)
    assert abs(delta) < MAX_DELTA, err_msg


def dissect_nonbonded_energy(reference_system, positions, alchemical_atoms):
    """Dissect the contributions to NonbondedForce of the reference system by atom group
    and sterics/electrostatics.

    Parameters
    ----------
    reference_system : simtk.openmm.System
        The reference system with the NonbondedForce to dissect.
    positions : simtk.openmm.unit.Quantity of dimension [nparticles,3] with units compatible with Angstroms
        The positions to test.
    alchemical_atoms : set of int
        The indices of the alchemical atoms.

    Returns
    -------
    tuple of simtk.openmm.unit.Quantity with units compatible with kJ/mol
        All contributions to the potential energy of NonbondedForce in the order:
        nn_particle_sterics: particle sterics interactions between nonalchemical atoms
        aa_particle_sterics: particle sterics interactions between alchemical atoms
        na_particle_sterics: particle sterics interactions between nonalchemical-alchemical atoms
        nn_particle_electro: (direct space) particle electrostatics interactions between nonalchemical atoms
        aa_particle_electro: (direct space) particle electrostatics interactions between alchemical atoms
        na_particle_electro: (direct space) particle electrostatics interactions between nonalchemical-alchemical atoms
        nn_exception_sterics: particle sterics 1,4 exceptions between nonalchemical atoms
        aa_exception_sterics: particle sterics 1,4 exceptions between alchemical atoms
        na_exception_sterics: particle sterics 1,4 exceptions between nonalchemical-alchemical atoms
        nn_exception_electro: particle electrostatics 1,4 exceptions between nonalchemical atoms
        aa_exception_electro: particle electrostatics 1,4 exceptions between alchemical atoms
        na_exception_electro: particle electrostatics 1,4 exceptions between nonalchemical-alchemical atoms
        nn_reciprocal_energy: electrostatics of reciprocal space between nonalchemical atoms
        aa_reciprocal_energy: electrostatics of reciprocal space between alchemical atoms
        na_reciprocal_energy: electrostatics of reciprocal space between nonalchemical-alchemical atoms

    """

    def turn_off(force, sterics=False, electrostatics=False,
                 exceptions=False, only_atoms=frozenset()):
        if len(only_atoms) == 0:  # if empty, turn off all particles
            only_atoms = set(range(force.getNumParticles()))
        e_coeff = 0.0 if sterics else 1.0
        c_coeff = 0.0 if electrostatics else 1.0
        if exceptions:  # Turn off exceptions
            for exception_index in range(force.getNumExceptions()):
                [iatom, jatom, charge, sigma, epsilon] = force.getExceptionParameters(exception_index)
                if iatom in only_atoms or jatom in only_atoms:
                    force.setExceptionParameters(exception_index, iatom, jatom, c_coeff*charge,
                                                 sigma, e_coeff*epsilon)
        else:  # Turn off particle interactions
            for particle_index in range(force.getNumParticles()):
                if particle_index in only_atoms:
                    [charge, sigma, epsilon] = force.getParticleParameters(particle_index)
                    force.setParticleParameters(particle_index, c_coeff*charge, sigma, e_coeff*epsilon)

    def restore_system(reference_system):
        system = copy.deepcopy(reference_system)
        nonbonded_force = system.getForces()[0]
        return system, nonbonded_force

    nonalchemical_atoms = set(range(reference_system.getNumParticles())).difference(alchemical_atoms)

    # Remove all forces but NonbondedForce
    reference_system = copy.deepcopy(reference_system)  # don't modify original system
    forces_to_remove = list()
    for force_index, force in enumerate(reference_system.getForces()):
        if force.__class__.__name__ != 'NonbondedForce':
            forces_to_remove.append(force_index)
        else:
            force.setForceGroup(0)
            force.setReciprocalSpaceForceGroup(31)  # separate PME reciprocal from direct space
    for force_index in reversed(forces_to_remove):
        reference_system.removeForce(force_index)
    assert len(reference_system.getForces()) == 1

    # Compute particle interactions between different groups of atoms
    # ----------------------------------------------------------------
    system, nonbonded_force = restore_system(reference_system)

    # Compute total energy from nonbonded interactions
    tot_energy = compute_energy(system, positions)
    tot_reciprocal_energy = compute_energy(system, positions, force_group={31})

    # Compute contributions from particle sterics
    turn_off(nonbonded_force, sterics=True, only_atoms=alchemical_atoms)
    tot_energy_no_alchem_particle_sterics = compute_energy(system, positions)
    system, nonbonded_force = restore_system(reference_system)  # Restore alchemical sterics
    turn_off(nonbonded_force, sterics=True, only_atoms=nonalchemical_atoms)
    tot_energy_no_nonalchem_particle_sterics = compute_energy(system, positions)
    turn_off(nonbonded_force, sterics=True)
    tot_energy_no_particle_sterics = compute_energy(system, positions)

    tot_particle_sterics = tot_energy - tot_energy_no_particle_sterics
    nn_particle_sterics = tot_energy_no_alchem_particle_sterics - tot_energy_no_particle_sterics
    aa_particle_sterics = tot_energy_no_nonalchem_particle_sterics - tot_energy_no_particle_sterics
    na_particle_sterics = tot_particle_sterics - nn_particle_sterics - aa_particle_sterics

    # Compute contributions from particle electrostatics
    system, nonbonded_force = restore_system(reference_system)  # Restore sterics
    turn_off(nonbonded_force, electrostatics=True, only_atoms=alchemical_atoms)
    tot_energy_no_alchem_particle_electro = compute_energy(system, positions)
    nn_reciprocal_energy = compute_energy(system, positions, force_group={31})
    system, nonbonded_force = restore_system(reference_system)  # Restore alchemical electrostatics
    turn_off(nonbonded_force, electrostatics=True, only_atoms=nonalchemical_atoms)
    tot_energy_no_nonalchem_particle_electro = compute_energy(system, positions)
    aa_reciprocal_energy = compute_energy(system, positions, force_group={31})
    turn_off(nonbonded_force, electrostatics=True)
    tot_energy_no_particle_electro = compute_energy(system, positions)

    na_reciprocal_energy = tot_reciprocal_energy - nn_reciprocal_energy - aa_reciprocal_energy
    tot_particle_electro = tot_energy - tot_energy_no_particle_electro

    nn_particle_electro = tot_energy_no_alchem_particle_electro - tot_energy_no_particle_electro
    aa_particle_electro = tot_energy_no_nonalchem_particle_electro - tot_energy_no_particle_electro
    na_particle_electro = tot_particle_electro - nn_particle_electro - aa_particle_electro
    nn_particle_electro -= nn_reciprocal_energy
    aa_particle_electro -= aa_reciprocal_energy
    na_particle_electro -= na_reciprocal_energy

    # Compute exceptions between different groups of atoms
    # -----------------------------------------------------

    # Compute contributions from exceptions sterics
    system, nonbonded_force = restore_system(reference_system)  # Restore particle interactions
    turn_off(nonbonded_force, sterics=True, exceptions=True, only_atoms=alchemical_atoms)
    tot_energy_no_alchem_exception_sterics = compute_energy(system, positions)
    system, nonbonded_force = restore_system(reference_system)  # Restore alchemical sterics
    turn_off(nonbonded_force, sterics=True, exceptions=True, only_atoms=nonalchemical_atoms)
    tot_energy_no_nonalchem_exception_sterics = compute_energy(system, positions)
    turn_off(nonbonded_force, sterics=True, exceptions=True)
    tot_energy_no_exception_sterics = compute_energy(system, positions)

    tot_exception_sterics = tot_energy - tot_energy_no_exception_sterics
    nn_exception_sterics = tot_energy_no_alchem_exception_sterics - tot_energy_no_exception_sterics
    aa_exception_sterics = tot_energy_no_nonalchem_exception_sterics - tot_energy_no_exception_sterics
    na_exception_sterics = tot_exception_sterics - nn_exception_sterics - aa_exception_sterics

    # Compute contributions from exceptions electrostatics
    system, nonbonded_force = restore_system(reference_system)  # Restore exceptions sterics
    turn_off(nonbonded_force, electrostatics=True, exceptions=True, only_atoms=alchemical_atoms)
    tot_energy_no_alchem_exception_electro = compute_energy(system, positions)
    system, nonbonded_force = restore_system(reference_system)  # Restore alchemical electrostatics
    turn_off(nonbonded_force, electrostatics=True, exceptions=True, only_atoms=nonalchemical_atoms)
    tot_energy_no_nonalchem_exception_electro = compute_energy(system, positions)
    turn_off(nonbonded_force, electrostatics=True, exceptions=True)
    tot_energy_no_exception_electro = compute_energy(system, positions)

    tot_exception_electro = tot_energy - tot_energy_no_exception_electro
    nn_exception_electro = tot_energy_no_alchem_exception_electro - tot_energy_no_exception_electro
    aa_exception_electro = tot_energy_no_nonalchem_exception_electro - tot_energy_no_exception_electro
    na_exception_electro = tot_exception_electro - nn_exception_electro - aa_exception_electro

    assert tot_particle_sterics == nn_particle_sterics + aa_particle_sterics + na_particle_sterics
    assert_almost_equal(tot_particle_electro, nn_particle_electro + aa_particle_electro +
                        na_particle_electro + nn_reciprocal_energy + aa_reciprocal_energy + na_reciprocal_energy,
                        'Inconsistency during dissection of nonbonded contributions:')
    assert tot_exception_sterics == nn_exception_sterics + aa_exception_sterics + na_exception_sterics
    assert tot_exception_electro == nn_exception_electro + aa_exception_electro + na_exception_electro
    assert_almost_equal(tot_energy, tot_particle_sterics + tot_particle_electro +
                        tot_exception_sterics + tot_exception_electro,
                        'Inconsistency during dissection of nonbonded contributions:')

    return nn_particle_sterics, aa_particle_sterics, na_particle_sterics,\
           nn_particle_electro, aa_particle_electro, na_particle_electro,\
           nn_exception_sterics, aa_exception_sterics, na_exception_sterics,\
           nn_exception_electro, aa_exception_electro, na_exception_electro,\
           nn_reciprocal_energy, aa_reciprocal_energy, na_reciprocal_energy


def compute_direct_space_correction(nonbonded_force, alchemical_atoms, positions):
    """
    Compute the correction added by OpenMM to the direct space to account for
    exception in reciprocal space energy.

    Parameters
    ----------
    nonbonded_force : simtk.openmm.NonbondedForce
        The nonbonded force to compute the direct space correction.
    alchemical_atoms : set
        Set of alchemical particles in the force.
    positions : numpy.array
        Position of the particles.

    Returns
    -------
    aa_correction : simtk.openmm.unit.Quantity with units compatible with kJ/mol
        The correction to the direct spaced caused by exceptions between alchemical atoms.
    na_correction : simtk.openmm.unit.Quantity with units compatible with kJ/mol
        The correction to the direct spaced caused by exceptions between nonalchemical-alchemical atoms.

    """
    energy_unit = unit.kilojoule_per_mole
    aa_correction = 0.0
    na_correction = 0.0

    # If there is no reciprocal space, the correction is 0.0
    if nonbonded_force.getNonbondedMethod() not in [openmm.NonbondedForce.Ewald, openmm.NonbondedForce.PME]:
        return aa_correction * energy_unit, na_correction * energy_unit

    # Get alpha ewald parameter
    alpha_ewald, _, _, _ = nonbonded_force.getPMEParameters()
    if alpha_ewald / alpha_ewald.unit == 0.0:
        cutoff_distance = nonbonded_force.getCutoffDistance()
        tolerance = nonbonded_force.getEwaldErrorTolerance()
        alpha_ewald = (1.0 / cutoff_distance) * np.sqrt(-np.log(2.0*tolerance))
    alpha_ewald = alpha_ewald.value_in_unit_system(unit.md_unit_system)
    assert alpha_ewald != 0.0

    for exception_id in range(nonbonded_force.getNumExceptions()):
        # Get particles parameters in md unit system
        iatom, jatom, _, _, _ = nonbonded_force.getExceptionParameters(exception_id)
        icharge, _, _ = nonbonded_force.getParticleParameters(iatom)
        jcharge, _, _ = nonbonded_force.getParticleParameters(jatom)
        icharge = icharge.value_in_unit_system(unit.md_unit_system)
        jcharge = jcharge.value_in_unit_system(unit.md_unit_system)

        # Compute the correction and take care of numerical instabilities
        r = np.linalg.norm(positions[iatom] - positions[jatom])  # distance between atoms
        alpha_r = alpha_ewald * r
        if alpha_r > 1e-6:
            correction = ONE_4PI_EPS0 * icharge * jcharge * scipy.special.erf(alpha_r) / r
        else:  # for small alpha_r we linearize erf()
            correction = ONE_4PI_EPS0 * alpha_ewald * icharge * jcharge * 2.0 / np.sqrt(np.pi)

        # Assign correction to correct group
        if iatom in alchemical_atoms and jatom in alchemical_atoms:
            aa_correction += correction
        elif iatom in alchemical_atoms or jatom in alchemical_atoms:
            na_correction += correction

    return aa_correction * energy_unit, na_correction * energy_unit


# =============================================================================
# SUBROUTINES FOR TESTING
# =============================================================================

def compare_system_energies(reference_system, alchemical_system, alchemical_regions, positions):
    """Check that the energies of reference and alchemical systems are close.

    This takes care of ignoring the reciprocal space when the nonbonded
    method is an Ewald method.

    """
    force_group = -1  # Default we compare the energy of all groups.

    # Check nonbonded method. Comparing with PME is more complicated
    # because the alchemical system does not take into account the
    # reciprocal space.
    # TODO remove this when PME will include reciprocal space fixed.
    ewald_force = None
    for force in reference_system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            nonbonded_method = force.getNonbondedMethod()
            if nonbonded_method == openmm.NonbondedForce.PME or nonbonded_method == openmm.NonbondedForce.Ewald:
                ewald_force = force
                break

    if ewald_force is not None:
        # Separate the reciprocal space force in a different group.
        reference_system = copy.deepcopy(reference_system)
        alchemical_system = copy.deepcopy(alchemical_system)
        for system in [reference_system, alchemical_system]:
            for force in system.getForces():
                force.setForceGroup(0)
                if isinstance(force, openmm.NonbondedForce):
                    force.setReciprocalSpaceForceGroup(31)

        # We compare only the direct space energy
        force_group = {0}

        # Compute the reciprocal space correction added to the direct space
        # energy due to the exceptions of the alchemical atoms.
        alchemical_atoms = alchemical_regions.alchemical_atoms
        aa_correction, na_correction = compute_direct_space_correction(ewald_force, alchemical_atoms, positions)

    # Compute potential of the direct space.
    potentials = [compute_energy(system, positions, force_group=force_group)
                  for system in [reference_system, alchemical_system]]

    # Add the direct space correction.
    if ewald_force is not None:
        potentials.append(aa_correction + na_correction)
    else:
        potentials.append(0.0 * GLOBAL_ENERGY_UNIT)

    # Check that error is small.
    delta = potentials[1] - potentials[2] - potentials[0]
    if abs(delta) > MAX_DELTA:
        print("========")
        for description, potential in zip(['reference', 'alchemical', 'PME correction'], potentials):
            print("{}: {} ".format(description, potential))
        print("delta    : {}".format(delta))
        err_msg = "Maximum allowable deviation exceeded (was {:.8f} kcal/mol; allowed {:.8f} kcal/mol)."
        raise Exception(err_msg.format(delta / unit.kilocalories_per_mole, MAX_DELTA / unit.kilocalories_per_mole))


def check_interacting_energy_components(reference_system, alchemical_system, alchemical_regions, positions):
    """Compare full and alchemically-modified system energies by energy component.

    Parameters
    ----------
    reference_system : simtk.openmm.System
        The reference system.
    alchemical_system : simtk.openmm.System
        The alchemically modified system to test.
    alchemical_regions : AlchemicalRegion.
       The alchemically modified region.
    positions : n_particlesx3 array-like of simtk.openmm.unit.Quantity
        The positions to test (units of length).

    """

    reference_system = copy.deepcopy(reference_system)
    alchemical_system = copy.deepcopy(alchemical_system)

    # Find nonbonded method
    for nonbonded_force in reference_system.getForces():
        if isinstance(nonbonded_force, openmm.NonbondedForce):
            nonbonded_method = nonbonded_force.getNonbondedMethod()
            break

    # Get energy components of reference system's nonbonded force
    print("Dissecting reference system's nonbonded force")
    energy_components = dissect_nonbonded_energy(reference_system, positions,
                                                 alchemical_regions.alchemical_atoms)
    nn_particle_sterics, aa_particle_sterics, na_particle_sterics,\
    nn_particle_electro, aa_particle_electro, na_particle_electro,\
    nn_exception_sterics, aa_exception_sterics, na_exception_sterics,\
    nn_exception_electro, aa_exception_electro, na_exception_electro,\
    nn_reciprocal_energy, aa_reciprocal_energy, na_reciprocal_energy = energy_components

    # Dissect unmodified nonbonded force in alchemical system
    print("Dissecting alchemical system's unmodified nonbonded force")
    energy_components = dissect_nonbonded_energy(alchemical_system, positions,
                                                 alchemical_regions.alchemical_atoms)
    unmod_nn_particle_sterics, unmod_aa_particle_sterics, unmod_na_particle_sterics,\
    unmod_nn_particle_electro, unmod_aa_particle_electro, unmod_na_particle_electro,\
    unmod_nn_exception_sterics, unmod_aa_exception_sterics, unmod_na_exception_sterics,\
    unmod_nn_exception_electro, unmod_aa_exception_electro, unmod_na_exception_electro,\
    unmod_nn_reciprocal_energy, unmod_aa_reciprocal_energy, unmod_na_reciprocal_energy = energy_components

    # Get alchemically-modified energy components
    print("Computing alchemical system components energies")
    alchemical_state = AlchemicalState.from_system(alchemical_system)
    alchemical_state.set_alchemical_parameters(1.0)
    energy_components = AlchemicalFactory.get_energy_components(alchemical_system, alchemical_state,
                                                                positions, platform=GLOBAL_ALCHEMY_PLATFORM)
    na_custom_particle_sterics = energy_components['alchemically modified NonbondedForce for non-alchemical/alchemical sterics']
    aa_custom_particle_sterics = energy_components['alchemically modified NonbondedForce for alchemical/alchemical sterics']
    na_custom_particle_electro = energy_components['alchemically modified NonbondedForce for non-alchemical/alchemical electrostatics']
    aa_custom_particle_electro = energy_components['alchemically modified NonbondedForce for alchemical/alchemical electrostatics']
    na_custom_exception_sterics = energy_components['alchemically modified BondForce for non-alchemical/alchemical sterics exceptions']
    aa_custom_exception_sterics = energy_components['alchemically modified BondForce for alchemical/alchemical sterics exceptions']
    na_custom_exception_electro = energy_components['alchemically modified BondForce for non-alchemical/alchemical electrostatics exceptions']
    aa_custom_exception_electro = energy_components['alchemically modified BondForce for alchemical/alchemical electrostatics exceptions']

    # Test that all NonbondedForce contributions match
    # -------------------------------------------------

    # All contributions from alchemical atoms in unmodified nonbonded force are turned off
    energy_unit = unit.kilojoule_per_mole
    err_msg = 'Non-zero contribution from unmodified NonbondedForce alchemical atoms: '
    assert_almost_equal(unmod_aa_particle_sterics, 0.0 * energy_unit, err_msg)
    assert_almost_equal(unmod_na_particle_sterics, 0.0 * energy_unit, err_msg)
    assert_almost_equal(unmod_aa_exception_sterics, 0.0 * energy_unit, err_msg)
    assert_almost_equal(unmod_na_exception_sterics, 0.0 * energy_unit, err_msg)
    assert_almost_equal(unmod_aa_particle_electro, 0.0 * energy_unit, err_msg)
    assert_almost_equal(unmod_na_particle_electro, 0.0 * energy_unit, err_msg)
    assert_almost_equal(unmod_aa_exception_electro, 0.0 * energy_unit, err_msg)
    assert_almost_equal(unmod_na_exception_electro, 0.0 * energy_unit, err_msg)
    assert_almost_equal(unmod_aa_reciprocal_energy, 0.0 * energy_unit, err_msg)
    assert_almost_equal(unmod_na_reciprocal_energy, 0.0 * energy_unit, err_msg)

    # Check sterics interactions match
    assert_almost_equal(nn_particle_sterics, unmod_nn_particle_sterics,
                        'Non-alchemical/non-alchemical atoms particle sterics')
    assert_almost_equal(nn_exception_sterics, unmod_nn_exception_sterics,
                        'Non-alchemical/non-alchemical atoms exceptions sterics')
    assert_almost_equal(aa_particle_sterics, aa_custom_particle_sterics,
                        'Alchemical/alchemical atoms particle sterics')
    assert_almost_equal(aa_exception_sterics, aa_custom_exception_sterics,
                        'Alchemical/alchemical atoms exceptions sterics')
    assert_almost_equal(na_particle_sterics, na_custom_particle_sterics,
                        'Non-alchemical/alchemical atoms particle sterics')
    assert_almost_equal(na_exception_sterics, na_custom_exception_sterics,
                        'Non-alchemical/alchemical atoms exceptions sterics')

    # Check electrostatics interactions
    assert_almost_equal(nn_particle_electro, unmod_nn_particle_electro,
                        'Non-alchemical/non-alchemical atoms particle electrostatics')
    assert_almost_equal(nn_exception_electro, unmod_nn_exception_electro,
                        'Non-alchemical/non-alchemical atoms exceptions electrostatics')
    if nonbonded_method == openmm.NonbondedForce.PME or nonbonded_method == openmm.NonbondedForce.Ewald:
        # TODO check ALL reciprocal energies if/when they'll be implemented
        # assert_almost_equal(aa_reciprocal_energy, unmod_aa_reciprocal_energy)
        # assert_almost_equal(na_reciprocal_energy, unmod_na_reciprocal_energy)
        assert_almost_equal(nn_reciprocal_energy, unmod_nn_reciprocal_energy,
                            'Non-alchemical/non-alchemical atoms reciprocal space energy')

        # Get direct space correction due to reciprocal space exceptions
        aa_correction, na_correction = compute_direct_space_correction(nonbonded_force,
                                                                       alchemical_regions.alchemical_atoms,
                                                                       positions)
        aa_particle_electro += aa_correction
        na_particle_electro += na_correction

        # Check direct space energy
        assert_almost_equal(aa_particle_electro, aa_custom_particle_electro,
                            'Alchemical/alchemical atoms particle electrostatics')
        assert_almost_equal(na_particle_electro, na_custom_particle_electro,
                            'Non-alchemical/alchemical atoms particle electrostatics')
    else:
        # Reciprocal space energy should be null in this case
        assert nn_reciprocal_energy == unmod_nn_reciprocal_energy == 0.0 * energy_unit
        assert aa_reciprocal_energy == unmod_aa_reciprocal_energy == 0.0 * energy_unit
        assert na_reciprocal_energy == unmod_na_reciprocal_energy == 0.0 * energy_unit

        # Check direct space energy
        assert_almost_equal(aa_particle_electro, aa_custom_particle_electro,
                            'Alchemical/alchemical atoms particle electrostatics')
        assert_almost_equal(na_particle_electro, na_custom_particle_electro,
                            'Non-alchemical/alchemical atoms particle electrostatics')
    assert_almost_equal(aa_exception_electro, aa_custom_exception_electro,
                        'Alchemical/alchemical atoms exceptions electrostatics')
    assert_almost_equal(na_exception_electro, na_custom_exception_electro,
                        'Non-alchemical/alchemical atoms exceptions electrostatics')

    # Check forces other than nonbonded
    # ----------------------------------
    for force_name in ['HarmonicBondForce', 'HarmonicAngleForce', 'PeriodicTorsionForce', 'GBSAOBCForce']:
        alchemical_forces_energies = [energy for label, energy in energy_components.items() if force_name in label]
        reference_force_energy = compute_energy_force(reference_system, positions, force_name)

        # There should be no force in the alchemical system if force_name is missing from the reference
        if reference_force_energy is None:
            assert len(alchemical_forces_energies) == 0, str(alchemical_forces_energies)
            continue

        # Check that the energies match
        tot_alchemical_forces_energies = 0.0 * energy_unit
        for energy in alchemical_forces_energies:
            tot_alchemical_forces_energies += energy
        assert_almost_equal(reference_force_energy, tot_alchemical_forces_energies,
                            '{} energy '.format(force_name))


def check_noninteracting_energy_components(alchemical_system, alchemical_regions, positions):
    """Check non-interacting energy components are zero when appropriate.

    Parameters
    ----------
    alchemical_system : simtk.openmm.System
        The alchemically modified system to test.
    alchemical_regions : AlchemicalRegion.
       The alchemically modified region.
    positions : n_particlesx3 array-like of simtk.openmm.unit.Quantity
        The positions to test (units of length).

    """
    alchemical_system = copy.deepcopy(alchemical_system)

    # Set state to non-interacting.
    alchemical_state = AlchemicalState.from_system(alchemical_system)
    alchemical_state.set_alchemical_parameters(0.0)
    energy_components = AlchemicalFactory.get_energy_components(alchemical_system, alchemical_state,
                                                                positions, platform=GLOBAL_ALCHEMY_PLATFORM)

    def assert_zero_energy(label):
        print('testing {}'.format(label))
        value = energy_components[label]
        assert abs(value / GLOBAL_ENERGY_UNIT) == 0.0, ("'{}' should have zero energy in annihilated alchemical"
                                                         " state, but energy is {}").format(label, str(value))

    # Check that non-alchemical/alchemical particle interactions and 1,4 exceptions have been annihilated
    assert_zero_energy('alchemically modified NonbondedForce for non-alchemical/alchemical sterics')
    assert_zero_energy('alchemically modified NonbondedForce for non-alchemical/alchemical electrostatics')
    assert_zero_energy('alchemically modified BondForce for non-alchemical/alchemical sterics exceptions')
    assert_zero_energy('alchemically modified BondForce for non-alchemical/alchemical electrostatics exceptions')

    # Check that alchemical/alchemical particle interactions and 1,4 exceptions have been annihilated
    if alchemical_regions.annihilate_sterics:
        assert_zero_energy('alchemically modified NonbondedForce for alchemical/alchemical sterics')
        assert_zero_energy('alchemically modified BondForce for alchemical/alchemical sterics exceptions')
    if alchemical_regions.annihilate_electrostatics:
        assert_zero_energy('alchemically modified NonbondedForce for alchemical/alchemical electrostatics')
        assert_zero_energy('alchemically modified BondForce for alchemical/alchemical electrostatics exceptions')

    # Check valence terms
    for force_name in ['HarmonicBondForce', 'HarmonicAngleForce', 'PeriodicTorsionForce', 'GBSAOBCForce']:
        force_label = 'alchemically modified ' + force_name
        if force_label in energy_components:
            assert_zero_energy(force_label)


# =============================================================================
# BENCHMARKING AND DEBUG FUNCTIONS
# =============================================================================

def benchmark(reference_system, alchemical_regions, positions, nsteps=500,
              timestep=1.0*unit.femtoseconds):
    """
    Benchmark performance of alchemically modified system relative to original system.

    Parameters
    ----------
    reference_system : simtk.openmm.System
        The reference System object to compare with.
    alchemical_regions : AlchemicalRegion
        The region to alchemically modify.
    positions : n_particlesx3 array-like of simtk.unit.Quantity
        The initial positions (units of distance).
    nsteps : int, optional
        Number of molecular dynamics steps to use for benchmarking (default is 500).
    timestep : simtk.unit.Quantity, optional
        Timestep to use for benchmarking (units of time, default is 1.0*unit.femtoseconds).

    """
    timer = utils.Timer()

    # Create the perturbed system.
    factory = AlchemicalFactory()
    timer.start('Create alchemical system')
    alchemical_system = factory.create_alchemical_system(reference_system, alchemical_regions)
    timer.stop('Create alchemical system')

    # Create an alchemically-perturbed state corresponding to nearly fully-interacting.
    # NOTE: We use a lambda slightly smaller than 1.0 because the AlchemicalFactory may
    # not use Custom*Force softcore versions if lambda = 1.0 identically.
    alchemical_state = AlchemicalState.from_system(alchemical_system)
    alchemical_state.set_alchemical_parameters(1.0 - 1.0e-6)

    # Create integrators.
    reference_integrator = openmm.VerletIntegrator(timestep)
    alchemical_integrator = openmm.VerletIntegrator(timestep)

    # Create contexts for sampling.
    if GLOBAL_ALCHEMY_PLATFORM:
        reference_context = openmm.Context(reference_system, reference_integrator, GLOBAL_ALCHEMY_PLATFORM)
        alchemical_context = openmm.Context(alchemical_system, alchemical_integrator, GLOBAL_ALCHEMY_PLATFORM)
    else:
        reference_context = openmm.Context(reference_system, reference_integrator)
        alchemical_context = openmm.Context(alchemical_system, alchemical_integrator)
    reference_context.setPositions(positions)
    alchemical_context.setPositions(positions)

    # Make sure all kernels are compiled.
    reference_integrator.step(1)
    alchemical_integrator.step(1)

    # Run simulations.
    timer.start('Run reference system')
    reference_integrator.step(nsteps)
    timer.stop('Run reference system')

    timer.start('Run alchemical system')
    alchemical_integrator.step(nsteps)
    timer.stop('Run alchemical system')

    timer.report_timing()


def overlap_check(reference_system, alchemical_regions, positions, nsteps=50, nsamples=200,
                  cached_trajectory_filename=None):
    """
    Test overlap between reference system and alchemical system by running a short simulation.

    Parameters
    ----------
    reference_system : simtk.openmm.System
        The reference System object to compare with.
    alchemical_regions : AlchemicalRegion
        The region to alchemically modify.
    positions : n_particlesx3 array-like of simtk.unit.Quantity
        The initial positions (units of distance).
    nsteps : int, optional
        Number of molecular dynamics steps between samples (default is 50).
    nsamples : int, optional
        Number of samples to collect (default is 100).
    cached_trajectory_filename : str, optional
        If not None, this file will be used to cache intermediate results with pickle.

    """
    temperature = 300.0 * unit.kelvin
    pressure = 1.0 * unit.atmospheres
    collision_rate = 5.0 / unit.picoseconds
    timestep = 2.0 * unit.femtoseconds
    kT = kB * temperature

    # Add a barostat if possible.
    reference_system = copy.deepcopy(reference_system)
    if reference_system.usesPeriodicBoundaryConditions():
        reference_system.addForce(openmm.MonteCarloBarostat(pressure, temperature))

    # Create a fully-interacting alchemical state.
    factory = AlchemicalFactory()
    alchemical_system = factory.create_alchemical_system(reference_system, alchemical_regions)

    # Create integrators.
    reference_integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
    alchemical_integrator = openmm.VerletIntegrator(timestep)

    # Create contexts.
    if GLOBAL_ALCHEMY_PLATFORM:
        reference_context = openmm.Context(reference_system, reference_integrator, GLOBAL_ALCHEMY_PLATFORM)
        alchemical_context = openmm.Context(alchemical_system, alchemical_integrator, GLOBAL_ALCHEMY_PLATFORM)
    else:
        reference_context = openmm.Context(reference_system, reference_integrator)
        alchemical_context = openmm.Context(alchemical_system, alchemical_integrator)

    # Initialize data structure or load if from cache.
    # du_n[n] is the potential energy difference of sample n.
    if cached_trajectory_filename is not None:
        try:
            with open(cached_trajectory_filename, 'rb') as f:
                data = pickle.load(f)
        except FileNotFoundError:
            data = dict(du_n=[])
            # Create directory if it doesn't exist.
            directory = os.path.dirname(cached_trajectory_filename)
            if not os.path.exists(directory):
                os.makedirs(directory)
        else:
            positions = data['positions']
            reference_context.setPeriodicBoxVectors(*data['box_vectors'])
    else:
        data = dict(du_n=[])

    # Collect simulation data.
    iteration = len(data['du_n'])
    reference_context.setPositions(positions)
    print()
    for sample in range(iteration, nsamples):
        print('\rSample {}/{}'.format(sample+1, nsamples), end='')
        sys.stdout.flush()

        # Run dynamics.
        reference_integrator.step(nsteps)

        # Get reference energies.
        reference_state = reference_context.getState(getEnergy=True, getPositions=True)
        reference_potential = reference_state.getPotentialEnergy()
        if np.isnan(reference_potential/kT):
            raise Exception("Reference potential is NaN")

        # Get alchemical energies.
        alchemical_context.setPeriodicBoxVectors(*reference_state.getPeriodicBoxVectors())
        alchemical_context.setPositions(reference_state.getPositions(asNumpy=True))
        alchemical_state = alchemical_context.getState(getEnergy=True)
        alchemical_potential = alchemical_state.getPotentialEnergy()
        if np.isnan(alchemical_potential/kT):
            raise Exception("Alchemical potential is NaN")

        # Update and cache data.
        data['du_n'].append((alchemical_potential - reference_potential) / kT)
        if cached_trajectory_filename is not None:
            # Save only last iteration positions and vectors.
            data['positions'] = reference_state.getPositions()
            data['box_vectors'] = reference_state.getPeriodicBoxVectors()
            with open(cached_trajectory_filename, 'wb') as f:
                pickle.dump(data, f)

    # Discard data to equilibration and subsample.
    du_n = np.array(data['du_n'])
    from pymbar import timeseries, EXP
    t0, g, Neff = timeseries.detectEquilibration(du_n)
    indices = timeseries.subsampleCorrelatedData(du_n, g=g)
    du_n = du_n[indices]

    # Compute statistics.
    DeltaF, dDeltaF = EXP(du_n)

    # Raise an exception if the error is larger than 3kT.
    MAX_DEVIATION = 3.0  # kT
    report = ('\nDeltaF = {:12.3f} +- {:12.3f} kT ({:3.2f} samples, g = {:3.1f}); '
              'du mean {:.3f} kT stddev {:.3f} kT').format(DeltaF, dDeltaF, Neff, g, du_n.mean(), du_n.std())
    print(report)
    if dDeltaF > MAX_DEVIATION:
        raise Exception(report)


def rstyle(ax):
    """Styles x,y axes to appear like ggplot2

    Must be called after all plot and axis manipulation operations have been
    carried out (needs to know final tick spacing)

    From:
    http://nbviewer.ipython.org/github/wrobstory/climatic/blob/master/examples/ggplot_styling_for_matplotlib.ipynb

    """
    import pylab
    import matplotlib
    import matplotlib.pyplot as plt

    #Set the style of the major and minor grid lines, filled blocks
    ax.grid(True, 'major', color='w', linestyle='-', linewidth=1.4)
    ax.grid(True, 'minor', color='0.99', linestyle='-', linewidth=0.7)
    ax.patch.set_facecolor('0.90')
    ax.set_axisbelow(True)

    #Set minor tick spacing to 1/2 of the major ticks
    ax.xaxis.set_minor_locator((pylab.MultipleLocator((plt.xticks()[0][1] - plt.xticks()[0][0]) / 2.0)))
    ax.yaxis.set_minor_locator((pylab.MultipleLocator((plt.yticks()[0][1] - plt.yticks()[0][0]) / 2.0)))

    #Remove axis border
    for child in ax.get_children():
        if isinstance(child, matplotlib.spines.Spine):
            child.set_alpha(0)

    #Restyle the tick lines
    for line in ax.get_xticklines() + ax.get_yticklines():
        line.set_markersize(5)
        line.set_color("gray")
        line.set_markeredgewidth(1.4)

    #Remove the minor tick lines
    for line in (ax.xaxis.get_ticklines(minor=True) +
                 ax.yaxis.get_ticklines(minor=True)):
        line.set_markersize(0)

    #Only show bottom left ticks, pointing out of axis
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')


def lambda_trace(reference_system, alchemical_regions, positions, nsteps=100):
    """
    Compute potential energy as a function of lambda.

    """

    # Create a factory to produce alchemical intermediates.
    factory = AlchemicalFactory()
    alchemical_system = factory.create_alchemical_system(reference_system, alchemical_regions)
    alchemical_state = AlchemicalState.from_system(alchemical_system)

    # Take equally-sized steps.
    delta = 1.0 / nsteps

    # Compute unmodified energy.
    u_original = compute_energy(reference_system, positions)

    # Scan through lambda values.
    lambda_i = np.zeros([nsteps+1], np.float64)  # lambda values for u_i

    # u_i[i] is the potential energy for lambda_i[i]
    u_i = unit.Quantity(np.zeros([nsteps+1], np.float64), unit.kilocalories_per_mole)
    for i in range(nsteps+1):
        lambda_i[i] = 1.0-i*delta
        alchemical_state.set_alchemical_parameters(lambda_i[i])
        alchemical_state.apply_to_system(alchemical_system)
        u_i[i] = compute_energy(alchemical_system, positions)
        logger.info("{:12.9f} {:24.8f} kcal/mol".format(lambda_i[i], u_i[i] / GLOBAL_ENERGY_UNIT))

    # Write figure as PDF.
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    with PdfPages('lambda-trace.pdf') as pdf:
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        plt.plot(1, u_original / unit.kilocalories_per_mole, 'ro', label='unmodified')
        plt.plot(lambda_i, u_i / unit.kilocalories_per_mole, 'k.', label='alchemical')
        plt.title('T4 lysozyme L99A + p-xylene : AMBER96 + OBC GBSA')
        plt.ylabel('potential (kcal/mol)')
        plt.xlabel('lambda')
        ax.legend()
        rstyle(ax)
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()


def generate_trace(test_system):
    lambda_trace(test_system['test'].system, test_system['test'].positions, test_system['receptor_atoms'], test_system['ligand_atoms'])


# =============================================================================
# TEST ALCHEMICAL FACTORY SUITE
# =============================================================================

def test_resolve_alchemical_region():
    """Test the method AlchemicalFactory._resolve_alchemical_region."""
    test_cases = [
        (testsystems.AlanineDipeptideVacuum(), range(22), 9, 36, 48),
        (testsystems.AlanineDipeptideVacuum(), range(11, 22), 4, 21, 31),
        (testsystems.LennardJonesCluster(), range(27), 0, 0, 0)
    ]

    for i, (test_case, atoms, n_bonds, n_angles, n_torsions) in enumerate(test_cases):
        system = test_case.system

        # Default arguments are converted to empty list.
        alchemical_region = AlchemicalRegion(alchemical_atoms=atoms)
        resolved_region = AlchemicalFactory._resolve_alchemical_region(system, alchemical_region)
        for region in ['bonds', 'angles', 'torsions']:
            assert getattr(resolved_region, 'alchemical_' + region) == set()

        # Numpy arrays are converted to sets.
        alchemical_region = AlchemicalRegion(alchemical_atoms=np.array(atoms),
                                             alchemical_bonds=np.array(range(n_bonds)),
                                             alchemical_angles=np.array(range(n_angles)),
                                             alchemical_torsions=np.array(range(n_torsions)))
        resolved_region = AlchemicalFactory._resolve_alchemical_region(system, alchemical_region)
        for region in ['atoms', 'bonds', 'angles', 'torsions']:
            assert isinstance(getattr(resolved_region, 'alchemical_' + region), frozenset)

        # Bonds, angles and torsions are inferred correctly.
        alchemical_region = AlchemicalRegion(alchemical_atoms=atoms, alchemical_bonds=True,
                                             alchemical_angles=True, alchemical_torsions=True)
        resolved_region = AlchemicalFactory._resolve_alchemical_region(system, alchemical_region)
        for j, region in enumerate(['bonds', 'angles', 'torsions']):
            assert len(getattr(resolved_region, 'alchemical_' + region)) == test_cases[i][j+2]

        # An exception is if indices are not part of the system.
        alchemical_region = AlchemicalRegion(alchemical_atoms=[10000000])
        with nose.tools.assert_raises(ValueError):
            AlchemicalFactory._resolve_alchemical_region(system, alchemical_region)

        # An exception is raised if nothing is defined.
        alchemical_region = AlchemicalRegion()
        with nose.tools.assert_raises(ValueError):
            AlchemicalFactory._resolve_alchemical_region(system, alchemical_region)


class TestAlchemicalFactory(object):
    """Test AlchemicalFactory class."""

    @classmethod
    def setup_class(cls):
        """Create test systems and shared objects."""
        cls.define_systems()
        cls.define_regions()
        cls.generate_cases()

    @classmethod
    def define_systems(cls):
        """Create shared test systems in cls.test_systems for the test suite."""
        cls.test_systems = dict()

        # Basic test systems: Lennard-Jones and water particles only.
        # Test also dispersion correction and switch off ("on" values
        # for these options are tested in HostGuestExplicit system).
        cls.test_systems['LennardJonesCluster'] = testsystems.LennardJonesCluster()
        cls.test_systems['LennardJonesFluid with dispersion correction'] = \
            testsystems.LennardJonesFluid(nparticles=100, dispersion_correction=True)
        cls.test_systems['WaterBox with reaction field, no switch, no dispersion correction'] = \
            testsystems.WaterBox(dispersion_correction=False, switch=False, nonbondedMethod=openmm.app.CutoffPeriodic)

        # Vacuum and implicit.
        cls.test_systems['AlanineDipeptideVacuum'] = testsystems.AlanineDipeptideVacuum()
        cls.test_systems['AlanineDipeptideImplicit'] = testsystems.AlanineDipeptideImplicit()
        cls.test_systems['TolueneImplicit'] = testsystems.TolueneImplicit()

        # Explicit test system: PME and CutoffPeriodic.
        cls.test_systems['AlanineDipeptideExplicit with CutoffPeriodic'] = \
            testsystems.AlanineDipeptideExplicit(nonbondedMethod=openmm.app.CutoffPeriodic)
        cls.test_systems['HostGuestExplicit with PME'] = \
            testsystems.HostGuestExplicit(nonbondedMethod=openmm.app.PME)

    @classmethod
    def define_regions(cls):
        """Create shared AlchemicalRegions for test systems in cls.test_regions."""
        cls.test_regions = dict()
        cls.test_regions['LennardJonesCluster'] = AlchemicalRegion(alchemical_atoms=range(2))
        cls.test_regions['LennardJonesFluid'] = AlchemicalRegion(alchemical_atoms=range(10))
        cls.test_regions['WaterBox'] = AlchemicalRegion(alchemical_atoms=range(3))
        cls.test_regions['Toluene'] = AlchemicalRegion(alchemical_atoms=range(2))  # Only partially modified.
        cls.test_regions['AlanineDipeptide'] = AlchemicalRegion(alchemical_atoms=range(22))
        cls.test_regions['HostGuestExplicit'] = AlchemicalRegion(alchemical_atoms=range(126, 156))

    @classmethod
    def generate_cases(cls):
        """Generate all test cases in cls.test_cases combinatorially."""
        cls.test_cases = dict()
        factory = AlchemicalFactory()

        # We generate all possible combinations of annihilate_sterics/electrostatics
        # for each test system. We also annihilate bonds, angles and torsions every
        # 3 test cases so that we test it at least one for each test system and for
        # each combination of annihilate_sterics/electrostatics.
        n_test_cases = 0
        for test_system_name, test_system in cls.test_systems.items():

            # Find standard alchemical region.
            for region_name, region in cls.test_regions.items():
                if region_name in test_system_name:
                    break
            assert region_name in test_system_name

            # Create all combinations of annihilate_sterics/electrostatics.
            for annihilate_sterics, annihilate_electrostatics in itertools.product((True, False), repeat=2):
                region = region._replace(annihilate_sterics=annihilate_sterics,
                                         annihilate_electrostatics=annihilate_electrostatics)

                # Create test name.
                test_case_name = test_system_name[:]
                if annihilate_sterics:
                    test_case_name += ', annihilated sterics'
                if annihilate_electrostatics:
                    test_case_name += ', annihilated electrostatics'

                # Annihilate bonds and angles every three test_cases.
                if n_test_cases % 3 == 0:
                    region = region._replace(alchemical_bonds=True, alchemical_angles=True,
                                             alchemical_torsions=True)
                    test_case_name += ', annihilated bonds, angles and torsions'

                # Add different softcore parameters every five test_cases.
                if n_test_cases % 5 == 0:
                    region = region._replace(softcore_alpha=1.0, softcore_beta=1.0, softcore_a=1.0, softcore_b=1.0,
                                             softcore_c=1.0, softcore_d=1.0, softcore_e=1.0, softcore_f=1.0)
                    test_case_name += ', modified softcore parameters'

                # Pre-generate alchemical system
                alchemical_system = factory.create_alchemical_system(test_system.system, region)
                cls.test_cases[test_case_name] = (test_system, alchemical_system, region)

                n_test_cases += 1

    def test_fully_interacting_energy(self):
        """Compare the energies of reference and fully interacting alchemical system."""
        for test_name, (test_system, alchemical_system, alchemical_region) in self.test_cases.items():
            reference_system = test_system.system
            positions = test_system.positions
            f = partial(compare_system_energies, reference_system, alchemical_system, alchemical_region, positions)
            f.description = "Testing fully interacting energy of {}".format(test_name)
            yield f

    def test_noninteracting_energy_components(self):
        """Check all forces annihilated/decoupled when their lambda variables are zero."""
        for test_name, (test_system, alchemical_system, alchemical_region) in self.test_cases.items():
            positions = test_system.positions
            f = partial(check_noninteracting_energy_components, alchemical_system, alchemical_region, positions)
            f.description = "Testing non-interacting energy of {}".format(test_name)
            yield f

    @attr('slow')
    def test_fully_interacting_energy_components(self):
        """Test interacting state energy by force component."""
        # This is a very expensive but very informative test. We can
        # run this locally when test_fully_interacting_energies() fails.
        test_cases_names = [test_name for test_name in self.test_cases
                            if 'Explicit' in test_name]
        for test_name in test_cases_names:
            test_system, alchemical_system, alchemical_region = self.test_cases[test_name]
            reference_system = test_system.system
            positions = test_system.positions
            f = partial(check_interacting_energy_components, reference_system, alchemical_system,
                        alchemical_region, positions)
            f.description = "Testing energy components of %s..." % test_name
            yield f

    @attr('slow')
    def test_platforms(self):
        """Test interacting and noninteracting energies on all platforms."""
        global GLOBAL_ALCHEMY_PLATFORM
        old_global_platform = GLOBAL_ALCHEMY_PLATFORM

        # Do not repeat tests on the platform already tested.
        if old_global_platform is None:
            default_platform_name = utils.get_fastest_platform().getName()
        else:
            default_platform_name = old_global_platform.getName()
        platforms = [platform for platform in utils.get_available_platforms()
                     if platform.getName() != default_platform_name]

        # Test interacting and noninteracting energies on all platforms.
        for platform in platforms:
            GLOBAL_ALCHEMY_PLATFORM = platform
            for test_name, (test_system, alchemical_system, alchemical_region) in self.test_cases.items():
                reference_system = test_system.system
                positions = test_system.positions
                f = partial(compare_system_energies, reference_system, alchemical_system, alchemical_region, positions)
                f.description = "Test fully interacting energy of {} on {}".format(test_name, platform.getName())
                yield f
                f = partial(check_noninteracting_energy_components, alchemical_system, alchemical_region, positions)
                f.description = "Test non-interacting energy of {} on {}".format(test_name, platform.getName())
                yield f

        # Restore global platform
        GLOBAL_ALCHEMY_PLATFORM = old_global_platform

    @attr('slow')
    def test_overlap(self):
        """Tests overlap between reference and alchemical systems."""
        for test_name, (test_system, alchemical_system, alchemical_region) in self.test_cases.items():
            reference_system = test_system.system
            positions = test_system.positions
            # cached_trajectory_filename = os.path.join(os.environ['HOME'], '.cache', 'alchemy', 'tests',
            #                                           test_name + '.pickle')
            cached_trajectory_filename = None
            f = partial(overlap_check, reference_system, alchemical_region, positions,
                        cached_trajectory_filename=cached_trajectory_filename)
            f.description = "Testing reference/alchemical overlap for {}".format(test_name)
            yield f


@attr('slow')
class TestAlchemicalFactorySlow(TestAlchemicalFactory):
    """Test AlchemicalFactory class with a more comprehensive set of systems."""

    @classmethod
    def define_systems(cls):
        """Create test systems and shared objects."""
        cls.test_systems = dict()
        cls.test_systems['LennardJonesFluid without dispersion correction'] = \
            testsystems.LennardJonesFluid(nparticles=100, dispersion_correction=False)
        cls.test_systems['DischargedWaterBox with reaction field, no switch, no dispersion correction'] = \
            testsystems.DischargedWaterBox(dispersion_correction=False, switch=False,
                                           nonbondedMethod=openmm.app.CutoffPeriodic)
        cls.test_systems['WaterBox with reaction field, no switch, dispersion correction'] = \
            testsystems.WaterBox(dispersion_correction=False, switch=True, nonbondedMethod=openmm.app.CutoffPeriodic)
        cls.test_systems['WaterBox with reaction field, switch, no dispersion correction'] = \
            testsystems.WaterBox(dispersion_correction=False, switch=True, nonbondedMethod=openmm.app.CutoffPeriodic)
        cls.test_systems['WaterBox with PME, switch, dispersion correction'] = \
            testsystems.WaterBox(dispersion_correction=True, switch=True, nonbondedMethod=openmm.app.PME)

        # Big systems.
        cls.test_systems['LysozymeImplicit'] = testsystems.LysozymeImplicit()
        cls.test_systems['DHFRExplicit with reaction field'] = \
            testsystems.DHFRExplicit(nonbondedMethod=openmm.app.CutoffPeriodic)
        cls.test_systems['SrcExplicit with reaction field'] = \
            testsystems.SrcExplicit(nonbondedMethod=openmm.app.CutoffPeriodic)
        cls.test_systems['SrcImplicit'] = testsystems.SrcImplicit()

    @classmethod
    def define_regions(cls):
        super(TestAlchemicalFactorySlow, cls).define_regions()
        cls.test_regions['LysozymeImplicit'] = AlchemicalRegion(alchemical_atoms=range(2603, 2621))
        cls.test_regions['DHFRExplicit'] = AlchemicalRegion(alchemical_atoms=range(0, 2849))
        cls.test_regions['Src'] = AlchemicalRegion(alchemical_atoms=range(0, 21))


# =============================================================================
# TEST ALCHEMICAL STATE
# =============================================================================

class TestAlchemicalState(object):
    """Test AlchemicalState compatibility with CompoundThermodynamicState."""

    @classmethod
    def setup_class(cls):
        """Create test systems and shared objects."""
        alanine_vacuum = testsystems.AlanineDipeptideVacuum()
        factory = AlchemicalFactory()

        # System with only lambda_sterics and lambda_electrostatics.
        alchemical_region = AlchemicalRegion(alchemical_atoms=range(22))
        alchemical_alanine_system = factory.create_alchemical_system(alanine_vacuum.system, alchemical_region)
        cls.alanine_state = states.ThermodynamicState(alchemical_alanine_system,
                                                      temperature=300*unit.kelvin)

        # System with all lambdas except for lambda_restraints.
        alchemical_region = AlchemicalRegion(alchemical_atoms=range(22), alchemical_torsions=True,
                                             alchemical_angles=True, alchemical_bonds=True)
        fully_alchemical_alanine_system = factory.create_alchemical_system(alanine_vacuum.system, alchemical_region)
        cls.full_alanine_state = states.ThermodynamicState(fully_alchemical_alanine_system,
                                                           temperature=300*unit.kelvin)

        # Test case: (ThermodynamicState, defined_lambda_parameters)
        cls.test_cases = [
            (cls.alanine_state, {'lambda_sterics', 'lambda_electrostatics'}),
            (cls.full_alanine_state, {'lambda_sterics', 'lambda_electrostatics', 'lambda_bonds',
                                      'lambda_angles', 'lambda_torsions'})
        ]

    @staticmethod
    def test_constructor():
        """Test AlchemicalState constructor behave as expected."""
        # Raise an exception if parameter is not recognized.
        with nose.tools.assert_raises(AlchemicalStateError):
            AlchemicalState(lambda_electro=1.0)

        # Properties are initialized correctly.
        test_cases = [{},
                      {'lambda_sterics': 0.5, 'lambda_angles': 0.5},
                      {'lambda_electrostatics': 1.0}]
        for test_kwargs in test_cases:
            alchemical_state = AlchemicalState(**test_kwargs)
            for parameter in AlchemicalState._get_supported_parameters():
                if parameter in test_kwargs:
                    assert getattr(alchemical_state, parameter) == test_kwargs[parameter]
                else:
                    assert getattr(alchemical_state, parameter) is None

    def test_from_system_constructor(self):
        """Test AlchemicalState.from_system constructor."""
        # A non-alchemical system raises an error.
        with nose.tools.assert_raises(AlchemicalStateError):
            AlchemicalState.from_system(testsystems.AlanineDipeptideVacuum().system)

        # Valid parameters are 1.0 by default in AlchemicalFactory,
        # and all the others must be None.
        for state, defined_lambdas in self.test_cases:
            alchemical_state = AlchemicalState.from_system(state.system)
            for parameter in AlchemicalState._get_supported_parameters():
                property_value = getattr(alchemical_state, parameter)
                if parameter in defined_lambdas:
                    assert property_value == 1.0, '{}: {}'.format(parameter, property_value)
                else:
                    assert property_value is None, '{}: {}'.format(parameter, property_value)

    @staticmethod
    def test_equality_operator():
        """Test equality operator between AlchemicalStates."""
        state1 = AlchemicalState(lambda_electrostatics=1.0)
        state2 = AlchemicalState(lambda_electrostatics=1.0)
        state3 = AlchemicalState(lambda_electrostatics=0.9)
        state4 = AlchemicalState(lambda_electrostatics=0.9, lambda_sterics=1.0)
        assert state1 == state2
        assert state2 != state3
        assert state3 != state4

    def test_apply_to_system(self):
        """Test method AlchemicalState.apply_to_system()."""
        # Do not modify cached test cases.
        test_cases = copy.deepcopy(self.test_cases)

        # Test precondition: all parameters are 1.0.
        for state, defined_lambdas in test_cases:
            kwargs = dict.fromkeys(defined_lambdas, 1.0)
            alchemical_state = AlchemicalState(**kwargs)
            assert alchemical_state == AlchemicalState.from_system(state.system)

        # apply_to_system() modifies the state.
        for state, defined_lambdas in test_cases:
            kwargs = dict.fromkeys(defined_lambdas, 0.5)
            alchemical_state = AlchemicalState(**kwargs)
            system = state.system
            alchemical_state.apply_to_system(system)
            system_state = AlchemicalState.from_system(system)
            assert system_state == alchemical_state

        # Raise an error if an extra parameter is defined in the system.
        for state, defined_lambdas in test_cases:
            defined_lambdas = set(defined_lambdas)  # Copy
            defined_lambdas.pop()  # Remove one element.
            kwargs = dict.fromkeys(defined_lambdas, 1.0)
            alchemical_state = AlchemicalState(**kwargs)
            with nose.tools.assert_raises(AlchemicalStateError):
                alchemical_state.apply_to_system(state.system)

        # Raise an error if an extra parameter is defined in the state.
        for state, defined_lambdas in test_cases:
            defined_lambdas = set(defined_lambdas)  # Copy
            defined_lambdas.add('lambda_restraints')  # Add extra parameter.
            kwargs = dict.fromkeys(defined_lambdas, 1.0)
            alchemical_state = AlchemicalState(**kwargs)
            with nose.tools.assert_raises(AlchemicalStateError):
                alchemical_state.apply_to_system(state.system)

    def test_check_system_consistency(self):
        """Test method AlchemicalState.check_system_consistency()."""
        # Raise error if system has MORE lambda parameters.
        alchemical_state = AlchemicalState.from_system(self.alanine_state.system)
        with nose.tools.assert_raises(AlchemicalStateError):
            alchemical_state.check_system_consistency(self.full_alanine_state.system)

        # Raise error if system has LESS lambda parameters.
        alchemical_state = AlchemicalState.from_system(self.full_alanine_state.system)
        with nose.tools.assert_raises(AlchemicalStateError):
            alchemical_state.check_system_consistency(self.alanine_state.system)

        # Raise error if system has different lambda values.
        alchemical_state.lambda_bonds = 0.5
        with nose.tools.assert_raises(AlchemicalStateError):
            alchemical_state.check_system_consistency(self.full_alanine_state.system)

    def test_apply_to_context(self):
        """Test method AlchemicalState.apply_to_context."""
        integrator = openmm.VerletIntegrator(1.0*unit.femtosecond)

        # Raise error if Context has more parameters than AlchemicalState.
        alchemical_state = AlchemicalState.from_system(self.alanine_state.system)
        context = self.full_alanine_state.create_context(copy.deepcopy(integrator))
        with nose.tools.assert_raises(AlchemicalStateError):
            alchemical_state.apply_to_context(context)

        # Raise error if AlchemicalState is applied to a Context with missing parameters.
        alchemical_state = AlchemicalState.from_system(self.full_alanine_state.system)
        context = self.alanine_state.create_context(copy.deepcopy(integrator))
        with nose.tools.assert_raises(AlchemicalStateError):
            alchemical_state.apply_to_context(context)

        # Correctly sets Context's parameters.
        alchemical_state = AlchemicalState.from_system(self.full_alanine_state.system)
        context = self.full_alanine_state.create_context(copy.deepcopy(integrator))
        alchemical_state.set_alchemical_parameters(0.5)
        alchemical_state.apply_to_context(context)
        for parameter_name, parameter_value in context.getParameters().items():
            if parameter_name in alchemical_state._parameters:
                assert parameter_value == 0.5

    def test_standardize_system(self):
        """Test method AlchemicalState.standardize_system."""
        # First create a non-standard system.
        system = copy.deepcopy(self.full_alanine_state.system)
        alchemical_state = AlchemicalState.from_system(system)
        alchemical_state.set_alchemical_parameters(0.5)
        alchemical_state.apply_to_system(system)

        # Check that _standardize_system() sets all parameters back to 1.0.
        AlchemicalState._standardize_system(system)
        standard_alchemical_state = AlchemicalState.from_system(system)
        assert alchemical_state != standard_alchemical_state
        for parameter_name, value in alchemical_state._parameters.items():
            standard_value = getattr(standard_alchemical_state, parameter_name)
            assert (value is None and standard_value is None) or (standard_value == 1.0)

    def test_alchemical_functions(self):
        """Test alchemical variables and functions work correctly."""
        system = copy.deepcopy(self.full_alanine_state.system)
        alchemical_state = AlchemicalState.from_system(system)

        # Add two alchemical variables to the state.
        alchemical_state.set_alchemical_variable('lambda', 1.0)
        alchemical_state.set_alchemical_variable('lambda2', 0.5)
        assert alchemical_state.get_alchemical_variable('lambda') == 1.0
        assert alchemical_state.get_alchemical_variable('lambda2') == 0.5

        # Cannot call an alchemical variable as a supported parameter.
        with nose.tools.assert_raises(AlchemicalStateError):
            alchemical_state.set_alchemical_variable('lambda_sterics', 0.5)

        # Assign string alchemical functions to parameters.
        alchemical_state.lambda_sterics = AlchemicalFunction('lambda')
        alchemical_state.lambda_electrostatics = AlchemicalFunction('(lambda + lambda2) / 2.0')
        assert alchemical_state.lambda_sterics == 1.0
        assert alchemical_state.lambda_electrostatics == 0.75

        # Setting alchemical variables updates alchemical parameter as well.
        alchemical_state.set_alchemical_variable('lambda2', 0)
        assert alchemical_state.lambda_electrostatics == 0.5

    def test_constructor_compound_state(self):
        """The AlchemicalState is set on construction of the CompoundState."""
        test_cases = copy.deepcopy(self.test_cases)

        # Test precondition: the original systems are in fully interacting state.
        for state, defined_lambdas in test_cases:
            system_state = AlchemicalState.from_system(state.system)
            kwargs = dict.fromkeys(defined_lambdas, 1.0)
            assert system_state == AlchemicalState(**kwargs)

        # CompoundThermodynamicState set the system state in constructor.
        for state, defined_lambdas in test_cases:
            kwargs = dict.fromkeys(defined_lambdas, 0.5)
            alchemical_state = AlchemicalState(**kwargs)
            compound_state = states.CompoundThermodynamicState(state, [alchemical_state])
            system_state = AlchemicalState.from_system(compound_state.system)
            assert system_state == alchemical_state

    def test_lambda_properties_compound_state(self):
        """Lambda properties setters/getters work in the CompoundState system."""
        test_cases = copy.deepcopy(self.test_cases)

        for state, defined_lambdas in test_cases:
            undefined_lambdas = AlchemicalState._get_supported_parameters() - defined_lambdas
            alchemical_state = AlchemicalState.from_system(state.system)
            compound_state = states.CompoundThermodynamicState(state, [alchemical_state])

            # Undefined properties raise an exception when assigned.
            for parameter_name in undefined_lambdas:
                assert getattr(compound_state, parameter_name) is None
                with nose.tools.assert_raises(AlchemicalStateError):
                    setattr(compound_state, parameter_name, 0.4)
                setattr(compound_state, parameter_name, None)  # Keep state consistent.

            # Defined properties can be assigned and read.
            for parameter_name in defined_lambdas:
                assert getattr(compound_state, parameter_name) == 1.0
                setattr(compound_state, parameter_name, 0.5)
                assert getattr(compound_state, parameter_name) == 0.5

            # System global variables are updated correctly
            system_alchemical_state = AlchemicalState.from_system(compound_state.system)
            for parameter_name in defined_lambdas:
                assert getattr(system_alchemical_state, parameter_name) == 0.5

            # Same for parameters setters.
            compound_state.set_alchemical_parameters(1.0)
            system_alchemical_state = AlchemicalState.from_system(compound_state.system)
            for parameter_name in defined_lambdas:
                assert getattr(compound_state, parameter_name) == 1.0
                assert getattr(system_alchemical_state, parameter_name) == 1.0

            # Same for alchemical variables setters.
            compound_state.set_alchemical_variable('lambda', 0.25)
            for parameter_name in defined_lambdas:
                setattr(compound_state, parameter_name, AlchemicalFunction('lambda'))
            system_alchemical_state = AlchemicalState.from_system(compound_state.system)
            for parameter_name in defined_lambdas:
                assert getattr(compound_state, parameter_name) == 0.25
                assert getattr(system_alchemical_state, parameter_name) == 0.25

    def test_set_system_compound_state(self):
        """Setting inconsistent system in compound state raise errors."""
        alanine_state = copy.deepcopy(self.alanine_state)
        alchemical_state = AlchemicalState.from_system(alanine_state.system)
        compound_state = states.CompoundThermodynamicState(alanine_state, [alchemical_state])

        # We create an inconsistent state that has different parameters.
        incompatible_state = copy.deepcopy(alchemical_state)
        incompatible_state.lambda_electrostatics = 0.5

        # Setting an inconsistent alchemical system raise an error.
        system = compound_state.system
        incompatible_state.apply_to_system(system)
        with nose.tools.assert_raises(AlchemicalStateError):
            compound_state.system = system

        # Same for set_system when called with default arguments.
        with nose.tools.assert_raises(AlchemicalStateError):
            compound_state.set_system(system)

        # This doesn't happen if we fix the state.
        compound_state.set_system(system, fix_state=True)
        assert AlchemicalState.from_system(compound_state.system) != incompatible_state

    def test_method_compatibility_compound_state(self):
        """Compatibility between states is handled correctly in compound state."""
        alanine_state = copy.deepcopy(self.alanine_state)
        alchemical_state = AlchemicalState.from_system(alanine_state.system)
        compound_state = states.CompoundThermodynamicState(alanine_state, [alchemical_state])

        # A compatible state has the same defined lambda parameters,
        # but their values can be different.
        alchemical_state_compatible = copy.deepcopy(alchemical_state)
        alchemical_state_compatible.lambda_electrostatics = 0.5
        compound_state_compatible = states.CompoundThermodynamicState(copy.deepcopy(alanine_state),
                                                                      [alchemical_state_compatible])

        # An incompatible state has a different set of defined lambdas.
        full_alanine_state = copy.deepcopy(self.full_alanine_state)
        alchemical_state_incompatible = AlchemicalState.from_system(full_alanine_state.system)
        compound_state_incompatible = states.CompoundThermodynamicState(full_alanine_state,
                                                                        [alchemical_state_incompatible])

        # Test states compatibility.
        assert compound_state.is_state_compatible(compound_state_compatible)
        assert not compound_state.is_state_compatible(compound_state_incompatible)

        # Test context compatibility.
        integrator = openmm.VerletIntegrator(1.0*unit.femtosecond)
        context = compound_state_compatible.create_context(copy.deepcopy(integrator))
        assert compound_state.is_context_compatible(context)

        context = compound_state_incompatible.create_context(copy.deepcopy(integrator))
        assert not compound_state.is_context_compatible(context)


# =============================================================================
# MAIN FOR MANUAL DEBUGGING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
