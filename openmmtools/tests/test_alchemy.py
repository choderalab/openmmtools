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
import zlib
import pickle
import itertools
from functools import partial

import nose
import scipy
from nose.plugins.attrib import attr

from openmmtools import testsystems, forces
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
# GLOBAL_ALCHEMY_PLATFORM = openmm.Platform.getPlatformByName('OpenCL') # DEBUG: Use OpenCL over CPU platform for testing since OpenCL is deterministic, while CPU is not


# =============================================================================
# TESTING UTILITIES
# =============================================================================

def create_context(system, integrator, platform=None):
    """Create a Context.

    If platform is None, GLOBAL_ALCHEMY_PLATFORM is used.

    """
    if platform is None:
        platform = GLOBAL_ALCHEMY_PLATFORM
    if platform is not None:
        context = openmm.Context(system, integrator, platform)
    else:
        context = openmm.Context(system, integrator)
    return context


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
    context = create_context(system, integrator, platform)
    context.setPositions(positions)
    state = context.getState(getEnergy=True, groups=force_group)
    potential = state.getPotentialEnergy()
    del context, integrator, state
    return potential


def minimize(system, positions, platform=None, tolerance=1.0*unit.kilocalories_per_mole/unit.angstroms, maxIterations=50):
    """Minimize the energy of the given system.

    Parameters
    ----------
    platform : simtk.openmm.Platform or None, optional
        If None, the global GLOBAL_ALCHEMY_PLATFORM will be used.
    tolerance : simtk.unit.Quantity with units compatible with energy/distance, optional, default = 1*kilocalories_per_mole/angstroms
        Minimization tolerance
    maxIterations : int, optional, default=50
        Maximum number of iterations for minimization

    Returns
    -------
    minimized_positions : simtk.openmm.Quantity with shape [nparticle,3] with units compatible with distance
        The energy-minimized positions.

    """
    timestep = 1.0 * unit.femtoseconds
    integrator = openmm.VerletIntegrator(timestep)
    context = create_context(system, integrator, platform)
    context.setPositions(positions)
    openmm.LocalEnergyMinimizer.minimize(context, tolerance, maxIterations)
    minimized_positions = context.getState(getPositions=True).getPositions(asNumpy=True)
    del context, integrator
    return minimized_positions


def compute_force_energy(system, positions, force_name):
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


def turn_off_nonbonded(system, sterics=False, electrostatics=False,
                       exceptions=False, only_atoms=frozenset()):
    """Turn off sterics and/or electrostatics interactions.

    This affects only NonbondedForce and non-alchemical CustomNonbondedForces.

    If `exceptions` is True, only the exceptions are turned off.
    Support also system that have gone through replace_reaction_field.
    The `system` must have only nonbonded forces.
    If `only_atoms` is specified, only the those atoms will be turned off.

    """
    if len(only_atoms) == 0:  # if empty, turn off all particles
        only_atoms = set(range(system.getNumParticles()))
    epsilon_coeff = 0.0 if sterics else 1.0
    charge_coeff = 0.0 if electrostatics else 1.0

    if exceptions:  # Turn off exceptions
        force_idx, nonbonded_force = forces.find_forces(system, openmm.NonbondedForce, only_one=True)

        # Exceptions.
        for exception_index in range(nonbonded_force.getNumExceptions()):
            iatom, jatom, charge, sigma, epsilon = nonbonded_force.getExceptionParameters(exception_index)
            if iatom in only_atoms or jatom in only_atoms:
                nonbonded_force.setExceptionParameters(exception_index, iatom, jatom,
                                                       charge_coeff*charge, sigma, epsilon_coeff*epsilon)

        # Offset exceptions.
        for offset_index in range(nonbonded_force.getNumExceptionParameterOffsets()):
            (parameter, exception_index, chargeprod_scale,
                sigma_scale, epsilon_scale) = nonbonded_force.getExceptionParameterOffset(offset_index)
            iatom, jatom, _, _, _ = nonbonded_force.getExceptionParameters(exception_index)
            if iatom in only_atoms or jatom in only_atoms:
                nonbonded_force.setExceptionParameterOffset(offset_index, parameter, exception_index,
                                                            charge_coeff*chargeprod_scale, sigma_scale,
                                                            epsilon_coeff*epsilon_scale)

    else:
        # Turn off particle interactions
        for force in system.getForces():
            # Handle only a Nonbonded and a CustomNonbonded (for RF).
            if not (isinstance(force, openmm.CustomNonbondedForce) and 'lambda' not in force.getEnergyFunction() or
                        isinstance(force, openmm.NonbondedForce)):
                continue

            # Particle interactions.
            for particle_index in range(force.getNumParticles()):
                if particle_index in only_atoms:
                    # Convert tuple parameters to list to allow changes.
                    parameters = list(force.getParticleParameters(particle_index))
                    parameters[0] *= charge_coeff  # charge
                    try:  # CustomNonbondedForce
                        force.setParticleParameters(particle_index, parameters)
                    except TypeError:  # NonbondedForce
                        parameters[2] *= epsilon_coeff  # epsilon
                        force.setParticleParameters(particle_index, *parameters)

            # Offset particle interactions.
            if isinstance(force, openmm.NonbondedForce):
                for offset_index in range(force.getNumParticleParameterOffsets()):
                    (parameter, particle_index, charge_scale,
                        sigma_scale, epsilon_scale) = force.getParticleParameterOffset(offset_index)
                    if particle_index in only_atoms:
                        force.setParticleParameterOffset(offset_index, parameter, particle_index,
                                                         charge_coeff*charge_scale, sigma_scale,
                                                         epsilon_coeff*epsilon_scale)



def dissect_nonbonded_energy(reference_system, positions, alchemical_atoms):
    """Dissect the nonbonded energy contributions of the reference system
    by atom group and sterics/electrostatics.

    This works also for systems objects whose CutoffPeriodic force
    has been replaced by a CustomNonbondedForce to set c_rf = 0.

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
    nonalchemical_atoms = set(range(reference_system.getNumParticles())).difference(alchemical_atoms)

    # Remove all forces but NonbondedForce and eventually the
    # CustomNonbondedForce used to model reaction field.
    reference_system = copy.deepcopy(reference_system)  # don't modify original system
    forces_to_remove = list()
    for force_index, force in enumerate(reference_system.getForces()):
        force.setForceGroup(0)
        if isinstance(force, openmm.NonbondedForce):
            force.setReciprocalSpaceForceGroup(30)  # separate PME reciprocal from direct space
        # We keep only CustomNonbondedForces that are not alchemically modified.
        elif not (isinstance(force, openmm.CustomNonbondedForce) and
                          'lambda' not in force.getEnergyFunction()):
            forces_to_remove.append(force_index)

    for force_index in reversed(forces_to_remove):
        reference_system.removeForce(force_index)
    assert len(reference_system.getForces()) <= 2

    # Compute particle interactions between different groups of atoms
    # ----------------------------------------------------------------
    system = copy.deepcopy(reference_system)

    # Compute total energy from nonbonded interactions
    tot_energy = compute_energy(system, positions)
    tot_reciprocal_energy = compute_energy(system, positions, force_group={30})

    # Compute contributions from particle sterics
    turn_off_nonbonded(system, sterics=True, only_atoms=alchemical_atoms)
    tot_energy_no_alchem_particle_sterics = compute_energy(system, positions)
    system = copy.deepcopy(reference_system)  # Restore alchemical sterics
    turn_off_nonbonded(system, sterics=True, only_atoms=nonalchemical_atoms)
    tot_energy_no_nonalchem_particle_sterics = compute_energy(system, positions)
    turn_off_nonbonded(system, sterics=True)
    tot_energy_no_particle_sterics = compute_energy(system, positions)

    tot_particle_sterics = tot_energy - tot_energy_no_particle_sterics
    nn_particle_sterics = tot_energy_no_alchem_particle_sterics - tot_energy_no_particle_sterics
    aa_particle_sterics = tot_energy_no_nonalchem_particle_sterics - tot_energy_no_particle_sterics
    na_particle_sterics = tot_particle_sterics - nn_particle_sterics - aa_particle_sterics

    # Compute contributions from particle electrostatics
    system = copy.deepcopy(reference_system)  # Restore sterics
    turn_off_nonbonded(system, electrostatics=True, only_atoms=alchemical_atoms)
    tot_energy_no_alchem_particle_electro = compute_energy(system, positions)
    nn_reciprocal_energy = compute_energy(system, positions, force_group={30})
    system = copy.deepcopy(reference_system)  # Restore alchemical electrostatics
    turn_off_nonbonded(system, electrostatics=True, only_atoms=nonalchemical_atoms)
    tot_energy_no_nonalchem_particle_electro = compute_energy(system, positions)
    aa_reciprocal_energy = compute_energy(system, positions, force_group={30})
    turn_off_nonbonded(system, electrostatics=True)
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
    system = copy.deepcopy(reference_system)  # Restore particle interactions
    turn_off_nonbonded(system, sterics=True, exceptions=True, only_atoms=alchemical_atoms)
    tot_energy_no_alchem_exception_sterics = compute_energy(system, positions)
    system = copy.deepcopy(reference_system)  # Restore alchemical sterics
    turn_off_nonbonded(system, sterics=True, exceptions=True, only_atoms=nonalchemical_atoms)
    tot_energy_no_nonalchem_exception_sterics = compute_energy(system, positions)
    turn_off_nonbonded(system, sterics=True, exceptions=True)
    tot_energy_no_exception_sterics = compute_energy(system, positions)

    tot_exception_sterics = tot_energy - tot_energy_no_exception_sterics
    nn_exception_sterics = tot_energy_no_alchem_exception_sterics - tot_energy_no_exception_sterics
    aa_exception_sterics = tot_energy_no_nonalchem_exception_sterics - tot_energy_no_exception_sterics
    na_exception_sterics = tot_exception_sterics - nn_exception_sterics - aa_exception_sterics

    # Compute contributions from exceptions electrostatics
    system = copy.deepcopy(reference_system)  # Restore exceptions sterics
    turn_off_nonbonded(system, electrostatics=True, exceptions=True, only_atoms=alchemical_atoms)
    tot_energy_no_alchem_exception_electro = compute_energy(system, positions)
    system = copy.deepcopy(reference_system)  # Restore alchemical electrostatics
    turn_off_nonbonded(system, electrostatics=True, exceptions=True, only_atoms=nonalchemical_atoms)
    tot_energy_no_nonalchem_exception_electro = compute_energy(system, positions)
    turn_off_nonbonded(system, electrostatics=True, exceptions=True)
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

    # Convert quantity positions into floats.
    if isinstance(positions, unit.Quantity):
        positions = positions.value_in_unit_system(unit.md_unit_system)

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


def is_alchemical_pme_treatment_exact(alchemical_system):
    """Return True if the given alchemical system models PME exactly."""
    # If exact PME is here, the NonbondedForce defines a
    # lambda_electrostatics variable.
    _, nonbonded_force = forces.find_forces(alchemical_system, openmm.NonbondedForce,
                                            only_one=True)
    for parameter_idx in range(nonbonded_force.getNumGlobalParameters()):
        parameter_name = nonbonded_force.getGlobalParameterName(parameter_idx)
        if parameter_name == 'lambda_electrostatics':
            return True
    return False


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
    # because the alchemical system with direct-space treatment of PME
    # does not take into account the reciprocal space.
    force_idx, nonbonded_force = forces.find_forces(reference_system, openmm.NonbondedForce, only_one=True)
    nonbonded_method = nonbonded_force.getNonbondedMethod()
    is_direct_space_pme = (nonbonded_method in [openmm.NonbondedForce.PME, openmm.NonbondedForce.Ewald] and
                           not is_alchemical_pme_treatment_exact(alchemical_system))

    if is_direct_space_pme:
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
        aa_correction, na_correction = compute_direct_space_correction(nonbonded_force, alchemical_atoms, positions)

    # Compute potential of the direct space.
    potentials = [compute_energy(system, positions, force_group=force_group)
                  for system in [reference_system, alchemical_system]]

    # Add the direct space correction.
    if is_direct_space_pme:
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
    energy_unit = unit.kilojoule_per_mole
    reference_system = copy.deepcopy(reference_system)
    alchemical_system = copy.deepcopy(alchemical_system)
    is_exact_pme = is_alchemical_pme_treatment_exact(alchemical_system)

    # Find nonbonded method
    _, nonbonded_force = forces.find_forces(reference_system, openmm.NonbondedForce, only_one=True)
    nonbonded_method = nonbonded_force.getNonbondedMethod()

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
    energy_components = AbsoluteAlchemicalFactory.get_energy_components(alchemical_system, alchemical_state,
                                                                        positions, platform=GLOBAL_ALCHEMY_PLATFORM)

    # Sterics particle and exception interactions are always modeled with a custom force.
    na_custom_particle_sterics = energy_components['alchemically modified NonbondedForce for non-alchemical/alchemical sterics']
    aa_custom_particle_sterics = energy_components['alchemically modified NonbondedForce for alchemical/alchemical sterics']
    na_custom_exception_sterics = energy_components['alchemically modified BondForce for non-alchemical/alchemical sterics exceptions']
    aa_custom_exception_sterics = energy_components['alchemically modified BondForce for alchemical/alchemical sterics exceptions']

    # With exact treatment of PME, we use the NonbondedForce offset for electrostatics.
    try:
        na_custom_particle_electro = energy_components['alchemically modified NonbondedForce for non-alchemical/alchemical electrostatics']
        aa_custom_particle_electro = energy_components['alchemically modified NonbondedForce for alchemical/alchemical electrostatics']
        na_custom_exception_electro = energy_components['alchemically modified BondForce for non-alchemical/alchemical electrostatics exceptions']
        aa_custom_exception_electro = energy_components['alchemically modified BondForce for alchemical/alchemical electrostatics exceptions']
    except KeyError:
        assert is_exact_pme

    # Test that all NonbondedForce contributions match
    # -------------------------------------------------

    # All contributions from alchemical atoms in unmodified nonbonded force are turned off
    err_msg = 'Non-zero contribution from unmodified NonbondedForce alchemical atoms: '
    assert_almost_equal(unmod_aa_particle_sterics, 0.0 * energy_unit, err_msg)
    assert_almost_equal(unmod_na_particle_sterics, 0.0 * energy_unit, err_msg)
    assert_almost_equal(unmod_aa_exception_sterics, 0.0 * energy_unit, err_msg)
    assert_almost_equal(unmod_na_exception_sterics, 0.0 * energy_unit, err_msg)
    if not is_exact_pme:
        # With exact PME treatment these are tested below.
        assert_almost_equal(unmod_aa_particle_electro, 0.0 * energy_unit, err_msg)
        assert_almost_equal(unmod_na_particle_electro, 0.0 * energy_unit, err_msg)
        assert_almost_equal(unmod_aa_reciprocal_energy, 0.0 * energy_unit, err_msg)
        assert_almost_equal(unmod_na_reciprocal_energy, 0.0 * energy_unit, err_msg)
        assert_almost_equal(unmod_aa_exception_electro, 0.0 * energy_unit, err_msg)
        assert_almost_equal(unmod_na_exception_electro, 0.0 * energy_unit, err_msg)

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
    # With exact treatment of PME, the electrostatics of alchemical-alchemical
    # atoms is modeled with NonbondedForce offsets.
    if is_exact_pme:
        # Reciprocal space.
        assert_almost_equal(aa_reciprocal_energy, unmod_aa_reciprocal_energy,
                            'Alchemical/alchemical atoms reciprocal space energy')
        assert_almost_equal(na_reciprocal_energy, unmod_na_reciprocal_energy,
                            'Non-alchemical/alchemical atoms reciprocal space energy')
        # Direct space.
        assert_almost_equal(aa_particle_electro, unmod_aa_particle_electro,
                            'Alchemical/alchemical atoms particle electrostatics')
        assert_almost_equal(na_particle_electro, unmod_na_particle_electro,
                            'Non-alchemical/alchemical atoms particle electrostatics')
        # Exceptions.
        assert_almost_equal(aa_exception_electro, unmod_aa_exception_electro,
                            'Alchemical/alchemical atoms exceptions electrostatics')
        assert_almost_equal(na_exception_electro, unmod_na_exception_electro,
                            'Non-alchemical/alchemical atoms exceptions electrostatics')
    # With direct space PME, the custom forces model only the
    # direct space of alchemical-alchemical interactions.
    else:
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
        # Check exceptions.
        assert_almost_equal(aa_exception_electro, aa_custom_exception_electro,
                            'Alchemical/alchemical atoms exceptions electrostatics')
        assert_almost_equal(na_exception_electro, na_custom_exception_electro,
                            'Non-alchemical/alchemical atoms exceptions electrostatics')

    # With Ewald methods, the NonbondedForce should always hold the
    # reciprocal space energy of nonalchemical-nonalchemical atoms.
    if nonbonded_method in [openmm.NonbondedForce.PME, openmm.NonbondedForce.Ewald]:
        # Reciprocal space.
        assert_almost_equal(nn_reciprocal_energy, unmod_nn_reciprocal_energy,
                            'Non-alchemical/non-alchemical atoms reciprocal space energy')
    else:
        # Reciprocal space energy should be null in this case
        assert nn_reciprocal_energy == unmod_nn_reciprocal_energy == 0.0 * energy_unit
        assert aa_reciprocal_energy == unmod_aa_reciprocal_energy == 0.0 * energy_unit
        assert na_reciprocal_energy == unmod_na_reciprocal_energy == 0.0 * energy_unit

    # Check forces other than nonbonded
    # ----------------------------------
    for force_name in ['HarmonicBondForce', 'HarmonicAngleForce', 'PeriodicTorsionForce',
                       'GBSAOBCForce', 'CustomGBForce']:
        alchemical_forces_energies = [energy for label, energy in energy_components.items() if force_name in label]
        reference_force_energy = compute_force_energy(reference_system, positions, force_name)

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


def check_noninteracting_energy_components(reference_system, alchemical_system, alchemical_regions, positions):
    """Check non-interacting energy components are zero when appropriate.

    Parameters
    ----------
    reference_system : simtk.openmm.System
        The reference system (not alchemically modified).
    alchemical_system : simtk.openmm.System
        The alchemically modified system to test.
    alchemical_regions : AlchemicalRegion.
       The alchemically modified region.
    positions : n_particlesx3 array-like of simtk.openmm.unit.Quantity
        The positions to test (units of length).

    """
    alchemical_system = copy.deepcopy(alchemical_system)
    is_exact_pme = is_alchemical_pme_treatment_exact(alchemical_system)

    # Set state to non-interacting.
    alchemical_state = AlchemicalState.from_system(alchemical_system)
    alchemical_state.set_alchemical_parameters(0.0)
    energy_components = AbsoluteAlchemicalFactory.get_energy_components(alchemical_system, alchemical_state,
                                                                        positions, platform=GLOBAL_ALCHEMY_PLATFORM)

    def assert_zero_energy(label):
        print('testing {}'.format(label))
        value = energy_components[label]
        assert abs(value / GLOBAL_ENERGY_UNIT) == 0.0, ("'{}' should have zero energy in annihilated alchemical"
                                                        " state, but energy is {}").format(label, str(value))

    # Check that non-alchemical/alchemical particle interactions and 1,4 exceptions have been annihilated
    assert_zero_energy('alchemically modified BondForce for non-alchemical/alchemical sterics exceptions')
    assert_zero_energy('alchemically modified NonbondedForce for non-alchemical/alchemical sterics')
    if is_exact_pme:
        assert 'alchemically modified NonbondedForce for non-alchemical/alchemical electrostatics' not in energy_components
        assert 'alchemically modified BondForce for non-alchemical/alchemical electrostatics exceptions' not in energy_components
    else:
        assert_zero_energy('alchemically modified NonbondedForce for non-alchemical/alchemical electrostatics')
        assert_zero_energy('alchemically modified BondForce for non-alchemical/alchemical electrostatics exceptions')

    # Check that alchemical/alchemical particle interactions and 1,4 exceptions have been annihilated
    if alchemical_regions.annihilate_sterics:
        assert_zero_energy('alchemically modified NonbondedForce for alchemical/alchemical sterics')
        assert_zero_energy('alchemically modified BondForce for alchemical/alchemical sterics exceptions')
    if alchemical_regions.annihilate_electrostatics:
        if is_exact_pme:
            assert 'alchemically modified NonbondedForce for alchemical/alchemical electrostatics' not in energy_components
            assert 'alchemically modified BondForce for alchemical/alchemical electrostatics exceptions' not in energy_components
        else:
            assert_zero_energy('alchemically modified NonbondedForce for alchemical/alchemical electrostatics')
            assert_zero_energy('alchemically modified BondForce for alchemical/alchemical electrostatics exceptions')

    # Check valence terms
    for force_name in ['HarmonicBondForce', 'HarmonicAngleForce', 'PeriodicTorsionForce']:
        force_label = 'alchemically modified ' + force_name
        if force_label in energy_components:
            assert_zero_energy(force_label)

    # Check implicit solvent force.
    for force_name in ['CustomGBForce', 'GBSAOBCForce']:
        label = 'alchemically modified ' + force_name

        # Check if the system has an implicit solvent force.
        try:
            alchemical_energy = energy_components[label]
        except KeyError:  # No implicit solvent.
            continue

        # If all alchemical particles are modified, the alchemical energy should be zero.
        if len(alchemical_regions.alchemical_atoms) == reference_system.getNumParticles():
            assert_zero_energy(label)
            continue

        # Otherwise compare the alchemical energy with a
        # reference system with only non-alchemical particles.
        # Find implicit solvent force in reference system.
        for reference_force in reference_system.getForces():
            if reference_force.__class__.__name__ == force_name:
                break

        system = openmm.System()
        force = reference_force.__class__()

        # For custom GB forces, we need to copy all computed values,
        # energy terms, parameters, tabulated functions and exclusions.
        if isinstance(force, openmm.CustomGBForce):
            for index in range(reference_force.getNumPerParticleParameters()):
                name = reference_force.getPerParticleParameterName(index)
                force.addPerParticleParameter(name)
            for index in range(reference_force.getNumComputedValues()):
                computed_value = reference_force.getComputedValueParameters(index)
                force.addComputedValue(*computed_value)
            for index in range(reference_force.getNumEnergyTerms()):
                energy_term = reference_force.getEnergyTermParameters(index)
                force.addEnergyTerm(*energy_term)
            for index in range(reference_force.getNumGlobalParameters()):
                name = reference_force.getGlobalParameterName(index)
                default_value = reference_force.getGlobalParameterDefaultValue(index)
                force.addGlobalParameter(name, default_value)
            for function_index in range(reference_force.getNumTabulatedFunctions()):
                name = reference_force.getTabulatedFunctionName(function_index)
                function = reference_force.getTabulatedFunction(function_index)
                function_copy = copy.deepcopy(function)
                force.addTabulatedFunction(name, function_copy)
            for exclusion_index in range(reference_force.getNumExclusions()):
                particles = reference_force.getExclusionParticles(exclusion_index)
                force.addExclusion(*particles)

        # Create a system with only the non-alchemical particles.
        for particle_index in range(reference_system.getNumParticles()):
            if particle_index not in alchemical_regions.alchemical_atoms:
                # Add particle to System.
                mass = reference_system.getParticleMass(particle_index)
                system.addParticle(mass)

                # Add particle to Force..
                parameters = reference_force.getParticleParameters(particle_index)
                try:  # GBSAOBCForce
                    force.addParticle(*parameters)
                except NotImplementedError:  # CustomGBForce
                    force.addParticle(parameters)

        system.addForce(force)

        # Get positions for all non-alchemical particles.
        non_alchemical_positions = [pos for i, pos in enumerate(positions)
                                    if i not in alchemical_regions.alchemical_atoms]

        # Compute reference force energy.
        reference_force_energy = compute_force_energy(system, non_alchemical_positions, force_name)
        assert_almost_equal(reference_force_energy, alchemical_energy,
                            'reference {}, alchemical {}'.format(reference_force_energy, alchemical_energy))


def check_split_force_groups(system):
    """Check that force groups are split correctly."""
    force_groups_by_lambda = {}
    lambdas_by_force_group = {}

    # Separate forces groups by lambda parameters that AlchemicalState supports.
    for force, lambda_name, _ in AlchemicalState._get_system_controlled_parameters(
            system, parameters_name_suffix=None):
        force_group = force.getForceGroup()
        try:
            force_groups_by_lambda[lambda_name].add(force_group)
        except KeyError:
            force_groups_by_lambda[lambda_name] = {force_group}
        try:
            lambdas_by_force_group[force_group].add(lambda_name)
        except KeyError:
            lambdas_by_force_group[force_group] = {lambda_name}

    # Check that force group 0 doesn't hold alchemical forces.
    assert 0 not in force_groups_by_lambda

    # There are as many alchemical force groups as not-None lambda variables.
    alchemical_state = AlchemicalState.from_system(system)
    valid_lambdas = {lambda_name for lambda_name in alchemical_state._get_controlled_parameters()
                     if getattr(alchemical_state, lambda_name) is not None}
    assert valid_lambdas == set(force_groups_by_lambda.keys())

    # Check that force groups and lambda variables are in 1-to-1 correspondence.
    assert len(force_groups_by_lambda) == len(lambdas_by_force_group)
    for d in [force_groups_by_lambda, lambdas_by_force_group]:
        for value in d.values():
            assert len(value) == 1

    # With exact treatment of PME, the NonbondedForce must
    # be in the lambda_electrostatics force group.
    if is_alchemical_pme_treatment_exact(system):
        force_idx, nonbonded_force = forces.find_forces(system, openmm.NonbondedForce, only_one=True)
        assert force_groups_by_lambda['lambda_electrostatics'] == {nonbonded_force.getForceGroup()}


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
    factory = AbsoluteAlchemicalFactory()
    timer.start('Create alchemical system')
    alchemical_system = factory.create_alchemical_system(reference_system, alchemical_regions)
    timer.stop('Create alchemical system')

    # Create an alchemically-perturbed state corresponding to nearly fully-interacting.
    # NOTE: We use a lambda slightly smaller than 1.0 because the AbsoluteAlchemicalFactory
    # may not use Custom*Force softcore versions if lambda = 1.0 identically.
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
    print('Running reference system...')
    timer.start('Run reference system')
    reference_integrator.step(nsteps)
    timer.stop('Run reference system')

    print('Running alchemical system...')
    timer.start('Run alchemical system')
    alchemical_integrator.step(nsteps)
    timer.stop('Run alchemical system')
    print('Done.')

    timer.report_timing()


def benchmark_alchemy_from_pdb():
    """CLI entry point for benchmarking alchemical performance from a PDB file.
    """
    logging.basicConfig(level=logging.DEBUG)

    import mdtraj
    import argparse
    from simtk.openmm import app

    parser = argparse.ArgumentParser(description='Benchmark performance of alchemically-modified system.')
    parser.add_argument('-p', '--pdb', metavar='PDBFILE', type=str, action='store', required=True,
                        help='PDB file to benchmark; only protein forcefields supported for now (no small molecules)')
    parser.add_argument('-s', '--selection', metavar='SELECTION', type=str, action='store', default='not water',
                        help='MDTraj DSL describing alchemical region (default: "not water")')
    parser.add_argument('-n', '--nsteps', metavar='STEPS', type=int, action='store', default=1000,
                        help='Number of benchmarking steps (default: 1000)')
    args = parser.parse_args()
    # Read the PDB file
    print('Loading PDB file...')
    pdbfile = app.PDBFile(args.pdb)
    print('Loading forcefield...')
    forcefield = app.ForceField('amber99sbildn.xml', 'tip3p.xml')
    print('Adding missing hydrogens...')
    modeller = app.Modeller(pdbfile.topology, pdbfile.positions)
    modeller.addHydrogens(forcefield)
    print('Creating System...')
    reference_system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.PME)
    # Minimize
    print('Minimizing...')
    positions = minimize(reference_system, modeller.positions)
    # Select alchemical regions
    mdtraj_topology = mdtraj.Topology.from_openmm(modeller.topology)
    alchemical_atoms = mdtraj_topology.select(args.selection)
    alchemical_region = AlchemicalRegion(alchemical_atoms=alchemical_atoms)
    print('There are %d atoms in the alchemical region.' % len(alchemical_atoms))
    # Benchmark
    print('Benchmarking...')
    benchmark(reference_system, alchemical_region, positions, nsteps=args.nsteps, timestep=1.0*unit.femtoseconds)


def overlap_check(reference_system, alchemical_system, positions, nsteps=50, nsamples=200,
                  cached_trajectory_filename=None, name=""):
    """
    Test overlap between reference system and alchemical system by running a short simulation.

    Parameters
    ----------
    reference_system : simtk.openmm.System
        The reference System object to compare with.
    alchemical_system : simtk.openmm.System
        Alchemically-modified system.
    positions : n_particlesx3 array-like of simtk.unit.Quantity
        The initial positions (units of distance).
    nsteps : int, optional
        Number of molecular dynamics steps between samples (default is 50).
    nsamples : int, optional
        Number of samples to collect (default is 100).
    cached_trajectory_filename : str, optional, default=None
        If not None, this file will be used to cache intermediate results with pickle.
    name : str, optional, default=None
        Name of test system being evaluated.

    """
    temperature = 300.0 * unit.kelvin
    pressure = 1.0 * unit.atmospheres
    collision_rate = 5.0 / unit.picoseconds
    timestep = 2.0 * unit.femtoseconds
    kT = kB * temperature

    # Minimize
    positions = minimize(reference_system, positions)

    # Add a barostat if possible.
    reference_system = copy.deepcopy(reference_system)
    if reference_system.usesPeriodicBoundaryConditions():
        reference_system.addForce(openmm.MonteCarloBarostat(pressure, temperature))

    # Create integrators.
    reference_integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
    alchemical_integrator = openmm.VerletIntegrator(timestep)

    # Create contexts.
    reference_context = create_context(reference_system, reference_integrator)
    alchemical_context = create_context(alchemical_system, alchemical_integrator)

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
    factory = AbsoluteAlchemicalFactory()
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
    """Test the method AbsoluteAlchemicalFactory._resolve_alchemical_region."""
    test_cases = [
        (testsystems.AlanineDipeptideVacuum(), range(22), 9, 36, 48),
        (testsystems.AlanineDipeptideVacuum(), range(11, 22), 4, 21, 31),
        (testsystems.LennardJonesCluster(), range(27), 0, 0, 0)
    ]

    for i, (test_case, atoms, n_bonds, n_angles, n_torsions) in enumerate(test_cases):
        system = test_case.system

        # Default arguments are converted to empty list.
        alchemical_region = AlchemicalRegion(alchemical_atoms=atoms)
        resolved_region = AbsoluteAlchemicalFactory._resolve_alchemical_region(system, alchemical_region)
        for region in ['bonds', 'angles', 'torsions']:
            assert getattr(resolved_region, 'alchemical_' + region) == set()

        # Numpy arrays are converted to sets.
        alchemical_region = AlchemicalRegion(alchemical_atoms=np.array(atoms),
                                             alchemical_bonds=np.array(range(n_bonds)),
                                             alchemical_angles=np.array(range(n_angles)),
                                             alchemical_torsions=np.array(range(n_torsions)))
        resolved_region = AbsoluteAlchemicalFactory._resolve_alchemical_region(system, alchemical_region)
        for region in ['atoms', 'bonds', 'angles', 'torsions']:
            assert isinstance(getattr(resolved_region, 'alchemical_' + region), frozenset)

        # Bonds, angles and torsions are inferred correctly.
        alchemical_region = AlchemicalRegion(alchemical_atoms=atoms, alchemical_bonds=True,
                                             alchemical_angles=True, alchemical_torsions=True)
        resolved_region = AbsoluteAlchemicalFactory._resolve_alchemical_region(system, alchemical_region)
        for j, region in enumerate(['bonds', 'angles', 'torsions']):
            assert len(getattr(resolved_region, 'alchemical_' + region)) == test_cases[i][j+2]

        # An exception is if indices are not part of the system.
        alchemical_region = AlchemicalRegion(alchemical_atoms=[10000000])
        with nose.tools.assert_raises(ValueError):
            AbsoluteAlchemicalFactory._resolve_alchemical_region(system, alchemical_region)

        # An exception is raised if nothing is defined.
        alchemical_region = AlchemicalRegion()
        with nose.tools.assert_raises(ValueError):
            AbsoluteAlchemicalFactory._resolve_alchemical_region(system, alchemical_region)


class TestAbsoluteAlchemicalFactory(object):
    """Test AbsoluteAlchemicalFactory class."""

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
        cls.test_systems['TIP3P WaterBox with reaction field, no switch, no dispersion correction'] = \
            testsystems.WaterBox(dispersion_correction=False, switch=False, nonbondedMethod=openmm.app.CutoffPeriodic)
        cls.test_systems['TIP4P-EW WaterBox and NaCl with PME'] = \
            testsystems.WaterBox(nonbondedMethod=openmm.app.PME, model='tip4pew', ionic_strength=200*unit.millimolar)

        # Vacuum and implicit.
        cls.test_systems['AlanineDipeptideVacuum'] = testsystems.AlanineDipeptideVacuum()
        cls.test_systems['AlanineDipeptideImplicit'] = testsystems.AlanineDipeptideImplicit()
        cls.test_systems['TolueneImplicitOBC2'] = testsystems.TolueneImplicitOBC2()
        cls.test_systems['TolueneImplicitGBn'] = testsystems.TolueneImplicitGBn()

        # Explicit test system: PME and CutoffPeriodic.
        #cls.test_systems['AlanineDipeptideExplicit with CutoffPeriodic'] = \
        #    testsystems.AlanineDipeptideExplicit(nonbondedMethod=openmm.app.CutoffPeriodic)
        cls.test_systems['HostGuestExplicit with PME'] = \
            testsystems.HostGuestExplicit(nonbondedMethod=openmm.app.PME)
        cls.test_systems['HostGuestExplicit with CutoffPeriodic'] = \
            testsystems.HostGuestExplicit(nonbondedMethod=openmm.app.CutoffPeriodic)

    @classmethod
    def define_regions(cls):
        """Create shared AlchemicalRegions for test systems in cls.test_regions."""
        cls.test_regions = dict()
        cls.test_regions['LennardJonesCluster'] = AlchemicalRegion(alchemical_atoms=range(2))
        cls.test_regions['LennardJonesFluid'] = AlchemicalRegion(alchemical_atoms=range(10))
        cls.test_regions['TIP3P WaterBox'] = AlchemicalRegion(alchemical_atoms=range(3))
        cls.test_regions['TIP4P-EW WaterBox and NaCl'] = AlchemicalRegion(alchemical_atoms=range(4))  # Modify ions.
        cls.test_regions['Toluene'] = AlchemicalRegion(alchemical_atoms=range(6))  # Only partially modified.
        cls.test_regions['AlanineDipeptide'] = AlchemicalRegion(alchemical_atoms=range(22))
        cls.test_regions['HostGuestExplicit'] = AlchemicalRegion(alchemical_atoms=range(126, 156))

    @classmethod
    def generate_cases(cls):
        """Generate all test cases in cls.test_cases combinatorially."""
        cls.test_cases = dict()
        direct_space_factory = AbsoluteAlchemicalFactory(alchemical_pme_treatment='direct-space',
                                                         alchemical_rf_treatment='switched')
        exact_pme_factory = AbsoluteAlchemicalFactory(alchemical_pme_treatment='exact')

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
            assert region_name in test_system_name, test_system_name

            # Find nonbonded method.
            force_idx, nonbonded_force = forces.find_forces(test_system.system, openmm.NonbondedForce, only_one=True)
            nonbonded_method = nonbonded_force.getNonbondedMethod()

            # Create all combinations of annihilate_sterics/electrostatics.
            for annihilate_sterics, annihilate_electrostatics in itertools.product((True, False), repeat=2):
                # Create new region that we can modify.
                test_region = region._replace(annihilate_sterics=annihilate_sterics,
                                              annihilate_electrostatics=annihilate_electrostatics)

                # Create test name.
                test_case_name = test_system_name[:]
                if annihilate_sterics:
                    test_case_name += ', annihilated sterics'
                if annihilate_electrostatics:
                    test_case_name += ', annihilated electrostatics'

                # Annihilate bonds and angles every three test_cases.
                if n_test_cases % 3 == 0:
                    test_region = test_region._replace(alchemical_bonds=True, alchemical_angles=True,
                                                       alchemical_torsions=True)
                    test_case_name += ', annihilated bonds, angles and torsions'

                # Add different softcore parameters every five test_cases.
                if n_test_cases % 5 == 0:
                    test_region = test_region._replace(softcore_alpha=1.0, softcore_beta=1.0, softcore_a=1.0, softcore_b=1.0,
                                                       softcore_c=1.0, softcore_d=1.0, softcore_e=1.0, softcore_f=1.0)
                    test_case_name += ', modified softcore parameters'

                # Pre-generate alchemical system.
                alchemical_system = direct_space_factory.create_alchemical_system(test_system.system, test_region)

                # Add test case.
                cls.test_cases[test_case_name] = (test_system, alchemical_system, test_region)
                n_test_cases += 1

                # If we don't use softcore electrostatics and we annihilate charges
                # we can test also exact PME treatment. We don't increase n_test_cases
                # purposely to keep track of which tests are added above.
                if (test_region.softcore_beta == 0.0 and annihilate_electrostatics and
                            nonbonded_method in [openmm.NonbondedForce.PME, openmm.NonbondedForce.Ewald]):
                    alchemical_system = exact_pme_factory.create_alchemical_system(test_system.system, test_region)
                    test_case_name += ', exact PME'
                    cls.test_cases[test_case_name] = (test_system, alchemical_system, test_region)

            # If the test system uses reaction field replace reaction field
            # of the reference system to allow comparisons.
            if nonbonded_method == openmm.NonbondedForce.CutoffPeriodic:
                forcefactories.replace_reaction_field(test_system.system, return_copy=False,
                                                      switch_width=direct_space_factory.switch_width)

    def filter_cases(self, condition_func, max_number=None):
        """Return the list of test cases that satisfy condition_func(test_case_name)."""
        if max_number is None:
            max_number = len(self.test_cases)

        test_cases = {}
        for test_name, test_case in self.test_cases.items():
            if condition_func(test_name):
                test_cases[test_name] = test_case
            if len(test_cases) >= max_number:
                break
        return test_cases

    def test_split_force_groups(self):
        """Forces having different lambda variables should have a different force group."""
        # Select 1 implicit, 1 explicit, and 1 exact PME explicit test case randomly.
        test_cases = self.filter_cases(lambda x: 'Implicit' in x, max_number=1)
        test_cases.update(self.filter_cases(lambda x: 'Explicit ' in x and 'exact PME' in x, max_number=1))
        test_cases.update(self.filter_cases(lambda x: 'Explicit ' in x and 'exact PME' not in x, max_number=1))
        for test_name, (test_system, alchemical_system, alchemical_region) in test_cases.items():
            f = partial(check_split_force_groups, alchemical_system)
            f.description = "Testing force splitting among groups of {}".format(test_name)
            yield f

    def test_fully_interacting_energy(self):
        """Compare the energies of reference and fully interacting alchemical system."""
        for test_name, (test_system, alchemical_system, alchemical_region) in self.test_cases.items():
            f = partial(compare_system_energies, test_system.system,
                        alchemical_system, alchemical_region, test_system.positions)
            f.description = "Testing fully interacting energy of {}".format(test_name)
            yield f

    def test_noninteracting_energy_components(self):
        """Check all forces annihilated/decoupled when their lambda variables are zero."""
        for test_name, (test_system, alchemical_system, alchemical_region) in self.test_cases.items():
            f = partial(check_noninteracting_energy_components, test_system.system, alchemical_system,
                        alchemical_region, test_system.positions)
            f.description = "Testing non-interacting energy of {}".format(test_name)
            yield f

    @attr('slow')
    def test_fully_interacting_energy_components(self):
        """Test interacting state energy by force component."""
        # This is a very expensive but very informative test. We can
        # run this locally when test_fully_interacting_energies() fails.
        test_cases = self.filter_cases(lambda x: 'Explicit' in x)
        for test_name, (test_system, alchemical_system, alchemical_region) in test_cases.items():
            f = partial(check_interacting_energy_components, test_system.system, alchemical_system,
                        alchemical_region, test_system.positions)
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
                f = partial(compare_system_energies, test_system.system, alchemical_system,
                            alchemical_region, test_system.positions)
                f.description = "Test fully interacting energy of {} on {}".format(test_name, platform.getName())
                yield f
                f = partial(check_noninteracting_energy_components, test_system.system, alchemical_system,
                            alchemical_region, test_system.positions)
                f.description = "Test non-interacting energy of {} on {}".format(test_name, platform.getName())
                yield f

        # Restore global platform
        GLOBAL_ALCHEMY_PLATFORM = old_global_platform

    @attr('slow')
    def test_overlap(self):
        """Tests overlap between reference and alchemical systems."""
        for test_name, (test_system, alchemical_system, alchemical_region) in self.test_cases.items():
            #cached_trajectory_filename = os.path.join(os.environ['HOME'], '.cache', 'alchemy', 'tests',
            #                                           test_name + '.pickle')
            cached_trajectory_filename = None
            f = partial(overlap_check, test_system.system, alchemical_system, test_system.positions,
                        cached_trajectory_filename=cached_trajectory_filename, name=test_name)
            f.description = "Testing reference/alchemical overlap for {}".format(test_name)
            yield f


class TestDispersionlessAlchemicalFactory(object):
    """
    Only test overlap for dispersionless alchemical factory, since energy agreement
    will be poor.
    """
    @classmethod
    def setup_class(cls):
        """Create test systems and shared objects."""
        cls.define_systems()
        cls.define_regions()
        cls.generate_cases()

    @classmethod
    def define_systems(cls):
        """Create test systems and shared objects."""
        cls.test_systems = dict()
        cls.test_systems['LennardJonesFluid with dispersion correction'] = \
            testsystems.LennardJonesFluid(nparticles=100, dispersion_correction=True)

    @classmethod
    def define_regions(cls):
        """Create shared AlchemicalRegions for test systems in cls.test_regions."""
        cls.test_regions = dict()
        cls.test_regions['LennardJonesFluid'] = AlchemicalRegion(alchemical_atoms=range(10))

    @classmethod
    def generate_cases(cls):
        """Generate all test cases in cls.test_cases combinatorially."""
        cls.test_cases = dict()
        factory = AbsoluteAlchemicalFactory(disable_alchemical_dispersion_correction=True)

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

            # Create all combinations of annihilate_sterics.
            for annihilate_sterics in itertools.product((True, False), repeat=1):
                region = region._replace(annihilate_sterics=annihilate_sterics,
                                         annihilate_electrostatics=True)

                # Create test name.
                test_case_name = test_system_name[:]
                if annihilate_sterics:
                    test_case_name += ', annihilated sterics'

                # Pre-generate alchemical system
                alchemical_system = factory.create_alchemical_system(test_system.system, region)
                cls.test_cases[test_case_name] = (test_system, alchemical_system, region)

                n_test_cases += 1

    def test_overlap(self):
        """Tests overlap between reference and alchemical systems."""
        for test_name, (test_system, alchemical_system, alchemical_region) in self.test_cases.items():
            #cached_trajectory_filename = os.path.join(os.environ['HOME'], '.cache', 'alchemy', 'tests',
            #                                           test_name + '.pickle')
            cached_trajectory_filename = None
            f = partial(overlap_check, test_system.system, alchemical_system, test_system.positions,
                        cached_trajectory_filename=cached_trajectory_filename, name=test_name)
            f.description = "Testing reference/alchemical overlap for no alchemical dispersion {}".format(test_name)
            yield f


@attr('slow')
class TestAbsoluteAlchemicalFactorySlow(TestAbsoluteAlchemicalFactory):
    """Test AbsoluteAlchemicalFactory class with a more comprehensive set of systems."""

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
        cls.test_systems['SrcExplicit with PME'] = \
            testsystems.SrcExplicit(nonbondedMethod=openmm.app.PME)
        cls.test_systems['SrcExplicit with reaction field'] = \
            testsystems.SrcExplicit(nonbondedMethod=openmm.app.CutoffPeriodic)
        cls.test_systems['SrcImplicit'] = testsystems.SrcImplicit()

    @classmethod
    def define_regions(cls):
        super(TestAbsoluteAlchemicalFactorySlow, cls).define_regions()
        cls.test_regions['WaterBox'] = AlchemicalRegion(alchemical_atoms=range(3))
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
        alanine_explicit = testsystems.AlanineDipeptideExplicit()
        factory = AbsoluteAlchemicalFactory()
        factory_exact_pme = AbsoluteAlchemicalFactory(alchemical_pme_treatment='exact')

        cls.alanine_alchemical_atoms = list(range(22))
        cls.alanine_test_system = alanine_explicit

        # System with only lambda_sterics and lambda_electrostatics.
        alchemical_region = AlchemicalRegion(alchemical_atoms=cls.alanine_alchemical_atoms)
        alchemical_alanine_system = factory.create_alchemical_system(alanine_vacuum.system, alchemical_region)
        cls.alanine_state = states.ThermodynamicState(alchemical_alanine_system,
                                                      temperature=300*unit.kelvin)

        # System with lambda_sterics and lambda_electrostatics and exact PME treatment.
        alchemical_alanine_system_exact_pme = factory_exact_pme.create_alchemical_system(alanine_explicit.system,
                                                                                         alchemical_region)
        cls.alanine_state_exact_pme = states.ThermodynamicState(alchemical_alanine_system_exact_pme,
                                                                temperature=300*unit.kelvin,
                                                                pressure=1.0*unit.atmosphere)

        # System with all lambdas.
        alchemical_region = AlchemicalRegion(alchemical_atoms=cls.alanine_alchemical_atoms,
                                             alchemical_torsions=True, alchemical_angles=True,
                                             alchemical_bonds=True)
        fully_alchemical_alanine_system = factory.create_alchemical_system(alanine_vacuum.system, alchemical_region)
        cls.full_alanine_state = states.ThermodynamicState(fully_alchemical_alanine_system,
                                                           temperature=300*unit.kelvin)

        # Test case: (ThermodynamicState, defined_lambda_parameters)
        cls.test_cases = [
            (cls.alanine_state, {'lambda_sterics', 'lambda_electrostatics'}),
            (cls.alanine_state_exact_pme, {'lambda_sterics', 'lambda_electrostatics'}),
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
            for parameter in AlchemicalState._get_controlled_parameters():
                if parameter in test_kwargs:
                    assert getattr(alchemical_state, parameter) == test_kwargs[parameter]
                else:
                    assert getattr(alchemical_state, parameter) is None

    def test_from_system_constructor(self):
        """Test AlchemicalState.from_system constructor."""
        # A non-alchemical system raises an error.
        with nose.tools.assert_raises(AlchemicalStateError):
            AlchemicalState.from_system(testsystems.AlanineDipeptideVacuum().system)

        # Valid parameters are 1.0 by default in AbsoluteAlchemicalFactory,
        # and all the others must be None.
        for state, defined_lambdas in self.test_cases:
            alchemical_state = AlchemicalState.from_system(state.system)
            for parameter in AlchemicalState._get_controlled_parameters():
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
            if 'lambda_bonds' in defined_lambdas:
                continue
            defined_lambdas = set(defined_lambdas)  # Copy
            defined_lambdas.add('lambda_bonds')  # Add extra parameter.
            kwargs = dict.fromkeys(defined_lambdas, 1.0)
            alchemical_state = AlchemicalState(**kwargs)
            with nose.tools.assert_raises(AlchemicalStateError):
                alchemical_state.apply_to_system(state.system)

    def test_check_system_consistency(self):
        """Test method AlchemicalState.check_system_consistency()."""
        # A system is consistent with itself.
        alchemical_state = AlchemicalState.from_system(self.alanine_state.system)
        alchemical_state.check_system_consistency(self.alanine_state.system)

        # Raise error if system has MORE lambda parameters.
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
        for state in [self.full_alanine_state, self.alanine_state_exact_pme]:
            alchemical_state = AlchemicalState.from_system(state.system)
            context = state.create_context(copy.deepcopy(integrator))
            alchemical_state.set_alchemical_parameters(0.5)
            alchemical_state.apply_to_context(context)
            for parameter_name, parameter_value in context.getParameters().items():
                if parameter_name in alchemical_state._parameters:
                    assert parameter_value == 0.5
            del context

    def test_standardize_system(self):
        """Test method AlchemicalState.standardize_system."""
        test_cases = [self.full_alanine_state, self.alanine_state_exact_pme]

        for state in test_cases:
            # First create a non-standard system.
            system = copy.deepcopy(state.system)
            alchemical_state = AlchemicalState.from_system(system)
            alchemical_state.set_alchemical_parameters(0.5)
            alchemical_state.apply_to_system(system)

            # Test pre-condition: The state of the System has been changed.
            assert AlchemicalState.from_system(system).lambda_electrostatics == 0.5

            # Check that _standardize_system() sets all parameters back to 1.0.
            alchemical_state._standardize_system(system)
            standard_alchemical_state = AlchemicalState.from_system(system)
            assert alchemical_state != standard_alchemical_state
            for parameter_name, value in alchemical_state._parameters.items():
                standard_value = getattr(standard_alchemical_state, parameter_name)
                assert (value is None and standard_value is None) or (standard_value == 1.0)

    def test_find_force_groups_to_update(self):
        """Test method AlchemicalState._find_force_groups_to_update."""
        test_cases = [self.full_alanine_state, self.alanine_state_exact_pme]

        for thermodynamic_state in test_cases:
            system = copy.deepcopy(thermodynamic_state.system)
            alchemical_state = AlchemicalState.from_system(system)
            alchemical_state2 = copy.deepcopy(alchemical_state)

            # Each lambda should be separated in its own force group.
            expected_force_groups = {}
            for force, lambda_name, _ in AlchemicalState._get_system_controlled_parameters(
                    system, parameters_name_suffix=None):
                expected_force_groups[lambda_name] = force.getForceGroup()

            integrator = openmm.VerletIntegrator(2.0*unit.femtoseconds)
            context = create_context(system, integrator)

            # No force group should be updated if we don't move.
            assert alchemical_state._find_force_groups_to_update(context, alchemical_state2, memo={}) == set()

            # Change the lambdas one by one and check that the method
            # recognize that the force group energy must be updated.
            for lambda_name in AlchemicalState._get_controlled_parameters():
                # Check that the system defines the global variable.
                if getattr(alchemical_state, lambda_name) is None:
                    continue

                # Change the current state.
                setattr(alchemical_state2, lambda_name, 0.0)
                force_group = expected_force_groups[lambda_name]
                assert alchemical_state._find_force_groups_to_update(context, alchemical_state2, memo={}) == {force_group}
                setattr(alchemical_state2, lambda_name, 1.0)  # Reset current state.
            del context

    def test_alchemical_functions(self):
        """Test alchemical variables and functions work correctly."""
        system = copy.deepcopy(self.full_alanine_state.system)
        alchemical_state = AlchemicalState.from_system(system)

        # Add two alchemical variables to the state.
        alchemical_state.set_function_variable('lambda', 1.0)
        alchemical_state.set_function_variable('lambda2', 0.5)
        assert alchemical_state.get_function_variable('lambda') == 1.0
        assert alchemical_state.get_function_variable('lambda2') == 0.5

        # Cannot call an alchemical variable as a supported parameter.
        with nose.tools.assert_raises(AlchemicalStateError):
            alchemical_state.set_function_variable('lambda_sterics', 0.5)

        # Assign string alchemical functions to parameters.
        alchemical_state.lambda_sterics = AlchemicalFunction('lambda')
        alchemical_state.lambda_electrostatics = AlchemicalFunction('(lambda + lambda2) / 2.0')
        assert alchemical_state.lambda_sterics == 1.0
        assert alchemical_state.lambda_electrostatics == 0.75

        # Setting alchemical variables updates alchemical parameter as well.
        alchemical_state.set_function_variable('lambda2', 0)
        assert alchemical_state.lambda_electrostatics == 0.5

    # ---------------------------------------------------
    # Integration tests with CompoundThermodynamicStates
    # ---------------------------------------------------

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
            alchemical_state = AlchemicalState.from_system(state.system)
            compound_state = states.CompoundThermodynamicState(state, [alchemical_state])

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
            compound_state.set_function_variable('lambda', 0.25)
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
        test_cases = [self.alanine_state, self.alanine_state_exact_pme]

        # An incompatible state has a different set of defined lambdas.
        full_alanine_state = copy.deepcopy(self.full_alanine_state)
        alchemical_state_incompatible = AlchemicalState.from_system(full_alanine_state.system)
        compound_state_incompatible = states.CompoundThermodynamicState(full_alanine_state,
                                                                        [alchemical_state_incompatible])

        for state in test_cases:
            state = copy.deepcopy(state)
            alchemical_state = AlchemicalState.from_system(state.system)
            compound_state = states.CompoundThermodynamicState(state, [alchemical_state])

            # A compatible state has the same defined lambda parameters,
            # but their values can be different.
            alchemical_state_compatible = copy.deepcopy(alchemical_state)
            assert alchemical_state.lambda_electrostatics != 0.5  # Test pre-condition.
            alchemical_state_compatible.lambda_electrostatics = 0.5
            compound_state_compatible = states.CompoundThermodynamicState(copy.deepcopy(state),
                                                                          [alchemical_state_compatible])

            # Test states compatibility.
            assert compound_state.is_state_compatible(compound_state_compatible)
            assert not compound_state.is_state_compatible(compound_state_incompatible)

            # Test context compatibility.
            integrator = openmm.VerletIntegrator(1.0*unit.femtosecond)
            context = compound_state_compatible.create_context(copy.deepcopy(integrator))
            assert compound_state.is_context_compatible(context)

            context = compound_state_incompatible.create_context(copy.deepcopy(integrator))
            assert not compound_state.is_context_compatible(context)

    @staticmethod
    def _check_compatibility(state1, state2, context_state1, is_compatible):
        """Check the compatibility of states and contexts between 2 states."""
        # Compatibility should be commutative
        assert state1.is_state_compatible(state2) is is_compatible
        assert state2.is_state_compatible(state1) is is_compatible

        # Test context incompatibility is commutative.
        context_state2 = state2.create_context(openmm.VerletIntegrator(1.0*unit.femtosecond))
        assert state2.is_context_compatible(context_state1) is is_compatible
        assert state1.is_context_compatible(context_state2) is is_compatible
        del context_state2

    def test_method_reduced_potential_compound_state(self):
        """Test CompoundThermodynamicState.reduced_potential_at_states() method.

        Computing the reduced potential singularly and with the class
        method should give the same result.
        """
        # Build a mixed collection of compatible and incompatible thermodynamic states.
        thermodynamic_states = [
            copy.deepcopy(self.alanine_state),
            copy.deepcopy(self.alanine_state_exact_pme)
        ]

        alchemical_states = [
            AlchemicalState(lambda_electrostatics=1.0, lambda_sterics=1.0),
            AlchemicalState(lambda_electrostatics=0.5, lambda_sterics=1.0),
            AlchemicalState(lambda_electrostatics=0.5, lambda_sterics=0.0),
            AlchemicalState(lambda_electrostatics=1.0, lambda_sterics=1.0)
        ]

        compound_states = []
        for thermo_state in thermodynamic_states:
            for alchemical_state in alchemical_states:
                compound_states.append(states.CompoundThermodynamicState(
                    copy.deepcopy(thermo_state), [copy.deepcopy(alchemical_state)]))

        # Group thermodynamic states by compatibility.
        compatible_groups, _ = states.group_by_compatibility(compound_states)
        assert len(compatible_groups) == 2

        # Compute the reduced potentials.
        expected_energies = []
        obtained_energies = []
        for compatible_group in compatible_groups:
            # Create context.
            integrator = openmm.VerletIntegrator(2.0*unit.femtoseconds)
            context = compatible_group[0].create_context(integrator)
            context.setPositions(self.alanine_test_system.positions[:compatible_group[0].n_particles])

            # Compute with single-state method.
            for state in compatible_group:
                state.apply_to_context(context)
                expected_energies.append(state.reduced_potential(context))

            # Compute with multi-state method.
            compatible_energies = states.ThermodynamicState.reduced_potential_at_states(context, compatible_group)

            # The first and the last state must be equal.
            assert np.isclose(compatible_energies[0], compatible_energies[-1])
            obtained_energies.extend(compatible_energies)

        assert np.allclose(np.array(expected_energies), np.array(obtained_energies))

    def test_serialization(self):
        """Test AlchemicalState serialization alone and in a compound state."""
        alchemical_state = AlchemicalState(lambda_electrostatics=0.5, lambda_angles=None)
        alchemical_state.set_function_variable('lambda', 0.0)
        alchemical_state.lambda_sterics = AlchemicalFunction('lambda')

        # Test serialization/deserialization of AlchemicalState.
        serialization = utils.serialize(alchemical_state)
        deserialized_state = utils.deserialize(serialization)
        original_pickle = pickle.dumps(alchemical_state)
        deserialized_pickle = pickle.dumps(deserialized_state)
        assert original_pickle == deserialized_pickle

        # Test serialization/deserialization of AlchemicalState in CompoundState.
        test_cases = [copy.deepcopy(self.alanine_state), copy.deepcopy(self.alanine_state_exact_pme)]
        for thermodynamic_state in test_cases:
            compound_state = states.CompoundThermodynamicState(thermodynamic_state, [alchemical_state])

            # The serialized system is standard.
            serialization = utils.serialize(compound_state)
            serialized_standard_system = serialization['thermodynamic_state']['standard_system']
            # Decompress the serialized_system
            serialized_standard_system = zlib.decompress(serialized_standard_system).decode(
                states.ThermodynamicState._ENCODING)
            assert serialized_standard_system.__hash__() == compound_state._standard_system_hash

            # The object is deserialized correctly.
            deserialized_state = utils.deserialize(serialization)
            assert pickle.dumps(compound_state) == pickle.dumps(deserialized_state)


# =============================================================================
# MAIN FOR MANUAL DEBUGGING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
