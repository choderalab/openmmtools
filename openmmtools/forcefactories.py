#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Factories to manipulate OpenMM System forces.

"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import copy

import numpy as np
import mdtraj
from simtk import openmm, unit

from openmmtools import forces


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def replace_reaction_field(reference_system, switch_width=1.0*unit.angstrom,
                           return_copy=True, shifted=False):
    """Return a system converted to use a switched reaction-field electrostatics using :class:`openmmtools.forces.UnshiftedReactionField`.

    This will add an `UnshiftedReactionFieldForce` or `SwitchedReactionFieldForce`
    for each `NonbondedForce` that utilizes `CutoffPeriodic`.

    Note that `AbsoluteAlchemicalFactory.create_alchemical_system()` can NOT
    handle the resulting `System` object yet since the `CustomNonbondedForce`
    are not recognized and re-coded.

    Parameters
    ----------
    reference_system : simtk.openmm.System
        The system to use as a reference. This will be modified only if
        `return_copy` is `False`.
    switch_width : simtk.unit.Quantity, default=1.0*angstrom
        Switch width for electrostatics (units of distance).
    return_copy : bool, optional, default=True
        If `True`, `reference_system` is not modified, and a copy is returned.
        Setting it to `False` speeds up the function execution but modifies
        the `reference_system` object.
    shifted : bool, optional, default=False
        If `True`, a shifted reaction-field will be used.

    Returns
    -------
    system : simtk.openmm.System
        System with reaction-field converted to c_rf = 0

    """
    if return_copy:
        system = copy.deepcopy(reference_system)
    else:
        system = reference_system

    if shifted:
        force_constructor = getattr(forces, 'SwitchedReactionFieldForce')
    else:
        force_constructor = getattr(forces, 'UnshiftedReactionFieldForce')

    # Add an reaction field for each CutoffPeriodic NonbondedForce.
    for reference_force in forces.find_forces(system, openmm.NonbondedForce).values():
        if reference_force.getNonbondedMethod() == openmm.NonbondedForce.CutoffPeriodic:
            reaction_field_force = force_constructor.from_nonbonded_force(reference_force, switch_width=switch_width)
            system.addForce(reaction_field_force)

            # Remove particle electrostatics from reference force, but leave exceptions.
            for particle_index in range(reference_force.getNumParticles()):
                charge, sigma, epsilon = reference_force.getParticleParameters(particle_index)
                reference_force.setParticleParameters(particle_index, abs(0.0*charge), sigma, epsilon)

    return system


# =============================================================================
# RESTRAIN ATOMS
# =============================================================================

def restrain_atoms_by_dsl(thermodynamic_state, sampler_state, topology, atoms_dsl, **kwargs):
    # Make sure the topology is an MDTraj topology.
    if isinstance(topology, mdtraj.Topology):
        mdtraj_topology = topology
    else:
        mdtraj_topology = mdtraj.Topology.from_openmm(topology)

    # Determine indices of the atoms to restrain.
    restrained_atoms = mdtraj_topology.select(atoms_dsl).tolist()
    restrain_atoms(thermodynamic_state, sampler_state, restrained_atoms, **kwargs)


def restrain_atoms(thermodynamic_state, sampler_state, restrained_atoms, sigma=3.0*unit.angstroms):
    """Apply a soft harmonic restraint to the given atoms.

    This modifies the ``ThermodynamicState`` object.

    Parameters
    ----------
    thermodynamic_state : openmmtools.states.ThermodynamicState
        The thermodynamic state with the system. This will be modified.
    sampler_state : openmmtools.states.SamplerState
        The sampler state with the positions.
    topology : mdtraj.Topology or simtk.openmm.Topology
        The topology of the system.
    atoms_dsl : str
        The MDTraj DSL string for selecting the atoms to restrain.
    sigma : simtk.unit.Quantity, optional
        Controls the strength of the restrain. The smaller, the tighter
        (units of distance, default is 3.0*angstrom).

    """
    K = thermodynamic_state.kT / sigma**2  # Spring constant.
    system = thermodynamic_state.system  # This is a copy.

    # Check that there are atoms to restrain.
    if len(restrained_atoms) == 0:
        raise ValueError('No atoms to restrain.')

    # We need to translate the restrained molecule to the origin
    # to avoid MonteCarloBarostat rejections (see openmm#1854).
    if thermodynamic_state.pressure is not None:
        # First, determine all the molecule atoms. Reference platform is the cheapest to allocate?
        reference_platform = openmm.Platform.getPlatformByName('Reference')
        integrator = openmm.VerletIntegrator(1.0*unit.femtosecond)
        context = openmm.Context(system, integrator, reference_platform)
        molecules_atoms = context.getMolecules()
        del context, integrator

        # Make sure the atoms to restrain belong only to a single molecule.
        molecules_atoms = [set(molecule_atoms) for molecule_atoms in molecules_atoms]
        restrained_atoms_set = set(restrained_atoms)
        restrained_molecule_atoms = None
        for molecule_atoms in molecules_atoms:
            if restrained_atoms_set.issubset(molecule_atoms):
                # Convert set to list to use it as numpy array indices.
                restrained_molecule_atoms = list(molecule_atoms)
                break
        if restrained_molecule_atoms is None:
            raise ValueError('Cannot match the restrained atoms to any molecule. Restraining '
                             'two molecules is not supported when using a MonteCarloBarostat.')

        # Translate system so that the center of geometry is in
        # the origin to reduce the barostat rejections.
        distance_unit = sampler_state.positions.unit
        centroid = np.mean(sampler_state.positions[restrained_molecule_atoms,:] / distance_unit, axis=0)
        sampler_state.positions -= centroid * distance_unit

    # Create a CustomExternalForce to restrain all atoms.
    if thermodynamic_state.is_periodic:
        energy_expression = '(K/2)*periodicdistance(x, y, z, x0, y0, z0)^2' # periodic distance
    else:
        energy_expression = '(K/2)*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)' # non-periodic distance
    restraint_force = openmm.CustomExternalForce(energy_expression)
    # Adding the spring constant as a global parameter allows us to turn it off if desired
    restraint_force.addGlobalParameter('K', K)
    restraint_force.addPerParticleParameter('x0')
    restraint_force.addPerParticleParameter('y0')
    restraint_force.addPerParticleParameter('z0')
    for index in restrained_atoms:
        parameters = sampler_state.positions[index,:].value_in_unit_system(unit.md_unit_system)
        restraint_force.addParticle(index, parameters)

    # Update thermodynamic state.
    system.addForce(restraint_force)
    thermodynamic_state.system = system


def is_lj_active_in_nonbonded_force(nonbonded_force):
    for i in range(nonbonded_force.getNumParticles()):
        q, sig, eps = nonbonded_force.getParticleParameters(i)
        vdw = (eps.value_in_unit(unit.kilojoule_per_mole) * sig.value_in_unit(unit.nanometer))
        if vdw > 1e-5:
            return True
    return False


def get_forces_that_define_vdw(system):
    """Get all forces that define van der Waals interactions"""
    nonbonded_forces = []
    custom_nonbonded_forces = []
    nbfix_forces = []
    custom_bonded_forces = []
    for force in system.getForces():
        # NONBONDED FORCES
        if isinstance(force, openmm.NonbondedForce):
            # check if all vdW potentials in the nonbonded force are zero
            if is_lj_active_in_nonbonded_force(force):
                nonbonded_forces.append(force)
        # CUSTOM NONBONDED FORCES AND NBFIXES
        if isinstance(force, openmm.CustomNonbondedForce):
            particle_parameters = [force.getPerParticleParameterName(i)
                                   for i in range(force.getNumPerParticleParameters())]
            tabulated_functions = [force.getTabulatedFunctionName(i)
                                   for i in range(force.getNumTabulatedFunctions())]
            if ("epsilon" in particle_parameters) and ("sigma" in particle_parameters):
                custom_bonded_forces.append(force)
                continue
            if ("acoef" in tabulated_functions) and ("bcoef" in tabulated_functions)\
                    and ("type" in particle_parameters):
                nbfix_forces.append(force)
                continue
        # CUSTOM BOND FORCES (TYPICALLY FOR 1-4 INTERACTIONS)
        if isinstance(force, openmm.CustomBondForce):
            bond_parameters = [force.getBondParameters()
                               for i in range(force.getNumPerBondParameters())]
            if ("epsilon" in bond_parameters) and ("sigma" in bond_parameters):
                custom_bonded_forces.append(force)
                continue

    return nonbonded_forces, custom_nonbonded_forces, nbfix_forces, custom_bonded_forces


LJ_POTENTIAL = "4*epsilon*((sigma/r)^12 - (sigma/r)^6);"
LJ_POTENTIAL2 = "(a/r6)^2-b/r6; r6=r^6;a=acoef(type1, type2);b=bcoef(type1, type2)"
LORENTZ_BERTHELOT_RULES = "epsilon=sqrt(epsilon1*epsilon2); sigma=0.5*(sigma1+sigma2)"
CHARMM_VSWITCH_ENERGY_EXPRESSION = "(cutoff^2 - r^2)^2 * (cutoff^2 + 2*r^2 - 3*switch^2) / (cutoff^2 - switch^2)^3"
CHARMM_VFSWITCH_ENERGY_EXPRESSION = ""


def separate_vdw_from_elec(system, lj_potential=LJ_POTENTIAL, mixing_rules=LORENTZ_BERTHELOT_RULES):
    """
    """

    # get NonbondedForce
    nb_forces, custom_nb_forces, nbfix_forces, custom_b_forces = get_forces_that_define_vdw(system)

    for nb_force in nb_forces:

        vdw = openmm.CustomNonbondedForce(lj_potential + mixing_rules)
        vdw.addPerParticleParameter("sigma")
        vdw.addPerParticleParameter("epsilon")

        # 1-4 interactions
        vdw14 = openmm.CustomBondForce(lj_potential)
        vdw14.addPerBondParameter("sigma")
        vdw14.addPerBondParameter("epsilon")

        # add particles to custom forces
        for i in range(nb_force.getNumParticles()):
            q, sig, eps = nb_force.getParticleParameters(i)
            particle_id = vdw.addParticle([sig, eps])
            assert particle_id == i
            nb_force.setParticleParameters(i, q, 0.0, 0.0)

        for i in range(nb_force.getNumExceptions()):
            atom1, atom2, q, sig, eps = nb_force.getExceptionParameters(i)
            nb_force.setExceptionParameters(i, atom1, atom2, q, 0.0, 0.0)
            vdw.addExclusion(atom1, atom2)
            vdw14.addBond(atom1, atom2, [sig, eps])

        vdw.setCutoffDistance(nb_force.getCutoffDistance())
        vdw.setNonbondedMethod(vdw.CutoffPeriodic)
        vdw.setSwitchingDistance(nb_force.getSwitchingDistance())
        vdw.setUseLongRangeCorrection(nb_force.getUseDispersionCorrection())

        system.addForce(vdw)
        system.addForce(vdw14)

        custom_nb_forces.append(vdw)
        custom_b_forces.append(vdw14)

    # If vdW forces are defined in a CustomNonbondedForce, assert that they
    # electrostatics are not defined in the same CustomNonbondedForce
    for custom_force in custom_nb_forces + nbfix_forces + custom_b_forces:
        global_parameters = [custom_force.getGlobalParameterName(i)
                             for i in range(custom_force.getNumGlobalParameters())]
        assert "q" not in global_parameters

    return custom_nb_forces, nbfix_forces, custom_b_forces


def apply_custom_switching_function(custom_nb_force, switch_distance, cutoff_distance,
                                    switching_function=CHARMM_VSWITCH_ENERGY_EXPRESSION):
    """Apply a custom switching function to a CustomNonbondedForce.

       Parameters
    ----------
    thermodynamic_state : openmmtools.states.ThermodynamicState
        The thermodynamic state with the system. This will be modified.
    sampler_state : openmmtools.states.SamplerState
        The sampler state with the positions.

    """
    energy_string = custom_nb_force.getEnergyFunction()
    energy_expressions = energy_string.split(';')
    energy_expressions = [
        "step(cutoff-r) * ( 1 + step(r-switch)*(switch_function-1) ) * energy_function",
        "switch_function=" + switching_function,
        "energy_function=" + energy_expressions[0]
    ] + energy_expressions[1:]

    custom_nb_force.setEnergyFunction("; ".join(energy_expressions))
    if isinstance(custom_nb_force, openmm.CustomNonbondedForce):
        custom_nb_force.setUseSwitchingFunction(False)  # switch off OpenMMs customary switching function
    custom_nb_force.addGlobalParameter("cutoff", cutoff_distance)
    custom_nb_force.addGlobalParameter("switch", switch_distance)

    return custom_nb_force


def use_custom_vdw_switching_function(system, switch_distance, cutoff_distance,
                                      switching_function=CHARMM_VSWITCH_ENERGY_EXPRESSION):
    custom_nb_forces, nbfix_forces, custom_b_forces = separate_vdw_from_elec(
        system, LJ_POTENTIAL, LORENTZ_BERTHELOT_RULES)
    for custom_nb_force in custom_nb_forces:
        apply_custom_switching_function(custom_nb_force, switch_distance, cutoff_distance, switching_function)
    for nbfix_force in nbfix_forces:
        apply_custom_switching_function(nbfix_force, switch_distance, cutoff_distance, switching_function)
    for custom_b_force in custom_b_forces:
        apply_custom_switching_function(custom_b_force, switch_distance, cutoff_distance, switching_function)


def use_vdw_with_charmm_force_switch(system, cutoff_distance, switch_distance):
    """Use the CHARMM force switching function for Lennard-Jones cutoffs.

    Modifies the system's Lennard-Jones interactions. The LJ part of the NonbondedForce
    object is replaced by a CustomNonbondedForce. Any other CustomNonbondedForce in the
    system is

    Parameters
    ----------
    thermodynamic_state : openmmtools.states.ThermodynamicState
        The thermodynamic state with the system. This will be modified.
    sampler_state : openmmtools.states.SamplerState
        The sampler state with the positions.
    topology : mdtraj.Topology or simtk.openmm.Topology
        The topology of the system.
    atoms_dsl : str
        The MDTraj DSL string for selecting the atoms to restrain.
    sigma : simtk.unit.Quantity, optional
        Controls the strength of the restrain. The smaller, the tighter
        (units of distance, default is 3.0*angstrom).

    """
    energy_string = (
        "step(switch-r)         * (A*(r^(-12) + dv12) - B*(r^(-6) + dv6))"                           # if r <= switch: ...
        "+ (1.0-step(switch-r)) * (k12*(r^(-6) - cutoff^(-6))^2 - k6*(r^(-3) - cutoff^(-3))^2);"     # else: ...
        "k6      = B * cutoff^3/(cutoff^3 - switch^3);"
        "k12     = A * cutoff^6/(cutoff^6 - switch^6);"
        "dv6     = -1.0/(cutoff*switch)^3;"
        "dv12    = -1.0/(cutoff*switch)^6;"
        "A       = 4.0*epsilon*sigma^12;"
        "B       = 4.0*epsilon*sigma^6;"
    )
    custom_nb_forces, nbfix_forces, custom_b_forces = separate_vdw_from_elec(
        system, lj_potential=energy_string, mixing_rules=LORENTZ_BERTHELOT_RULES)
    for custom_nb_force in custom_nb_forces:
        custom_nb_force.setUseSwitchingFunction(False)  # switch off OpenMMs customary switching function
    for nbfix_force in nbfix_forces:
        nbfix_force.setEnergyFunction("")
    for custom_force in custom_nb_forces + nbfix_forces + custom_b_forces:
        custom_force.addGlobalParameter("cutoff", cutoff_distance)
        custom_force.addGlobalParameter("switch", switch_distance)



if __name__ == '__main__':
    import doctest
    doctest.testmod()
