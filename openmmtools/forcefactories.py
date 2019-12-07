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


# ==================== CUSTOM LENNARD-JONES POTENTIALS AND SWITCHING FUNCTIONS ==========================

"""
Attributes
----------
LJ_POTENTIAL: str
    OpenMM energy expression for a standard 6-12 LJ potential in terms of A and B:
    "A/r^12 - B/r^6;"
A_B_FROM_EPSILON_SIGMA: str
    OpenMM energy expression for calculating A and B in the LJ potential from sigma and epsilon:
    "A = 4.0*epsilon*sigma^12; B = 4.0*epsilon*sigma^6;"
A_B_FROM_ACOEF_BCOEF: str
    OpenMM energy expression for calculating A and B from the acoef and bcoef fields that
    define NBFIXes in openmm.app.CharmmPsfFile.createSystem:
    "A = acoef(type1, type2)^2; B = bcoef(type1, type2);"
LORENTZ_BERTEHLOT_RULES: str
    OpenMM energy expression for calculating sigma and epsilon by Lorentz-Berthelot mixing rules
    (geometric mean for epsilon, arithmetic mean for sigma):
    "epsilon=sqrt(epsilon1*epsilon2); sigma=0.5*(sigma1+sigma2)"
CHARMM_VSWITCH_ENERGY_EXPRESSION: str
    OpenMM energy expressions for the VSWITCH switching function that is used in CHARMM
    (Steinbach and Brooks, JCC 15 (7), pp. 667-683, 1994):
    "(cutoff^2 - r^2)^2 * (cutoff^2 + 2*r^2 - 3*switch^2) / (cutoff^2 - switch^2)^3"
"""
LJ_POTENTIAL = "A/r^12 - B/r^6;"
A_B_FROM_EPSILON_SIGMA = "A = 4.0*epsilon*sigma^12; B = 4.0*epsilon*sigma^6;"
A_B_FROM_ACOEF_BCOEF = "A = acoef(type1, type2)^2; B = bcoef(type1, type2);"
LORENTZ_BERTHELOT_RULES = "epsilon=sqrt(epsilon1*epsilon2); sigma=0.5*(sigma1+sigma2)"
CHARMM_VSWITCH_ENERGY_EXPRESSION = "(cutoff^2 - r^2)^2 * (cutoff^2 + 2*r^2 - 3*switch^2) / (cutoff^2 - switch^2)^3"


def is_lj_active_in_nonbonded_force(nonbonded_force):
    """Check if the LJ potential is defined in the openmm.NonbondedForce or if all terms are zero.

    Parameters
    ----------
    nonbonded_force : openmm.NonbondedForce
        The nonbonded force of a openmm.System.

    Returns
    -------
    is_lj_active : bool
    """
    for i in range(nonbonded_force.getNumParticles()):
        q, sig, eps = nonbonded_force.getParticleParameters(i)
        vdw = (eps.value_in_unit(unit.kilojoule_per_mole) * sig.value_in_unit(unit.nanometer))
        if vdw > 1e-5:
            return True
    for i in range(nonbonded_force.getNumExceptions()):
        particle1, particle2, qq, sig, eps = nonbonded_force.getExceptionParameters(i)
        vdw = (eps.value_in_unit(unit.kilojoule_per_mole) * sig.value_in_unit(unit.nanometer))
        if vdw > 1e-5:
            return True
    return False


def get_forces_that_define_vdw(system):
    """Get all forces that define van der Waals interactions.

    Identifies all forces in the system that define van der Waals interactions. These include the LJ part
    in the openmm.NonbondedForce; all openmm.CustomNonbondedForce and openmm.CustomBondForce instances that
    have per-particle-parameters 'sigma' and 'epsilon'
    (the bond forces are often used to define custom 1-4 interactions);
    all openmm.CustomNonbondedForce instances that have a per-particle-parameter 'type'
    and tabulated functions 'acoef' and 'bcoef' (these define NBFIXes in openmm.app.CharmmPsfFile.createSystem).

    Parameters
    ----------
    system : openmm.System
        An openmm.System.

    Returns
    -------
    nonbonded_forces : list of openmm.NonbondedForce
        This list is empty, if all Lennard-Jones interactions in the NonbondedForce are zero.
        Otherwise it contains one element.
    custom_nonbonded_forces : list of openmm.CustomNonbondedForce
        A list of all custom nonbonded forces that define Lennard-Jones interactions.
    nbfix_forces : list of openmm.CustomNonbondedForce
        A list of all custom nonbonded forces that define NBFIXes (i.e. pair-specific Lennard-Jones interactions).
    custom_bond_forces : list of openmm.CustomBondForce
        A list of all custom bonded forces that define Lennard-Jones interactions (typically 1-4 interactions).
    """
    nonbonded_forces = []
    custom_nonbonded_forces = []
    nbfix_forces = []
    custom_bond_forces = []
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
                custom_bond_forces.append(force)
                continue
            if (("acoef" in tabulated_functions) and ("bcoef" in tabulated_functions)
                    and ("type" in particle_parameters)):
                nbfix_forces.append(force)
                continue
        # CUSTOM BOND FORCES (TYPICALLY FOR 1-4 INTERACTIONS)
        if isinstance(force, openmm.CustomBondForce):
            bond_parameters = [force.getBondParameters()
                               for i in range(force.getNumPerBondParameters())]
            if ("epsilon" in bond_parameters) and ("sigma" in bond_parameters):
                custom_bond_forces.append(force)
                continue

    return nonbonded_forces, custom_nonbonded_forces, nbfix_forces, custom_bond_forces


def vdw_as_custom_forces(system, lj_potential=LJ_POTENTIAL+A_B_FROM_EPSILON_SIGMA,
                         lj_nbfix_potential=LJ_POTENTIAL+A_B_FROM_ACOEF_BCOEF,
                         mixing_rules=LORENTZ_BERTHELOT_RULES):
    """Modify the system so that all Lennard-Jones interactions are defined in custom forces.

    Sets all LJ parameters in the openmm.NonbondedForce to zero and adds them as custom forces to the system instead.
    This allows modifying the functional form of the LJ potential, as well as incorporation of custom mixing rules and
    switching functions.

    Parameters
    ----------
    system : openmm.System
        The System instance to be modified.
    lj_potential : str, optional, default=LJ_POTENTIAL+A_B_FROM_EPSILON_SIGMA
        An energy expression that defines the Lennard-Jones potential in terms of parameters 'epsilon' and 'sigma'.
        The interactions defined in instances of openmm.CustomNonbondedForce use the energy expression + mixing rules.
        The interactions defined in instances of openmm.CustomBondForce use the energy expression without mixing rules.
    lj_nbfix_potential : str, optional, default=LJ_POTENTIAL+A_B_FROM_ACOEF_BCOEF
        An energy expression that defines the pair-specific Lennard-Jones potential in terms of the two atom types
        'type1' and 'type2' and the tabulated functions 'acoef' and 'bcoef', where acoef=sqrt(A) and bcoef=B in
        the Lennard-Jones potential U = A/r^12 - B/r^6. These expressions are used in the NBFIXes in
        openmm.app.CharmmPsfFile.createSystem.
    mixing_rules : str, optional, default=LORENTZ_BERTHELOT_RULES
        An energy expression that defines the mixing rules used to obtain 'epsilon' and 'sigma' from per-particle
        parameters 'epsilon1', 'sigma1', 'epsilon2', and 'sigma2'. Note that the mixing rules are not applied to the
        the nbfix_forces and custom_bond_forces.

    Returns
    -------
    custom_nonbonded_forces: list of openmm.CustomNonbondedForce
        A list of all custom nonbonded forces that define Lennard-Jones interactions.
    nbfix_forces: list of openmm.CustomNonbondedForce
        A list of all custom nonbonded forces that define NBFIXes (i.e. pair-specific Lennard-Jones interactions).
    custom_bond_forces: list of openmm.CustomBondForce
        A list of all custom bonded forces that define Lennard-Jones interactions (typically 1-4 interactions).

    Examples
    --------

    Replace the customary 6-12 potential by a 6-14 potential with multiplicative mixing rules.

    >>> from openmmtools import testsystems
    >>> lj_fluid = testsystems.LennardJonesFluid(nparticles=100)
    >>> pot = "A/r^14 - B/r^6;" + A_B_FROM_EPSILON_SIGMA
    >>> pot_nb = "A/r^14 - B/r^6;" + A_B_FROM_ACOEF_BCOEF
    >>> mix = "sigma=sqrt(sigma1*sigma2); epsilon=sqrt(epsilon1*epsilon2)"
    >>> _ = vdw_as_custom_forces(lj_fluid.system, lj_potential=pot, lj_nbfix_potential=pot_nb, mixing_rules=mix)
    >>> # proceed with the modified system, e.g.:
    >>> integrator = openmm.LangevinIntegrator(300*unit.kelvin, 5.0/unit.picosecond, 1.0*unit.femtosecond)
    >>> context = openmm.Context(lj_fluid.system, integrator)
    """

    # get NonbondedForce
    nonbonded_forces, custom_nonbonded_forces, nbfix_forces, custom_bond_forces = get_forces_that_define_vdw(system)

    for nb_force in nonbonded_forces:

        vdw = openmm.CustomNonbondedForce("")  # Energy expressions are defined later
        vdw.addPerParticleParameter("sigma")
        vdw.addPerParticleParameter("epsilon")

        # 1-4 interactions
        vdw14 = openmm.CustomBondForce("")  # Energy expressions are defined later
        vdw14.addPerBondParameter("sigma")
        vdw14.addPerBondParameter("epsilon")

        # add particles to custom forces
        for i in range(nb_force.getNumParticles()):
            q, sig, eps = nb_force.getParticleParameters(i)
            particle_id = vdw.addParticle([sig, eps])
            assert particle_id == i
            nb_force.setParticleParameters(i, q, 1.0, 0.0)

        for i in range(nb_force.getNumExceptions()):
            atom1, atom2, q, sig, eps = nb_force.getExceptionParameters(i)
            nb_force.setExceptionParameters(i, atom1, atom2, q, 1.0, 0.0)
            vdw.addExclusion(atom1, atom2)
            vdw14.addBond(atom1, atom2, [sig, eps])

        vdw.setCutoffDistance(nb_force.getCutoffDistance())
        vdw.setNonbondedMethod(vdw.CutoffPeriodic)
        vdw.setSwitchingDistance(nb_force.getSwitchingDistance())
        vdw.setUseLongRangeCorrection(nb_force.getUseDispersionCorrection())

        system.addForce(vdw)
        system.addForce(vdw14)

        custom_nonbonded_forces.append(vdw)
        custom_bond_forces.append(vdw14)

    # If vdW forces are defined in a CustomNonbondedForce, assert that they
    # electrostatics are not defined in the same CustomNonbondedForce
    for custom_force in custom_nonbonded_forces + nbfix_forces + custom_bond_forces:
        global_parameters = [custom_force.getGlobalParameterName(i)
                             for i in range(custom_force.getNumGlobalParameters())]
        assert "q" not in global_parameters

    # Define Energy Expressions
    for custom_nb_force in custom_nonbonded_forces:
        custom_nb_force.setEnergyFunction(lj_potential + mixing_rules)
    for custom_b_force in custom_bond_forces:
        custom_b_force.setEnergyFunction(lj_potential)
    for nbfix_force in nbfix_forces:
        nbfix_force.setEnergyFunction(lj_nbfix_potential)

    return custom_nonbonded_forces, nbfix_forces, custom_bond_forces


def apply_custom_switching_function(custom_force, switch_distance, cutoff_distance,
                                    switching_function=CHARMM_VSWITCH_ENERGY_EXPRESSION):
    """Apply a custom switching function to a force.

    Switching functions bring the potential energy to zero between the switch distance and the cutoff distance.
    The energy expression of the custom force, U(r), is changed to:
    U_switched(r) = { U(r),         if r < switch ,
                    { U(r) * S(r),  if switch <= r <= cutoff,
                    { 0,            if r > cutoff,
    where S(r) is the switching function.

    Parameters
    ----------
    custom_force : openmm.Force
        A custom force, whose energy expression is defined in terms of the radius 'r'.
        Could be an instance of openmm.CustomNonbondedForce or openmm.CustomBondForce.
    switch_distance: unit.Quantity
        The distance at which the switching begins.
    cutoff_distance: unit.Quantity
        The distance from where on the potential is zero.
    switching_function: str, optional, default=CHARMM_VSWITCH_ENERGY_EXPRESSION
        An OpenMM energy expression that defines the switching function. The switching function, S(r), is defined in
        terms of three variables ('r', 'cutoff', 'switch'). It has to satisfy two conditions:
        S(cutoff)=0 and S(switch)=1.
    """
    energy_string = custom_force.getEnergyFunction()
    energy_expressions = energy_string.split(';')
    energy_expressions = [
        "step(cutoff-r) * ( 1 + step(r-switch)*(switch_function-1) ) * energy_function",
        "switch_function=" + switching_function,
        "energy_function=" + energy_expressions[0]
    ] + energy_expressions[1:]

    custom_force.setEnergyFunction("; ".join(energy_expressions))
    if isinstance(custom_force, openmm.CustomNonbondedForce):
        custom_force.setUseSwitchingFunction(False)  # switch off OpenMMs customary switching function
    custom_force.addGlobalParameter("cutoff", cutoff_distance)
    custom_force.addGlobalParameter("switch", switch_distance)


def use_custom_vdw_switching_function(system, switch_distance, cutoff_distance,
                                      switching_function=CHARMM_VSWITCH_ENERGY_EXPRESSION):
    """Use a custom switching function for all Lennard-Jones interactions in the system.

    U_switched(r) = { U(r),         if r < switch ,
                    { U(r) * S(r),  if switch <= r <= cutoff,
                    { 0,            if r > cutoff,
    For a definition of what is considered a Lennard-Jones interactions, see the documentation of
    get_forces_that_define_vdw.

    Parameters
    ----------
    system: openmm.System
        The System instance to be modified.
    switch_distance: unit.Quantity
        The distance at which the switching begins.
    cutoff_distance: unit.Quantity
        The distance from where on the potential is zero.
    switching_function: str, optional, default=CHARMM_VSWITCH_ENERGY_EXPRESSION
        An OpenMM energy expression that defines the switching function. The switching function, S(r), is defined in
        terms of three variables ('r', 'cutoff', 'switch'). It has to satisfy two conditions:
        S(cutoff)=0 and S(switch)=1.

    Examples
    --------

    Use CHARMM's VSWITCH function, as defined in Steinbach and Brooks, JCC 15 (7), pp. 667-683, 1994.
    (This is the default if the switching_function argument is not specified).

    >>> from openmmtools import testsystems
    >>> lj_fluid = testsystems.LennardJonesFluid(nparticles=100, cutoff=1.2*unit.nanometer)
    >>> use_custom_vdw_switching_function(lj_fluid.system, switch_distance=1.0*unit.nanometer, cutoff_distance=1.2*unit.nanometer)
    >>> # proceed with the modified system, e.g.:
    >>> integrator = openmm.LangevinIntegrator(300*unit.kelvin, 5.0/unit.picosecond, 1.0*unit.femtosecond)
    >>> context = openmm.Context(lj_fluid.system, integrator)
    """
    custom_nb_forces, nbfix_forces, custom_b_forces = vdw_as_custom_forces(
        system, lj_potential=LJ_POTENTIAL+A_B_FROM_EPSILON_SIGMA,
        lj_nbfix_potential=LJ_POTENTIAL+A_B_FROM_ACOEF_BCOEF, mixing_rules=LORENTZ_BERTHELOT_RULES)
    for custom_nb_force in custom_nb_forces:
        apply_custom_switching_function(custom_nb_force, switch_distance, cutoff_distance, switching_function)
    for nbfix_force in nbfix_forces:
        apply_custom_switching_function(nbfix_force, switch_distance, cutoff_distance, switching_function)
    for custom_b_force in custom_b_forces:
        apply_custom_switching_function(custom_b_force, switch_distance, cutoff_distance, switching_function)


def use_vdw_with_charmm_force_switch(system, switch_distance, cutoff_distance):
    """Use the CHARMM force switching function for Lennard-Jones cutoffs.

    Modifies the system's Lennard-Jones interactions to use the force switching scheme by
    Steinbach and Brooks, JCC 15 (7), pp. 667-683, 1994.
    The force switch modifies the potential form of the LJ potential altogether and is therefore
    a special case that is not easy to implement via the more general function use_custom_vdw_switching_function.

    Parameters
    ----------
    system: openmm.System
        The System instance to be modified.
    switch_distance: unit.Quantity
        The distance at which the switching begins.
    cutoff_distance: unit.Quantity
        The distance from where on the potential is zero.

    Examples
    --------

    >>> from openmmtools import testsystems
    >>> lj_fluid = testsystems.LennardJonesFluid(nparticles=100, cutoff=1.2*unit.nanometer)
    >>> use_vdw_with_charmm_force_switch(lj_fluid.system, switch_distance=1.0*unit.nanometer, cutoff_distance=1.2*unit.nanometer)
    >>> # proceed with the modified system, e.g.:
    >>> integrator = openmm.LangevinIntegrator(300*unit.kelvin, 5.0/unit.picosecond, 1.0*unit.femtosecond)
    >>> context = openmm.Context(lj_fluid.system, integrator)
    """
    energy_string = (
        "r_leq_cutoff * (r_leq_switch * short_range + (1.0-r_leq_switch) * long_range);"
        "long_range   = k12*(r^(-6) - cutoff^(-6))^2 - k6*(r^(-3) - cutoff^(-3))^2;"
        "short_range  = A*(r^(-12) + dv12) - B*(r^(-6) + dv6);"
        "r_leq_switch = step(switch-r);"
        "r_leq_cutoff = step(cutoff-r);"
        "k6           = B * cutoff^3/(cutoff^3 - switch^3);"
        "k12          = A * cutoff^6/(cutoff^6 - switch^6);"
        "dv6          = -1.0/(cutoff*switch)^3;"
        "dv12         = -1.0/(cutoff*switch)^6;"
    )
    custom_nb_forces, nbfix_forces, custom_b_forces = vdw_as_custom_forces(
        system, lj_potential=energy_string+A_B_FROM_EPSILON_SIGMA,
        lj_nbfix_potential=energy_string+A_B_FROM_ACOEF_BCOEF, mixing_rules=LORENTZ_BERTHELOT_RULES)
    for custom_nb_force in custom_nb_forces:
        custom_nb_force.setUseSwitchingFunction(False)  # switch off OpenMMs customary switching function
    for custom_force in custom_nb_forces + nbfix_forces + custom_b_forces:
        custom_force.addGlobalParameter("cutoff", cutoff_distance)
        custom_force.addGlobalParameter("switch", switch_distance)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
