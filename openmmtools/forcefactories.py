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


from openmmtools.constants import ONE_4PI_EPS0
default_energy_expression = "(4*epsilon*((sigma/r)^12-(sigma/r)^6) + k * chargeprod/r); sigma=0.5*(sigma1+sigma2); epsilon=sqrt(epsilon1*epsilon2); chargeprod=charge1*charge2; k={};".format(ONE_4PI_EPS0)


def clone_nonbonded_parameters(nonbonded_force, energy_expression=default_energy_expression):
    """Creates a new CustomNonbonded force with the same global parameters,
    per-particle parameters, and exception parameters as """

    #TODO: assert that nonbonded_force.getNonbondedMethod() indicates reaction field

    # call constructor
    new_force = openmm.CustomNonbondedForce(energy_expression)
    new_force.addPerParticleParameter('charge')
    new_force.addPerParticleParameter('sigma')
    new_force.addPerParticleParameter('epsilon')

    # go through all of the setter and getter methods
    new_force.setCutoffDistance(nonbonded_force.getCutoffDistance())
    #new_force.setEwaldErrorTolerance(nonbonded_force.getEwaldErrorTolerance())
    #new_force.setForceGroup(nonbonded_force.getForceGroup())
    #new_force.setNonbondedMethod(nonbonded_force.getNonbondedMethod()) # If PME, will be an Illegal value for CustomNonbonded
    #new_force.setPMEParameters(*nonbonded_force.getPMEParameters())
    #new_force.setReactionFieldDielectric(nonbonded_force.getReactionFieldDielectric())
    #new_force.setReciprocalSpaceForceGroup(nonbonded_force.getReciprocalSpaceForceGroup())
    new_force.setSwitchingDistance(nonbonded_force.getSwitchingDistance())
    #new_force.setUseDispersionCorrection(nonbonded_force.getUseDispersionCorrection())
    new_force.setUseSwitchingFunction(nonbonded_force.getUseSwitchingFunction())

    # now add all the particle parameters
    num_particles = nonbonded_force.getNumParticles()
    for i in range(num_particles):
        new_force.addParticle(nonbonded_force.getParticleParameters(i))

    # now add all the exceptions
    #    for exception_index in range(nonbonded_force.getNumExceptions()):
    #        new_force.addException(nonbonded_force.getExceptionParameters(exception_index))

    # TODO: There's probably a cleaner, more Pythonic way to do this
    # (this was the most obvious way, but may not work in the future if the OpenMM API changes)

    return new_force

from copy import deepcopy
from openmmtools.forces import find_forces

def split_using_exceptions(system, md_topology):
    """Construct a new system where force group 0 contains slow, expensive solvent-solvent
    interactions, and force group 1 contains fast, cheaper solute-solute and solute-solvent
    interactions.

    TODO: correct this:
    Force group 0: default nonbonded force with exceptions for solute-solvent pairs
    Force group 1:
    * CustomNonbonded force with interaction group for solute-solvent pairs
    * non-Nonbonded forces

    TODO: more informative docstring
    """

    # create a copy of the original system: only touch this
    new_system = deepcopy(system)

    # find the default nonbonded force
    force_index, nb_force = find_forces(new_system, openmm.NonbondedForce, only_one=True)
    # assert('Cutoff' in nb_force.getNonbondedMethod()) # TODO: this, less jankily

    # create copies containing
    nb_except_solute_solvent = nb_force
    nb_only_solute_solvent = clone_nonbonded_parameters(nb_force)

    nb_except_solute_solvent.setForceGroup(0)

    # identify solvent-solute pairs
    # find the pairs (i,j) where i in solute, j in solvent

    # NOTE: these need to be python ints -- not np.int64s -- when passing to addInteractionGroup later!
    solvent_indices = list(map(int, md_topology.select('water')))
    solute_indices = list(map(int, md_topology.select('not water')))

    # for each (i,j) set chargeprod, sigma, epsilon to 0, causing these pairs to be omitted completely from force calculation
    for i in solvent_indices:
        for j in solute_indices:
            nb_except_solute_solvent.addException(i, j, 0, 0, 0)

    nb_only_solute_solvent.addInteractionGroup(set1=solute_indices, set2=solvent_indices)

    nb_except_solute_solvent.setForceGroup(0)
    nb_only_solute_solvent.setForceGroup(1)

    # handle non-NonbondedForce's
    for force in new_system.getForces():
        if 'Nonbonded' not in force.__class__.__name__:
            force.setForceGroup(1)

    return new_system


if __name__ == '__main__':
    import doctest
    doctest.testmod()
