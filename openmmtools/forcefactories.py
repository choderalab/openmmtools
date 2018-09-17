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


if __name__ == '__main__':
    import doctest
    doctest.testmod()
