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
from simtk import openmm, unit

from openmmtools import forces


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def replace_reaction_field(reference_system, switch_width=1.0*unit.angstrom,
                           return_copy=True):
    """Return a system converted to use a switched reaction-field electrostatics.

    This will add an `UnshiftedReactionFieldForce` for each `NonbondedForce`
    that utilizes `CutoffPeriodic`.

    Note that `AbsoluteAlchemicalFactory.create_alchemical_system()` can NOT
    handle the resulting `System` object yet since the `CustomNonbondedForce`
    are not recognized and re-coded.

    Parameters
    ----------
    reference_system : simtk.openmm.System
        The system to use as a reference. This will be modified only if
        `return_copy` is `False`.
    switch_width : simtk.unit.Quantity, default 1.0*angstrom
        Switch width for electrostatics (units of distance).
    return_copy : bool
        If `True`, `reference_system` is not modified, and a copy is returned.
        Setting it to `False` speeds up the function execution but modifies
        the `reference_system` object.

    Returns
    -------
    system : simtk.openmm.System
        System with reaction-field converted to c_rf = 0

    """
    if return_copy:
        system = copy.deepcopy(reference_system)
    else:
        system = reference_system

    # Add an UnshiftedReactionFieldForce for each CutoffPeriodic NonbondedForce.
    for reference_force in forces.iterate_nonbonded_forces(system):
        if reference_force.getNonbondedMethod() == openmm.NonbondedForce.CutoffPeriodic:
            reaction_field_force = forces.UnshiftedReactionFieldForce.from_nonbonded_force(reference_force,
                                                                                           switch_width=switch_width)
            system.addForce(reaction_field_force)

            # Remove particle electrostatics from reference force, but leave exceptions.
            for particle_index in range(reference_force.getNumParticles()):
                charge, sigma, epsilon = reference_force.getParticleParameters(particle_index)
                reference_force.setParticleParameters(particle_index, abs(0.0*charge), sigma, epsilon)

    return system


# =============================================================================
# RESTRAIN ATOMS
# =============================================================================

def restrain_atoms(thermodynamic_state, sampler_state, mdtraj_topology,
                   atoms_dsl, sigma=3.0*unit.angstroms):
    """Apply a soft harmonic restraint to the given atoms.

    This modifies the ``ThermodynamicState`` object.

    Parameters
    ----------
    thermodynamic_state : openmmtools.states.ThermodynamicState
        The thermodynamic state with the system. This will be modified.
    sampler_state : openmmtools.states.SamplerState
        The sampler state with the positions.
    mdtraj_topology : mdtraj.Topology
        The topology.
    atoms_dsl : str
        The MDTraj DSL string for selecting the atoms to restrain.
    sigma : simtk.unit.Quantity, optional
        Controls the strength of the restrain. The smaller, the tighter
        (units of distance, default is 3.0*angstrom).

    """
    K = thermodynamic_state.kT / sigma  # Spring constant.
    system = thermodynamic_state.system  # This is a copy.

    # Translate the system to the origin to avoid
    # MonteCarloBarostat rejections (see openmm#1854).
    protein_atoms = mdtraj_topology.select('protein')
    distance_unit = sampler_state.positions.unit
    centroid = np.mean(sampler_state.positions[protein_atoms,:] / distance_unit, 0) * distance_unit
    sampler_state.positions -= centroid

    # Create a CustomExternalForce to restrain all atoms.
    restraint_force = openmm.CustomExternalForce('(K/2)*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)')
    restrained_atoms = mdtraj_topology.select(atoms_dsl).tolist()
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
