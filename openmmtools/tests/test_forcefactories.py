#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test force factories in forcefactories.py.

"""

# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

from functools import partial

from openmmtools.forcefactories import *
from openmmtools import testsystems, states


# =============================================================================
# CONSTANTS
# =============================================================================

MAX_FORCE_RELATIVE_ERROR = 1.0e-6  # maximum allowable relative force error
GLOBAL_FORCE_UNIT = unit.kilojoules_per_mole / unit.nanometers  # controls printed units
GLOBAL_FORCES_PLATFORM = None  # This is used in every calculation.


# =============================================================================
# TESTING UTILITIES
# =============================================================================

def create_context(system, integrator, platform=None):
    """Create a Context.

    If platform is None, GLOBAL_ALCHEMY_PLATFORM is used.

    """
    if platform is None:
        platform = GLOBAL_FORCES_PLATFORM
    if platform is not None:
        context = openmm.Context(system, integrator, platform)
    else:
        context = openmm.Context(system, integrator)
    return context


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_forces(system, positions, platform=None, force_group=-1):
    """Compute forces of the system in the given positions.

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
    state = context.getState(getForces=True, groups=force_group)
    forces = state.getForces(asNumpy=True)
    del context, integrator, state
    return forces


def compare_system_forces(reference_system, alchemical_system, positions, name="", platform=None):
    """Check that the forces of reference and modified systems are close.

    Parameters
    ----------
    reference_system : simtk.openmm.System
        Reference System
    alchemical_system : simtk.openmm.System
        System to compare to reference
    positions : simtk.unit.Quantity of shape [nparticles,3] with units of distance
        The particle positions to use
    name : str, optional, default=""
        System name to use for debugging.
    platform : simtk.openmm.Platform, optional, default=None
        If specified, use this platform

    """
    # Compute forces
    reference_force = compute_forces(reference_system, positions, platform=platform) / GLOBAL_FORCE_UNIT
    alchemical_force = compute_forces(alchemical_system, positions, platform=platform) / GLOBAL_FORCE_UNIT

    # Check that error is small.
    def magnitude(vec):
        return np.sqrt(np.mean(np.sum(vec**2, axis=1)))

    relative_error = magnitude(alchemical_force - reference_force) / magnitude(reference_force)
    if np.any(np.abs(relative_error) > MAX_FORCE_RELATIVE_ERROR):
        err_msg = ("Maximum allowable relative force error exceeded (was {:.8f}; allowed {:.8f}).\n"
                   "alchemical_force = {:.8f}, reference_force = {:.8f}, difference = {:.8f}")
        raise Exception(err_msg.format(relative_error, MAX_FORCE_RELATIVE_ERROR, magnitude(alchemical_force),
                                       magnitude(reference_force), magnitude(alchemical_force-reference_force)))


def generate_new_positions(system, positions, platform=None, nsteps=50):
    """Generate new positions by taking a few steps from the old positions.

    Parameters
    ----------
    platform : simtk.openmm.Platform or None, optional
        If None, the global GLOBAL_ALCHEMY_PLATFORM will be used.
    nsteps : int, optional, default=50
        Number of steps of dynamics to take.

    Returns
    -------
    new_positions : simtk.unit.Quantity of shape [nparticles,3] with units compatible with distance
        New positions

    """
    temperature = 300 * unit.kelvin
    collision_rate = 90 / unit.picoseconds
    timestep = 1.0 * unit.femtoseconds
    integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
    context = create_context(system, integrator, platform)
    context.setPositions(positions)
    integrator.step(nsteps)
    new_positions = context.getState(getPositions=True).getPositions(asNumpy=True)
    del context, integrator
    return new_positions


# =============================================================================
# TEST FORCE FACTORIES FUNCTIONS
# =============================================================================

def test_restrain_atoms():
    """Check that the restrained molecule's centroid is in the origin."""
    host_guest = testsystems.HostGuestExplicit()
    topology = mdtraj.Topology.from_openmm(host_guest.topology)
    sampler_state = states.SamplerState(positions=host_guest.positions)
    thermodynamic_state = states.ThermodynamicState(host_guest.system, temperature=300*unit.kelvin,
                                                    pressure=1.0*unit.atmosphere)

    # Restrain all the host carbon atoms.
    restrained_atoms = [atom.index for atom in topology.atoms
                        if atom.element.symbol is 'C' and atom.index <= 125]
    restrain_atoms(thermodynamic_state, sampler_state, restrained_atoms)

    # Compute host center_of_geometry.
    centroid = np.mean(sampler_state.positions[:126], axis=0)
    assert np.allclose(centroid, np.zeros(3))

def test_replace_reaction_field():
    """Check that replacing reaction-field electrostatics with Custom*Force
    yields minimal force differences with original system.

    Note that we cannot test for energy consistency or energy overlap because
    which atoms are within the cutoff will cause energy difference to vary wildly.

    """
    test_cases = [
        testsystems.AlanineDipeptideExplicit(nonbondedMethod=openmm.app.CutoffPeriodic),
        testsystems.HostGuestExplicit(nonbondedMethod=openmm.app.CutoffPeriodic)
    ]
    platform = openmm.Platform.getPlatformByName('Reference')
    for test_system in test_cases:
        test_name = test_system.__class__.__name__

        # Replace reaction field.
        modified_rf_system = replace_reaction_field(test_system.system, switch_width=None)

        # Make sure positions are not at minimum.
        positions = generate_new_positions(test_system.system, test_system.positions)

        # Test forces.
        f = partial(compare_system_forces, test_system.system, modified_rf_system, positions,
                    name=test_name, platform=platform)
        f.description = "Testing replace_reaction_field on system {}".format(test_name)
        yield f

    for test_system in test_cases:
        test_name = test_system.__class__.__name__

        # Replace reaction field.
        modified_rf_system = replace_reaction_field(test_system.system, switch_width=None, shifted=True)

        # Make sure positions are not at minimum.
        positions = generate_new_positions(test_system.system, test_system.positions)

        # Test forces.
        f = partial(compare_system_forces, test_system.system, modified_rf_system, positions,
                    name=test_name, platform=platform)
        f.description = "Testing replace_reaction_field on system {} with shifted=True".format(test_name)
        yield f
