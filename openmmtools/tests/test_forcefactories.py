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
from simtk.openmm import Context, LangevinIntegrator

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


class LJPairTestBox(testsystems.LennardJonesPair):
    """A helper class to facilitate energy evaluations of a Lennard-Jones interaction."""
    def __init__(self, mass=1.0 * unit.atomic_mass_unit, epsilon=0.25*unit.kilojoule_per_mole,
                 sigma=0.5*unit.nanometer, box_size=100.0*unit.nanometer):
        super(LJPairTestBox, self).__init__(mass=mass, epsilon=epsilon, sigma=sigma)
        self.box_size = box_size
        self.system.getForce(0).setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
        self.system.getForce(0).setUseDispersionCorrection(False)
        self.system.getForce(0).setCutoffDistance(1.5 * unit.nanometer)
        self.system.setDefaultPeriodicBoxVectors(*np.eye(3)*self.box_size)
        self.update_context()

    def update_context(self):
        self.context = Context(self.system, LangevinIntegrator(
            300.0*unit.kelvin, 5.0/unit.picosecond, 1.0*unit.femtosecond))

    def get_energy(self, r_nm=1.0):
        self.context.setPositions(
            unit.Quantity(value=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, r_nm]]),
                          unit=unit.nanometer))
        state = self.context.getState(getEnergy=True)
        return state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)


def test_use_custom_switch_function():
    """Test if custom switch function is effective."""
    # system without switch
    ljpair = LJPairTestBox(epsilon=0.0*unit.kilojoule_per_mole)
    custom_force = openmm.CustomNonbondedForce("1")
    custom_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    custom_force.setCutoffDistance(1.5 * unit.nanometer)
    custom_force.addParticle()
    custom_force.addParticle()
    ljpair.system.addForce(custom_force)
    ljpair.update_context()
    assert abs(ljpair.get_energy(1.25) - 1.0) < 1e-8

    # add switch
    custom_force.setUseSwitchingFunction(True)
    custom_force.setSwitchingDistance(1.0 * unit.nanometer)
    apply_custom_switching_function(custom_force, 1.0*unit.nanometer, 1.5*unit.nanometer,
                                    switching_function="(r-cutoff)/(switch-cutoff)")
    ljpair.update_context()
    assert abs(ljpair.get_energy(1.25) - 0.5) < 1e-8


def test_use_vdw_with_charmm_vswitch():
    """Check Lennard-Jones switching using CHARMM's VSWITCH function."""
    ljpair = LJPairTestBox()
    assert is_lj_active_in_nonbonded_force(ljpair.system.getForce(0))
    use_custom_vdw_switching_function(ljpair.system, 1.0*unit.nanometer, 1.5*unit.nanometer)
    assert not is_lj_active_in_nonbonded_force(ljpair.system.getForce(0))
    [nb, cnb, cb] = ljpair.system.getForces()
    for i in range(ljpair.system.getNumForces()):
        try:
            print(i,ljpair.system.getForce(i).getEnergyFunction())
        except:
            pass
    ljpair.update_context()

    def switch(r, a=1.0, b=1.5):
        """Steinbach and Brooks """
        if r < a:
            return 1.0
        elif r > b:
            return 0.0
        else:
            return (b ** 2 - r ** 2) ** 2 * (b ** 2 + 2 * r ** 2 - 3 * a ** 2) / (b ** 2 - a ** 2) ** 3

    def lj_switch(r, sigma=ljpair.sigma, epsilon=ljpair.epsilon):
        sigma = sigma.value_in_unit(unit.nanometer)
        epsilon = epsilon.value_in_unit(unit.kilojoule_per_mole)
        return switch(r) * (4*epsilon*((sigma/r)**12 - (sigma/r)**6))

    for r in np.linspace(0.5, 2.0, 10, True):
        assert abs(ljpair.get_energy(r) - lj_switch(r)) < 1e-7


def test_use_vdw_with_charmm_force_switch():
    """Check Lennard-Jones energy from CHARMM's force switching scheme."""
    cutoff = 1.5 #*unit.nanometer
    switch = 1.0 #*unit.nanometer
    ljpair = LJPairTestBox()
    assert is_lj_active_in_nonbonded_force(ljpair.system.getForce(0))
    use_vdw_with_charmm_force_switch(ljpair.system, switch*unit.nanometer, cutoff*unit.nanometer)
    assert not is_lj_active_in_nonbonded_force(ljpair.system.getForce(0))
    ljpair.update_context()

    def target_energy(r, epsilon=ljpair.epsilon.value_in_unit(unit.kilojoule_per_mole),
                      sigma=ljpair.sigma.value_in_unit(unit.nanometer), switch=switch, cutoff=cutoff):
        A = 4 * epsilon * sigma**12
        B = 4 * epsilon * sigma**6
        k6 = B * cutoff ** 3 / (cutoff ** 3 - switch ** 3)
        k12 = A * cutoff ** 6 / (cutoff ** 6 - switch ** 6)
        dv6 = -1 / (cutoff * switch) ** 3
        dv12 = -1 / (cutoff * switch) ** 6
        if r < switch:
            return A * (1.0 / r ** 12 + dv12) - B * (1.0 / r ** 6 + dv6)
        elif r > cutoff:
            return 0.0
        else:
            return k12 * (r ** (-6) - cutoff ** (-6)) ** 2 - k6 * (r ** (-3) - cutoff ** (-3)) ** 2

    for r in np.linspace(0.8, 1.5, 200, True):
        assert abs(target_energy(r) - ljpair.get_energy(r)) < 1e-7


def test_no_vdw_interactions_after_switch():
    for switching in [use_vdw_with_charmm_force_switch, use_custom_vdw_switching_function]:
        testsystem = testsystems.CharmmSolvated(annihilate_vdw=True)
        switching(testsystem.system, 1.0*unit.nanometer, 1.2*unit.nanometer)
        for i,f in enumerate(testsystem.system.getForces()):
            f.setForceGroup(i)
        context = Context(testsystem.system, openmm.VerletIntegrator(1.0*unit.femtosecond))
        context.setPositions(testsystem.positions)
        for i,f in enumerate(testsystem.system.getForces()):
            if isinstance(f, openmm.CustomNonbondedForce):
                assert np.isclose(
                    0, context.getState(getEnergy=True, groups={i}).
                        getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole), rtol=0)


#TODO: Testing vs CHARMM energies
