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

import os
from functools import partial

import nose
import scipy
from simtk.openmm import app
from nose.plugins.attrib import attr

from openmmtools import testsystems
from openmmtools.alchemy import *

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA  # Boltzmann constant
temperature = 300.0 * unit.kelvin  # reference temperature
# MAX_DELTA = 0.01 * kB * temperature # maximum allowable deviation
MAX_DELTA = 1.0 * kB * temperature  # maximum allowable deviation


# =============================================================================
# SUBROUTINES FOR TESTING
# =============================================================================

def config_root_logger(verbose, log_file_path=None, mpicomm=None):
    """Setup the the root logger's configuration.
     The log messages are printed in the terminal and saved in the file specified
     by log_file_path (if not None) and printed. Note that logging use sys.stdout
     to print logging.INFO messages, and stderr for the others. The root logger's
     configuration is inherited by the loggers created by logging.getLogger(name).
     Different formats are used to display messages on the terminal and on the log
     file. For example, in the log file every entry has a timestamp which does not
     appear in the terminal. Moreover, the log file always shows the module that
     generate the message, while in the terminal this happens only for messages
     of level WARNING and higher.
    Parameters
    ----------
    verbose : bool
        Control the verbosity of the messages printed in the terminal. The logger
        displays messages of level logging.INFO and higher when verbose=False.
        Otherwise those of level logging.DEBUG and higher are printed.
    log_file_path : str, optional, default = None
        If not None, this is the path where all the logger's messages of level
        logging.DEBUG or higher are saved.
    mpicomm : mpi4py.MPI.COMM communicator, optional, default=None
        If specified, this communicator will be used to determine node rank.
    """

    class TerminalFormatter(logging.Formatter):
        """
        Simplified format for INFO and DEBUG level log messages.
        This allows to keep the logging.info() and debug() format separated from
        the other levels where more information may be needed. For example, for
        warning and error messages it is convenient to know also the module that
        generates them.
        """

        # This is the cleanest way I found to make the code compatible with both
        # Python 2 and Python 3
        simple_fmt = logging.Formatter('%(asctime)-15s: %(message)s')
        default_fmt = logging.Formatter('%(asctime)-15s: %(levelname)s - %(name)s - %(message)s')

        def format(self, record):
            if record.levelno <= logging.INFO:
                return self.simple_fmt.format(record)
            else:
                return self.default_fmt.format(record)

    # Check if root logger is already configured
    n_handlers = len(logging.root.handlers)
    if n_handlers > 0:
        root_logger = logging.root
        for i in range(n_handlers):
            root_logger.removeHandler(root_logger.handlers[0])

    # If this is a worker node, don't save any log file
    if mpicomm:
        rank = mpicomm.rank
    else:
        rank = 0

    if rank != 0:
        log_file_path = None

    # Add handler for stdout and stderr messages
    terminal_handler = logging.StreamHandler()
    terminal_handler.setFormatter(TerminalFormatter())
    if rank != 0:
        terminal_handler.setLevel(logging.WARNING)
    elif verbose:
        terminal_handler.setLevel(logging.DEBUG)
    else:
        terminal_handler.setLevel(logging.INFO)
    logging.root.addHandler(terminal_handler)

    # Add file handler to root logger
    if log_file_path is not None:
        file_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(file_format))
        logging.root.addHandler(file_handler)

    # Do not handle logging.DEBUG at all if unnecessary
    if log_file_path is not None:
        logging.root.setLevel(logging.DEBUG)
    else:
        logging.root.setLevel(terminal_handler.level)


def dump_xml(system=None, integrator=None, state=None):
    """
    Dump system, integrator, and state to XML for debugging.
    """
    from simtk.openmm import XmlSerializer

    def write_file(filename, contents):
        outfile = open(filename, 'w')
        outfile.write(contents)
        outfile.close()
    if system:
        write_file('system.xml', XmlSerializer.serialize(system))
    if integrator:
        write_file('integrator.xml', XmlSerializer.serialize(integrator))
    if state:
        write_file('state.xml', XmlSerializer.serialize(state))


def compute_energy(system, positions, platform=None, precision=None, force_group=-1):
    timestep = 1.0 * unit.femtoseconds
    integrator = openmm.VerletIntegrator(timestep)
    if platform:
        platform_name = platform.getName()
        if precision:
            if platform_name == 'CUDA':
                platform.setDefaultPropertyValue('CudaPrecision', precision)
            elif platform_name == 'OpenCL':
                platform.setDefaultPropertyValue('OpenCLPrecision', precision)
        context = openmm.Context(system, integrator, platform)
    else:
        context = openmm.Context(system, integrator)
    context.setPositions(positions)
    state = context.getState(getEnergy=True, groups=force_group)
    potential = state.getPotentialEnergy()
    del context, integrator, state
    return potential


def compute_energy_force(system, positions, force_name, platform=None, precision=None):
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

    force_energy = compute_energy(system, positions, platform=platform,
                                  precision=precision, force_group=2**force_name_index)
    del system
    return force_energy


def check_waterbox(platform=None, precision=None, nonbondedMethod=openmm.NonbondedForce.CutoffPeriodic):
    """Compare annihilated states in vacuum and a large box.
    """
    platform_name = platform.getName()
    testsystem = testsystems.WaterBox()
    system = testsystem.system
    positions = testsystem.positions

    # Use reaction field
    for force in system.getForces():
        if force.__class__.__name__ == 'NonbondedForce':
            force.setNonbondedMethod(nonbondedMethod)

    factory_args = {'ligand_atoms': [], 'receptor_atoms': [],
                    'annihilate_sterics': False, 'annihilate_electrostatics': True}

    # Create alchemically-modified system
    factory = AbsoluteAlchemicalFactory(system, **factory_args)
    alchemical_system = factory.createPerturbedSystem()

    # Compare energies
    system_energy = compute_energy(system, positions, platform=platform, precision=precision)
    alchemical_1_energy = compute_energy(alchemical_system, positions, platform=platform, precision=precision)

    # Set lambda = 0
    lambda_value = 0.0
    alchemical_state = AlchemicalState(lambda_electrostatics=lambda_value, lambda_sterics=lambda_value,
                                       lambda_torsions=lambda_value)
    AbsoluteAlchemicalFactory.perturbSystem(alchemical_system, alchemical_state)
    alchemical_0_energy = compute_energy(alchemical_system, positions, platform=platform, precision=precision)

    # Check deviation.
    logger.info('========')
    logger.info('Platform {}'.format(platform_name))
    logger.info('Alchemically-modified WaterBox with no alchemical atoms')
    logger.info('real system : {:8.3f} kcal/mol'.format(system_energy / unit.kilocalories_per_mole))
    logger.info('lambda = 1  : {:8.3f} kcal/mol'.format(alchemical_1_energy / unit.kilocalories_per_mole))
    logger.info('lambda = 0  : {:8.3f} kcal/mol'.format(alchemical_0_energy / unit.kilocalories_per_mole))
    delta = alchemical_1_energy - alchemical_0_energy
    logger.info("ERROR       : {:8.3f} kcal/mol".format(delta / unit.kilocalories_per_mole))
    if abs(delta) > MAX_DELTA:
        raise Exception(("Maximum allowable deviation on platform {} exceeded "
                         "(was {:.8f} kcal/mol; allowed {.8f} kcal/mol);").format(
            platform_name, delta / unit.kilocalories_per_mole, MAX_DELTA / unit.kilocalories_per_mole))


def test_waterbox():
    """Compare annihilated states in vacuum and a large box.
    """
    for platform_index in range(openmm.Platform.getNumPlatforms()):
        for nonbondedMethod in [openmm.NonbondedForce.PME, openmm.NonbondedForce.CutoffPeriodic]:
            platform = openmm.Platform.getPlatform(platform_index)
            f = partial(check_waterbox, platform=platform, nonbondedMethod=nonbondedMethod)
            platform_name = platform.getName()
            nonbondedMethod_name = 'PME' if (nonbondedMethod == openmm.NonbondedForce.PME) else 'CutoffPeriodic'
            f.description = ('Comparing waterbox annihilated states for platform {} and '
                             'nonbondedMethod {}').format(platform_name, nonbondedMethod_name)
            yield f


def compare_platforms(system, positions, factory_args=dict()):
    # Create annihilated version of vacuum system.
    factory = AbsoluteAlchemicalFactory(system, **factory_args)
    alchemical_system = factory.createPerturbedSystem()

    def set_lambda(alchemical_system, lambda_value):
        alchemical_state = AlchemicalState(lambda_electrostatics=lambda_value, lambda_sterics=lambda_value,
                                           lambda_torsions=lambda_value)
        AbsoluteAlchemicalFactory.perturbSystem(alchemical_system, alchemical_state)

    # Compare energies
    energies = dict()
    platform_names = list()
    for platform_index in range(openmm.Platform.getNumPlatforms()):
        platform = openmm.Platform.getPlatform(platform_index)
        platform_name = platform.getName()
        if platform_name != 'Reference':
            platform_names.append(platform_name)
        energies[platform_name] = dict()
        energies[platform_name]['full'] = compute_energy(system, positions, platform=platform)
        set_lambda(alchemical_system, 1.0)
        energies[platform_name]['lambda = 1'] = compute_energy(alchemical_system, positions, platform=platform)
        set_lambda(alchemical_system, 0.0)
        energies[platform_name]['lambda = 0'] = compute_energy(alchemical_system, positions, platform=platform)

    # Check deviations.
    for platform_name in platform_names:
        for energy_name in ['full', 'lambda = 1', 'lambda = 0']:
            delta = energies[platform_name][energy_name] - energies['Reference'][energy_name]
            if abs(delta) > MAX_DELTA:
                raise Exception(("Maximum allowable deviation on platform {} exceeded "
                                 "(was {:.8f} kcal/mol; allowed {:.8f} kcal/mol).").format(
                    platform_name, delta / unit.kilocalories_per_mole, MAX_DELTA / unit.kilocalories_per_mole))


def notest_denihilated_states(platform_name=None, precision=None):
    """Compare annihilated electrostatics / decoupled sterics states in vacuum and a large box.
    """
    testsystem = testsystems.TolueneVacuum()
    vacuum_system = testsystem.system
    positions = testsystem.positions

    factory_args = {'ligand_atoms': range(0, 15), 'receptor_atoms': [],
                    'annihilate_sterics': False, 'annihilate_electrostatics': True}

    # Create annihilated version of vacuum system.
    factory = AbsoluteAlchemicalFactory(vacuum_system, **factory_args)
    vacuum_alchemical_system = factory.createPerturbedSystem()

    # Make copy of system that has periodic boundaries and uses reaction field.
    periodic_system = copy.deepcopy(vacuum_system)
    box_edge = 18.5 * unit.angstroms
    from simtk.openmm import Vec3
    periodic_system.setDefaultPeriodicBoxVectors(Vec3(box_edge, 0, 0), Vec3(0, box_edge, 0),
                                                 Vec3(0, 0, box_edge))
    for force in periodic_system.getForces():
        if force.__class__.__name__ == 'NonbondedForce':
            force.setNonbondedMethod(openmm.NonbondedForce.PME)
            force.setCutoffDistance(9.0 * unit.angstroms)
            force.setUseDispersionCorrection(False)
            force.setReactionFieldDielectric(1.0)
    factory = AbsoluteAlchemicalFactory(periodic_system, **factory_args)
    periodic_alchemical_system = factory.createPerturbedSystem()

    # Compare energies
    platform = None
    if platform_name:
        platform = openmm.Platform.getPlatformByName(platform_name)

    vacuum_alchemical_1_energy = compute_energy(vacuum_alchemical_system, positions,
                                                platform=platform, precision=precision)
    periodic_alchemical_1_energy = compute_energy(periodic_alchemical_system, positions,
                                                  platform=platform, precision=precision)

    # Set lambda = 0
    lambda_value = 0.0
    alchemical_state = AlchemicalState(lambda_electrostatics=lambda_value, lambda_sterics=lambda_value,
                                       lambda_torsions=lambda_value)
    AbsoluteAlchemicalFactory.perturbSystem(vacuum_alchemical_system, alchemical_state)
    AbsoluteAlchemicalFactory.perturbSystem(periodic_alchemical_system, alchemical_state)

    vacuum_alchemical_0_energy = compute_energy(vacuum_alchemical_system, positions,
                                                platform=platform, precision=precision)
    periodic_alchemical_0_energy = compute_energy(periodic_alchemical_system, positions,
                                                  platform=platform, precision=precision)

    vacuum_alchemical_energy_diff = (vacuum_alchemical_1_energy - vacuum_alchemical_0_energy)
    logger.info('vacuum   lambda = 1 : %8.3f kcal/mol' % (vacuum_alchemical_1_energy / unit.kilocalories_per_mole))
    logger.info('vacuum   lambda = 0 : %8.3f kcal/mol' % (vacuum_alchemical_0_energy / unit.kilocalories_per_mole))
    logger.info('difference          : %8.3f kcal/mol' % (vacuum_alchemical_energy_diff / unit.kilocalories_per_mole))

    periodic_alchemical_energy_diff = periodic_alchemical_1_energy - periodic_alchemical_0_energy
    logger.info('periodic lambda = 1 : %8.3f kcal/mol' % (periodic_alchemical_1_energy / unit.kilocalories_per_mole))
    logger.info('periodic lambda = 0 : %8.3f kcal/mol' % (periodic_alchemical_0_energy / unit.kilocalories_per_mole))
    logger.info('difference          : %8.3f kcal/mol' % (periodic_alchemical_energy_diff / unit.kilocalories_per_mole))

    delta = (vacuum_alchemical_1_energy - vacuum_alchemical_0_energy) - (periodic_alchemical_1_energy - periodic_alchemical_0_energy)
    if abs(delta) > MAX_DELTA:
        raise Exception(("Maximum allowable difference lambda=1 energy and lambda=0 energy in vacuum "
                         "and periodic box exceeded (was {:.8f} kcal/mol; allowed {:.8f} kcal/mol).").format(
            delta / unit.kilocalories_per_mole, MAX_DELTA / unit.kilocalories_per_mole))


def assert_almost_equal(energy1, energy2, err_msg):
        delta = energy1 - energy2
        err_msg += ' interactions do not match! Reference {}, alchemical {},' \
                   ' difference {}'.format(energy1, energy2, delta)
        assert abs(delta) < MAX_DELTA, err_msg


def dissect_nonbonded_energy(reference_system, positions, alchemical_atoms, platform=None):
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
    platform : simtk.openmm.Platform
        The platform used to compute energies.

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
    tot_energy = compute_energy(system, positions, platform)
    tot_reciprocal_energy = compute_energy(system, positions, platform, force_group={31})

    # Compute contributions from particle sterics
    turn_off(nonbonded_force, sterics=True, only_atoms=alchemical_atoms)
    tot_energy_no_alchem_particle_sterics = compute_energy(system, positions, platform)
    system, nonbonded_force = restore_system(reference_system)  # Restore alchemical sterics
    turn_off(nonbonded_force, sterics=True, only_atoms=nonalchemical_atoms)
    tot_energy_no_nonalchem_particle_sterics = compute_energy(system, positions, platform)
    turn_off(nonbonded_force, sterics=True)
    tot_energy_no_particle_sterics = compute_energy(system, positions, platform)

    tot_particle_sterics = tot_energy - tot_energy_no_particle_sterics
    nn_particle_sterics = tot_energy_no_alchem_particle_sterics - tot_energy_no_particle_sterics
    aa_particle_sterics = tot_energy_no_nonalchem_particle_sterics - tot_energy_no_particle_sterics
    na_particle_sterics = tot_particle_sterics - nn_particle_sterics - aa_particle_sterics

    # Compute contributions from particle electrostatics
    system, nonbonded_force = restore_system(reference_system)  # Restore sterics
    turn_off(nonbonded_force, electrostatics=True, only_atoms=alchemical_atoms)
    tot_energy_no_alchem_particle_electro = compute_energy(system, positions, platform)
    nn_reciprocal_energy = compute_energy(system, positions, platform, force_group={31})
    system, nonbonded_force = restore_system(reference_system)  # Restore alchemical electrostatics
    turn_off(nonbonded_force, electrostatics=True, only_atoms=nonalchemical_atoms)
    tot_energy_no_nonalchem_particle_electro = compute_energy(system, positions, platform)
    aa_reciprocal_energy = compute_energy(system, positions, platform, force_group={31})
    turn_off(nonbonded_force, electrostatics=True)
    tot_energy_no_particle_electro = compute_energy(system, positions, platform)

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
    tot_energy_no_alchem_exception_sterics = compute_energy(system, positions, platform)
    system, nonbonded_force = restore_system(reference_system)  # Restore alchemical sterics
    turn_off(nonbonded_force, sterics=True, exceptions=True, only_atoms=nonalchemical_atoms)
    tot_energy_no_nonalchem_exception_sterics = compute_energy(system, positions, platform)
    turn_off(nonbonded_force, sterics=True, exceptions=True)
    tot_energy_no_exception_sterics = compute_energy(system, positions, platform)

    tot_exception_sterics = tot_energy - tot_energy_no_exception_sterics
    nn_exception_sterics = tot_energy_no_alchem_exception_sterics - tot_energy_no_exception_sterics
    aa_exception_sterics = tot_energy_no_nonalchem_exception_sterics - tot_energy_no_exception_sterics
    na_exception_sterics = tot_exception_sterics - nn_exception_sterics - aa_exception_sterics

    # Compute contributions from exceptions electrostatics
    system, nonbonded_force = restore_system(reference_system)  # Restore exceptions sterics
    turn_off(nonbonded_force, electrostatics=True, exceptions=True, only_atoms=alchemical_atoms)
    tot_energy_no_alchem_exception_electro = compute_energy(system, positions, platform)
    system, nonbonded_force = restore_system(reference_system)  # Restore alchemical electrostatics
    turn_off(nonbonded_force, electrostatics=True, exceptions=True, only_atoms=nonalchemical_atoms)
    tot_energy_no_nonalchem_exception_electro = compute_energy(system, positions, platform)
    turn_off(nonbonded_force, electrostatics=True, exceptions=True)
    tot_energy_no_exception_electro = compute_energy(system, positions, platform)

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


def check_interacting_energy_components(factory, positions, platform=None):
    """Compare full and alchemically-modified system energies by energy component.

    Parameters
    ----------
    factory : AbsoluteAlchemicalFactory
        The factory to test.
    positions : simtk.openmm.unit.Quantity of dimension [nparticles,3] with units compatible with Angstroms
        The positions to test.
    platform : simtk.openmm.Platform, optional
        The platform used to compute energies.

    """

    reference_system = copy.deepcopy(factory.reference_system)
    alchemical_system = copy.deepcopy(factory.alchemically_modified_system)

    # Find nonbonded method
    for nonbonded_force in reference_system.getForces():
        if isinstance(nonbonded_force, openmm.NonbondedForce):
            nonbonded_method = nonbonded_force.getNonbondedMethod()
            break

    # Get energy components of reference system's nonbonded force
    print("Dissecting reference system's nonbonded force")
    energy_components = dissect_nonbonded_energy(reference_system, positions,
                                                 factory.ligand_atomset, platform)
    nn_particle_sterics, aa_particle_sterics, na_particle_sterics,\
    nn_particle_electro, aa_particle_electro, na_particle_electro,\
    nn_exception_sterics, aa_exception_sterics, na_exception_sterics,\
    nn_exception_electro, aa_exception_electro, na_exception_electro,\
    nn_reciprocal_energy, aa_reciprocal_energy, na_reciprocal_energy = energy_components

    # Dissect unmodified nonbonded force in alchemical system
    print("Dissecting alchemical system's unmodified nonbonded force")
    energy_components = dissect_nonbonded_energy(alchemical_system, positions,
                                                 factory.ligand_atomset, platform)
    unmod_nn_particle_sterics, unmod_aa_particle_sterics, unmod_na_particle_sterics,\
    unmod_nn_particle_electro, unmod_aa_particle_electro, unmod_na_particle_electro,\
    unmod_nn_exception_sterics, unmod_aa_exception_sterics, unmod_na_exception_sterics,\
    unmod_nn_exception_electro, unmod_aa_exception_electro, unmod_na_exception_electro,\
    unmod_nn_reciprocal_energy, unmod_aa_reciprocal_energy, unmod_na_reciprocal_energy = energy_components

    # Get alchemically-modified energy components
    print("Computing alchemical system components energies")
    alchemical_state = factory.FullyInteractingAlchemicalState()
    energy_components = factory.getEnergyComponents(alchemical_state, positions, use_all_parameters=False)
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
                                                                       factory.ligand_atomset, positions)
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
        reference_force_energy = compute_energy_force(reference_system, positions,
                                                      force_name, platform=platform)

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


def check_noninteracting_energy_components(factory, positions, platform=None):
    """Check noninteracting energy components are zero when appropriate.

    Parameters
    ----------
    factory : AbsoluteAlchemicalFactory
        The factory to test.
    positions : simtk.openmm.unit.Quantity of dimension [nparticles,3] with units compatible with Angstroms
        The positions to test.
    platform : simtk.openmm.Platform, optional
        The platform used to compute energies.

    """
    alchemical_state = factory.NoninteractingAlchemicalState()
    energy_components = factory.getEnergyComponents(alchemical_state, positions, use_all_parameters=False)
    energy_unit = unit.kilojoule_per_mole

    def assert_zero_energy(label):
        print('testing %s' % label)
        value = energy_components[label]
        assert abs(value / energy_unit) == 0.0, ("'{}' should have zero energy in annihilated alchemical"
                                                 " state, but energy is {}").format(label, str(value))

    # Check that non-alchemical/alchemical particle interactions and 1,4 exceptions have been annihilated
    assert_zero_energy('alchemically modified NonbondedForce for non-alchemical/alchemical sterics')
    assert_zero_energy('alchemically modified NonbondedForce for non-alchemical/alchemical electrostatics')
    assert_zero_energy('alchemically modified BondForce for non-alchemical/alchemical sterics exceptions')
    assert_zero_energy('alchemically modified BondForce for non-alchemical/alchemical electrostatics exceptions')

    # Check that alchemical/alchemical particle interactions and 1,4 exceptions have been annihilated
    if factory.annihilate_sterics:
        assert_zero_energy('alchemically modified NonbondedForce for alchemical/alchemical sterics')
        assert_zero_energy('alchemically modified BondForce for alchemical/alchemical sterics exceptions')
    if factory.annihilate_electrostatics:
        assert_zero_energy('alchemically modified NonbondedForce for alchemical/alchemical electrostatics')
        assert_zero_energy('alchemically modified BondForce for alchemical/alchemical electrostatics exceptions')

    # Check valence terms
    for force_name in ['HarmonicBondForce', 'HarmonicAngleForce', 'PeriodicTorsionForce', 'GBSAOBCForce']:
        force_label = 'alchemically modified ' + force_name
        if force_label in energy_components:
            assert_zero_energy(force_label)


def compareSystemEnergies(positions, systems, descriptions, platform=None, precision=None):
    # Compare energies.
    timestep = 1.0 * unit.femtosecond

    if platform:
        platform_name = platform.getName()
        if precision:
            if platform_name == 'CUDA':
                platform.setDefaultPropertyValue('CudaPrecision', precision)
            elif platform_name == 'OpenCL':
                platform.setDefaultPropertyValue('OpenCLPrecision', precision)

    potentials = list()
    states = list()
    for system in systems:
        #dump_xml(system=system)
        integrator = openmm.VerletIntegrator(timestep)
        #dump_xml(integrator=integrator)
        if platform:
            context = openmm.Context(system, integrator, platform)
        else:
            context = openmm.Context(system, integrator)
        context.setPositions(positions)
        state = context.getState(getEnergy=True, getPositions=True)
        #dump_xml(system=system, integrator=integrator, state=state)
        potential = state.getPotentialEnergy()
        potentials.append(potential)
        states.append(state)
        del context, integrator, state

    logger.info("========")
    for i in range(len(systems)):
        logger.info("%32s : %24.8f kcal/mol" % (descriptions[i], potentials[i] / unit.kilocalories_per_mole))
        if (i > 0):
            delta = potentials[i] - potentials[0]
            logger.info("%32s : %24.8f kcal/mol" % ('ERROR', delta / unit.kilocalories_per_mole))
            if (abs(delta) > MAX_DELTA):
                raise Exception(("Maximum allowable deviation exceeded "
                                 "(was {:.8f} kcal/mol; allowed {:.8f} kcal/mol).").format(
                    delta / unit.kilocalories_per_mole, MAX_DELTA / unit.kilocalories_per_mole))

    return potentials

def alchemical_factory_check(reference_system, positions, platform_name=None, precision=None, factory_args=None):
    """
    Compare energies of reference system and fully-interacting alchemically modified system.

    ARGUMENTS

    reference_system : simtk.openmm.System
       The reference System object to compare with
    positions : simtk.unit.Quantity of dimentsion [natoms,3] with units compatible with angstroms
       The positions to assess energetics for
    precision : str, optional, default=None
       Precision model, or default if not specified. ('single', 'double', 'mixed')
    factory_args : dict(), optional, default=None
       Arguments passed to AbsoluteAlchemicalFactory.

    """

    # Create a factory to produce alchemical intermediates.
    logger.info("Creating alchemical factory...")
    initial_time = time.time()
    factory = AbsoluteAlchemicalFactory(reference_system, **factory_args)
    final_time = time.time()
    elapsed_time = final_time - initial_time
    logger.info("AbsoluteAlchemicalFactory initialization took %.3f s" % elapsed_time)

    platform = None
    if platform_name:
        platform = openmm.Platform.getPlatformByName(platform_name)

    # Check energies for fully interacting system.
    print('check fully interacting interacting energy components...')
    check_interacting_energy_components(factory, positions, platform)

    # Check energies for noninteracting system.
    print('check noninteracting energy components...')
    check_noninteracting_energy_components(factory, positions, platform)


def benchmark(reference_system, positions, platform_name=None, nsteps=500,
              timestep=1.0*unit.femtoseconds, factory_args=None):
    """
    Benchmark performance of alchemically modified system relative to original system.

    Parameters
    ----------
    reference_system : simtk.openmm.System
       The reference System object to compare with
    positions : simtk.unit.Quantity with units compatible with nanometers
       The positions to assess energetics for.
    platform_name : str, optional, default=None
       The name of the platform to use for benchmarking.
    nsteps : int, optional, default=500
       Number of molecular dynamics steps to use for benchmarking.
    timestep : simtk.unit.Quantity with units compatible with femtoseconds, optional, default=1*femtoseconds
       Timestep to use for benchmarking.
    factory_args : dict(), optional, default=None
       Arguments passed to AbsoluteAlchemicalFactory.

    """

    # Create a factory to produce alchemical intermediates.
    logger.info("Creating alchemical factory...")
    initial_time = time.time()
    factory = AbsoluteAlchemicalFactory(reference_system, **factory_args)
    final_time = time.time()
    elapsed_time = final_time - initial_time
    logger.info("AbsoluteAlchemicalFactory initialization took %.3f s" % elapsed_time)

    # Create an alchemically-perturbed state corresponding to nearly fully-interacting.
    # NOTE: We use a lambda slightly smaller than 1.0 because the AlchemicalFactory does
    # not use Custom*Force softcore versions if lambda = 1.0 identically.
    lambda_value = 1.0 - 1.0e-6
    alchemical_state = AlchemicalState(lambda_electrostatics=lambda_value, lambda_sterics=lambda_value,
                                       lambda_torsions=lambda_value)

    platform = None
    if platform_name:
        platform = openmm.Platform.getPlatformByName(platform_name)

    # Create the perturbed system.
    logger.info("Creating alchemically-modified state...")
    initial_time = time.time()
    alchemical_system = factory.createPerturbedSystem(alchemical_state)
    final_time = time.time()
    elapsed_time = final_time - initial_time
    # Compare energies.
    logger.info("Computing reference energies...")
    reference_integrator = openmm.VerletIntegrator(timestep)
    if platform:
        reference_context = openmm.Context(reference_system, reference_integrator, platform)
    else:
        reference_context = openmm.Context(reference_system, reference_integrator)
    reference_context.setPositions(positions)
    reference_state = reference_context.getState(getEnergy=True)
    reference_potential = reference_state.getPotentialEnergy()
    logger.info("Computing alchemical energies...")
    alchemical_integrator = openmm.VerletIntegrator(timestep)
    if platform:
        alchemical_context = openmm.Context(alchemical_system, alchemical_integrator, platform)
    else:
        alchemical_context = openmm.Context(alchemical_system, alchemical_integrator)
    alchemical_context.setPositions(positions)
    alchemical_state = alchemical_context.getState(getEnergy=True)
    alchemical_potential = alchemical_state.getPotentialEnergy()
    delta = alchemical_potential - reference_potential

    # Make sure all kernels are compiled.
    reference_integrator.step(1)
    alchemical_integrator.step(1)

    # Time simulations.
    logger.info("Simulating reference system...")
    initial_time = time.time()
    reference_integrator.step(nsteps)
    reference_state = reference_context.getState(getEnergy=True)
    reference_potential = reference_state.getPotentialEnergy()
    final_time = time.time()
    reference_time = final_time - initial_time
    logger.info("Simulating alchemical system...")
    initial_time = time.time()
    alchemical_integrator.step(nsteps)
    alchemical_state = alchemical_context.getState(getEnergy=True)
    alchemical_potential = alchemical_state.getPotentialEnergy()
    final_time = time.time()
    alchemical_time = final_time - initial_time

    logger.info("TIMINGS")
    logger.info("reference system       : %12.3f s for %8d steps (%12.3f ms/step)" % (reference_time, nsteps, reference_time/nsteps*1000))
    logger.info("alchemical system      : %12.3f s for %8d steps (%12.3f ms/step)" % (alchemical_time, nsteps, alchemical_time/nsteps*1000))
    logger.info("alchemical simulation is %12.3f x slower than unperturbed system" % (alchemical_time / reference_time))

    return delta


def overlap_check(reference_system, positions, platform_name=None, precision=None,
                  nsteps=50, nsamples=200, factory_args=None, cached_trajectory_filename=None):
    """
    Test overlap between reference system and alchemical system by running a short simulation.

    Parameters
    ----------
    reference_system : simtk.openmm.System
       The reference System object to compare with
    positions : simtk.unit.Quantity with units compatible with nanometers
       The positions to assess energetics for.
    platform_name : str, optional, default=None
       The name of the platform to use for benchmarking.
    nsteps : int, optional, default=50
       Number of molecular dynamics steps between samples.
    nsamples : int, optional, default=100
       Number of samples to collect.
    factory_args : dict(), optional, default=None
       Arguments passed to AbsoluteAlchemicalFactory.
    cached_trajectory_filename : str, optional, default=None
       If specified, attempt to cache (or reuse) trajectory.

    """
    temperature = 300.0 * unit.kelvin
    pressure = 1.0 * unit.atmospheres
    collision_rate = 5.0 / unit.picoseconds
    timestep = 2.0 * unit.femtoseconds
    kT = (kB * temperature)

    # Add a barostat
    reference_system = copy.deepcopy(reference_system)
    reference_system.addForce( openmm.MonteCarloBarostat(pressure, temperature) )

    # Create a fully-interacting alchemical state.
    factory = AbsoluteAlchemicalFactory(reference_system, **factory_args)
    alchemical_state = AlchemicalState()
    alchemical_system = factory.createPerturbedSystem(alchemical_state)

    # Select platform.
    platform = None
    if platform_name:
        platform = openmm.Platform.getPlatformByName(platform_name)

    # Create integrators.
    reference_integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
    alchemical_integrator = openmm.VerletIntegrator(timestep)

    # Create contexts.
    if platform:
        reference_context = openmm.Context(reference_system, reference_integrator, platform)
        alchemical_context = openmm.Context(alchemical_system, alchemical_integrator, platform)
    else:
        reference_context = openmm.Context(reference_system, reference_integrator)
        alchemical_context = openmm.Context(alchemical_system, alchemical_integrator)

    ncfile = None
    if cached_trajectory_filename:
        cache_mode = 'write'

        # Try reading from cache
        from netCDF4 import Dataset
        if os.path.exists(cached_trajectory_filename):
            try:
                ncfile = Dataset(cached_trajectory_filename, 'r')
                if (ncfile.variables['positions'].shape == (nsamples, reference_system.getNumParticles(), 3)
                    and ncfile.variables['box_vectors'].shape == (nsamples, 3, 3)):
                    # Read the cache if everything matches
                    cache_mode = 'read'
            except Exception as e:
                pass

        if cache_mode == 'write':
            # If anything went wrong, create a new cache.
            try:
                (pathname, filename) = os.path.split(cached_trajectory_filename)
                if not os.path.exists(pathname): os.makedirs(pathname)
                ncfile = Dataset(cached_trajectory_filename, 'w', format='NETCDF4')
                ncfile.createDimension('samples', 0)
                ncfile.createDimension('atoms', reference_system.getNumParticles())
                ncfile.createDimension('spatial', 3)
                ncfile.createVariable('positions', 'f4', ('samples', 'atoms', 'spatial'))
                ncfile.createVariable('box_vectors', 'f4', ('samples', 'spatial', 'spatial'))
            except Exception as e:
                logger.info(str(e))
                logger.info('Could not create a trajectory cache (%s).' % cached_trajectory_filename)
                ncfile = None

    # Collect simulation data.
    reference_context.setPositions(positions)
    du_n = np.zeros([nsamples], np.float64)  # du_n[n] is the
    print()
    import click
    with click.progressbar(range(nsamples)) as bar:
        for sample in bar:
            if cached_trajectory_filename and (cache_mode == 'read'):
                # Load cached frames.
                positions = unit.Quantity(ncfile.variables['positions'][sample, :, :], unit.nanometers)
                box_vectors = unit.Quantity(ncfile.variables['box_vectors'][sample, :, :], unit.nanometers)
                reference_context.setPeriodicBoxVectors(box_vectors[0,:], box_vectors[1, :], box_vectors[2, :])
                reference_context.setPositions(positions)
            else:
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

            du_n[sample] = (alchemical_potential - reference_potential) / kT

            if cached_trajectory_filename and (cache_mode == 'write') and (ncfile is not None):
                ncfile.variables['positions'][sample, :, :] = reference_state.getPositions(asNumpy=True) / unit.nanometers
                ncfile.variables['box_vectors'][sample, :, :] = reference_state.getPeriodicBoxVectors(asNumpy=True) / unit.nanometers

    # Clean up.
    del reference_context, alchemical_context
    if cached_trajectory_filename and (ncfile is not None):
        ncfile.close()

    # Discard data to equilibration and subsample.
    from pymbar import timeseries
    t0, g, Neff = timeseries.detectEquilibration(du_n)
    indices = timeseries.subsampleCorrelatedData(du_n, g=g)
    du_n = du_n[indices]

    # Compute statistics.
    from pymbar import EXP
    DeltaF, dDeltaF = EXP(du_n)

    # Raise an exception if the error is larger than 3kT.
    MAX_DEVIATION = 3.0  # kT
    report = ('DeltaF = {:12.3f} +- {:12.3f} kT ({:5d} samples, g = {:6.1f}); '
              'du mean {:.3f} kT stddev {:.3f} kT').format(DeltaF, dDeltaF, Neff, g, du_n.mean(), du_n.std())
    logger.info(report)
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


def lambda_trace(reference_system, positions, platform_name=None, precision=None, nsteps=100, factory_args=None):
    """
    Compute potential energy as a function of lambda.

    """

    # Create a factory to produce alchemical intermediates.
    factory = AbsoluteAlchemicalFactory(reference_system, **factory_args)

    platform = None
    if platform_name:
        # Get platform.
        platform = openmm.Platform.getPlatformByName(platform_name)

    if precision:
        if platform_name == 'CUDA':
            platform.setDefaultPropertyValue('CudaPrecision', precision)
        elif platform_name == 'OpenCL':
            platform.setDefaultPropertyValue('OpenCLPrecision', precision)

    # Take equally-sized steps.
    delta = 1.0 / nsteps

    def compute_potential(system, positions, platform=None):
        timestep = 1.0 * unit.femtoseconds
        integrator = openmm.VerletIntegrator(timestep)
        if platform:
            context = openmm.Context(system, integrator, platform)
        else:
            context = openmm.Context(system, integrator)
        context.setPositions(positions)
        state = context.getState(getEnergy=True)
        potential = state.getPotentialEnergy()
        del integrator, context
        return potential

    # Compute unmodified energy.
    u_original = compute_potential(reference_system, positions, platform)

    # Scan through lambda values.
    lambda_i = np.zeros([nsteps+1], np.float64)  # lambda values for u_i
    # u_i[i] is the potential energy for lambda_i[i]
    u_i = unit.Quantity(np.zeros([nsteps+1], np.float64), unit.kilocalories_per_mole)
    for i in range(nsteps+1):
        lambda_value = 1.0-i*delta # compute lambda value for this step
        alchemical_system = factory.createPerturbedSystem(AlchemicalState(lambda_electrostatics=lambda_value,
                                                                          lambda_sterics=lambda_value,
                                                                          lambda_torsions=lambda_value))
        lambda_i[i] = lambda_value
        u_i[i] = compute_potential(alchemical_system, positions, platform)
        logger.info("%12.9f %24.8f kcal/mol" % (lambda_i[i], u_i[i] / unit.kilocalories_per_mole))

    # Write figure as PDF.
    import pylab
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

    return


def generate_trace(test_system):
    lambda_trace(test_system['test'].system, test_system['test'].positions, test_system['receptor_atoms'], test_system['ligand_atoms'])
    return


# =============================================================================
# TEST ALCHEMICAL STATE
# =============================================================================

class TestAlchemicalState(object):
    """Test AlchemicalState compatibility with CompoundThermodynamicState."""

    @classmethod
    def setup_class(cls):
        """Create test systems and shared objects."""
        alanine_vacuum = testsystems.AlanineDipeptideVacuum()

        # System with only lambda_sterics and lambda_electrostatics.
        alchemical_factory = AbsoluteAlchemicalFactory(reference_system=alanine_vacuum.system,
                                                       ligand_atoms=range(0, 22))
        alchemical_alanine_system = alchemical_factory.alchemically_modified_system
        cls.alanine_state = states.ThermodynamicState(alchemical_alanine_system,
                                                      temperature=300*unit.kelvin)

        # System with all lambdas except for lambda_restraints.
        alchemical_factory = AbsoluteAlchemicalFactory(reference_system=alanine_vacuum.system,
                                                       ligand_atoms=range(0, 22), alchemical_torsions=True,
                                                       alchemical_angles=True, alchemical_bonds=True)
        fully_alchemical_alanine_system = alchemical_factory.alchemically_modified_system
        cls.full_alanine_state = states.ThermodynamicState(fully_alchemical_alanine_system,
                                                           temperature=300*unit.kelvin)

        # Test case: (ThermodynamicState, defined_lambda_parameters)
        cls.test_cases = [
            (cls.alanine_state, {'lambda_sterics', 'lambda_electrostatics'}),
            (cls.full_alanine_state, {'lambda_sterics', 'lambda_electrostatics', 'lambda_bonds',
                                      'lambda_angles', 'lambda_torsions'})
        ]

    @staticmethod
    def test_sanitize_expression():
        """Test that lambda variable is substituted correctly."""
        test_cases = [('lambda', '_AlchemicalFunction__lambda'),
                      ('(lambda)', '(_AlchemicalFunction__lambda)'),
                      ('( lambda )', '( _AlchemicalFunction__lambda )'),
                      ('lambda_sterics', 'lambda_sterics'),
                      ('sterics_lambda', 'sterics_lambda'),
                      ('2+lambda-lambda_angles', '2+_AlchemicalFunction__lambda-lambda_angles'),
                      ('2+lambda-lambda_angles/lambda',
                       '2+_AlchemicalFunction__lambda-lambda_angles/_AlchemicalFunction__lambda')]
        for expression, result in test_cases:
            substituted_expression = AlchemicalFunction._sanitize_expression(expression)
            assert substituted_expression == result, '{}, {}, {}'.format(expression, substituted_expression, result)

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

        # Valid parameters are 1.0 by default in AbsoluteAlchemicalFactory,
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

        # Check that standardize_system() sets all parameters back to 1.0.
        AlchemicalState.standardize_system(system)
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
# TEST SYSTEM DEFINITIONS
# =============================================================================

accuracy_testsystem_names = [
    'Lennard-Jones cluster',
    'Lennard-Jones cluster numpy atom set',
    'Lennard-Jones fluid without dispersion correction',
    'Lennard-Jones fluid with dispersion correction',
    'TIP3P with reaction field, no charges, no switch, no dispersion correction',
    'TIP3P with reaction field, switch, no dispersion correction',
    'TIP3P with reaction field, switch, dispersion correction',
    # 'alanine dipeptide in vacuum with annihilated sterics',  TODO temporarily deactivated until openmm#1554 gets fixed
    'toluene in implicit solvent',
]

overlap_testsystem_names = [
    'HostGuest in explicit solvent with PME',
    'TIP3P with PME, no switch, no dispersion correction', # PME still lacks reciprocal space component; known energy comparison failure
    'Lennard-Jones cluster',
    'Lennard-Jones fluid without dispersion correction',
    'Lennard-Jones fluid with dispersion correction',
    'TIP3P with reaction field, no charges, no switch, no dispersion correction',
    'TIP3P with reaction field, switch, no dispersion correction',
    'TIP3P with reaction field, switch, dispersion correction',
    'alanine dipeptide in vacuum with annihilated sterics',
    'toluene in implicit solvent',
]

overlap_testsystem_names = [
    'HostGuest in explicit solvent with PME',
    'TIP3P with PME, no switch, no dispersion correction',  # PME still lacks reciprocal space component; known energy comparison failure
]

test_systems = dict()

# Generate host-guest test systems combinatorially.
for nonbonded_method in [openmm.app.CutoffPeriodic, openmm.app.PME]:
    nonbonded_treatment = 'CutoffPeriodic' if (nonbonded_method == openmm.app.CutoffPeriodic) else 'PME'
    for annihilate_sterics in [False, True]:
        sterics_treatment = 'annihilated' if annihilate_sterics else 'decoupled'
        for annihilate_electrostatics in [False, True]:
            electrostatics_treatment = 'annihilated' if annihilate_electrostatics else 'decoupled'
            name = 'host-guest system in explicit solvent with %s using %s sterics and %s electrostatics' % (nonbonded_treatment, sterics_treatment, electrostatics_treatment)
            test_systems[name] = {
                'test': testsystems.HostGuestExplicit(nonbondedMethod=nonbonded_method),
                'factory_args': {'ligand_atoms': range(126, 156), 'receptor_atoms': range(0, 126),
                                 'annihilate_sterics': annihilate_sterics,
                                 'annihilate_electrostatics': annihilate_electrostatics}}
            accuracy_testsystem_names.append(name)

test_systems['Lennard-Jones cluster'] = {
    'test': testsystems.LennardJonesCluster(),
    'factory_args': {'ligand_atoms': range(0, 1), 'receptor_atoms': range(1, 2)}}
test_systems['Lennard-Jones cluster numpy atom set'] = {
    'test': testsystems.LennardJonesCluster(),
    'factory_args': {'ligand_atoms': np.array(range(0, 1), np.int64),
                     'receptor_atoms': np.array(range(1, 2), np.int32)}}
test_systems['Lennard-Jones cluster with modified softcore parameters'] = {
    'test': testsystems.LennardJonesCluster(),
    'factory_args': {'ligand_atoms': range(0, 1), 'receptor_atoms': range(1, 2),
                     'softcore_alpha': 1, 'softcore_beta': 1, 'softcore_a': 2, 'softcore_b': 2,
                     'softcore_c': 2, 'softcore_d': 2, 'softcore_e': 2, 'softcore_f': 2}}
test_systems['Lennard-Jones fluid without dispersion correction'] = {
    'test': testsystems.LennardJonesFluid(dispersion_correction=False),
    'factory_args': {'ligand_atoms': range(0, 1), 'receptor_atoms': range(1, 2)}}
test_systems['Lennard-Jones fluid with dispersion correction'] = {
    'test': testsystems.LennardJonesFluid(dispersion_correction=True),
    'factory_args': {'ligand_atoms': range(0, 1), 'receptor_atoms': range(1, 2)}}
test_systems['TIP3P with reaction field, no charges, no switch, no dispersion correction'] = {
    'test': testsystems.DischargedWaterBox(dispersion_correction=False, switch=False,
                                           nonbondedMethod=app.CutoffPeriodic),
    'factory_args': {'ligand_atoms': range(0, 3), 'receptor_atoms': range(3, 6)}}
test_systems['TIP3P with reaction field, switch, no dispersion correction'] = {
    'test': testsystems.WaterBox(dispersion_correction=False, switch=True, nonbondedMethod=app.CutoffPeriodic),
    'factory_args': {'ligand_atoms': range(0, 3), 'receptor_atoms': range(3, 6)}}
test_systems['TIP3P with reaction field, no switch, dispersion correction'] = {
    'test': testsystems.WaterBox(dispersion_correction=True, switch=False, nonbondedMethod=app.CutoffPeriodic),
    'factory_args': {'ligand_atoms': range(0, 3), 'receptor_atoms': range(3, 6)}}
test_systems['TIP3P with reaction field, no switch, dispersion correction, no alchemical atoms'] = {
    'test': testsystems.WaterBox(dispersion_correction=True, switch=False, nonbondedMethod=app.CutoffPeriodic),
    'factory_args': {'ligand_atoms': [], 'receptor_atoms': []}}
test_systems['TIP3P with reaction field, switch, dispersion correction'] = {
    'test': testsystems.WaterBox(dispersion_correction=True, switch=True, nonbondedMethod=app.CutoffPeriodic),
    'factory_args': {'ligand_atoms': range(0, 3), 'receptor_atoms': range(3, 6)}}
test_systems['TIP3P with reaction field, switch, dispersion correction, no alchemical atoms'] = {
    'test': testsystems.WaterBox(dispersion_correction=True, switch=True, nonbondedMethod=app.CutoffPeriodic),
    'factory_args': {'ligand_atoms': [], 'receptor_atoms': []}}
test_systems['TIP3P with reaction field, switch, dispersion correctionm, electrostatics scaling followed by softcore Lennard-Jones'] = {
    'test': testsystems.WaterBox(dispersion_correction=True, switch=True, nonbondedMethod=app.CutoffPeriodic),
    'factory_args': {'ligand_atoms': range(0, 3), 'receptor_atoms': range(3, 6), 'softcore_beta': 0.0,
                     'alchemical_functions': { 'lambda_sterics': '2*lambda * step(0.5 - lambda)',
                                               'lambda_electrostatics': '2*(lambda - 0.5) * step(lambda - 0.5)'}}}
test_systems['alanine dipeptide in vacuum'] = {
    'test': testsystems.AlanineDipeptideVacuum(),
    'factory_args': {'ligand_atoms': range(0, 22), 'receptor_atoms': range(22, 22)}}
test_systems['alanine dipeptide in vacuum with annihilated bonds, angles, and torsions'] = {
    'test': testsystems.AlanineDipeptideVacuum(),
    'factory_args': {'ligand_atoms': range(0, 22), 'receptor_atoms': range(22, 22),
    'alchemical_torsions': True, 'alchemical_angles': True, 'alchemical_bonds': True}}
test_systems['alanine dipeptide in vacuum with annihilated sterics'] = {
    'test': testsystems.AlanineDipeptideVacuum(),
    'factory_args': {'ligand_atoms': range(0, 22), 'receptor_atoms': range(22, 22),
    'annihilate_sterics': True, 'annihilate_electrostatics': True}}
test_systems['alanine dipeptide in OBC GBSA'] = {
    'test': testsystems.AlanineDipeptideImplicit(),
    'factory_args': {'ligand_atoms': range(0, 22), 'receptor_atoms': range(22, 22)}}
test_systems['alanine dipeptide in OBC GBSA, with sterics annihilated'] = {
    'test': testsystems.AlanineDipeptideImplicit(),
    'factory_args': {'ligand_atoms': range(0, 22), 'receptor_atoms': range(22, 22),
    'annihilate_sterics': True, 'annihilate_electrostatics': True}}
test_systems['alanine dipeptide in TIP3P with reaction field'] = {
    'test': testsystems.AlanineDipeptideExplicit(nonbondedMethod=app.CutoffPeriodic),
    'factory_args': {'ligand_atoms': range(0, 22), 'receptor_atoms': range(22, 22)}}
test_systems['T4 lysozyme L99A with p-xylene in OBC GBSA'] = {
    'test': testsystems.LysozymeImplicit(),
    'factory_args': {'ligand_atoms': range(2603, 2621), 'receptor_atoms': range(0, 2603)}}
test_systems['DHFR in explicit solvent with reaction field, annihilated'] = {
    'test': testsystems.DHFRExplicit(nonbondedMethod=app.CutoffPeriodic),
    'factory_args': {'ligand_atoms': range(0, 2849), 'receptor_atoms': [],
    'annihilate_sterics': True, 'annihilate_electrostatics': True}}
test_systems['Src in TIP3P with reaction field, with Src sterics annihilated'] = {
    'test': testsystems.SrcExplicit(nonbondedMethod=app.CutoffPeriodic),
    'factory_args': {'ligand_atoms': range(0, 4428), 'receptor_atoms': [],
    'annihilate_sterics': True, 'annihilate_electrostatics': True}}
test_systems['Src in GBSA'] = {
    'test': testsystems.SrcImplicit(),
    'factory_args': {'ligand_atoms': range(0, 4427), 'receptor_atoms': [],
    'annihilate_sterics': False, 'annihilate_electrostatics': False}}
test_systems['Src in GBSA, with Src sterics annihilated'] = {
    'test': testsystems.SrcImplicit(),
    'factory_args': {'ligand_atoms': range(0, 4427), 'receptor_atoms': [],
    'annihilate_sterics': True, 'annihilate_electrostatics': True}}

# Problematic tests: PME is not fully implemented yet
test_systems['TIP3P with PME, no switch, no dispersion correction'] = {
    'test': testsystems.WaterBox(dispersion_correction=False, switch=False, nonbondedMethod=app.PME),
    'factory_args': {'ligand_atoms': range(0, 3), 'receptor_atoms': range(3, 6)}}
test_systems['TIP3P with PME, no switch, no dispersion correction, no alchemical atoms'] = {
    'test': testsystems.WaterBox(dispersion_correction=False, switch=False, nonbondedMethod=app.PME),
    'factory_args': {'ligand_atoms': [], 'receptor_atoms': []}}

test_systems['HostGuest in explicit solvent with PME'] = {
    'test': testsystems.HostGuestExplicit(nonbondedCutoff=9.0*unit.angstroms, use_dispersion_correction=True,
                                          nonbondedMethod=app.PME, switch_width=1.5*unit.angstroms,
                                          ewaldErrorTolerance=1.0e-6),
    'factory_args': {'ligand_atoms': range(126, 156), 'receptor_atoms': range(0, 126)}}

test_systems['toluene in implicit solvent'] = {
    'test': testsystems.TolueneImplicit(),
    'factory_args': {'ligand_atoms': [0, 1], 'receptor_atoms': list(),
                     'alchemical_torsions': True, 'alchemical_angles': True, 'annihilate_sterics': True,
                     'annihilate_electrostatics': True}}

# Slow tests
# test_systems['Src in OBC GBSA'] = {
#     'test': testsystems.SrcImplicit(),
#     'ligand_atoms': range(0, 21), 'receptor_atoms': range(21,7208)}
# test_systems['Src in TIP3P with reaction field'] = {
#     'test': testsystems.SrcExplicit(nonbondedMethod=app.CutoffPeriodic),
#     'ligand_atoms': range(0, 21), 'receptor_atoms': range(21,4091)}


# =============================================================================
# Test various options to AbsoluteAlchemicalFactory
# =============================================================================

def test_alchemical_functions():
    """
    Testing alchemical slave functions
    """
    alchemical_functions = {'lambda_sterics': 'lambda', 'lambda_electrostatics': 'lambda',
                             'lambda_bonds': 'lambda', 'lambda_angles': 'lambda', 'lambda_torsions': 'lambda'}
    name = 'Lennard-Jones fluid with dispersion correction'
    test_system = copy.deepcopy(test_systems[name])
    reference_system = test_system['test'].system
    positions = test_system['test'].positions
    factory_args = test_system['factory_args']
    factory_args['alchemical_functions'] = alchemical_functions
    factory = AbsoluteAlchemicalFactory(reference_system, **factory_args)
    alchemical_system = factory.createPerturbedSystem()
    compareSystemEnergies(positions, [reference_system, alchemical_system], ['reference', 'alchemical'])


def test_softcore_parameters():
    """
    Testing softcore parameters
    """
    name = 'Lennard-Jones fluid with dispersion correction'
    test_system = copy.deepcopy(test_systems[name])
    reference_system = test_system['test'].system
    positions = test_system['test'].positions
    factory_args = test_system['factory_args']
    factory_args.update({'softcore_alpha': 1.0, 'softcore_beta': 1.0, 'softcore_a': 1.0, 'softcore_b': 1.0,
                         'softcore_c': 1.0, 'softcore_d': 1.0, 'softcore_e': 1.0, 'softcore_f': 1.0})
    factory = AbsoluteAlchemicalFactory(reference_system, **factory_args)
    alchemical_system = factory.createPerturbedSystem()
    compareSystemEnergies(positions, [reference_system, alchemical_system], ['reference', 'alchemical'])


# =============================================================================
# NOSETEST GENERATORS
# =============================================================================

@attr('slow')
def test_overlap():
    """
    Generate nose tests for overlap for all alchemical test systems.
    """
    for name in overlap_testsystem_names:
        test_system = test_systems[name]
        reference_system = test_system['test'].system
        positions = test_system['test'].positions
        factory_args = test_system['factory_args']
        cached_trajectory_filename = os.path.join(os.environ['HOME'], '.cache', 'alchemy', 'tests', name + '.nc')
        f = partial(overlap_check, reference_system, positions, factory_args=factory_args,
                    cached_trajectory_filename=cached_trajectory_filename)
        f.description = "Testing reference/alchemical overlap for %s..." % name
        yield f


# TODO REMOVE slow attribute when openmm#1588 gets fixed
@attr('slow')
def test_alchemical_accuracy():
    """
    Generate nose tests for overlap for all alchemical test systems.
    """
    for name in accuracy_testsystem_names:
        test_system = test_systems[name]
        reference_system = test_system['test'].system
        positions = test_system['test'].positions
        factory_args = test_system['factory_args']
        f = partial(alchemical_factory_check, reference_system, positions, factory_args=factory_args)
        f.description = "Testing alchemical fidelity of %s..." % name
        yield f


def test_platforms():
    """
    Generate nosetests for comparing platform energies...
    """
    for name in accuracy_testsystem_names:
        test_system = test_systems[name]
        reference_system = test_system['test'].system
        positions = test_system['test'].positions
        factory_args = test_system['factory_args']
        f = partial(compare_platforms, reference_system, positions, factory_args=factory_args)
        f.description = "Comparing platforms for alchemically-modified forms of %s..." % name
        yield f


# =============================================================================
# MAIN FOR MANUAL DEBUGGING
# =============================================================================

if __name__ == "__main__":
    # generate_trace(test_systems['TIP3P with reaction field, switch, dispersion correction'])
    config_root_logger(True)

    logging.basicConfig(level=logging.INFO)
    test_alchemical_accuracy()
