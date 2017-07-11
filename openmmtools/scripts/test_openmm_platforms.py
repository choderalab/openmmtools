#!/usr/local/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Test all test systems on different platforms to ensure differences in potential energy and
forces are small among platforms.

DESCRIPTION

COPYRIGHT

@author John D. Chodera <jchodera@gmail.com>

All code in this repository is released under the MIT License.

This program is free software: you can redistribute it and/or modify it under
the terms of the MIT License.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the MIT License for more details.

You should have received a copy of the MIT License along with this program.

TODO

"""

#=============================================================================================
# PYTHON 3 COMPATIBILITY CRAP
#=============================================================================================

from __future__ import print_function

#=============================================================================================
# ENABLE LOGGING
#=============================================================================================

import logging
logger = logging.getLogger(__name__)

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
        simple_fmt = logging.Formatter('%(message)s')
        default_fmt = logging.Formatter('%(levelname)s - %(name)s - %(message)s')

        def format(self, record):
            if record.levelno <= logging.INFO:
                return self.simple_fmt.format(record)
            else:
                return self.default_fmt.format(record)

    # Check if root logger is already configured
    n_handlers = len(logging.root.handlers)
    if n_handlers > 0:
        root_logger = logging.root
        for i in xrange(n_handlers):
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
        #file_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
        file_format = '%(asctime)s: %(message)s'
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(file_format))
        logging.root.addHandler(file_handler)

    # Do not handle logging.DEBUG at all if unnecessary
    if log_file_path is not None:
        logging.root.setLevel(logging.DEBUG)
    else:
        logging.root.setLevel(terminal_handler.level)

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import os
import os.path
import sys
import math

import simtk.unit as units
import simtk.openmm as openmm

from openmmtools import testsystems

#=============================================================================================
# SUBROUTINES
#=============================================================================================

# These settings control what tolerance is allowed between platforms and the Reference platform.
ENERGY_TOLERANCE = 0.06*units.kilocalories_per_mole # energy difference tolerance
FORCE_RMSE_TOLERANCE = 0.06*units.kilocalories_per_mole/units.angstrom # per-particle force root-mean-square error tolerance

def assert_approximately_equal(computed_potential, expected_potential, tolerance=ENERGY_TOLERANCE):
    """
    Check whether computed potential is acceptably close to expected value, using an error tolerance.

    ARGUMENTS

    computed_potential (simtk.unit.Quantity in units of energy) - computed potential energy
    expected_potential (simtk.unit.Quantity in units of energy) - expected

    OPTIONAL ARGUMENTS

    tolerance (simtk.unit.Quantity in units of energy) - acceptable tolerance

    EXAMPLES

    >>> assert_approximately_equal(0.0000 * units.kilocalories_per_mole, 0.0001 * units.kilocalories_per_mole, tolerance=0.06*units.kilocalories_per_mole)

    """

    # Compute error.
    error = (computed_potential - expected_potential)

    # Raise an exception if the error is larger than the tolerance.
    if abs(error) > tolerance:
        raise Exception("Computed potential %s, expected %s.  Error %s is larger than acceptable tolerance of %s." % (computed_potential, expected_potential, error, tolerance))

    return

def compute_potential_and_force(system, positions, platform):
    """
    Compute the energy and force for the given system and positions in the designated platform.

    ARGUMENTS

    system (simtk.openmm.System) - the system for which the energy is to be computed
    positions (simtk.unit.Quantity of Nx3 numpy.array in units of distance) - positions for which energy and force are to be computed
    platform (simtk.openmm.Platform) - platform object to be used to compute the energy and force

    RETURNS

    potential (simtk.unit.Quantity in energy/mole) - the potential
    force (simtk.unit.Quantity of Nx3 numpy.array in units of energy/mole/distance) - the force

    """

    # Create a Context.
    kB = units.BOLTZMANN_CONSTANT_kB
    temperature = 298.0 * units.kelvin
    kT = kB * temperature
    beta = 1.0 / kT
    collision_rate = 90.0 / units.picosecond
    timestep = 1.0 * units.femtosecond
    integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
    context = openmm.Context(system, integrator, platform)
    # Set positions
    context.setPositions(positions)
    # Evaluate the potential energy.
    state = context.getState(getEnergy=True, getForces=True)
    potential = state.getPotentialEnergy()
    force = state.getForces(asNumpy=True)

    return [potential, force]

def compute_potential_and_force_by_force_index(system, positions, platform, force_index):
    """
    Compute the energy and force for the given system and positions in the designated platform for the given force index.

    ARGUMENTS

    system (simtk.openmm.System) - the system for which the energy is to be computed
    positions (simtk.unit.Quantity of Nx3 numpy.array in units of distance) - positions for which energy and force are to be computed
    platform (simtk.openmm.Platform) - platform object to be used to compute the energy and force
    force_index (int) - index of force to be computed (all others ignored)

    RETURNS

    potential (simtk.unit.Quantity in energy/mole) - the potential
    force (simtk.unit.Quantity of Nx3 numpy.array in units of energy/mole/distance) - the force

    """

    forces = [ system.getForce(index) for index in range(system.getNumForces()) ]

    # Get original force groups.
    groups = [ force.getForceGroup() for force in forces ]

    # Set force groups so only specified force_index contributes.
    for force in forces:
        force.setForceGroup(1)
    forces[force_index].setForceGroup(0) # bitmask of 1 should select only desired force

    # Create a Context.
    kB = units.BOLTZMANN_CONSTANT_kB
    temperature = 298.0 * units.kelvin
    kT = kB * temperature
    beta = 1.0 / kT
    collision_rate = 90.0 / units.picosecond
    timestep = 1.0 * units.femtosecond
    integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
    context = openmm.Context(system, integrator, platform)
    # Set positions
    context.setPositions(positions)
    # Evaluate the potential energy.
    state = context.getState(getEnergy=True, getForces=True, groups=1)
    potential = state.getPotentialEnergy()
    force = state.getForces(asNumpy=True)

    # Restore original force groups.
    for index in range(system.getNumForces()):
        forces[index].setForceGroup(groups[index])

    return [potential, force]

def compute_potential_and_force_by_force_group(system, positions, platform, force_group):
    """
    Compute the energy and force for the given system and positions in the designated platform for the given force group.

    ARGUMENTS

    system (simtk.openmm.System) - the system for which the energy is to be computed
    positions (simtk.unit.Quantity of Nx3 numpy.array in units of distance) - positions for which energy and force are to be computed
    platform (simtk.openmm.Platform) - platform object to be used to compute the energy and force
    force_group (int) - index of force group to be computed (all others ignored)

    RETURNS

    potential (simtk.unit.Quantity in energy/mole) - the potential
    force (simtk.unit.Quantity of Nx3 numpy.array in units of energy/mole/distance) - the force

    """

    forces = [ system.getForce(index) for index in range(system.getNumForces()) ]

    # Create a Context.
    kB = units.BOLTZMANN_CONSTANT_kB
    temperature = 298.0 * units.kelvin
    kT = kB * temperature
    beta = 1.0 / kT
    collision_rate = 90.0 / units.picosecond
    timestep = 1.0 * units.femtosecond
    integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
    context = openmm.Context(system, integrator, platform)
    # Set positions
    context.setPositions(positions)
    # Evaluate the potential energy.
    groupmask = 1 << (force_group + 1)
    state = context.getState(getEnergy=True, getForces=True, groups=groupmask)
    potential = state.getPotentialEnergy()
    force = state.getForces(asNumpy=True)

    return [potential, force]

def get_all_subclasses(cls):
    """
    Return all subclasses of a specified class.

    Parameters
    ----------
    cls : class
       The class for which all subclasses are to be returned.

    Returns
    -------
    all_subclasses : list of class
       List of all subclasses of `cls`.

    """
       
    all_subclasses = []

    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))

    return all_subclasses

#=============================================================================================
# MAIN AND TESTS
#=============================================================================================

def main():
    import doctest
    import argparse

    parser = argparse.ArgumentParser(description="Check OpenMM computed energies and forces across all platforms for a suite of test systems.")
    parser.add_argument('-o', '--outfile', dest='logfile', action='store', type=str, default=None)
    parser.add_argument('-v', dest='verbose', action='store_true')
    args = parser.parse_args()

    verbose = args.verbose # Don't display extra debug information.
    config_root_logger(verbose, log_file_path=args.logfile)

    # Print version.
    logger.info("OpenMM version: %s" % openmm.version.version)
    logger.info("")

    # List all available platforms
    logger.info("Available platforms:")
    for platform_index in range(openmm.Platform.getNumPlatforms()):
        platform = openmm.Platform.getPlatform(platform_index)
        logger.info("%5d %s" % (platform_index, platform.getName()))
    logger.info("")

    # Test all systems on Reference platform.
    platform = openmm.Platform.getPlatformByName("Reference")
    print('Testing Reference platform...')
    doctest.testmod()

    # Compute energy error made on all test systems for other platforms.
    # Make a count of how often set tolerance is exceeded.
    tests_failed = 0 # number of times tolerance is exceeded
    tests_passed = 0 # number of times tolerance is not exceeded
    logger.info("%16s%16s %16s          %16s          %16s          %16s" % ("platform", "precision", "potential", "error", "force mag", "rms error"))
    reference_platform = openmm.Platform.getPlatformByName("Reference")
    testsystem_classes = get_all_subclasses(testsystems.TestSystem)
    for testsystem_class in testsystem_classes:
        class_name = testsystem_class.__name__

        try:
            testsystem = testsystem_class()
        except ImportError as e:
            logger.info(e)
            logger.info("Skipping %s due to missing dependency" % class_name)
            continue

        # Create test system instance.
        testsystem = testsystem_class()
        [system, positions] = [testsystem.system, testsystem.positions]

        logger.info("%s (%d atoms)" % (class_name, testsystem.system.getNumParticles()))

        # Compute reference potential and force
        [reference_potential, reference_force] = compute_potential_and_force(system, positions, reference_platform)

        # Test all platforms.
        test_success = True
        for platform_index in range(openmm.Platform.getNumPlatforms()):
            try:
                platform = openmm.Platform.getPlatform(platform_index)
                platform_name = platform.getName()

                # Define precision models to test.
                if platform_name == 'Reference':
                    precision_models = ['double']
                else:
                    precision_models = ['single']
                    if platform.supportsDoublePrecision():
                        precision_models.append('double')

                for precision_model in precision_models:
                    # Set precision.
                    if platform_name == 'CUDA':
                        platform.setPropertyDefaultValue('CudaPrecision', precision_model)
                    if platform_name == 'OpenCL':
                        platform.setPropertyDefaultValue('OpenCLPrecision', precision_model)

                    # Compute potential and force.
                    [platform_potential, platform_force] = compute_potential_and_force(system, positions, platform)

                    # Compute error in potential.
                    potential_error = platform_potential - reference_potential

                    # Compute per-atom RMS (magnitude) and RMS error in force.
                    force_unit = units.kilocalories_per_mole / units.nanometers
                    natoms = system.getNumParticles()
                    force_mse = (((reference_force - platform_force) / force_unit)**2).sum() / natoms * force_unit**2
                    force_rmse = units.sqrt(force_mse)

                    force_ms = ((platform_force / force_unit)**2).sum() / natoms * force_unit**2
                    force_rms = units.sqrt(force_ms)

                    logger.info("%16s%16s %16.6f kcal/mol %16.6f kcal/mol %16.6f kcal/mol/nm %16.6f kcal/mol/nm" % (platform_name, precision_model, platform_potential / units.kilocalories_per_mole, potential_error / units.kilocalories_per_mole, force_rms / force_unit, force_rmse / force_unit))

                    # Mark whether tolerance is exceeded or not.
                    if abs(potential_error) > ENERGY_TOLERANCE:
                        test_success = False
                        logger.info("%32s WARNING: Potential energy error (%.6f kcal/mol) exceeds tolerance (%.6f kcal/mol).  Test failed." % ("", potential_error/units.kilocalories_per_mole, ENERGY_TOLERANCE/units.kilocalories_per_mole))
                    if abs(force_rmse) > FORCE_RMSE_TOLERANCE:
                        test_success = False
                        logger.info("%32s WARNING: Force RMS error (%.6f kcal/mol/nm) exceeds tolerance (%.6f kcal/mol/nm).  Test failed." % ("", force_rmse/force_unit, FORCE_RMSE_TOLERANCE/force_unit))
                        if verbose:
                            for atom_index in range(natoms):
                                for k in range(3):
                                    logger.info("%12.6f" % (reference_force[atom_index,k]/force_unit), end="")
                                logger.info(" : ", end="")
                                for k in range(3):
                                    logger.info("%12.6f" % (platform_force[atom_index,k]/force_unit), end="")
            except Exception as e:
                logger.info(e)

        if test_success:
            tests_passed += 1
        else:
            tests_failed += 1

        if (test_success is False):
            # Write XML files of failed tests to aid in debugging.
            logger.info("Writing failed test system to '%s'.{system,state}.xml ..." % testsystem.name)
            [system_xml, state_xml] = testsystem.serialize()
            xml_file = open(testsystem.name + '.system.xml', 'w')
            xml_file.write(system_xml)
            xml_file.close()
            xml_file = open(testsystem.name + '.state.xml', 'w')
            xml_file.write(state_xml)
            xml_file.close()

            
            # Place forces into different force groups.
            forces = [ system.getForce(force_index) for force_index in range(system.getNumForces()) ]
            force_group_names = dict()
            group_index = 0
            for force_index in range(system.getNumForces()):
                force_name = forces[force_index].__class__.__name__
                if force_name == 'NonbondedForce':
                    forces[force_index].setForceGroup(group_index+1)
                    force_group_names[group_index] = 'NonbondedForce (direct)'
                    group_index += 1
                    forces[force_index].setReciprocalSpaceForceGroup(group_index+1)
                    force_group_names[group_index] = 'NonbondedForce (reciprocal)'
                    group_index += 1
                else:
                    forces[force_index].setForceGroup(group_index+1)
                    force_group_names[group_index] = force_name
                    group_index += 1
            ngroups = len(force_group_names)

            # Test by force group.
            logger.info("Breakdown of discrepancies by Force component:")
            nforces = system.getNumForces()
            for force_group in range(ngroups):
                force_name = force_group_names[force_group]
                logger.info(force_name)
                [reference_potential, reference_force] = compute_potential_and_force_by_force_group(system, positions, reference_platform, force_group)
                logger.info("%16s%16s %16s          %16s          %16s          %16s" % ("platform", "precision", "potential", "error", "force mag", "rms error"))

                for platform_index in range(openmm.Platform.getNumPlatforms()):
                    try:
                        platform = openmm.Platform.getPlatform(platform_index)
                        platform_name = platform.getName()
                        
                        # Define precision models to test.
                        if platform_name == 'Reference':
                            precision_models = ['double']
                        else:
                            precision_models = ['single']
                            if platform.supportsDoublePrecision():
                                precision_models.append('double')

                        for precision_model in precision_models:
                            # Set precision.
                            if platform_name == 'CUDA':
                                platform.setPropertyDefaultValue('CudaPrecision', precision_model)
                            if platform_name == 'OpenCL':
                                platform.setPropertyDefaultValue('OpenCLPrecision', precision_model)
                                
                            # Compute potential and force.
                            [platform_potential, platform_force] = compute_potential_and_force_by_force_group(system, positions, platform, force_group)

                            # Compute error in potential.
                            potential_error = platform_potential - reference_potential

                            # Compute per-atom RMS (magnitude) and RMS error in force.
                            force_unit = units.kilocalories_per_mole / units.nanometers
                            natoms = system.getNumParticles()
                            force_mse = (((reference_force - platform_force) / force_unit)**2).sum() / natoms * force_unit**2
                            force_rmse = units.sqrt(force_mse)

                            force_ms = ((platform_force / force_unit)**2).sum() / natoms * force_unit**2
                            force_rms = units.sqrt(force_ms)

                            logger.info("%16s%16s %16.6f kcal/mol %16.6f kcal/mol %16.6f kcal/mol/nm %16.6f kcal/mol/nm" % (platform_name, precision_model, platform_potential / units.kilocalories_per_mole, potential_error / units.kilocalories_per_mole, force_rms / force_unit, force_rmse / force_unit))

                    except Exception as e:
                        logger.info(e)
                        pass
        logger.info("")

    logger.info("%d tests failed" % tests_failed)
    logger.info("%d tests passed" % tests_passed)

    if (tests_failed > 0):
        # Signal failure of test.
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
