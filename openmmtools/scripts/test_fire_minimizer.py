#!/usr/local/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Test the FIRE minimizer against the OpenMM minmizer.

DESCRIPTION

COPYRIGHT

@author John D. Chodera <jchodera@gmail.com>

All code in this repository is released under the GNU General Public License.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <http://www.gnu.org/licenses/>.

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

from simtk import openmm, unit

from openmmtools import testsystems

#=============================================================================================
# SUBROUTINES
#=============================================================================================

# These settings control what tolerance is allowed between platforms and the Reference platform.
ENERGY_TOLERANCE = 0.06*unit.kilocalories_per_mole # energy difference tolerance
FORCE_RMSE_TOLERANCE = 0.06*unit.kilocalories_per_mole/unit.angstrom # per-particle force root-mean-square error tolerance

def assert_approximately_equal(computed_potential, expected_potential, tolerance=ENERGY_TOLERANCE):
    """
    Check whether computed potential is acceptably close to expected value, using an error tolerance.

    ARGUMENTS

    computed_potential (simtk.unit.Quantity in units of energy) - computed potential energy
    expected_potential (simtk.unit.Quantity in units of energy) - expected

    OPTIONAL ARGUMENTS

    tolerance (simtk.unit.Quantity in units of energy) - acceptable tolerance

    EXAMPLES

    >>> assert_approximately_equal(0.0000 * unit.kilocalories_per_mole, 0.0001 * unit.kilocalories_per_mole, tolerance=0.06*unit.kilocalories_per_mole)

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
    kB = unit.BOLTZMANN_CONSTANT_kB
    temperature = 298.0 * unit.kelvin
    kT = kB * temperature
    beta = 1.0 / kT
    collision_rate = 90.0 / unit.picosecond
    timestep = 1.0 * unit.femtosecond
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
    kB = unit.BOLTZMANN_CONSTANT_kB
    temperature = 298.0 * unit.kelvin
    kT = kB * temperature
    beta = 1.0 / kT
    collision_rate = 90.0 / unit.picosecond
    timestep = 1.0 * unit.femtosecond
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
    kB = unit.BOLTZMANN_CONSTANT_kB
    temperature = 298.0 * unit.kelvin
    kT = kB * temperature
    beta = 1.0 / kT
    collision_rate = 90.0 / unit.picosecond
    timestep = 1.0 * unit.femtosecond
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

    parser = argparse.ArgumentParser(description="Benchmark the FIRE minimizer against the OpenMM minimizer for a suite of test systems.")
    parser.add_argument('-o', '--outfile', dest='logfile', action='store', type=str, default=None)
    parser.add_argument('-v', dest='verbose', action='store_true')
    parser.add_argument('-p', '--platform', dest='platformname', action='store', type=str, default=None)
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

    # Select specified platform.
    platform = None
    if args.platformname:
        logger.info("Using platform '%s'..." % args.platformname)
        platform = openmm.Platform.getPlatformByName(args.platformname)

    # Test performance on a variety of testsystems.
    tests_failed = 0 # number of times tolerance is exceeded
    tests_passed = 0 # number of times tolerance is not exceeded
    #logger.info("%16s%16s %16s          %16s          %16s          %16s" % ("platform", "precision", "potential", "error", "force mag", "rms error"))
    testsystem_classes = get_all_subclasses(testsystems.TestSystem)
    testsystem_classes = [getattr(testsystems, name) for name in ('LysozymeImplicit', 'AMOEBAProteinBox', 'SrcImplicit', 'SrcExplicit')]
    for testsystem_class in testsystem_classes:
        class_name = testsystem_class.__name__
        logger.info(class_name)

        try:
            testsystem = testsystem_class()
        except ImportError as e:
            logger.info(e)
            logger.info("Skipping %s due to missing dependency" % class_name)
            continue

        # Create test system instance.
        [system, initial_positions] = [testsystem.system, testsystem.positions]

        logger.info("%s (%d atoms)" % (class_name, testsystem.system.getNumParticles()))

        maxits = 100
        forcetol = 1.0 * unit.kilojoules_per_mole / unit.angstrom

        # Create integrator.
        from openmmtools import integrators
        logger.info('Creating FIRE integrator...')
        integrator = integrators.FIREMinimizationIntegrator(tolerance=forcetol)
        logger.info('Done.')

        # Build list of parameters.
        #global_variables = { integrator.getGlobalVariableName(index) : index for index in range(integrator.getNumGlobalVariables()) }

        # Create context.
        if platform:
            context = openmm.Context(system, integrator, platform)
        else:
            context = openmm.Context(system, integrator)

        # Minimize with FIRE minimizer
        import time
        context.setPositions(initial_positions)
        fire_initial_energy = context.getState(getEnergy=True).getPotentialEnergy() / unit.kilocalories_per_mole
        initial_time = time.time()
        integrator.step(maxits)
        final_time = time.time()
        elapsed_time = final_time - initial_time
        fire_final_energy = context.getState(getEnergy=True).getPotentialEnergy() / unit.kilocalories_per_mole
        logger.info("FIRE                 %8.3f s" % elapsed_time)
        logger.info("initial energy = %12.3f kcal/mol" % fire_initial_energy)
        logger.info("final energy   = %12.3f kcal/mol" % fire_final_energy)
        #logger.info("converged = %f" % integrator.getGlobalVariable(global_variables['converged']))

        # Time LocalEnergyMinimizer
        from simtk.openmm import LocalEnergyMinimizer
        context.setPositions(initial_positions)
        fire_initial_energy = context.getState(getEnergy=True).getPotentialEnergy() / unit.kilocalories_per_mole
        initial_time = time.time()
        LocalEnergyMinimizer.minimize(context, forcetol, maxits)
        final_time = time.time()
        elapsed_time = final_time - initial_time
        fire_final_energy = context.getState(getEnergy=True).getPotentialEnergy() / unit.kilocalories_per_mole
        logger.info("LocalEnergyMinimizer %8.3f s" % elapsed_time)
        logger.info("initial energy = %12.3f kcal/mol" % fire_initial_energy)
        logger.info("final energy   = %12.3f kcal/mol" % fire_final_energy)


        # Clean up
        logger.info("Cleaning up.")
        del context
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
