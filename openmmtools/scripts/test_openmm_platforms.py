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

    debug = False # Don't display extra debug information.

    # List all available platforms
    print("Available platforms:")
    for platform_index in range(openmm.Platform.getNumPlatforms()):
        platform = openmm.Platform.getPlatform(platform_index)
        print("%5d %s" % (platform_index, platform.getName()))
    print("")

    # Test all systems on Reference platform.
    platform = openmm.Platform.getPlatformByName("Reference")
    print('Testing Reference platform...')
    doctest.testmod()

    # Compute energy error made on all test systems for other platforms.
    # Make a count of how often set tolerance is exceeded.
    tests_failed = 0 # number of times tolerance is exceeded
    tests_passed = 0 # number of times tolerance is not exceeded
    print("%16s%16s %16s          %16s          %16s          %16s" % ("platform", "precision", "potential", "error", "force mag", "rms error"))
    reference_platform = openmm.Platform.getPlatformByName("Reference")
    testsystem_classes = get_all_subclasses(testsystems.TestSystem)
    for testsystem_class in testsystem_classes:
        class_name = testsystem_class.__name__

        try:
            testsystem = testsystem_class()
        except ImportError as e:
            print(e)
            print("Skipping %s due to missing dependency" % class_name)
            continue

        print("%s" % class_name)

        testsystem = testsystem_class()
        [system, positions] = [testsystem.system, testsystem.positions]
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

                    print("%16s%16s %16.6f kcal/mol %16.6f kcal/mol %16.6f kcal/mol/nm %16.6f kcal/mol/nm" % (platform_name, precision_model, platform_potential / units.kilocalories_per_mole, potential_error / units.kilocalories_per_mole, force_rms / force_unit, force_rmse / force_unit))

                    # Mark whether tolerance is exceeded or not.
                    if abs(potential_error) > ENERGY_TOLERANCE:
                        test_success = False
                        print("%32s WARNING: Potential energy error (%.6f kcal/mol) exceeds tolerance (%.6f kcal/mol).  Test failed." % ("", potential_error/units.kilocalories_per_mole, ENERGY_TOLERANCE/units.kilocalories_per_mole))
                    if abs(force_rmse) > FORCE_RMSE_TOLERANCE:
                        test_success = False
                        print("%32s WARNING: Force RMS error (%.6f kcal/mol/nm) exceeds tolerance (%.6f kcal/mol/nm).  Test failed." % ("", force_rmse/force_unit, FORCE_RMSE_TOLERANCE/force_unit))
                        if debug:
                            for atom_index in range(natoms):
                                for k in range(3):
                                    print("%12.6f" % (reference_force[atom_index,k]/force_unit), end="")
                                print(" : ", end="")
                                for k in range(3):
                                    print("%12.6f" % (platform_force[atom_index,k]/force_unit), end="")
            except Exception as e:
                print(e)

        if test_success:
            tests_passed += 1
        else:
            tests_failed += 1

        if (test_success is False):
            # Write XML files of failed tests to aid in debugging.
            print("Writing failed test system to '%s'.{system,state}.xml ..." % testsystem.name)
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
            print("Breakdown of discrepancies by Force component:")
            nforces = system.getNumForces()
            for force_group in range(ngroups):
                force_name = force_group_names[force_group]
                print(force_name)
                [reference_potential, reference_force] = compute_potential_and_force_by_force_group(system, positions, reference_platform, force_group)
                print("%16s%16s %16s          %16s          %16s          %16s" % ("platform", "precision", "potential", "error", "force mag", "rms error"))

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

                            print("%16s%16s %16.6f kcal/mol %16.6f kcal/mol %16.6f kcal/mol/nm %16.6f kcal/mol/nm" % (platform_name, precision_model, platform_potential / units.kilocalories_per_mole, potential_error / units.kilocalories_per_mole, force_rms / force_unit, force_rmse / force_unit))

                    except Exception as e:
                        print(e)
                        pass
        print("")

    print("%d tests failed" % tests_failed)
    print("%d tests passed" % tests_passed)

    if (tests_failed > 0):
        # Signal failure of test.
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
