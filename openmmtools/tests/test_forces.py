#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test Force classes in forces.py.

"""

# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import pickle

import nose.tools

from openmmtools import testsystems, states
from openmmtools.forces import *
from openmmtools.forces import _compute_sphere_volume, _compute_harmonic_radius


# =============================================================================
# CONSTANTS
# =============================================================================


# =============================================================================
# TESTING UTILITIES
# =============================================================================

def assert_pickles_equal(object1, object2):
    assert pickle.dumps(object1) == pickle.dumps(object2)


def assert_quantity_almost_equal(object1, object2):
    assert utils.is_quantity_close(object1, object2), '{} != {}'.format(object1, object2)


def assert_equal(*args, **kwargs):
    """Python 2 work-around to be able to yield nose.tools.assert_equal"""
    # TODO: Just yield nose.tools.assert_equal after we have dropped Python2 support.
    nose.tools.assert_equal(*args, **kwargs)


# =============================================================================
# UTILITY FUNCTIONS TESTS
# =============================================================================

def test_find_forces():
    """Generator of tests for the find_forces() utility function."""
    system = testsystems.TolueneVacuum().system

    # Add two CustomBondForces, one is restorable.
    restraint_force = HarmonicRestraintBondForce(spring_constant=1.0*unit.kilojoule_per_mole/unit.angstroms**2,
                                                 restrained_atom_index1=2, restrained_atom_index2=5)
    system.addForce(restraint_force)
    system.addForce(openmm.CustomBondForce('0.0'))

    def assert_forces_equal(found_forces, expected_force_classes):
        # Forces should be ordered by their index.
        assert list(found_forces.keys()) == sorted(found_forces.keys())
        found_forces = {(i, force.__class__) for i, force in found_forces.items()}
        nose.tools.assert_equal(found_forces, set(expected_force_classes))

    # Test find force without including subclasses.
    found_forces = find_forces(system, openmm.CustomBondForce)
    yield assert_forces_equal, found_forces, [(6, openmm.CustomBondForce)]

    # Test find force and include subclasses.
    found_forces = find_forces(system, openmm.CustomBondForce, include_subclasses=True)
    yield assert_forces_equal, found_forces, [(5, HarmonicRestraintBondForce),
                                              (6, openmm.CustomBondForce)]
    found_forces = find_forces(system, RadiallySymmetricRestraintForce, include_subclasses=True)
    yield assert_forces_equal, found_forces, [(5, HarmonicRestraintBondForce)]

    # Test exact name matching.
    found_forces = find_forces(system, 'HarmonicBondForce')
    yield assert_forces_equal, found_forces, [(0, openmm.HarmonicBondForce)]

    # Find all forces containing the word "Harmonic".
    found_forces = find_forces(system, '.*Harmonic.*')
    yield assert_forces_equal, found_forces, [(0, openmm.HarmonicBondForce),
                                              (1, openmm.HarmonicAngleForce),
                                              (5, HarmonicRestraintBondForce)]

    # Find all forces from the name including the subclasses.
    # Test find force and include subclasses.
    found_forces = find_forces(system, 'CustomBond.*', include_subclasses=True)
    yield assert_forces_equal, found_forces, [(5, HarmonicRestraintBondForce),
                                              (6, openmm.CustomBondForce)]

    # With check_multiple=True only one force is returned.
    force_idx, force = find_forces(system, openmm.NonbondedForce, only_one=True)
    yield assert_forces_equal, {force_idx: force}, [(3, openmm.NonbondedForce)]

    # An exception is raised with "only_one" if multiple forces are found.
    yield nose.tools.assert_raises, MultipleForcesError, find_forces, system, 'CustomBondForce', True, True

    # An exception is raised with "only_one" if the force wasn't found.
    yield nose.tools.assert_raises, NoForceFoundError, find_forces, system, 'NonExistentForce', True


# =============================================================================
# RESTRAINTS TESTS
# =============================================================================

class TestRadiallySymmetricRestraints(object):
    """Test radially symmetric receptor-ligand restraint classes."""

    @classmethod
    def setup_class(cls):
        cls.well_radius = 12.0 * unit.angstroms
        cls.spring_constant = 15000.0 * unit.joule/unit.mole/unit.nanometers**2
        cls.restrained_atom_indices1 = [2, 3, 4]
        cls.restrained_atom_indices2 = [10, 11]
        cls.restrained_atom_index1=12
        cls.restrained_atom_index2=2
        cls.custom_parameter_name = 'restraints_parameter'

        cls.restraints = [
            HarmonicRestraintForce(spring_constant=cls.spring_constant,
                                   restrained_atom_indices1=cls.restrained_atom_indices1,
                                   restrained_atom_indices2=cls.restrained_atom_indices2),
            HarmonicRestraintBondForce(spring_constant=cls.spring_constant,
                                       restrained_atom_index1=cls.restrained_atom_index1,
                                       restrained_atom_index2=cls.restrained_atom_index2),
            FlatBottomRestraintForce(spring_constant=cls.spring_constant, well_radius=cls.well_radius,
                                     restrained_atom_indices1=cls.restrained_atom_indices1,
                                     restrained_atom_indices2=cls.restrained_atom_indices2),
            FlatBottomRestraintBondForce(spring_constant=cls.spring_constant, well_radius=cls.well_radius,
                                         restrained_atom_index1=cls.restrained_atom_index1,
                                         restrained_atom_index2=cls.restrained_atom_index2),
            HarmonicRestraintForce(spring_constant=cls.spring_constant,
                                   restrained_atom_indices1=cls.restrained_atom_indices1,
                                   restrained_atom_indices2=cls.restrained_atom_indices2,
                                   controlling_parameter_name=cls.custom_parameter_name),
            FlatBottomRestraintBondForce(spring_constant=cls.spring_constant, well_radius=cls.well_radius,
                                         restrained_atom_index1=cls.restrained_atom_index1,
                                         restrained_atom_index2=cls.restrained_atom_index2,
                                         controlling_parameter_name=cls.custom_parameter_name),
        ]

    def test_restorable_forces(self):
        """Test that the restraint interface can be restored after serialization."""
        for restorable_force in self.restraints:
            force_serialization = openmm.XmlSerializer.serialize(restorable_force)
            deserialized_force = utils.RestorableOpenMMObject.deserialize_xml(force_serialization)
            yield assert_pickles_equal, restorable_force, deserialized_force

    def test_restraint_properties(self):
        """Test that properties work as expected."""
        for restraint in self.restraints:
            yield assert_quantity_almost_equal, restraint.spring_constant, self.spring_constant
            if isinstance(restraint, FlatBottomRestraintForceMixIn):
                yield assert_quantity_almost_equal, restraint.well_radius, self.well_radius

            if isinstance(restraint, RadiallySymmetricCentroidRestraintForce):
                yield assert_equal, restraint.restrained_atom_indices1, self.restrained_atom_indices1
                yield assert_equal, restraint.restrained_atom_indices2, self.restrained_atom_indices2
            else:
                assert isinstance(restraint, RadiallySymmetricBondRestraintForce)
                yield assert_equal, restraint.restrained_atom_indices1, [self.restrained_atom_index1]
                yield assert_equal, restraint.restrained_atom_indices2, [self.restrained_atom_index2]

    def test_controlling_parameter_name(self):
        """Test that the controlling parameter name enters the energy function correctly."""
        default_name_restraint = self.restraints[0]
        custom_name_restraints = self.restraints[-2:]

        assert default_name_restraint.controlling_parameter_name == 'lambda_restraints'
        energy_function = default_name_restraint.getEnergyFunction()
        assert 'lambda_restraints' in energy_function
        assert self.custom_parameter_name not in energy_function

        for custom_name_restraint in custom_name_restraints:
            assert custom_name_restraint.controlling_parameter_name == self.custom_parameter_name
            energy_function = custom_name_restraint.getEnergyFunction()
            assert 'lambda_restraints' not in energy_function
            assert self.custom_parameter_name in energy_function

    def test_compute_restraint_volume(self):
        """Test the calculation of the restraint volume."""
        testsystem = testsystems.TolueneVacuum()
        thermodynamic_state = states.ThermodynamicState(testsystem.system, 300*unit.kelvin)

        energy_cutoffs = np.linspace(0.0, 10.0, num=3)
        radius_cutoffs = np.linspace(0.0, 5.0, num=3) * unit.nanometers

        def assert_integrated_analytical_equal(restraint, square_well, radius_cutoff, energy_cutoff):
            args = [thermodynamic_state, square_well, radius_cutoff, energy_cutoff]

            # For flat-bottom, the calculation is only partially analytical.
            analytical_volume = restraint._compute_restraint_volume(*args)

            # Make sure there's no analytical component (from _determine_integral_limits)
            # in the numerical integration calculation.
            copied_restraint = copy.deepcopy(restraint)
            for parent_cls in [RadiallySymmetricCentroidRestraintForce, RadiallySymmetricBondRestraintForce]:
                if isinstance(copied_restraint, parent_cls):
                    copied_restraint.__class__ = parent_cls
            integrated_volume = copied_restraint._integrate_restraint_volume(*args)

            err_msg = '{}: square_well={}, radius_cutoff={}, energy_cutoff={}\n'.format(
                restraint.__class__.__name__, square_well, radius_cutoff, energy_cutoff)
            err_msg += 'integrated_volume={}, analytical_volume={}'.format(integrated_volume,
                                                                           analytical_volume)
            assert utils.is_quantity_close(integrated_volume, analytical_volume, rtol=1e-2), err_msg

        for restraint in self.restraints:
            # Test integrated and analytical agree with no cutoffs.
            yield assert_integrated_analytical_equal, restraint, False, None, None

            for square_well in [True, False]:
                # Try energies and distances singly and together.
                for energy_cutoff in energy_cutoffs:
                    yield assert_integrated_analytical_equal, restraint, square_well, None, energy_cutoff

                for radius_cutoff in radius_cutoffs:
                    yield assert_integrated_analytical_equal, restraint, square_well, radius_cutoff, None

                for energy_cutoff, radius_cutoff in zip(energy_cutoffs, radius_cutoffs):
                    yield assert_integrated_analytical_equal, restraint, square_well, radius_cutoff, energy_cutoff
                for energy_cutoff, radius_cutoff in zip(energy_cutoffs, reversed(radius_cutoffs)):
                    yield assert_integrated_analytical_equal, restraint, square_well, radius_cutoff, energy_cutoff

    def test_compute_standard_state_correction(self):
        """Test standard state correction works correctly in all ensembles."""
        toluene = testsystems.TolueneVacuum()
        alanine = testsystems.AlanineDipeptideExplicit()
        big_radius = 200.0 * unit.nanometers
        temperature = 300.0 * unit.kelvin

        # Limit the maximum volume to 1nm^3.
        distance_unit = unit.nanometers
        state_volume = 1.0 * distance_unit**3
        box_vectors = np.identity(3) * np.cbrt(state_volume / distance_unit**3) * distance_unit
        alanine.system.setDefaultPeriodicBoxVectors(*box_vectors)
        toluene.system.setDefaultPeriodicBoxVectors(*box_vectors)

        # Create systems in various ensembles (NVT, NPT and non-periodic).
        nvt_state = states.ThermodynamicState(alanine.system, temperature)
        npt_state = states.ThermodynamicState(alanine.system, temperature, 1.0*unit.atmosphere)
        nonperiodic_state = states.ThermodynamicState(toluene.system, temperature)

        def assert_equal_ssc(expected_restraint_volume, restraint, thermodynamic_state, square_well=False,
                             radius_cutoff=None, energy_cutoff=None, max_volume=None):
            expected_ssc = -math.log(STANDARD_STATE_VOLUME/expected_restraint_volume)
            ssc = restraint.compute_standard_state_correction(thermodynamic_state, square_well,
                                                              radius_cutoff, energy_cutoff, max_volume)
            err_msg = '{} computed SSC != expected SSC'.format(restraint.__class__.__name__)
            nose.tools.assert_equal(ssc, expected_ssc, msg=err_msg)

        for restraint in self.restraints:
            # In NPT ensemble, an exception is thrown if max_volume is not provided.
            with nose.tools.assert_raises_regexp(TypeError, "max_volume must be provided"):
                restraint.compute_standard_state_correction(npt_state)

            # With non-periodic systems and reweighting to square-well
            # potential, a cutoff must be given.
            with nose.tools.assert_raises_regexp(TypeError, "One between radius_cutoff"):
                restraint.compute_standard_state_correction(nonperiodic_state, square_well=True)
            # While there are no problems if we don't reweight to a square-well potential.
            restraint.compute_standard_state_correction(nonperiodic_state, square_well=False)

            # SSC is limited by max_volume (in NVT and NPT).
            assert_equal_ssc(state_volume, restraint, nvt_state, radius_cutoff=big_radius)
            assert_equal_ssc(state_volume, restraint, npt_state, radius_cutoff=big_radius,
                             max_volume='system')

            # SSC is not limited by max_volume with non periodic systems.
            expected_ssc = -math.log(STANDARD_STATE_VOLUME/state_volume)
            ssc = restraint.compute_standard_state_correction(nonperiodic_state, radius_cutoff=big_radius)
            assert expected_ssc < ssc, (restraint, expected_ssc, ssc)

            # Check reweighting to square-well potential.
            expected_volume = _compute_sphere_volume(big_radius)
            assert_equal_ssc(expected_volume, restraint, nonperiodic_state,
                             square_well=True, radius_cutoff=big_radius)

            energy_cutoff = 10 * nonperiodic_state.kT
            radius_cutoff = _compute_harmonic_radius(self.spring_constant, energy_cutoff)
            if isinstance(restraint, FlatBottomRestraintForceMixIn):
                radius_cutoff += self.well_radius
            expected_volume = _compute_sphere_volume(radius_cutoff)
            assert_equal_ssc(expected_volume, restraint, nonperiodic_state,
                             square_well=True, radius_cutoff=radius_cutoff)

            max_volume = 3.0 * unit.nanometers**3
            assert_equal_ssc(max_volume, restraint, nonperiodic_state,
                             square_well=True, max_volume=max_volume)
