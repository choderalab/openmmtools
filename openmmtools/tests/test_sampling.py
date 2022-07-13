#!/usr/local/bin/env python

"""
Test replicaexchange.py facility.

TODO

* Create a few simulation objects on simple systems (e.g. harmonic oscillators?) and run multiple tests on each object?

"""

# =============================================================================================
# GLOBAL IMPORTS
# =============================================================================================

import contextlib
import copy
import inspect
import math
import os
import pickle
import sys
from io import StringIO

import numpy as np
import scipy.integrate
import yaml
from nose.plugins.attrib import attr
from nose.tools import assert_raises
try:
    import openmm
    from openmm import unit
except ImportError:  # OpenMM < 7.6
    from simtk import openmm, unit
import mpiplus

import openmmtools as mmtools
from openmmtools import cache
from openmmtools import testsystems
from openmmtools.multistate import MultiStateReporter
from openmmtools.multistate import MultiStateSampler, MultiStateSamplerAnalyzer
from openmmtools.multistate import ReplicaExchangeSampler, ReplicaExchangeAnalyzer
from openmmtools.multistate import ParallelTemperingSampler, ParallelTemperingAnalyzer
from openmmtools.multistate import SAMSSampler, SAMSAnalyzer
from openmmtools.multistate.multistatereporter import _DictYamlLoader
from openmmtools.utils import temporary_directory

# quiet down some citation spam
MultiStateSampler._global_citation_silence = True

# ==============================================================================
# MODULE CONSTANTS
# ==============================================================================

kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA  # Boltzmann constant


# ==============================================================================
# SUBROUTINES
# ==============================================================================

def check_thermodynamic_states_equality(original_states, restored_states):
    """Check that the thermodynamic states are equivalent."""
    assert len(original_states) == len(restored_states), '{}, {}'.format(
        len(original_states), len(restored_states))

    for original_state, restored_state in zip(original_states, restored_states):
        assert original_state._standard_system_hash == restored_state._standard_system_hash
        assert original_state.temperature == restored_state.temperature
        assert original_state.pressure == restored_state.pressure

        if isinstance(original_state, mmtools.states.CompoundThermodynamicState):
            assert original_state.lambda_sterics == restored_state.lambda_sterics
            assert original_state.lambda_electrostatics == restored_state.lambda_electrostatics

# ==============================================================================
# Harmonic oscillator free energy test
# ==============================================================================

class TestHarmonicOscillatorsMultiStateSampler(object):
    """Test multistate sampler can compute free energies of harmonic oscillator"""

    # ------------------------------------
    # VARIABLES TO SET FOR EACH TEST CLASS
    # ------------------------------------

    N_SAMPLERS = 3
    N_STATES = 5 # number of thermodynamic states to sample; two additional unsampled states will be added
    N_ITERATIONS = 1000 # number of iterations
    SAMPLER = MultiStateSampler
    ANALYZER = MultiStateSamplerAnalyzer

    @classmethod
    def setup_class(cls):
        # Configure the global context cache to use the Reference platform
        from openmmtools import cache
        platform = openmm.Platform.getPlatformByName('Reference')
        cls.old_global_context_cache = cache.global_context_cache
        cache.global_context_cache = cache.ContextCache(platform=platform)

        # Set oscillators to mass of carbon atom
        mass = 12.0 * unit.amu

        # Translate the sampler states to be different one from each other.
        n_particles = 1
        positions = unit.Quantity(np.zeros([n_particles,3]), unit.angstroms)
        cls.sampler_states = [
            mmtools.states.SamplerState(positions=positions)
            for sampler_index in range(cls.N_SAMPLERS)]

        # Generate list of thermodynamic states and analytical free energies
        # This list includes both sampled and two unsampled states
        thermodynamic_states = list()
        temperature = 300 * unit.kelvin
        f_i = np.zeros([cls.N_STATES+2]) # f_i[state_index] is the dimensionless free energy of state `state_index`
        for state_index in range(cls.N_STATES + 2):
            sigma = (1.0 + 0.2 * state_index) * unit.angstroms # compute reasonable standard deviation with good overlap
            kT = kB * temperature # compute thermal energy
            K = kT / sigma**2 # compute spring constant
            testsystem = testsystems.HarmonicOscillator(K=K, mass=mass)
            thermodynamic_state = mmtools.states.ThermodynamicState(testsystem.system, temperature)
            thermodynamic_states.append(thermodynamic_state)

            # Store analytical reduced free energy
            f_i[state_index] = - np.log(2 * np.pi * (sigma / unit.angstroms)**2) * (3.0/2.0)

        # delta_f_ij_analytical[i,j] = f_i_analytical[j] - f_i_analytical[i]
        cls.f_i_analytical = f_i
        cls.delta_f_ij_analytical = f_i - f_i[:,np.newaxis]

        # Define sampled and unsampled states.
        cls.nstates = cls.N_STATES
        cls.unsampled_states = [thermodynamic_states[0], thermodynamic_states[-1]] # first and last
        cls.thermodynamic_states = thermodynamic_states[1:-1] # intermediate states

    @classmethod
    def teardown_class(cls):
        # Restore global context cache
        from openmmtools import cache
        cache.global_context_cache = cls.old_global_context_cache

    def run(self, include_unsampled_states=False):
        # Create and configure simulation object
        move = mmtools.mcmc.MCDisplacementMove(displacement_sigma=1.0*unit.angstroms)
        simulation = self.SAMPLER(mcmc_moves=move, number_of_iterations=self.N_ITERATIONS)

        # Define file for temporary storage.
        with temporary_directory() as tmp_dir:
            storage = os.path.join(tmp_dir, 'test_storage.nc')
            reporter = MultiStateReporter(storage, checkpoint_interval=self.N_ITERATIONS)

            if include_unsampled_states:
                simulation.create(self.thermodynamic_states, self.sampler_states, reporter,
                                  unsampled_thermodynamic_states=self.unsampled_states)
            else:
                simulation.create(self.thermodynamic_states, self.sampler_states, reporter)

            # Run simulation without debug logging
            import logging
            logger = logging.getLogger()
            logger.setLevel(logging.CRITICAL)
            simulation.run()

            # Create Analyzer.
            analyzer = self.ANALYZER(reporter)

            # Check if free energies have the right shape and deviations exceed tolerance
            delta_f_ij, delta_f_ij_stderr = analyzer.get_free_energy()
            nstates, _ = delta_f_ij.shape

            if include_unsampled_states:
                nstates_expected = self.N_STATES+2 # We expect N_STATES plus two additional states
                delta_f_ij_analytical = self.delta_f_ij_analytical # Use the whole matrix
            else:
                nstates_expected = self.N_STATES # We expect only N_STATES
                delta_f_ij_analytical = self.delta_f_ij_analytical[1:-1,1:-1] # Use only the intermediate, sampled states

            assert nstates == nstates_expected, \
                f'analyzer.get_free_energy() returned {delta_f_ij.shape} but expected {nstates_expected,nstates_expected}'

            error = np.abs(delta_f_ij - delta_f_ij_analytical)
            indices = np.where(delta_f_ij_stderr > 0.0)
            nsigma = np.zeros([nstates,nstates], np.float32)
            nsigma[indices] = error[indices] / delta_f_ij_stderr[indices]
            MAX_SIGMA = 6.0 # maximum allowed number of standard errors
            if np.any(nsigma > MAX_SIGMA):
                np.set_printoptions(precision=3)
                print("delta_f_ij")
                print(delta_f_ij)
                print("delta_f_ij_analytical")
                print(delta_f_ij_analytical)
                print("error")
                print(error)
                print("stderr")
                print(delta_f_ij_stderr)
                print("nsigma")
                print(nsigma)
                raise Exception("Dimensionless free energy difference exceeds MAX_SIGMA of %.1f" % MAX_SIGMA)

        # Clean up.
        del simulation

    def test_with_unsampled_states(self):
        self.run(include_unsampled_states=True)

    def test_without_unsampled_states(self):
        self.run(include_unsampled_states=False)

class TestHarmonicOscillatorsReplicaExchangeSampler(TestHarmonicOscillatorsMultiStateSampler):
    """Test replica-exchange sampler can compute free energies of harmonic oscillator"""

    # ------------------------------------
    # VARIABLES TO SET FOR EACH TEST CLASS
    # ------------------------------------

    N_SAMPLERS = 5
    N_STATES = 5
    SAMPLER = ReplicaExchangeSampler
    ANALYZER = ReplicaExchangeAnalyzer

class TestHarmonicOscillatorsSAMSSampler(TestHarmonicOscillatorsMultiStateSampler):
    """Test SAMS sampler can compute free energies of harmonic oscillator"""

    # ------------------------------------
    # VARIABLES TO SET FOR EACH TEST CLASS
    # ------------------------------------

    N_SAMPLERS = 1
    N_STATES = 5
    N_ITERATIONS = 1000 * N_STATES # number of iterations
    SAMPLER = SAMSSampler
    ANALYZER = SAMSAnalyzer

# ==============================================================================
# TEST REPORTER
# ==============================================================================

class TestReporter(object):
    """Test suite for Reporter class."""

    @staticmethod
    @contextlib.contextmanager
    def temporary_reporter(checkpoint_interval=1, checkpoint_storage=None, analysis_particle_indices=()):
        """Create and initialize a reporter in a temporary directory."""
        with temporary_directory() as tmp_dir_path:
            storage_file = os.path.join(tmp_dir_path, 'temp_dir/test_storage.nc')
            assert not os.path.isfile(storage_file)
            reporter = MultiStateReporter(storage=storage_file, open_mode='w',
                                          checkpoint_interval=checkpoint_interval,
                                          checkpoint_storage=checkpoint_storage,
                                          analysis_particle_indices=analysis_particle_indices)
            assert reporter.storage_exists(skip_size=True)
            yield reporter

    def test_store_thermodynamic_states(self):
        """Check correct storage of thermodynamic states."""
        # Thermodynamic states.
        temperature = 300*unit.kelvin
        alanine_system = testsystems.AlanineDipeptideImplicit().system
        alanine_explicit_system = testsystems.AlanineDipeptideExplicit().system
        thermodynamic_state_nvt = mmtools.states.ThermodynamicState(alanine_system, temperature)
        thermodynamic_state_nvt_compatible = mmtools.states.ThermodynamicState(alanine_system,
                                                                               temperature + 20*unit.kelvin)
        thermodynamic_state_npt = mmtools.states.ThermodynamicState(alanine_explicit_system,
                                                                    temperature, 1.0*unit.atmosphere)

        # Compound states.
        factory = mmtools.alchemy.AbsoluteAlchemicalFactory()
        alchemical_region = mmtools.alchemy.AlchemicalRegion(alchemical_atoms=range(22))
        alanine_alchemical = factory.create_alchemical_system(alanine_system, alchemical_region)
        alchemical_state_interacting = mmtools.alchemy.AlchemicalState.from_system(alanine_alchemical)
        alchemical_state_noninteracting = copy.deepcopy(alchemical_state_interacting)
        alchemical_state_noninteracting.set_alchemical_parameters(0.0)
        compound_state_interacting = mmtools.states.CompoundThermodynamicState(
            thermodynamic_state=mmtools.states.ThermodynamicState(alanine_alchemical, temperature),
            composable_states=[alchemical_state_interacting]
        )
        compound_state_noninteracting = mmtools.states.CompoundThermodynamicState(
            thermodynamic_state=mmtools.states.ThermodynamicState(alanine_alchemical, temperature),
            composable_states=[alchemical_state_noninteracting]
        )

        thermodynamic_states = [thermodynamic_state_nvt, thermodynamic_state_nvt_compatible,
                                thermodynamic_state_npt, compound_state_interacting,
                                compound_state_noninteracting]

        # Unsampled thermodynamic states.
        toluene_system = testsystems.TolueneVacuum().system
        toluene_state = mmtools.states.ThermodynamicState(toluene_system, temperature)
        unsampled_states = [copy.deepcopy(toluene_state), copy.deepcopy(toluene_state),
                            copy.deepcopy(compound_state_interacting)]

        with self.temporary_reporter() as reporter:
            # Check that after writing and reading, states are identical.
            reporter.write_thermodynamic_states(thermodynamic_states, unsampled_states)
            restored_states, restored_unsampled = reporter.read_thermodynamic_states()
            check_thermodynamic_states_equality(thermodynamic_states, restored_states)
            check_thermodynamic_states_equality(unsampled_states, restored_unsampled)

            # The latest writer only stores one full serialization per compatible state.
            ncgrp_states = reporter._storage_analysis.groups['thermodynamic_states']
            ncgrp_unsampled = reporter._storage_analysis.groups['unsampled_states']

            # Load representation of the states on the disk. There
            # should be only one full serialization per compatible state.
            def decompact_state_variable(variable):
                if variable.dtype == 'S1':
                    # Handle variables stored in fixed_dimensions
                    data_chars = variable[:]
                    data_str = data_chars.tostring().decode()
                else:
                    data_str = str(variable[0])
                return data_str
            states_serialized = []
            for state_id in range(len(thermodynamic_states)):
                state_str = decompact_state_variable(ncgrp_states.variables['state' + str(state_id)])
                state_dict = yaml.load(state_str, Loader=_DictYamlLoader)
                states_serialized.append(state_dict)
            unsampled_serialized = []
            for state_id in range(len(unsampled_states)):
                unsampled_str = decompact_state_variable(ncgrp_unsampled.variables['state' + str(state_id)])
                unsampled_dict = yaml.load(unsampled_str, Loader=_DictYamlLoader)
                unsampled_serialized.append(unsampled_dict)

            # Two of the three ThermodynamicStates are compatible.
            assert 'standard_system' in states_serialized[0]
            assert 'standard_system' not in states_serialized[1]
            state_compatible_to_1 = states_serialized[1]['_Reporter__compatible_state']
            assert state_compatible_to_1 == 'thermodynamic_states/0'
            assert 'standard_system' in states_serialized[2]

            # The two CompoundThermodynamicStates are compatible.
            assert 'standard_system' in states_serialized[3]['thermodynamic_state']
            thermodynamic_state_4 = states_serialized[4]['thermodynamic_state']
            assert thermodynamic_state_4['_Reporter__compatible_state'] == 'thermodynamic_states/3'

            # The first two unsampled states are incompatible with everything else
            # but compatible to each other, while the third unsampled state is
            # compatible with the alchemical states.
            assert 'standard_system' in unsampled_serialized[0]
            state_compatible_to_1 = unsampled_serialized[1]['_Reporter__compatible_state']
            assert state_compatible_to_1 == 'unsampled_states/0'
            thermodynamic_state_2 = unsampled_serialized[2]['thermodynamic_state']
            assert thermodynamic_state_2['_Reporter__compatible_state'] == 'thermodynamic_states/3'

    def test_write_sampler_states(self):
        """Check correct storage of sampler states."""
        analysis_particles = (1, 2)
        with self.temporary_reporter(analysis_particle_indices=analysis_particles, checkpoint_interval=2) as reporter:
            # Create sampler states.
            alanine_test = testsystems.AlanineDipeptideVacuum()
            positions = alanine_test.positions
            sampler_states = [mmtools.states.SamplerState(positions=positions)
                              for _ in range(2)]

            # Check that after writing and reading, states are identical.
            for iteration in range(3):
                reporter.write_sampler_states(sampler_states, iteration=iteration)
                reporter.write_last_iteration(iteration)
            restored_sampler_states = reporter.read_sampler_states(iteration=0)
            for state, restored_state in zip(sampler_states, restored_sampler_states):
                assert np.allclose(state.positions, restored_state.positions)
                # By default stored velocities are zeros if not present in origin sampler_state
                assert np.allclose(np.zeros(state.positions.shape), restored_state.velocities)
                assert np.allclose(state.box_vectors / unit.nanometer, restored_state.box_vectors / unit.nanometer)
            # Check that the analysis particles are written off checkpoint whereas full trajectory is not
            restored_analysis_states = reporter.read_sampler_states(iteration=1, analysis_particles_only=True)
            restored_checkpoint_states = reporter.read_sampler_states(iteration=1)
            assert type(restored_analysis_states) is list
            for state in restored_analysis_states:
                assert state.positions.shape == (len(analysis_particles), 3)
                assert state.velocities.shape == (len(analysis_particles), 3)
            assert restored_checkpoint_states is None
            # Check that the analysis particles are written separate from the checkpoint particles
            restored_analysis_states = reporter.read_sampler_states(iteration=2, analysis_particles_only=True)
            restored_checkpoint_states = reporter.read_sampler_states(iteration=2)
            assert len(restored_analysis_states) == len(restored_checkpoint_states)
            for analysis_state, checkpoint_state in zip(restored_analysis_states, restored_checkpoint_states):
                # This assert is multiple purpose: Positions are identical; Velocities are indetical and zeros
                # (since unspecified); Analysis shape is correct
                # Will raise a ValueError for np.allclose(x,y) if x.shape != y.shape
                # Will raise AssertionError if the values are not allclose
                assert np.allclose(analysis_state.positions, checkpoint_state.positions[analysis_particles, :])
                assert np.allclose(analysis_state.velocities, checkpoint_state.velocities[analysis_particles, :])
                assert np.allclose(analysis_state.box_vectors / unit.nanometer,
                                   checkpoint_state.box_vectors / unit.nanometer)

    def test_analysis_particle_mismatch(self):
        """Test that previously stored analysis particles is higher priority."""
        blank_analysis_particles = ()
        set1_analysis_particles = (0, 1)
        set2_analysis_particles = (0, 2)
        # Does not use the temp reporter since we close and reopen reporter a few times
        with temporary_directory() as tmp_dir_path:
            # Test that starting with a blank analysis cannot be overwritten
            blank_file = os.path.join(tmp_dir_path, 'temp_dir/blank_analysis.nc')
            reporter = MultiStateReporter(storage=blank_file, open_mode='w',
                                          analysis_particle_indices=blank_analysis_particles)
            reporter.close()
            del reporter
            new_blank_reporter = MultiStateReporter(storage=blank_file, open_mode='r',
                                                    analysis_particle_indices=set1_analysis_particles)
            assert new_blank_reporter.analysis_particle_indices == blank_analysis_particles
            del new_blank_reporter
            # Test that starting from an initial set of particles and passing in a blank does not overwrite
            set1_file = os.path.join(tmp_dir_path, 'temp_dir/set1_analysis.nc')
            set1_reporter = MultiStateReporter(storage=set1_file, open_mode='w',
                                               analysis_particle_indices=set1_analysis_particles)
            set1_reporter.close()  # Don't delete, we'll need it for another test
            new_set1_reporter = MultiStateReporter(storage=set1_file, open_mode='r',
                                                   analysis_particle_indices=blank_analysis_particles)
            assert new_set1_reporter.analysis_particle_indices == set1_analysis_particles
            del new_set1_reporter
            # Test that passing in a different set than the initial returns the initial set
            new2_set1_reporter = MultiStateReporter(storage=set1_file, open_mode='r',
                                                    analysis_particle_indices=set2_analysis_particles)
            assert new2_set1_reporter.analysis_particle_indices == set1_analysis_particles

    def test_store_replica_thermodynamic_states(self):
        """Check storage of replica thermodynamic states indices."""
        with self.temporary_reporter() as reporter:
            for i, replica_states in enumerate([[2, 1, 0, 3], np.array([3, 1, 0, 2])]):
                reporter.write_replica_thermodynamic_states(replica_states, iteration=i)
                reporter.write_last_iteration(i)
                restored_replica_states = reporter.read_replica_thermodynamic_states(iteration=i)
                assert np.all(replica_states == restored_replica_states)

    def test_store_mcmc_moves(self):
        """Check storage of MCMC moves."""
        sequence_move = mmtools.mcmc.SequenceMove(move_list=[mmtools.mcmc.LangevinDynamicsMove(),
                                                             mmtools.mcmc.GHMCMove()],
                                                  context_cache=mmtools.cache.ContextCache(capacity=1))
        integrator_move = mmtools.mcmc.IntegratorMove(openmm.VerletIntegrator(1.0*unit.femtosecond),
                                                      n_steps=100)
        mcmc_moves = [sequence_move, integrator_move]
        with self.temporary_reporter() as reporter:
            reporter.write_mcmc_moves(mcmc_moves)
            restored_mcmc_moves = reporter.read_mcmc_moves()

            # Check that restored MCMCMoves are exactly the same.
            original_pickle = pickle.dumps(mcmc_moves)
            restored_pickle = pickle.dumps(restored_mcmc_moves)
            assert original_pickle == restored_pickle

    def test_store_energies(self):
        """Check storage of energies."""
        energy_thermodynamic_states = np.array(
            [[0, 2, 3],
             [1, 2, 0],
             [1, 2, 3]])
        energy_neighborhoods = np.array(
            [[0, 1, 1],
             [1, 1, 0],
             [1, 1, 3]]
        )
        energy_unsampled_states = np.array(
            [[1, 2],
             [2, 3.0],
             [3, 9.0]])

        with self.temporary_reporter() as reporter:
            reporter.write_energies(energy_thermodynamic_states, energy_neighborhoods, energy_unsampled_states, iteration=0)
            restored_energy_thermodynamic_states, restored_energy_neighborhoods, restored_energy_unsampled_states = reporter.read_energies(iteration=0)
            assert np.all(energy_thermodynamic_states == restored_energy_thermodynamic_states)
            assert np.all(energy_neighborhoods == restored_energy_neighborhoods)
            assert np.all(energy_unsampled_states == restored_energy_unsampled_states)

    def test_ensure_dimension_exists(self):
        """Test ensuring that a dimension exists works as expected."""
        with self.temporary_reporter() as reporter:
            # These should work fine
            reporter._ensure_dimension_exists('dim1', 0)
            reporter._ensure_dimension_exists('dim2', 1)
            # These should raise an exception
            assert_raises(ValueError, reporter._ensure_dimension_exists, 'dim1', 1)
            assert_raises(ValueError, reporter._ensure_dimension_exists, 'dim2', 2)

    def test_store_dict(self):
        """Check correct storage and restore of dictionaries."""

        def sorted_dict(d):
            d = copy.deepcopy(d)
            # Sort internal dictionaries.
            for k, v in d.items():
                if isinstance(v, dict):
                    d[k] = sorted_dict(v)
            return sorted(d.items())

        def compare_dicts(reference, restored):
            # We need a deterministically-ordered dict to compare pickles.
            sorted_reference = sorted_dict(reference)
            sorted_restored = sorted_dict(restored)
            assert pickle.dumps(sorted_reference) == pickle.dumps(sorted_restored)

        data = {
            'mybool': False,
            'mystring': 'test',
            'myinteger': 3, 'myfloat': 4.0,
            'mylist': [0, 1, 2, 3], 'mynumpyarray': np.array([2.0, 3, 4]),
            'mynestednumpyarray': np.array([[1, 2, 3], [4.0, 5, 6]]),
            'myquantity': 5.0 / unit.femtosecond,
            'myquantityarray': unit.Quantity(np.array([[1, 2, 3], [4.0, 5, 6]]), unit=unit.angstrom),
            'mynesteddict': {'field1': 'string', 'field2': {'field21': 3.0, 'field22': True}}
        }

        with self.temporary_reporter() as reporter:
            # Test both nested and single-string representations.
            for name, nested in [('testdict', False), ('nested', True)]:
                reporter._write_dict(name, data, nested=nested)
                restored_data = reporter.read_dict(name)
                compare_dicts(data, restored_data)

                # Test reading a keyword inside a dict.
                restored_data = reporter.read_dict(name + '/mynesteddict/field2')
                compare_dicts(data['mynesteddict']['field2'], restored_data)

                # write_dict supports updates, even with the nested representation
                # if the structure of the dictionary doesn't change.
                data['mybool'] = True
                data['mystring'] = 'substituted'
                reporter._write_dict(name, data, nested=nested)
                restored_data = reporter.read_dict(name)
                assert restored_data['mybool'] is True
                assert restored_data['mystring'] == 'substituted'

                # In nested representation, dictionaries are stored as groups and
                # values as variables. Otherwise, there's only a single variable.
                if nested:
                    dict_group = reporter._storage_analysis.groups[name]
                    assert 'mynesteddict' in dict_group.groups
                    assert 'mylist' in dict_group.variables
                else:
                    assert name in reporter._storage_analysis.variables

            # Write dict fixed_dimension creates static dimensions and read/writes correctly
            reporter._write_dict('fixed', data, fixed_dimension=True)
            restored_fixed_data = reporter.read_dict('fixed')
            compare_dicts(data, restored_fixed_data)

    def test_store_mixing_statistics(self):
        """Check mixing statistics are correctly stored."""
        n_accepted_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        n_proposed_matrix = np.array([[3, 3, 3], [6, 6, 6], [9, 9, 9]])
        with self.temporary_reporter() as reporter:
            reporter.write_mixing_statistics(n_accepted_matrix, n_proposed_matrix, iteration=0)
            restored_n_accepted, restored_n_proposed = reporter.read_mixing_statistics(iteration=0)
            assert np.all(n_accepted_matrix == restored_n_accepted)
            assert np.all(n_proposed_matrix == restored_n_proposed)


# ==============================================================================
# TEST MULTISTATE SAMPLERS
# ==============================================================================

class TestMultiStateSampler(object):
    """Base test suite for the multi-state classes"""

    # ------------------------------------
    # VARIABLES TO SET FOR EACH TEST CLASS
    # ------------------------------------

    N_SAMPLERS = 3
    N_STATES = 5
    SAMPLER = MultiStateSampler
    REPORTER = MultiStateReporter

    # --------------------------------------
    # Optional helper function to overwrite.
    # --------------------------------------

    @classmethod
    def call_sampler_create(cls, sampler, reporter,
                            thermodynamic_states,
                            sampler_states,
                            unsampled_states):
        """Helper function to call the create method for the sampler"""
        # Allows initial thermodynamic states to be handled by the built in methods
        sampler.create(thermodynamic_states, sampler_states, reporter,
                       unsampled_thermodynamic_states=unsampled_states)

    # --------------------------------
    # Tests overwritten by sub-classes
    # --------------------------------
    def test_stored_properties(self):
        """Test that storage is kept in sync with options."""
        # Intermediary step to testing stored properties
        self.actual_stored_properties_check()

    @classmethod
    def _compute_energies_independently(cls, sampler):
        """
        Helper function to compute energies by hand.
        This is overwritten by subclasses
        """
        thermodynamic_states = sampler._thermodynamic_states
        unsampled_states = sampler._unsampled_states
        sampler_states = sampler._sampler_states

        n_states = len(thermodynamic_states)
        n_replicas = len(sampler_states)
        # Compute the energies independently.
        energy_thermodynamic_states = np.zeros((n_replicas, n_states))
        energy_unsampled_states = np.zeros((n_replicas, len(unsampled_states)))
        for energies, states in [(energy_thermodynamic_states, thermodynamic_states),
                                 (energy_unsampled_states, unsampled_states)]:
            for i, sampler_state in enumerate(sampler_states):
                for j, state in enumerate(states):
                    context, integrator = mmtools.cache.global_context_cache.get_context(state)
                    sampler_state.apply_to_context(context)
                    energies[i][j] = state.reduced_potential(context)
        return energy_thermodynamic_states, energy_unsampled_states

    # --------------------------------
    # Common Test functions below here
    # --------------------------------

    @classmethod
    def setup_class(cls):
        """Shared test cases and variables."""
        # Test case with alanine in vacuum at 3 different positions and temperatures.
        # ---------------------------------------------------------------------------
        alanine_test = testsystems.AlanineDipeptideVacuum()

        # Translate the sampler states to be different one from each other.
        alanine_sampler_states = [
            mmtools.states.SamplerState(positions=alanine_test.positions + 10 * i * unit.nanometers)
            for i in range(cls.N_SAMPLERS)]

        # Set increasing temperature.
        temperatures = [(300 + 10 * i) * unit.kelvin for i in range(cls.N_STATES)]
        alanine_thermodynamic_states = [mmtools.states.ThermodynamicState(alanine_test.system, temperatures[i])
                                        for i in range(cls.N_STATES)]

        # No unsampled states for this test.
        cls.alanine_test = (alanine_thermodynamic_states, alanine_sampler_states, [])

        # Test case with host guest in implicit at 3 different positions and alchemical parameters.
        # -----------------------------------------------------------------------------------------
        hostguest_test = testsystems.HostGuestVacuum()
        factory = mmtools.alchemy.AbsoluteAlchemicalFactory()
        alchemical_region = mmtools.alchemy.AlchemicalRegion(alchemical_atoms=range(126, 156))
        hostguest_alchemical = factory.create_alchemical_system(hostguest_test.system, alchemical_region)

        # Translate the sampler states to be different one from each other.
        hostguest_sampler_states = [
            mmtools.states.SamplerState(positions=hostguest_test.positions + 10 * i * unit.nanometers)
            for i in range(cls.N_SAMPLERS)]

        # Create the three basic thermodynamic states.
        temperatures = [(300 + 10 * i) * unit.kelvin for i in range(cls.N_STATES)]
        hostguest_thermodynamic_states = [mmtools.states.ThermodynamicState(hostguest_alchemical, temperatures[i])
                                          for i in range(cls.N_STATES)]

        # Create alchemical states at different parameter values.
        alchemical_states = [mmtools.alchemy.AlchemicalState.from_system(hostguest_alchemical)
                             for _ in range(cls.N_STATES)]
        for i, alchemical_state in enumerate(alchemical_states):
            alchemical_state.set_alchemical_parameters(float(i) / (cls.N_STATES - 1))

        # Create compound states.
        hostguest_compound_states = list()
        for i in range(cls.N_STATES):
            hostguest_compound_states.append(
                mmtools.states.CompoundThermodynamicState(thermodynamic_state=hostguest_thermodynamic_states[i],
                                                          composable_states=[alchemical_states[i]])
            )

        # Unsampled states.
        nonalchemical_state = mmtools.states.ThermodynamicState(hostguest_test.system, temperatures[0])
        hostguest_unsampled_states = [copy.deepcopy(nonalchemical_state)]

        cls.hostguest_test = (hostguest_compound_states, hostguest_sampler_states, hostguest_unsampled_states)

        # Debugging Messages to sent to Nose with --nocapture enabled
        output_descr = "Testing Sampler: {}  -- States: {}  -- Samplers: {}".format(
            cls.SAMPLER.__name__, cls.N_STATES, cls.N_SAMPLERS)
        len_output = len(output_descr)
        print("#" * len_output)
        print(output_descr)
        print("#" * len_output)

    @staticmethod
    @contextlib.contextmanager
    def temporary_storage_path():
        """Generate a storage path in a temporary folder and share it.

        It makes it possible to run tests on multiple nodes with MPI.

        """
        mpicomm = mpiplus.get_mpicomm()
        with temporary_directory() as tmp_dir_path:
            storage_file_path = os.path.join(tmp_dir_path, 'test_storage.nc')
            if mpicomm is not None:
                storage_file_path = mpicomm.bcast(storage_file_path, root=0)
            yield storage_file_path

    @staticmethod
    def get_node_replica_ids(tot_n_replicas):
        """Return the indices of the replicas that this node is responsible for."""
        mpicomm = mpiplus.get_mpicomm()
        if mpicomm is None or mpicomm.rank == 0:
            return set(range(tot_n_replicas))
        else:
            return set(range(mpicomm.rank, tot_n_replicas, mpicomm.size))

    @staticmethod
    @contextlib.contextmanager
    def captured_output():
        new_out, new_err = StringIO(), StringIO()
        old_out, old_err = sys.stdout, sys.stderr
        try:
            sys.stdout, sys.stderr = new_out, new_err
            yield sys.stdout, sys.stderr
        finally:
            sys.stdout, sys.stderr = old_out, old_err

    @staticmethod
    def property_creator(name, on_disk_name, value, on_disk_value):
        """
        Helper to create additional properties to create for checking

        Makes a nested dict where the top key is the 'name', with one
        value as a dict where the sub-dict is of the form:
        {'value': value,
         'on_disk_name': on_disk_name,
         'on_disk_value': on_disk_value
        }
        """
        return {name: {

            'value': value,
            'on_disk_value': on_disk_value,
            'on_disk_name': on_disk_name
        }}

    def test_create(self):
        """Test creation of a new MultiState simulation.

        Checks that the storage file is correctly initialized with all the
        information needed. With MPI, this checks that only node 0 has an
        open Reporter for writing.

        """
        thermodynamic_states, sampler_states, unsampled_states = copy.deepcopy(self.alanine_test)
        n_states = len(thermodynamic_states)
        n_samplers = len(sampler_states)

        with self.temporary_storage_path() as storage_path:
            reporter = self.REPORTER(storage_path, checkpoint_interval=1)
            sampler = self.SAMPLER()
            if hasattr(sampler, 'replica_mixing_scheme'):
                sampler.replica_mixing_scheme = 'swap-neighbors'
            sampler.locality = 2
            self.call_sampler_create(sampler, reporter,
                                     thermodynamic_states,
                                     sampler_states, unsampled_states)

            # Check that reporter has reporter only if rank 0.
            mpicomm = mpiplus.get_mpicomm()
            if mpicomm is None or mpicomm.rank == 0:
                assert sampler._reporter.is_open()
            else:
                assert not sampler._reporter.is_open()

            # Ensure reporter is closed. Windows based system testing seems to get upset about this.
            reporter.close()

            # Open reporter to read stored data.
            reporter = self.REPORTER(storage_path, open_mode='r', checkpoint_interval=1)

            # The n_states sampler states have been distributed
            restored_sampler_states = reporter.read_sampler_states(iteration=0)
            restored_thermo_states, _ = reporter.read_thermodynamic_states()
            assert sampler.n_states == n_states, ("Mismatch: sampler.n_states = {} "
                                                  "but n_states = {}".format(sampler.n_states, n_states))
            assert sampler.n_replicas == n_samplers, ("Mismatch: sampler.n_replicas = {} "
                                                      "but n_samplers = {}".format(sampler.n_replicas, n_samplers))
            assert len(restored_sampler_states) == n_samplers
            assert len(restored_thermo_states) == n_states
            assert np.allclose(restored_sampler_states[0].positions, sampler._sampler_states[0].positions)

            # MCMCMove was stored correctly.
            restored_mcmc_moves = reporter.read_mcmc_moves()
            assert len(sampler._mcmc_moves) == n_states
            assert len(restored_mcmc_moves) == n_states
            for sampler_move, restored_move in zip(sampler._mcmc_moves, restored_mcmc_moves):
                assert isinstance(sampler_move, mmtools.mcmc.LangevinDynamicsMove)
                assert isinstance(restored_move, mmtools.mcmc.LangevinDynamicsMove)

            # Options have been stored.
            stored_options = reporter.read_dict('options')
            options_to_store = dict()
            for cls in inspect.getmro(type(sampler)):
                parameter_names, _, _, defaults, _, _, _ = inspect.getfullargspec(cls.__init__)
                if defaults:
                    for parameter_name in parameter_names[-len(defaults):]:
                        options_to_store[parameter_name] = getattr(sampler, '_' + parameter_name)
            options_to_store.pop('mcmc_moves')  # mcmc_moves are stored separately
            for key, value in options_to_store.items():
                if np.isscalar(value):
                    assert stored_options[key] == value, "stored_options['%s'] = %s, but value = %s" % (key, stored_options[key], value)
                    assert getattr(sampler, '_' + key) == value, "getattr(sampler, '%s') = %s, but value = %s" % ('_' + key, getattr(sampler, '_' + key), value)
                else:
                    assert np.all(stored_options[key] == value), "stored_options['%s'] = %s, but value = %s" % (key, stored_options[key], value)
                    assert np.all(getattr(sampler, '_' + key) == value), "getattr(sampler, '%s') = %s, but value = %s" % ('_' + key, getattr(sampler, '_' + key), value)

            # A default title has been added to the stored metadata.
            metadata = reporter.read_dict('metadata')
            assert len(metadata) == 1
            assert sampler.metadata['title'] == metadata['title']

    def test_citations(self):
        """Test that citations are displayed and suppressed as needed."""
        thermodynamic_states, sampler_states, unsampled_states = copy.deepcopy(self.alanine_test)

        with self.temporary_storage_path() as storage_path:
            sampler = self.SAMPLER()
            reporter = self.REPORTER(storage_path, checkpoint_interval=1)

            # Ensure global mute is on
            sampler._global_citation_silence = True

            # Trap expected output
            with self.captured_output() as (out, _):
                sampler._display_citations(overwrite_global=True)
                cite_string = out.getvalue().strip()
                self.call_sampler_create(sampler, reporter,
                                         thermodynamic_states, sampler_states,
                                         unsampled_states)
                # Reset internal flag
                sampler._have_displayed_citations_before = False
                # Test that the overwrite flag worked
                assert cite_string != ''
            # Test that the output is not generate when the global is set
            with self.captured_output() as (out, _):
                sampler._global_citation_silence = True
                sampler._display_citations()
                output = out.getvalue().strip()
                assert cite_string not in output
            # Test that the output is generated with the global is not set and previously un shown
            with self.captured_output() as (out, _):
                sampler._global_citation_silence = False
                sampler._have_displayed_citations_before = False
                sampler._display_citations()
                output = out.getvalue().strip()
                assert cite_string in output
            # Repeat to ensure the citations are not generated a second time
            with self.captured_output() as (out, _):
                sampler._global_citation_silence = False
                sampler._display_citations()
                output = out.getvalue().strip()
                assert cite_string not in output

    def test_from_storage(self):
        """Test that from_storage completely restore the sampler.

        Checks that the static constructor MultiStateSampler.from_storage()
        restores the simulation object in the exact same state as the last
        iteration. Except from the reporter and timing data attributes, that
        is _reporter and _timing_data, respectively.

        """
        # We don't want to restore reporter and timing data attributes
        __NON_RESTORABLE_ATTRIBUTES__ = ("_reporter", "_timing_data")
        thermodynamic_states, sampler_states, unsampled_states = copy.deepcopy(self.hostguest_test)
        n_replicas = len(sampler_states)

        with self.temporary_storage_path() as storage_path:
            number_of_iterations = 3
            move = mmtools.mcmc.LangevinDynamicsMove(n_steps=1)
            sampler = self.SAMPLER(mcmc_moves=move, number_of_iterations=number_of_iterations)
            if hasattr(sampler, 'replica_mixing_scheme'):
                # TODO: Test both 'swap-all' with locality=None and 'swap-neighbors' with locality=1
                sampler.replica_mixing_scheme = 'swap-neighbors' # required for non-global locality
            sampler.locality = 1
            reporter = self.REPORTER(storage_path, checkpoint_interval=1)
            self.call_sampler_create(sampler, reporter,
                                     thermodynamic_states, sampler_states,
                                     unsampled_states)

            # Test at the beginning and after few iterations.
            for iteration in range(2):
                # Store the state of the initial repex object (its __dict__). We leave the
                # reporter out because when the NetCDF file is copied, it runs into issues.
                original_dict = copy.deepcopy({k: v for k, v in sampler.__dict__.items()
                                               if k not in __NON_RESTORABLE_ATTRIBUTES__})

                # Delete repex to close reporter before creating a new one
                # to avoid weird issues with multiple NetCDF files open.
                del sampler
                reporter.close()
                sampler = self.SAMPLER.from_storage(reporter)
                restored_dict = copy.deepcopy({k: v for k, v in sampler.__dict__.items()
                                               if k not in __NON_RESTORABLE_ATTRIBUTES__})

                # Check thermodynamic states.
                original_ts = original_dict.pop('_thermodynamic_states')
                restored_ts = restored_dict.pop('_thermodynamic_states')
                check_thermodynamic_states_equality(original_ts, restored_ts)

                # Check unsampled thermodynamic states.
                original_us = original_dict.pop('_unsampled_states')
                restored_us = restored_dict.pop('_unsampled_states')
                check_thermodynamic_states_equality(original_us, restored_us)

                # The reporter of the restored simulation must be open only in node 0.
                mpicomm = mpiplus.get_mpicomm()
                if mpicomm is None or mpicomm.rank == 0:
                    assert sampler._reporter.is_open()
                else:
                    assert not sampler._reporter.is_open()

                # Each replica keeps only the info for the replicas it is
                # responsible for to minimize network traffic.
                node_replica_ids = self.get_node_replica_ids(n_replicas)

                # Check sampler states. Non 0 nodes only hold their positions.
                original_ss = original_dict.pop('_sampler_states')
                restored_ss = restored_dict.pop('_sampler_states')
                for replica_id, (original, restored) in enumerate(zip(original_ss, restored_ss)):
                    if replica_id in node_replica_ids:
                        assert np.allclose(original.positions, restored.positions)
                        assert np.all(original.box_vectors == restored.box_vectors)

                # Check energies. Non 0 nodes only hold their energies.
                original_neighborhoods = original_dict.pop('_neighborhoods')
                restored_neighborhoods = restored_dict.pop('_neighborhoods')
                original_ets = original_dict.pop('_energy_thermodynamic_states')
                restored_ets = restored_dict.pop('_energy_thermodynamic_states')
                original_eus = original_dict.pop('_energy_unsampled_states')
                restored_eus = restored_dict.pop('_energy_unsampled_states')
                for replica_id in node_replica_ids:
                    assert np.allclose(original_neighborhoods[replica_id], restored_neighborhoods[replica_id])
                    assert np.allclose(original_ets[replica_id], restored_ets[replica_id])
                    assert np.allclose(original_eus[replica_id], restored_eus[replica_id])

                # Only node 0 has updated accepted and proposed exchanges.
                original_accepted = original_dict.pop('_n_accepted_matrix')
                restored_accepted = restored_dict.pop('_n_accepted_matrix')
                original_proposed = original_dict.pop('_n_proposed_matrix')
                restored_proposed = restored_dict.pop('_n_proposed_matrix')
                if len(node_replica_ids) == n_replicas:
                    assert np.all(original_accepted == restored_accepted)
                    assert np.all(original_proposed == restored_proposed)

                # Test mcmc moves with pickle.
                original_mcmc_moves = original_dict.pop('_mcmc_moves')
                restored_mcmc_moves = restored_dict.pop('_mcmc_moves')
                if len(node_replica_ids) == n_replicas:
                    assert pickle.dumps(original_mcmc_moves) == pickle.dumps(restored_mcmc_moves)

                # Check all other arrays. Instantiate list so that we can pop from original_dict.
                for attr, original_value in list(original_dict.items()):
                    if isinstance(original_value, np.ndarray):
                        original_value = original_dict.pop(attr)
                        restored_value = restored_dict.pop(attr)
                        assert np.all(original_value == restored_value), '{}: {}\t{}'.format(
                            attr, original_value, restored_value)

                # Everything else should be a dict of builtins.
                assert original_dict == restored_dict

                # Run few iterations to see that we restore also after a while.
                if iteration == 0:
                    sampler.run(number_of_iterations)

    def actual_stored_properties_check(self, additional_properties=None):
        """Stored properties check which expects a keyword"""
        thermodynamic_states, sampler_states, unsampled_states = copy.deepcopy(self.alanine_test)

        with self.temporary_storage_path() as storage_path:
            sampler = self.SAMPLER(number_of_iterations=5)
            reporter = self.REPORTER(storage_path, checkpoint_interval=1)
            self.call_sampler_create(sampler, reporter,
                                     thermodynamic_states, sampler_states,
                                     unsampled_states)

            # Update options and check the storage is synchronized.
            sampler.number_of_iterations = float('inf')
            # Process Additional properties
            if additional_properties is not None:
                for add_property, property_data in additional_properties.items():
                    setattr(sampler, add_property, property_data['value'])

            # Displace positions of the first sampler state.
            sampler_states = sampler.sampler_states
            original_positions = copy.deepcopy(sampler_states[0].positions)
            displacement_vector = np.ones(3) * unit.angstroms
            sampler_states[0].positions += displacement_vector
            sampler.sampler_states = sampler_states

            mpicomm = mpiplus.get_mpicomm()
            if mpicomm is None or mpicomm.rank == 0:
                reporter.close()
                reporter = self.REPORTER(storage_path, open_mode='r')
                restored_options = reporter.read_dict('options')
                assert restored_options['number_of_iterations'] == float('inf')
                if additional_properties is not None:
                    for _, property_data in additional_properties.items():
                        on_disk_name = property_data['on_disk_name']
                        on_disk_value = property_data['on_disk_value']
                        restored_value = restored_options[on_disk_name]
                        if on_disk_value is None:
                            assert restored_value is on_disk_value, "Restored {} is not {}".format(restored_value,
                                                                                                   on_disk_value)
                        else:
                            assert restored_value == on_disk_value, "Restored {} != {}".format(restored_value,
                                                                                               on_disk_value)

                restored_sampler_states = reporter.read_sampler_states(iteration=0)
                assert np.allclose(restored_sampler_states[0].positions,
                                   original_positions + displacement_vector)

    def test_propagate_replicas(self):
        """Test method _propagate_replicas from MultiStateSampler.

        The purpose of this test is mainly to make sure that MPI doesn't mix
        the information of the propagated StateSamplers when it communicates
        the new positions and box vectors.

        """
        thermodynamic_states, sampler_states, unsampled_states = copy.deepcopy(self.alanine_test)
        n_replicas = len(sampler_states)
        if n_replicas == 1:
            # This test is intended for use with more than one replica
            return

        with self.temporary_storage_path() as storage_path:
            # For this test to work, positions should be the same but
            # translated, so that minimized positions should satisfy
            # the same condition.
            original_diffs = [np.average(sampler_states[i].positions - sampler_states[i+1].positions)
                              for i in range(n_replicas - 1)]
            assert not np.allclose(original_diffs, [0 for _ in range(n_replicas - 1)]), "sampler %s failed" % self.SAMPLER

            # Create a replica exchange that propagates only 1 femtosecond
            # per iteration so that positions won't change much.
            move = mmtools.mcmc.IntegratorMove(openmm.VerletIntegrator(1.0*unit.femtosecond), n_steps=1)
            sampler = self.SAMPLER(mcmc_moves=move)
            reporter = self.REPORTER(storage_path)
            self.call_sampler_create(sampler, reporter,
                                     thermodynamic_states, sampler_states,
                                     unsampled_states)

            # Propagate.
            sampler._propagate_replicas()

            # The relative positions between the new sampler states should
            # be still translated the same way (i.e. we are not assigning
            # the minimized positions to the incorrect sampler states).
            new_sampler_states = sampler._sampler_states
            new_diffs = [np.average(new_sampler_states[i].positions - new_sampler_states[i+1].positions)
                         for i in range(n_replicas - 1)]
            assert np.allclose(original_diffs, new_diffs, rtol=1e-4)

    def test_compute_energies(self):
        """Test method _compute_energies from MultiStateSampler.

        The purpose of this test is mainly to make sure that MPI doesn't mix
        the information of the thermodynamics and unsampled ThermodynamicStates
        when it communicates them to the other nodes.

        """
        thermodynamic_states, sampler_states, unsampled_states = copy.deepcopy(self.hostguest_test)
        n_states = len(thermodynamic_states)
        n_replicas = len(sampler_states)

        with self.temporary_storage_path() as storage_path:
            sampler = self.SAMPLER()
            reporter = self.REPORTER(storage_path, checkpoint_interval=1)
            self.call_sampler_create(sampler, reporter,
                                     thermodynamic_states, sampler_states,
                                     unsampled_states)

            # Let MultiStateSampler distribute the computation of energies among nodes.
            sampler._compute_energies()

            # Compute energies at all states
            energy_thermodynamic_states, energy_unsampled_states = self._compute_energies_independently(sampler)

            # Only node 0 has all the energies.
            mpicomm = mpiplus.get_mpicomm()
            if mpicomm is None or mpicomm.rank == 0:
                for replica_index in range(n_replicas):
                    neighborhood = sampler._neighborhoods[replica_index,:]
                    msg = f"{sampler} failed test_compute_energies:\n"
                    msg += f"{sampler._energy_thermodynamic_states}\n"
                    msg += f"{energy_thermodynamic_states}"
                    assert np.allclose(sampler._energy_thermodynamic_states[replica_index,neighborhood], energy_thermodynamic_states[replica_index,neighborhood]), msg
                assert np.allclose(sampler._energy_unsampled_states, energy_unsampled_states)

    def test_minimize(self):
        """Test MultiStateSampler minimize method.

        The purpose of this test is mainly to make sure that MPI doesn't mix
        the information of the minimized StateSamplers when it communicates
        the new positions. It also checks that the energies are effectively
        decreased.

        """
        thermodynamic_states, sampler_states, unsampled_states = copy.deepcopy(self.alanine_test)
        n_states = len(thermodynamic_states)
        n_replicas = len(sampler_states)
        if n_replicas == 1:
            # This test is intended for use with more than one replica
            return

        with self.temporary_storage_path() as storage_path:
            sampler = self.SAMPLER()
            reporter = self.REPORTER(storage_path, checkpoint_interval=1)
            self.call_sampler_create(sampler, reporter,
                                     thermodynamic_states, sampler_states,
                                     unsampled_states)

            # For this test to work, positions should be the same but
            # translated, so that minimized positions should satisfy
            # the same condition.
            original_diffs = [np.average(sampler_states[i].positions - sampler_states[i + 1].positions)
                              for i in range(n_replicas - 1)]
            assert not np.allclose(original_diffs, [0 for _ in range(n_replicas - 1)])

            # Compute initial energies.
            sampler._compute_energies()
            state_indices = sampler._replica_thermodynamic_states
            original_energies = [sampler._energy_thermodynamic_states[i, j] for i, j in enumerate(state_indices)]

            # Minimize.
            sampler.minimize()

            # The relative positions between the new sampler states should
            # be still translated the same way (i.e. we are not assigning
            # the minimized positions to the incorrect sampler states).
            new_sampler_states = sampler._sampler_states
            new_diffs = [np.average(new_sampler_states[i].positions - new_sampler_states[i + 1].positions)
                         for i in range(n_replicas - 1)]
            assert np.allclose(original_diffs, new_diffs, atol=0.1)

            # Each replica keeps only the info for the replicas it is
            # responsible for to minimize network traffic.
            node_replica_ids = self.get_node_replica_ids(n_replicas)

            # The energies have been minimized.
            sampler._compute_energies()
            for replica_index in node_replica_ids:
                state_index = sampler._replica_thermodynamic_states[replica_index]
                old_energy = original_energies[replica_index]
                new_energy = sampler._energy_thermodynamic_states[replica_index, state_index]
                assert new_energy <= old_energy, "Energies did not decrease: Replica {} was originally {}, now {}".format(replica_index, old_energy, new_energy)

            # The storage has been updated.
            reporter.close()
            if len(node_replica_ids) == n_states:
                reporter = self.REPORTER(storage_path, open_mode='r')
                stored_sampler_states = reporter.read_sampler_states(iteration=0)
                for new_state, stored_state in zip(new_sampler_states, stored_sampler_states):
                    assert np.allclose(new_state.positions, stored_state.positions)

    def test_equilibrate(self):
        """Test equilibration of MultiStateSampler simulation.

        During equilibration, we set temporarily different MCMCMoves. This checks
        that they are restored correctly. It also checks that the storage has the
        updated positions.

        """
        thermodynamic_states, sampler_states, unsampled_states = copy.deepcopy(self.alanine_test)
        n_replicas = len(sampler_states)

        with self.temporary_storage_path() as storage_path:
            # We create a ReplicaExchange with a GHMC move but use Langevin for equilibration.
            sampler = self.SAMPLER(mcmc_moves=mmtools.mcmc.GHMCMove())
            reporter = self.REPORTER(storage_path, checkpoint_interval=1)
            self.call_sampler_create(sampler, reporter,
                                     thermodynamic_states, sampler_states,
                                     unsampled_states)

            # Equilibrate
            equilibration_move = mmtools.mcmc.LangevinDynamicsMove(n_steps=1)
            sampler.equilibrate(n_iterations=10, mcmc_moves=equilibration_move)
            assert isinstance(sampler._mcmc_moves[0], mmtools.mcmc.GHMCMove)

            # Each replica keeps only the info for the replicas it is
            # responsible for to minimize network traffic.
            node_replica_ids = self.get_node_replica_ids(n_replicas)

            # The storage has been updated.
            reporter.close()
            if len(node_replica_ids) == n_replicas:
                reporter = self.REPORTER(storage_path, open_mode='r', checkpoint_interval=1)
                stored_sampler_states = reporter.read_sampler_states(iteration=0)
                for stored_state in stored_sampler_states:
                    assert any([np.allclose(new_state.positions, stored_state.positions) for new_state in sampler._sampler_states])

            # We are still at iteration 0.
            assert sampler._iteration == 0

    def test_run_extend(self):
        """Test methods run and extend of MultiStateSampler."""
        test_cases = [self.alanine_test, self.hostguest_test]

        for test_case in test_cases:
            thermodynamic_states, sampler_states, unsampled_states = copy.deepcopy(test_case)

            with self.temporary_storage_path() as storage_path:
                moves = mmtools.mcmc.SequenceMove([
                    mmtools.mcmc.LangevinDynamicsMove(n_steps=1),
                    mmtools.mcmc.MCRotationMove(),
                    mmtools.mcmc.GHMCMove(n_steps=1)
                ])

                sampler = self.SAMPLER(mcmc_moves=moves, number_of_iterations=2)
                reporter = self.REPORTER(storage_path, checkpoint_interval=1)
                self.call_sampler_create(sampler, reporter,
                                         thermodynamic_states, sampler_states,
                                         unsampled_states)

                # MultiStateSampler.run doesn't go past number_of_iterations.
                assert not sampler.is_completed
                sampler.run(n_iterations=3)
                assert sampler.iteration == 2
                assert sampler.is_completed

                # MultiStateSampler.extend does.
                sampler.extend(n_iterations=2)
                assert sampler.iteration == 4

                # Extract the sampled thermodynamic states
                # Only use propagated states since the last iteration is not subject to MCMC moves
                sampled_states = list(reporter.read_replica_thermodynamic_states()[1:].flat)

                # All replicas must have moves with updated statistics.
                for state_index, sequence_move in enumerate(sampler._mcmc_moves):
                    # LangevinDynamicsMove (index 0) doesn't have statistics.
                    for move_id in [1, 2]:
                        assert sequence_move.move_list[move_id].n_proposed == sampled_states.count(state_index)

                # The MCMCMoves statistics in the storage are updated.
                mpicomm = mpiplus.get_mpicomm()
                if mpicomm is None or mpicomm.rank == 0:
                    reporter.close()
                    reporter = self.REPORTER(storage_path, open_mode='r', checkpoint_interval=1)
                    restored_mcmc_moves = reporter.read_mcmc_moves()
                    for state_index, sequence_move in enumerate(restored_mcmc_moves):
                        # LangevinDynamicsMove (index 0) doesn't have statistic
                        for move_id in [1, 2]:
                            assert sequence_move.move_list[move_id].n_proposed == sampled_states.count(state_index)

    def test_checkpointing(self):
        """Test that checkpointing writes infrequently"""
        thermodynamic_states, sampler_states, unsampled_states = copy.deepcopy(self.alanine_test)

        with self.temporary_storage_path() as storage_path:
            # For this test, we simply check that the checkpointing writes on the interval
            # We don't care about the numbers, per se, but we do care about when things are written
            n_iterations = 3
            move = mmtools.mcmc.IntegratorMove(openmm.VerletIntegrator(1.0 * unit.femtosecond),
                                               n_steps=1)
            reporter = self.REPORTER(storage_path, checkpoint_interval=2)
            sampler = self.SAMPLER(mcmc_moves=move, number_of_iterations=n_iterations)
            self.call_sampler_create(sampler, reporter,
                                     thermodynamic_states, sampler_states,
                                     unsampled_states)
            # Propagate.
            sampler.run()
            reporter.close()
            reporter = self.REPORTER(storage_path, open_mode='r', checkpoint_interval=2)
            for i in range(n_iterations):
                energies, _, _ = reporter.read_energies(i)
                states = reporter.read_sampler_states(i)
                assert type(energies) is np.ndarray
                if reporter._calculate_checkpoint_iteration(i) is not None:
                    assert type(states[0].positions) is mmtools.utils.TrackedQuantity
                else:
                    assert states is None

    def test_resume_positions_velocities_from_storage(self):
        """Test that positions and velocities are the same when resuming a simulation from reporter storage file."""
        # TODO: Find a way to extend this test to use MPI since resuming velocities has a problem there.
        test_cases = [self.alanine_test, self.hostguest_test]

        for test_case in test_cases:
            thermodynamic_states, sampler_states, unsampled_states = copy.deepcopy(test_case)

            with self.temporary_storage_path() as storage_path:
                moves = mmtools.mcmc.SequenceMove([
                    mmtools.mcmc.LangevinDynamicsMove(n_steps=1),
                    mmtools.mcmc.MCRotationMove(),
                    mmtools.mcmc.GHMCMove(n_steps=1)
                ])

                sampler = self.SAMPLER(mcmc_moves=moves, number_of_iterations=3)
                reporter = self.REPORTER(storage_path, checkpoint_interval=1)
                self.call_sampler_create(sampler, reporter,
                                         thermodynamic_states, sampler_states,
                                         unsampled_states)
                # Run 3 iterations
                sampler.run(n_iterations=3)
                # store a copy of the original states
                original_states = sampler.sampler_states
                # Unallocate current objects and close reporter
                del sampler
                reporter.close()
                # recreate sampler from storage
                sampler = self.SAMPLER.from_storage(reporter)
                restored_states = sampler.sampler_states
                for original_state, restored_state in zip(original_states, restored_states):
                    assert np.allclose(original_state.positions, restored_state.positions)
                    assert np.allclose(original_state.velocities, restored_state.velocities)


    def test_last_iteration_functions(self):
        """Test that the last_iteration functions work right"""
        thermodynamic_states, sampler_states, unsampled_states = copy.deepcopy(self.alanine_test)
        with self.temporary_storage_path() as storage_path:
            # For this test, we simply check that the checkpointing writes on the interval
            # We don't care about the numbers, per se, but we do care about when things are written
            n_iterations = 10
            move = mmtools.mcmc.IntegratorMove(openmm.VerletIntegrator(1.0 * unit.femtosecond), n_steps=1)
            sampler = self.SAMPLER(mcmc_moves=move, number_of_iterations=n_iterations)
            reporter = self.REPORTER(storage_path, checkpoint_interval=2)
            self.call_sampler_create(sampler, reporter,
                                     thermodynamic_states, sampler_states,
                                     unsampled_states)
            # Propagate.
            sampler.run()
            reporter.close()
            reporter = self.REPORTER(storage_path, open_mode='a', checkpoint_interval=2)
            all_energies, _, _ = reporter.read_energies()
            # Break the checkpoint
            last_index = 4
            reporter.write_last_iteration(last_index)  # 5th iteration
            reporter.close()
            del reporter
            reporter = self.REPORTER(storage_path, open_mode='r', checkpoint_interval=2)
            # Check single positive index within range
            energies, _, _ = reporter.read_energies(1)
            assert np.all(energies == all_energies[1])
            # Check negative index was moved
            energies, _, _ = reporter.read_energies(-1)
            assert np.all(energies == all_energies[last_index])
            # Check slice
            energies, _, _ = reporter.read_energies()
            assert np.all(
                energies == all_energies[:last_index + 1])  # +1 to make sure we get the last index
            # Check negative slicing
            energies, _, _ = reporter.read_energies(slice(-1, None, -1))
            assert np.all(energies == all_energies[last_index::-1])
            # Errors
            with assert_raises(IndexError):
                reporter.read_energies(7)

    def test_separate_checkpoint_file(self):
        """Test that a separate checkpoint file can be created"""
        with self.temporary_storage_path() as storage_path:
            cp_file = 'checkpoint_file.nc'
            base, head = os.path.split(storage_path)
            cp_path = os.path.join(base, cp_file)
            reporter = self.REPORTER(storage_path, checkpoint_storage=cp_file, open_mode='w')
            reporter.close()
            assert os.path.isfile(storage_path)
            assert os.path.isfile(cp_path)

    def test_checkpoint_uuid_matching(self):
        """Test that checkpoint and storage files have the same UUID"""
        with self.temporary_storage_path() as storage_path:
            cp_file = 'checkpoint_file.nc'
            reporter = self.REPORTER(storage_path, checkpoint_storage=cp_file, open_mode='w')
            assert reporter._storage_checkpoint.UUID == reporter._storage_analysis.UUID

    def test_uuid_mismatch_errors(self):
        """Test that trying to use separate checkpoint file fails the UUID check"""
        with self.temporary_storage_path() as storage_path:
            file_base, ext = os.path.splitext(storage_path)
            storage_mod = file_base + '_mod' + ext
            cp_file_main = 'checkpoint_file.nc'
            cp_file_mod = 'checkpoint_mod.nc'
            reporter_main = self.REPORTER(storage_path, checkpoint_storage=cp_file_main, open_mode='w')
            reporter_main.close()
            reporter_mod = self.REPORTER(storage_mod, checkpoint_storage=cp_file_mod, open_mode='w')
            reporter_mod.close()
            del reporter_main, reporter_mod
            with assert_raises(IOError):
                self.REPORTER(storage_path, checkpoint_storage=cp_file_mod, open_mode='r')

    def test_analysis_opens_without_checkpoint(self):
        """Test that the analysis file can open without the checkpoint file"""
        with self.temporary_storage_path() as storage_path:
            cp_file = 'checkpoint_file.nc'
            cp_file_mod = 'checkpoint_mod.nc'
            reporter = self.REPORTER(storage_path, checkpoint_storage=cp_file, open_mode='w')
            reporter.close()
            del reporter
            self.REPORTER(storage_path, checkpoint_storage=cp_file_mod, open_mode='r')

    def test_storage_reporter_and_string(self):
        """Test that creating a MultiState by storage string and reporter is the same"""
        thermodynamic_states, sampler_states, unsampled_states = copy.deepcopy(self.alanine_test)
        with self.temporary_storage_path() as storage_path:
            n_iterations = 5
            move = mmtools.mcmc.IntegratorMove(openmm.VerletIntegrator(1.0 * unit.femtosecond), n_steps=1)
            sampler = self.SAMPLER(mcmc_moves=move, number_of_iterations=n_iterations)
            self.call_sampler_create(sampler, storage_path,
                                     thermodynamic_states, sampler_states,
                                     unsampled_states)
            # Propagate.
            sampler.run()
            energies_str, _, _ = sampler._reporter.read_energies()
            reporter = self.REPORTER(storage_path)
            del sampler
            sampler = self.SAMPLER.from_storage(reporter)
            energies_rep, _, _ = sampler._reporter.read_energies()
            assert np.all(energies_str == energies_rep)

    def test_online_analysis_works(self):
        """Test online analysis runs"""
        thermodynamic_states, sampler_states, unsampled_states = copy.deepcopy(self.alanine_test)
        with self.temporary_storage_path() as storage_path:
            n_iterations = 5
            online_interval = 1
            move = mmtools.mcmc.IntegratorMove(openmm.VerletIntegrator(1.0 * unit.femtosecond), n_steps=1)
            sampler = self.SAMPLER(mcmc_moves=move, number_of_iterations=n_iterations,
                                   online_analysis_interval=online_interval,
                                   online_analysis_minimum_iterations=3)
            self.call_sampler_create(sampler, storage_path,
                                     thermodynamic_states, sampler_states,
                                     unsampled_states)
            # Run
            sampler.run()

            def validate_this_test():
                # The stored values of online analysis should be up to date.
                last_written_free_energy = self.SAMPLER._read_last_free_energy(sampler._reporter, sampler.iteration)
                last_mbar_f_k, (last_free_energy, last_err_free_energy) = last_written_free_energy

                assert len(sampler._last_mbar_f_k) == len(thermodynamic_states)
                assert not np.all(sampler._last_mbar_f_k == 0)
                assert np.all(sampler._last_mbar_f_k == last_mbar_f_k)

                assert last_free_energy is not None

                # Error should not be 0 yet
                assert sampler._last_err_free_energy != 0

                assert sampler._last_err_free_energy == last_err_free_energy, \
                    ("SAMPLER %s : sampler._last_err_free_energy = %s, "
                     "last_err_free_energy = %s" % (self.SAMPLER.__name__,
                                                    sampler._last_err_free_energy,
                                                    last_err_free_energy)
                     )
            try:
                validate_this_test()
            except AssertionError as e:
                # Handle case where MBAR does not have a converged free energy yet by attempting to run longer
                # Only run up until we have sampled every state, or we hit some cycle limit
                cycle_limit = 20  # Put some upper limit of cycles
                cycles = 0
                while (not np.unique(sampler._reporter.read_replica_thermodynamic_states()).size == self.N_STATES
                       or cycles == cycle_limit):
                    sampler.extend(20)
                    try:
                        validate_this_test()
                    except AssertionError:
                        # If the max error count internally is reached, its a RuntimeError and won't be trapped
                        # So it will be raised correctly
                        pass
                    else:
                        # Test is good, let it pass by returning here
                        return
                # If we get here, we have not validated, raise original error
                raise e

    def test_online_analysis_stops(self):
        """Test online analysis will stop the simulation"""
        thermodynamic_states, sampler_states, unsampled_states = copy.deepcopy(self.alanine_test)
        with self.temporary_storage_path() as storage_path:
            n_iterations = 5
            online_interval = 1
            move = mmtools.mcmc.IntegratorMove(openmm.VerletIntegrator(1.0 * unit.femtosecond), n_steps=1)
            sampler = self.SAMPLER(mcmc_moves=move, number_of_iterations=n_iterations,
                                   online_analysis_interval=online_interval,
                                   online_analysis_minimum_iterations=0,
                                   online_analysis_target_error=np.inf)  # use infinite error to stop right away
            self.call_sampler_create(sampler, storage_path,
                                     thermodynamic_states, sampler_states,
                                     unsampled_states)
            # Run
            sampler.run()
            assert sampler._iteration < n_iterations
            assert sampler.is_completed

    def test_context_cache_default(self):
        """Test default behavior of context cache attributes."""
        sampler = self.SAMPLER()
        global_context_cache = cache.global_context_cache
        # Default is to use global context cache for both context cache attributes
        assert sampler.sampler_context_cache is global_context_cache
        assert sampler.energy_context_cache is global_context_cache

    def test_context_cache_energy_propagation(self):
        """Test specifying different context caches for energy and propagation in a short simulation."""
        thermodynamic_states, sampler_states, unsampled_states = copy.deepcopy(self.alanine_test)
        n_replicas = len(sampler_states)
        if n_replicas == 1:
            # This test is intended for use with more than one replica
            return

        with self.temporary_storage_path() as storage_path:
            # Create a replica exchange that propagates only 1 femtosecond
            # per iteration so that positions won't change much.
            move = mmtools.mcmc.IntegratorMove(openmm.VerletIntegrator(1.0 * unit.femtosecond), n_steps=1)
            sampler = self.SAMPLER(mcmc_moves=move)
            reporter = self.REPORTER(storage_path)
            self.call_sampler_create(sampler, reporter,
                                     thermodynamic_states, sampler_states,
                                     unsampled_states)
            # Set context cache attributes
            sampler.energy_context_cache = cache.ContextCache(capacity=None, time_to_live=None)
            sampler.sampler_context_cache = cache.ContextCache(capacity=None, time_to_live=None)
            # Compute energies
            sampler._compute_energies()
            # Check only energy context cache has been accessed
            assert sampler.energy_context_cache._lru._n_access > 0, \
                f"Expected more than 0 accesses, received {sampler.energy_context_cache._lru._n_access }."
            assert sampler.sampler_context_cache._lru._n_access == 0, \
                f"{sampler.sampler_context_cache._lru._n_access} accesses, expected 0."

            # Propagate replicas
            sampler._propagate_replicas()
            # Check propagation context cache has been accessed after propagation
            assert sampler.sampler_context_cache._lru._n_access > 0, \
                f"Expected more than 0 accesses, received {sampler.energy_context_cache._lru._n_access }."

    def test_real_time_analysis_yaml(self):
        """Test expected number of entries in real time analysis output yaml file."""
        thermodynamic_states, sampler_states, unsampled_states = copy.deepcopy(self.alanine_test)
        with self.temporary_storage_path() as storage_path:
            n_iterations = 13
            online_interval = 3
            expected_yaml_entries = int(n_iterations/online_interval)
            move = mmtools.mcmc.IntegratorMove(openmm.VerletIntegrator(1.0 * unit.femtosecond), n_steps=1)
            sampler = self.SAMPLER(mcmc_moves=move, number_of_iterations=n_iterations,
                                   online_analysis_interval=online_interval)
            self.call_sampler_create(sampler, storage_path,
                                     thermodynamic_states, sampler_states,
                                     unsampled_states)
            # Run
            sampler.run()
            # load file and check number of iterations
            storage_dir, reporter_filename = os.path.split(sampler._reporter._storage_analysis_file_path)
            # remove extension from filename
            yaml_prefix = os.path.splitext(reporter_filename)[0]
            output_filepath = f"{storage_dir}/{yaml_prefix}_real_time_analysis.yaml"
            with open(f"{storage_dir}/{yaml_prefix}_real_time_analysis.yaml") as yaml_file:
                yaml_contents = yaml.safe_load(yaml_file)
            # Make sure we get the correct number of entries
            assert len(yaml_contents) == expected_yaml_entries, \
                "Expected yaml entries do not match the actual number entries in the file."


#############

class TestExtraSamplersMultiStateSampler(TestMultiStateSampler):
    """Test MultiStateSampler with more samplers than states"""

    # ------------------------------------
    # VARIABLES TO SET FOR EACH TEST CLASS
    # ------------------------------------

    N_SAMPLERS = 5
    N_STATES = 3
    SAMPLER = MultiStateSampler
    REPORTER = MultiStateReporter


class TestReplicaExchange(TestMultiStateSampler):
    """Test suite for ReplicaExchange class."""

    # ------------------------------------
    # VARIABLES TO SET FOR EACH TEST CLASS
    # ------------------------------------

    N_SAMPLERS = 3
    N_STATES = 3
    SAMPLER = ReplicaExchangeSampler
    REPORTER = MultiStateReporter

    # --------------------------------------
    # Tests overwritten from base test suite
    # --------------------------------------

    def test_stored_properties(self):
        """Test that storage is kept in sync with options. Unique to ReplicaExchange"""
        additional_values = {}
        additional_values.update(self.property_creator('replica_mixing_scheme', 'replica_mixing_scheme', None, None))
        self.actual_stored_properties_check(additional_properties=additional_values)

    @attr('slow')  # Skip on Travis-CI
    def test_uniform_mixing(self):
        """Test that mixing is uniform for a sequence of harmonic oscillators.

        This test was implemented following choderalab/yank#1130. Briefly, when
        a repex calculation was distributed over multiple MPI processes, the
        odd replicas turned out to be much less diffusive than the even replicas.

        """
        temperature = 300.0 * unit.kelvin
        sigma = 1.0 * unit.angstrom  # Oscillator width
        #n_states = 50  # Number of harmonic oscillators.
        n_states = 6  # DEBUG
        n_states = 20  # DEBUG

        collision_rate = 10.0 / unit.picoseconds

        number_of_iterations = 2000
        number_of_iterations = 200 # DEBUG

        # Build an equidistant sequence of harmonic oscillators.
        sampler_states = []
        thermodynamic_states = []

        # The minima of the harmonic oscillators are 1 kT from each other.
        K = mmtools.constants.kB * temperature / sigma**2  # spring constant
        mass = 39.948*unit.amu # mass
        period = 2*np.pi*np.sqrt(mass/K)
        n_steps = 20  # Number of steps per iteration.
        timestep = period / n_steps
        spacing_sigma = 0.05
        oscillator = testsystems.HarmonicOscillator(K=K, mass=mass)

        for oscillator_idx in range(n_states):
            system = copy.deepcopy(oscillator.system)
            positions = copy.deepcopy(oscillator.positions)

            # Determine the position of the harmonic oscillator minimum.
            minimum_position = oscillator_idx * sigma * spacing_sigma
            minimum_position_unitless = minimum_position.value_in_unit_system(unit.md_unit_system)
            positions[0][0] = minimum_position

            # Create an oscillator starting from its minimum.
            force = system.getForce(0)
            assert force.getGlobalParameterName(1) == 'testsystems_HarmonicOscillator_x0'
            force.setGlobalParameterDefaultValue(1, minimum_position_unitless)

            thermodynamic_states.append(mmtools.states.ThermodynamicState(
                system=system, temperature=temperature))
            sampler_states.append(mmtools.states.SamplerState(positions))

        # Run a short repex simulation and gather data.
        with self.temporary_storage_path() as storage_path:
            # Create and run object.
            sampler = self.SAMPLER(
                mcmc_moves=mmtools.mcmc.LangevinDynamicsMove(timestep=timestep, collision_rate=collision_rate, n_steps=n_steps),
                number_of_iterations=number_of_iterations,
            )
            reporter = self.REPORTER(storage_path, checkpoint_interval=number_of_iterations)
            sampler.create(thermodynamic_states, sampler_states, reporter)
            #sampler.replica_mixing_scheme = 'swap-neighbors'
            sampler.replica_mixing_scheme = 'swap-all'
            sampler.run()

            # Retrieve from the reporter the mixing information before deleting.
            # Only the reporter from MPI node 0 should be open.
            n_accepted_matrix, n_proposed_matrix = mpiplus.run_single_node(
                task=reporter.read_mixing_statistics,
                rank=0, broadcast_result=True
            )
            replica_thermo_states = mpiplus.run_single_node(
                task=reporter.read_replica_thermodynamic_states,
                rank=0, broadcast_result=True
            )
            del sampler, reporter

        # No need to analyze the same data in multiple MPI processes.
        mpicomm = mpiplus.get_mpicomm()
        if mpicomm is not None and mpicomm.rank == 0:
            print('Acceptance matrix')
            print(n_accepted_matrix)
            print()

            # Count the number of visited states by each replica.
            replica_thermo_state_counts = np.empty(n_states)
            for replica_idx in range(n_states):
                state_trajectory = replica_thermo_states[:, replica_idx]
                #print(f"replica {replica_idx} : {''.join([ str(state) for state in state_trajectory ])}")
                n_visited_states = len(set(state_trajectory))
                replica_thermo_state_counts[replica_idx] = n_visited_states
                print(replica_idx, ':', n_visited_states)
            print()

            # Count the number of visited states by each MPI process.
            n_mpi_processes = mpicomm.size
            mpi_avg_thermo_state_counts = np.empty(n_mpi_processes)
            mpi_sem_thermo_state_counts = np.empty(n_mpi_processes)
            for mpi_idx in range(n_mpi_processes):
                # Find replicas assigned to this MPI process.
                replica_indices = list(i for i in range(n_states) if i % n_mpi_processes == mpi_idx)
                # Find the average number of states visited by
                # the replicas assigned to this MPI process.
                mpi_avg_thermo_state_counts[mpi_idx] = np.mean(replica_thermo_state_counts[replica_indices])
                mpi_sem_thermo_state_counts[mpi_idx] = np.std(replica_thermo_state_counts[replica_indices], ddof=1) / np.sqrt(len(replica_indices))

            # These should be roughly equal.
            print('MPI process mean number of thermo states visited:')
            for mpi_idx, (mean, sem) in enumerate(zip(mpi_avg_thermo_state_counts,
                                                      mpi_sem_thermo_state_counts)):
                print('{}: {} +- {}'.format(mpi_idx, mean, 2*sem))

            # Check if the confidence intervals overlap.
            def are_overlapping(interval1, interval2):
                return min(interval1[1], interval2[1]) - max(interval1[0], interval2[0]) > 0

            cis = [(mean-2*sem, mean+2*sem) for mean, sem in zip(mpi_avg_thermo_state_counts, mpi_sem_thermo_state_counts)]
            for i in range(1, len(cis)):
                assert are_overlapping(cis[0], cis[i])


class TestSingleReplicaSAMS(TestMultiStateSampler):
    """Test suite for SAMSSampler class."""

    # ------------------------------------
    # VARIABLES TO SET FOR EACH TEST CLASS
    # ------------------------------------

    N_SAMPLERS = 1
    N_STATES = 5
    SAMPLER = SAMSSampler
    REPORTER = MultiStateReporter

    # --------------------------------------
    # Tests overwritten from base test suite
    # --------------------------------------

    def test_stored_properties(self):
        """Test that storage is kept in sync with options. Unique to SAMSSampler"""
        additional_values = {}
        options = {
            'state_update_scheme': 'global-jump',
            'locality': None,
            'update_stages': 'one-stage',
            'weight_update_method': 'optimal',
            'adapt_target_probabilities': False,
            }
        for (name, value) in options.items():
            additional_values.update(self.property_creator(name, name, value, value))
        self.actual_stored_properties_check(additional_properties=additional_values)

    def test_state_histogram(self):
        """Ensure SAMS on-the-fly state histograms match actually visited states"""
        thermodynamic_states, sampler_states, unsampled_states = copy.deepcopy(self.alanine_test)
        with self.temporary_storage_path() as storage_path:
            # For this test, we simply check that the checkpointing writes on the interval
            # We don't care about the numbers, per se, but we do care about when things are written
            n_iterations = 10
            move = mmtools.mcmc.IntegratorMove(openmm.VerletIntegrator(1.0 * unit.femtosecond), n_steps=1)
            sampler = self.SAMPLER(mcmc_moves=move, number_of_iterations=n_iterations)
            reporter = self.REPORTER(storage_path, checkpoint_interval=2)
            self.call_sampler_create(sampler, reporter,
                                     thermodynamic_states, sampler_states,
                                     unsampled_states)
            # Propagate.
            sampler.run()
            reporter.close()
            reporter = self.REPORTER(storage_path, open_mode='a', checkpoint_interval=2)
            replica_thermodynamic_states = reporter.read_replica_thermodynamic_states()
            N_k, _ = np.histogram(replica_thermodynamic_states, bins=np.arange(-0.5, sampler.n_states + 0.5))
            assert np.all(sampler._state_histogram == N_k)

    # TODO: Test all update methods


class TestMultipleReplicaSAMS(TestSingleReplicaSAMS):
    """Test suite for SAMSSampler class."""

    # ------------------------------------
    # VARIABLES TO SET FOR EACH TEST CLASS
    # ------------------------------------

    N_SAMPLERS = 2

    # --------------------------------------
    # Tests overwritten from base test suite
    # --------------------------------------

    def test_stored_properties(self):
        """Test that storage is kept in sync with options. Unique to SAMSSampler"""
        additional_values = {}
        options = {
            'state_update_scheme': 'global-jump',
            'locality': None,
            'update_stages': 'two-stage',
            'weight_update_method' : 'rao-blackwellized',
            'adapt_target_probabilities': False,
            }
        for (name, value) in options.items():
            additional_values.update(self.property_creator(name, name, value, value))
        self.actual_stored_properties_check(additional_properties=additional_values)

    # TODO: Test all update methods


class TestParallelTempering(TestMultiStateSampler):

    # ------------------------------------
    # VARIABLES TO SET FOR EACH TEST CLASS
    # ------------------------------------
    try:
        from openmm import unit
    except ImportError:  # OpenMM < 7.6
        from simtk import unit
    N_SAMPLERS = 3
    N_STATES = 3
    SAMPLER = ParallelTemperingSampler
    REPORTER = MultiStateReporter
    MIN_TEMP = 300*unit.kelvin
    MAX_TEMP = 350*unit.kelvin

    # --------------------------------------
    # Optional helper function to overwrite.
    # --------------------------------------

    @classmethod
    def call_sampler_create(cls, sampler, reporter,
                            thermodynamic_states,
                            sampler_states,
                            unsampled_states):
        """
        Helper function to call the create method for the sampler
        ParallelTempering has a unique call
        """
        single_state = thermodynamic_states[0]
        # Allows initial thermodynamic states to be handled by the built in methods
        sampler.create(single_state, sampler_states, reporter,
                       min_temperature=cls.MIN_TEMP, max_temperature=cls.MAX_TEMP, n_temperatures=cls.N_STATES,
                       unsampled_thermodynamic_states=unsampled_states)

    def test_temperatures(self):
        """
        Test temperatures are created with desired range
        """
        thermodynamic_states, sampler_states, unsampled_states = copy.deepcopy(self.alanine_test)

        with self.temporary_storage_path() as storage_path:
            sampler = self.SAMPLER()
            reporter = self.REPORTER(storage_path, checkpoint_interval=1)

            self.call_sampler_create(sampler, reporter,
                thermodynamic_states, sampler_states,
                unsampled_states)
            try:
                from openmm import unit
            except ImportError:  # OpenMM < 7.6
                from simtk import unit
            temperatures = [state.temperature/unit.kelvin for state in sampler._thermodynamic_states] # in kelvin
            assert len(temperatures) == self.N_STATES, f"There are {len(temperatures)} thermodynamic states; expected {self.N_STATES}"
            assert np.isclose(min(temperatures), (self.MIN_TEMP/unit.kelvin)), f"Min temperature is {min(temperatures)} K; expected {(self.MIN_TEMP/unit.kelvin)} K"
            assert np.isclose(max(temperatures), (self.MAX_TEMP/unit.kelvin)), f"Max temperature is {max(temperatures)} K; expected {(self.MAX_TEMP/unit.kelvin)} K"

    # ----------------------------------
    # Methods overwritten from the Super
    # ----------------------------------

    @classmethod
    def _compute_energies_independently(cls, sampler):
        """
        Helper function to compute energies by hand.
        This is overwritten from Super.

        There is faster way to compute sampled states with ParallelTempering that is O(N) as is done in production,
        but the O(N^2) way should get it right as well and serves as a decent check
        """
        thermodynamic_states = sampler._thermodynamic_states
        unsampled_states = sampler._unsampled_states
        sampler_states = sampler._sampler_states

        n_states = len(thermodynamic_states)
        n_replicas = len(sampler_states)

        # Use the `ThermodynamicState.reduced_potential()` to ensure the fast
        # parallel tempering specific subclass implementation works as desired
        energy_thermodynamic_states = np.zeros((n_replicas, n_states))
        energy_unsampled_states = np.zeros((n_replicas, len(unsampled_states)))
        for energies, states in [(energy_thermodynamic_states, thermodynamic_states),
                                 (energy_unsampled_states, unsampled_states)]:
            for i, sampler_state in enumerate(sampler_states):
                for j, state in enumerate(states):
                    context, integrator = mmtools.cache.global_context_cache.get_context(state)
                    sampler_state.apply_to_context(context)
                    energies[i][j] = state.reduced_potential(context)
        return energy_thermodynamic_states, energy_unsampled_states


# ==============================================================================
# MAIN AND TESTS
# ==============================================================================

if __name__ == "__main__":

    # Test simple system of harmonic oscillators.
    # Disabled until we fix the test
    # test_replica_exchange()

    print('Creating class')
    repex = TestReplicaExchange()
    print('testing...')
    repex.test_uniform_mixing()
