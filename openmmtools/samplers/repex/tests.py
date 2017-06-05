#!/usr/local/bin/env python

"""
Tests for replica-exchange algorithms.

TODO

* Create a few simulation objects on simple systems (e.g. harmonic oscillators?) and run multiple tests on each object?

"""

# =============================================================================================
# GLOBAL IMPORTS
# =============================================================================================

import os
import math
import copy
import pickle
import inspect
import contextlib

import yaml
import numpy as np
import scipy.integrate
from simtk import openmm, unit
from nose.plugins.attrib import attr
from nose.tools import assert_raises

import openmmtools as mmtools
from openmmtools import testsystems
from openmmtools.constants import kB
from openmmtools.distributed import mpi
from openmmtools import utils

from openmmtools import utils
from openmmtools.distributed import mpi

from .repex import Reporter, ReplicaExchange, _DictYamlLoader
from . import analyze

from openmmtools.constants import kB

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

def compute_harmonic_oscillator_expectations(K, temperature):
    """Compute mean and variance of potential and kinetic energies for a 3D harmonic oscillator.

    Notes
    -----
    Numerical quadrature is used to compute the mean and standard deviation of the potential energy.
    Mean and standard deviation of the kinetic energy, as well as the absolute free energy, is computed analytically.

    Parameters
    ----------
    K : simtk.unit.Quantity
        Spring constant.
    temperature : simtk.unit.Quantity
        Temperature.

    Returns
    -------
    values : dict

    TODO

    Replace this with built-in analytical expectations for new repex.testsystems classes.

    """

    values = dict()

    # Compute thermal energy and inverse temperature from specified temperature.
    kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
    kT = kB * temperature  # thermal energy
    beta = 1.0 / kT  # inverse temperature

    # Compute standard deviation along one dimension.
    sigma = 1.0 / unit.sqrt(beta * K)

    # Define limits of integration along r.
    r_min = 0.0 * unit.nanometers  # initial value for integration
    r_max = 10.0 * sigma      # maximum radius to integrate to

    # Compute mean and std dev of potential energy.
    V = lambda r : (K/2.0) * (r*unit.nanometers)**2 / unit.kilojoules_per_mole # potential in kJ/mol, where r in nm
    q = lambda r : 4.0 * math.pi * r**2 * math.exp(-beta * (K/2.0) * (r*unit.nanometers)**2) # q(r), where r in nm
    (IqV2, dIqV2) = scipy.integrate.quad(lambda r : q(r) * V(r)**2, r_min / unit.nanometers, r_max / unit.nanometers)
    (IqV, dIqV)   = scipy.integrate.quad(lambda r : q(r) * V(r), r_min / unit.nanometers, r_max / unit.nanometers)
    (Iq, dIq)     = scipy.integrate.quad(lambda r : q(r), r_min / unit.nanometers, r_max / unit.nanometers)
    values['potential'] = dict()
    values['potential']['mean'] = (IqV / Iq) * unit.kilojoules_per_mole
    values['potential']['stddev'] = (IqV2 / Iq) * unit.kilojoules_per_mole

    # Compute mean and std dev of kinetic energy.
    values['kinetic'] = dict()
    values['kinetic']['mean'] = (3./2.) * kT
    values['kinetic']['stddev'] = math.sqrt(3./2.) * kT

    # Compute dimensionless free energy.
    # f = - \ln \int_{-\infty}^{+\infty} \exp[-\beta K x^2 / 2]
    #   = - \ln \int_{-\infty}^{+\infty} \exp[-x^2 / 2 \sigma^2]
    #   = - \ln [\sqrt{2 \pi} \sigma]
    values['f'] = - np.log(2 * np.pi * (sigma / unit.angstroms)**2) * (3.0/2.0)

    return values


# ==============================================================================
# TEST ANALYSIS REPLICA EXCHANGE
# ==============================================================================

@attr('slow')  # Skip on Travis-CI
def test_replica_exchange(verbose=False, verbose_simulation=False):
    """Free energies and average potential energies of a 3D harmonic oscillator are correctly computed."""
    # Define mass of carbon atom.
    mass = 12.0 * unit.amu

    sampler_states = list()
    thermodynamic_states = list()
    analytical_results = list()
    f_i_analytical = list()  # Dimensionless free energies.
    u_i_analytical = list()  # Reduced potentials.

    # Define thermodynamic states.
    Ks = [500.00, 400.0, 300.0] * unit.kilocalories_per_mole / unit.angstroms**2  # Spring constants.
    temperatures = [300.0, 350.0, 400.0] * unit.kelvin  # Temperatures.
    for (K, temperature) in zip(Ks, temperatures):
        # Create harmonic oscillator system.
        testsystem = testsystems.HarmonicOscillator(K=K, mass=mass, mm=openmm)

        # Create thermodynamic state and save positions.
        system, positions = [testsystem.system, testsystem.positions]
        sampler_states.append(mmtools.states.SamplerState(positions))
        thermodynamic_states.append(mmtools.states.ThermodynamicState(system=system, temperature=temperature))

        # Store analytical results.
        results = compute_harmonic_oscillator_expectations(K, temperature)
        analytical_results.append(results)
        f_i_analytical.append(results['f'])
        reduced_potential = results['potential']['mean'] / (kB * temperature)
        u_i_analytical.append(reduced_potential)

    # Compute analytical Delta_f_ij
    nstates = len(f_i_analytical)
    f_i_analytical = np.array(f_i_analytical)
    u_i_analytical = np.array(u_i_analytical)
    s_i_analytical = u_i_analytical - f_i_analytical
    Delta_f_ij_analytical = np.zeros([nstates, nstates], np.float64)
    Delta_u_ij_analytical = np.zeros([nstates, nstates], np.float64)
    Delta_s_ij_analytical = np.zeros([nstates, nstates], np.float64)
    for i in range(nstates):
        for j in range(nstates):
            Delta_f_ij_analytical[i, j] = f_i_analytical[j] - f_i_analytical[i]
            Delta_u_ij_analytical[i, j] = u_i_analytical[j] - u_i_analytical[i]
            Delta_s_ij_analytical[i, j] = s_i_analytical[j] - s_i_analytical[i]

    # Create and configure simulation object.
    move = mmtools.mcmc.LangevinDynamicsMove(timestep=2.0*unit.femtoseconds,
                                             collision_rate=20.0/unit.picosecond,
                                             n_steps=500, reassign_velocities=True)
    simulation = ReplicaExchange(mcmc_moves=move, number_of_iterations=200)

    # Define file for temporary storage.
    with mmtools.utils.temporary_directory() as tmp_dir:
        storage = os.path.join(tmp_dir, 'test_storage.nc')
        reporter = Reporter(storage, checkpoint_interval=1)
        simulation.create(thermodynamic_states, sampler_states, reporter)

        # Run simulation we keep the debug info off during the simulation
        # to not clog the output, and reactivate it for analysis.
        utils.config_root_logger(verbose_simulation)
        simulation.run()

        # Create Analyzer.
        analyzer = analyze.get_analyzer(storage)

        # TODO: Check if deviations exceed tolerance.
        Delta_f_ij, dDelta_f_ij = analyzer.get_free_energy()
        error = np.abs(Delta_f_ij - Delta_f_ij_analytical)
        indices = np.where(dDelta_f_ij > 0.0)
        nsigma = np.zeros([nstates,nstates], np.float32)
        nsigma[indices] = error[indices] / dDelta_f_ij[indices]
        MAX_SIGMA = 6.0 # maximum allowed number of standard errors
        if np.any(nsigma > MAX_SIGMA):
            print("Delta_f_ij")
            print(Delta_f_ij)
            print("Delta_f_ij_analytical")
            print(Delta_f_ij_analytical)
            print("error")
            print(error)
            print("stderr")
            print(dDelta_f_ij)
            print("nsigma")
            print(nsigma)
            raise Exception("Dimensionless free energy difference exceeds MAX_SIGMA of %.1f" % MAX_SIGMA)

        Delta_u_ij, dDelta_u_ij = analyzer.get_enthalpy()
        error = Delta_u_ij - Delta_u_ij_analytical
        nsigma = np.zeros([nstates,nstates], np.float32)
        nsigma[indices] = error[indices] / dDelta_f_ij[indices]
        if np.any(nsigma > MAX_SIGMA):
            print("Delta_u_ij")
            print(Delta_u_ij)
            print("Delta_u_ij_analytical")
            print(Delta_u_ij_analytical)
            print("error")
            print(error)
            print("nsigma")
            print(nsigma)
            raise Exception("Dimensionless potential energy difference exceeds MAX_SIGMA of %.1f" % MAX_SIGMA)

        # Clean up.
        del simulation

    if verbose:
        print("PASSED.")


# ==============================================================================
# TEST REPORTER
# ==============================================================================

class TestReporter(object):
    """Test suite for Reporter class."""

    @staticmethod
    @contextlib.contextmanager
    def temporary_reporter(checkpoint_interval=1, checkpoint_storage_file=None):
        """Create and initialize a reporter in a temporary directory."""
        with mmtools.utils.temporary_directory() as tmp_dir_path:
            storage_file = os.path.join(tmp_dir_path, 'temp_dir/test_storage.nc')
            assert not os.path.isfile(storage_file)
            reporter = Reporter(storage=storage_file, open_mode='w',
                                checkpoint_interval=checkpoint_interval,
                                checkpoint_storage_file=checkpoint_storage_file)
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
            states_serialized = []
            for state_id in range(len(thermodynamic_states)):
                state_str = ncgrp_states.variables['state' + str(state_id)][0]
                state_dict = yaml.load(state_str, Loader=_DictYamlLoader)
                states_serialized.append(state_dict)
            unsampled_serialized = []
            for state_id in range(len(unsampled_states)):
                unsampled_str = ncgrp_unsampled.variables['state' + str(state_id)][0]
                unsampled_dict = yaml.load(unsampled_str, Loader=_DictYamlLoader)
                unsampled_serialized.append(unsampled_dict)

            # Two of the three ThermodynamicStates are compatible.
            assert isinstance(states_serialized[0]['standard_system'], str)
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

    def test_store_sampler_states(self):
        """Check correct storage of thermodynamic states."""
        with self.temporary_reporter() as reporter:
            # Create sampler states.
            alanine_test = testsystems.AlanineDipeptideVacuum()
            positions = alanine_test.positions
            box_vectors = alanine_test.system.getDefaultPeriodicBoxVectors()
            sampler_states = [mmtools.states.SamplerState(positions=positions, box_vectors=box_vectors)
                              for _ in range(2)]

            # Check that after writing and reading, states are identical.
            reporter.write_sampler_states(sampler_states, iteration=0)
            reporter.write_last_iteration(0)
            restored_sampler_states = reporter.read_sampler_states(iteration=0)
            for state, restored_state in zip(sampler_states, restored_sampler_states):
                assert np.allclose(state.positions, restored_state.positions)
                assert np.allclose(state.box_vectors / unit.nanometer, restored_state.box_vectors / unit.nanometer)

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
        thermodynamic_states_energies = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
        unsampled_states_energies = np.array([[1, 2], [2, 3.0], [3, 9.0]])

        with self.temporary_reporter() as reporter:
            reporter.write_energies(thermodynamic_states_energies, unsampled_states_energies, iteration=0)
            restored_ts, restored_us = reporter.read_energies(iteration=0)
            assert np.all(thermodynamic_states_energies == restored_ts)
            assert np.all(unsampled_states_energies == restored_us)

    def test_store_dict(self):
        """Check correct storage and restore of dictionaries."""
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
            reporter.write_dict('testdict', data)
            restored_data = reporter.read_dict('testdict')
            for key, value in data.items():
                restored_value = restored_data[key]
                err_msg = '{}, {}'.format(value, restored_value)
                try:
                    assert value == restored_value, err_msg
                except ValueError:  # array-like
                    assert np.all(value == restored_value)
                else:
                    assert type(value) == type(restored_value), err_msg

            # write_dict supports updates.
            data['mybool'] = True
            data['mystring'] = 'substituted'
            reporter.write_dict('testdict', data)
            restored_data = reporter.read_dict('testdict')
            assert restored_data['mybool'] is True
            assert restored_data['mystring'] == 'substituted'

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
# TEST REPLICA EXCHANGE
# ==============================================================================

class TestReplicaExchange(object):
    """Test suite for ReplicaExchange class."""

    @classmethod
    def setup_class(cls):
        """Shared test cases and variables."""
        n_states = 3

        # Test case with alanine in vacuum at 3 different positions and temperatures.
        # ---------------------------------------------------------------------------
        alanine_test = testsystems.AlanineDipeptideVacuum()

        # Translate the sampler states to be different one from each other.
        alanine_sampler_states = [mmtools.states.SamplerState(alanine_test.positions + 10*i*unit.nanometers)
                                  for i in range(n_states)]

        # Set increasing temperature.
        temperatures = [(300 + 10*i) * unit.kelvin for i in range(n_states)]
        alanine_thermodynamic_states = [mmtools.states.ThermodynamicState(alanine_test.system, temperatures[i])
                                        for i in range(n_states)]

        # No unsampled states for this test.
        cls.alanine_test = (alanine_thermodynamic_states, alanine_sampler_states, [])

        # Test case with host guest in implicit at 3 different positions and alchemical parameters.
        # -----------------------------------------------------------------------------------------
        hostguest_test = testsystems.HostGuestVacuum()
        factory = mmtools.alchemy.AbsoluteAlchemicalFactory()
        alchemical_region = mmtools.alchemy.AlchemicalRegion(alchemical_atoms=range(126, 156))
        hostguest_alchemical = factory.create_alchemical_system(hostguest_test.system, alchemical_region)

        # Translate the sampler states to be different one from each other.
        hostguest_sampler_states = [mmtools.states.SamplerState(hostguest_test.positions + 10*i*unit.nanometers)
                                    for i in range(n_states)]

        # Create the three basic thermodynamic states.
        temperatures = [(300 + 10*i) * unit.kelvin for i in range(n_states)]
        hostguest_thermodynamic_states = [mmtools.states.ThermodynamicState(hostguest_alchemical, temperatures[i])
                                          for i in range(n_states)]

        # Create alchemical states at different parameter values.
        alchemical_states = [mmtools.alchemy.AlchemicalState.from_system(hostguest_alchemical)
                             for _ in range(n_states)]
        for i, alchemical_state in enumerate(alchemical_states):
            alchemical_state.set_alchemical_parameters(float(i) / (n_states - 1))

        # Create compound states.
        hostguest_compound_states = list()
        for i in range(n_states):
            hostguest_compound_states.append(
                mmtools.states.CompoundThermodynamicState(thermodynamic_state=hostguest_thermodynamic_states[i],
                                                          composable_states=[alchemical_states[i]])
            )

        # Unsampled states.
        nonalchemical_state = mmtools.states.ThermodynamicState(hostguest_test.system, temperatures[0])
        hostguest_unsampled_states = [copy.deepcopy(nonalchemical_state)]

        cls.hostguest_test = (hostguest_compound_states, hostguest_sampler_states, hostguest_unsampled_states)

    @staticmethod
    @contextlib.contextmanager
    def temporary_storage_path():
        """Generate a storage path in a temporary folder and share it.

        It makes it possible to run tests on multiple nodes with MPI.

        """
        mpicomm = mpi.get_mpicomm()
        with mmtools.utils.temporary_directory() as tmp_dir_path:
            storage_file_path = os.path.join(tmp_dir_path, 'test_storage.nc')
            if mpicomm is not None:
                storage_file_path = mpicomm.bcast(storage_file_path, root=0)
            yield storage_file_path

    @staticmethod
    def get_node_replica_ids(tot_n_replicas):
        """Return the indices of the replicas that this node is responsible for."""
        mpicomm = mpi.get_mpicomm()
        if mpicomm is None or mpicomm.rank == 0:
            return set(range(tot_n_replicas))
        else:
            return set(range(mpicomm.rank, tot_n_replicas, mpicomm.size))

    def test_repex_create(self):
        """Test creation of a new ReplicaExchange simulation.

        Checks that the storage file is correctly initialized with all the
        information needed. With MPI, this checks that only node 0 has an
        open Reporter for writing.

        """
        thermodynamic_states, sampler_states, unsampled_states = copy.deepcopy(self.alanine_test)
        n_states = len(thermodynamic_states)

        # Remove one sampler state to verify distribution over states.
        sampler_states = sampler_states[:-1]

        with self.temporary_storage_path() as storage_path:
            repex = ReplicaExchange()

            # Create simulation and storage file.
            reporter = Reporter(storage_path, checkpoint_interval=1)
            repex.create(thermodynamic_states, sampler_states, storage=reporter,
                         unsampled_thermodynamic_states=unsampled_states)

            # Check that reporter has reporter only if rank 0.
            mpicomm = mpi.get_mpicomm()
            if mpicomm is None or mpicomm.rank == 0:
                assert repex._reporter.is_open()
            else:
                assert not repex._reporter.is_open()

            # Open reporter to read stored data.
            reporter = Reporter(storage_path, open_mode='r', checkpoint_interval=1)

            # The n_states-1 sampler states have been distributed to n_states replica.
            restored_sampler_states = reporter.read_sampler_states(iteration=0)
            assert len(repex._sampler_states) == n_states
            assert len(restored_sampler_states) == n_states
            assert np.allclose(restored_sampler_states[0].positions, repex._sampler_states[0].positions)

            # MCMCMove was stored correctly.
            restored_mcmc_moves = reporter.read_mcmc_moves()
            assert len(repex._mcmc_moves) == n_states
            assert len(restored_mcmc_moves) == n_states
            for repex_move, restored_move in zip(repex._mcmc_moves, restored_mcmc_moves):
                assert isinstance(repex_move, mmtools.mcmc.LangevinDynamicsMove)
                assert isinstance(restored_move, mmtools.mcmc.LangevinDynamicsMove)

            # Options have been stored.
            option_names, _, _, defaults = inspect.getargspec(repex.__init__)
            option_names = option_names[2:]  # Discard 'self' and 'mcmc_moves' arguments.
            defaults = defaults[1:]  # Discard 'mcmc_moves' default.
            options = reporter.read_dict('options')
            assert len(options) == len(defaults)
            for key, value in zip(option_names, defaults):
                assert options[key] == value
                assert getattr(repex, '_' + key) == value

            # A default title has been added to the stored metadata.
            metadata = reporter.read_dict('metadata')
            assert len(metadata) == 1
            assert repex.metadata['title'] == metadata['title']

    def test_from_storage(self):
        """Test that from_storage completely restore ReplicaExchange.

        Checks that the static constructor ReplicaExchange.from_storage()
        restores the simulation object in the exact same state as the last
        iteration.

        """
        thermodynamic_states, sampler_states, unsampled_states = copy.deepcopy(self.hostguest_test)
        n_replicas = len(thermodynamic_states)

        with self.temporary_storage_path() as storage_path:
            number_of_iterations = 3
            move = mmtools.mcmc.LangevinDynamicsMove(n_steps=1)
            repex = ReplicaExchange(mcmc_moves=move, number_of_iterations=number_of_iterations)
            reporter = Reporter(storage_path, checkpoint_interval=1)
            repex.create(thermodynamic_states, sampler_states, storage=reporter,
                         unsampled_thermodynamic_states=unsampled_states)

            # Test at the beginning and after few iterations.
            for iteration in range(2):
                # Store the state of the initial repex object (its __dict__). We leave the
                # reporter out because when the NetCDF file is copied, it runs into issues.
                original_dict = copy.deepcopy({k: v for k, v in repex.__dict__.items() if not k == '_reporter'})

                # Delete repex to close reporter before creating a new one
                # to avoid weird issues with multiple NetCDF files open.
                del repex
                reporter.close()
                repex = ReplicaExchange.from_storage(reporter)
                restored_dict = copy.deepcopy({k: v for k, v in repex.__dict__.items() if not k == '_reporter'})

                # Check thermodynamic states.
                original_ts = original_dict.pop('_thermodynamic_states')
                restored_ts = restored_dict.pop('_thermodynamic_states')
                check_thermodynamic_states_equality(original_ts, restored_ts)

                # Check unsampled thermodynamic states.
                original_us = original_dict.pop('_unsampled_states')
                restored_us = restored_dict.pop('_unsampled_states')
                check_thermodynamic_states_equality(original_us, restored_us)

                # The reporter of the restored simulation must be open only in node 0.
                mpicomm = mpi.get_mpicomm()
                if mpicomm is None or mpicomm.rank == 0:
                    assert repex._reporter.is_open()
                else:
                    assert not repex._reporter.is_open()

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
                original_ets = original_dict.pop('_energy_thermodynamic_states')
                restored_ets = restored_dict.pop('_energy_thermodynamic_states')
                original_eus = original_dict.pop('_energy_unsampled_states')
                restored_eus = restored_dict.pop('_energy_unsampled_states')
                for replica_id in range(n_replicas):
                    if replica_id in node_replica_ids:
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

                # Remove cached values that are not resumed.
                original_dict.pop('_cached_transition_counts', None)
                original_dict.pop('_cached_last_replica_thermodynamic_states', None)

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
                    repex.run(number_of_iterations)

    def test_stored_properties(self):
        """Test that storage is kept in sync with options."""
        thermodynamic_states, sampler_states, unsampled_states = copy.deepcopy(self.alanine_test)

        with self.temporary_storage_path() as storage_path:
            repex = ReplicaExchange()
            reporter = Reporter(storage_path, checkpoint_interval=1)
            repex.create(thermodynamic_states, sampler_states, storage=reporter,
                         unsampled_thermodynamic_states=unsampled_states)

            # Update options and check the storage is synchronized.
            repex.number_of_iterations = 123
            repex.replica_mixing_scheme = 'none'

            # Displace positions of the first sampler state.
            sampler_states = repex.sampler_states
            original_positions = copy.deepcopy(sampler_states[0].positions)
            displacement_vector = np.ones(3) * unit.angstroms
            sampler_states[0].positions += displacement_vector
            repex.sampler_states = sampler_states

            mpicomm = mpi.get_mpicomm()
            if mpicomm is None or mpicomm.rank == 0:
                reporter.close()
                reporter = Reporter(storage_path, open_mode='r')
                restored_options = reporter.read_dict('options')
                assert restored_options['number_of_iterations'] == 123
                assert restored_options['replica_mixing_scheme'] == 'none'

                restored_sampler_states = reporter.read_sampler_states(iteration=0)
                assert np.allclose(restored_sampler_states[0].positions,
                                   original_positions + displacement_vector)

    def test_propagate_replicas(self):
        """Test method _propagate_replicas from ReplicaExchange.

        The purpose of this test is mainly to make sure that MPI doesn't mix
        the information of the propagated StateSamplers when it communicates
        the new positions and box vectors.

        """
        thermodynamic_states, sampler_states, unsampled_states = copy.deepcopy(self.alanine_test)
        n_states = len(thermodynamic_states)

        with self.temporary_storage_path() as storage_path:
            # For this test to work, positions should be the same but
            # translated, so that minimized positions should satisfy
            # the same condition.
            original_diffs = [np.average(sampler_states[i].positions - sampler_states[i+1].positions)
                              for i in range(n_states - 1)]
            assert not np.allclose(original_diffs, [0 for _ in range(n_states - 1)])

            # Create a replica exchange that propagates only 1 femtosecond
            # per iteration so that positions won't change much.
            move = mmtools.mcmc.IntegratorMove(openmm.VerletIntegrator(1.0*unit.femtosecond), n_steps=1)
            repex = ReplicaExchange(mcmc_moves=move)
            reporter = Reporter(storage_path)
            repex.create(thermodynamic_states, sampler_states, storage=reporter,
                         unsampled_thermodynamic_states=unsampled_states)

            # Propagate.
            repex._propagate_replicas()

            # The relative positions between the new sampler states should
            # be still translated the same way (i.e. we are not assigning
            # the minimized positions to the incorrect sampler states).
            new_sampler_states = repex._sampler_states
            new_diffs = [np.average(new_sampler_states[i].positions - new_sampler_states[i+1].positions)
                         for i in range(n_states - 1)]
            assert np.allclose(original_diffs, new_diffs)

    def test_compute_energies(self):
        """Test method _compute_energies from ReplicaExchange.

        The purpose of this test is mainly to make sure that MPI doesn't mix
        the information of the thermodynamics and unsampled ThermodynamicStates
        when it communicates them to the other nodes.

        """
        thermodynamic_states, sampler_states, unsampled_states = copy.deepcopy(self.hostguest_test)
        n_replicas = len(thermodynamic_states)

        with self.temporary_storage_path() as storage_path:
            repex = ReplicaExchange()
            reporter = Reporter(storage_path, checkpoint_interval=1)
            repex.create(thermodynamic_states, sampler_states, storage=reporter,
                         unsampled_thermodynamic_states=unsampled_states)

            # Let ReplicaExchange distribute the computation of energies among nodes.
            repex._compute_energies()

            # Compute the energies independently.
            energy_thermodynamic_states = np.zeros((n_replicas, n_replicas))
            energy_unsampled_states = np.zeros((n_replicas, len(unsampled_states)))
            for energies, states in [(energy_thermodynamic_states, thermodynamic_states),
                                     (energy_unsampled_states, unsampled_states)]:
                for i, sampler_state in enumerate(sampler_states):
                    for j, state in enumerate(states):
                        context, integrator = mmtools.cache.global_context_cache.get_context(state)
                        sampler_state.apply_to_context(context)
                        energies[i][j] = state.reduced_potential(context)

            # Only node 0 has all the energies.
            mpicomm = mpi.get_mpicomm()
            if mpicomm is None or mpicomm.rank == 0:
                assert np.allclose(repex._energy_thermodynamic_states, energy_thermodynamic_states)
                assert np.allclose(repex._energy_unsampled_states, energy_unsampled_states)

    def test_minimize(self):
        """Test ReplicaExchange minimize method.

        The purpose of this test is mainly to make sure that MPI doesn't mix
        the information of the minimized StateSamplers when it communicates
        the new positions. It also checks that the energies are effectively
        decreased.

        """
        thermodynamic_states, sampler_states, unsampled_states = copy.deepcopy(self.alanine_test)
        n_states = len(thermodynamic_states)

        with self.temporary_storage_path() as storage_path:
            repex = ReplicaExchange()
            reporter = Reporter(storage_path, checkpoint_interval=1)
            repex.create(thermodynamic_states, sampler_states, storage=reporter,
                         unsampled_thermodynamic_states=unsampled_states)

            # For this test to work, positions should be the same but
            # translated, so that minimized positions should satisfy
            # the same condition.
            original_diffs = [np.average(sampler_states[i].positions - sampler_states[i+1].positions)
                              for i in range(n_states - 1)]
            assert not np.allclose(original_diffs, [0 for _ in range(n_states - 1)])

            # Compute initial energies.
            repex._compute_energies()
            original_energies = [repex._energy_thermodynamic_states[i, i] for i in range(n_states)]

            # Minimize.
            repex.minimize()

            # The relative positions between the new sampler states should
            # be still translated the same way (i.e. we are not assigning
            # the minimized positions to the incorrect sampler states).
            new_sampler_states = repex._sampler_states
            new_diffs = [np.average(new_sampler_states[i].positions - new_sampler_states[i+1].positions)
                         for i in range(n_states - 1)]
            assert np.allclose(original_diffs, new_diffs)

            # Each replica keeps only the info for the replicas it is
            # responsible for to minimize network traffic.
            node_replica_ids = self.get_node_replica_ids(n_states)

            # The energies have been minimized.
            repex._compute_energies()
            for i in node_replica_ids:
                assert repex._energy_thermodynamic_states[i, i] < original_energies[i]

            # The storage has been updated.
            reporter.close()
            if len(node_replica_ids) == n_states:
                reporter = Reporter(storage_path, open_mode='r')
                stored_sampler_states = reporter.read_sampler_states(iteration=0)
                for new_state, stored_state in zip(new_sampler_states, stored_sampler_states):
                    assert np.allclose(new_state.positions, stored_state.positions)

    def test_equilibrate(self):
        """Test equilibration of ReplicaExchange simulation.

        During equilibration, we set temporarily different MCMCMoves. This checks
        that they are restored correctly. It also checks that the storage has the
        updated positions.

        """
        thermodynamic_states, sampler_states, unsampled_states = copy.deepcopy(self.alanine_test)
        n_states = len(thermodynamic_states)

        with self.temporary_storage_path() as storage_path:
            # We create a ReplicaExchange with a GHMC move but use Langevin for equilibration.
            repex = ReplicaExchange(mcmc_moves=mmtools.mcmc.GHMCMove())
            reporter = Reporter(storage_path, checkpoint_interval=1)
            repex.create(thermodynamic_states, sampler_states, storage=reporter,
                         unsampled_thermodynamic_states=unsampled_states)

            # Equilibrate.
            equilibration_move = mmtools.mcmc.LangevinDynamicsMove(n_steps=1)
            repex.equilibrate(n_iterations=10, mcmc_moves=equilibration_move)
            assert isinstance(repex._mcmc_moves[0], mmtools.mcmc.GHMCMove)

            # Each replica keeps only the info for the replicas it is
            # responsible for to minimize network traffic.
            node_replica_ids = self.get_node_replica_ids(n_states)

            # The storage has been updated.
            reporter.close()
            if len(node_replica_ids) == n_states:
                reporter = Reporter(storage_path, open_mode='r', checkpoint_interval=1)
                stored_sampler_states = reporter.read_sampler_states(iteration=0)
                for new_state, stored_state in zip(repex._sampler_states, stored_sampler_states):
                    assert np.allclose(new_state.positions, stored_state.positions)

            # We are still at iteration 0.
            assert repex._iteration == 0

    def test_run_extend(self):
        """Test methods run and extend of ReplicaExchange."""
        test_cases = [self.alanine_test, self.hostguest_test]

        for test_case in test_cases:
            thermodynamic_states, sampler_states, unsampled_states = copy.deepcopy(test_case)

            with self.temporary_storage_path() as storage_path:
                moves = mmtools.mcmc.SequenceMove([
                    mmtools.mcmc.LangevinDynamicsMove(n_steps=1),
                    mmtools.mcmc.MCRotationMove(),
                    mmtools.mcmc.GHMCMove(n_steps=1)
                ])
                repex = ReplicaExchange(mcmc_moves=moves, number_of_iterations=2)
                reporter = Reporter(storage_path, checkpoint_interval=1)
                repex.create(thermodynamic_states, sampler_states, storage=reporter,
                             unsampled_thermodynamic_states=unsampled_states)

                # ReplicaExchange.run doesn't go past number_of_iterations.
                repex.run(n_iterations=3)
                assert repex.iteration == 2

                # ReplicaExchange.extend does.
                repex.extend(n_iterations=2)
                assert repex.iteration == 4

                # All replicas must have moves with updated statistics.
                for sequence_move in repex._mcmc_moves:
                    # LangevinDynamicsMove doesn't have statistics.
                    for move_id in range(1, 2):
                        assert sequence_move.move_list[move_id].n_proposed == 4

                # The MCMCMoves statistics in the storage are updated.
                mpicomm = mpi.get_mpicomm()
                if mpicomm is None or mpicomm.rank == 0:
                    reporter.close()
                    reporter = Reporter(storage_path, open_mode='r', checkpoint_interval=1)
                    restored_mcmc_moves = reporter.read_mcmc_moves()
                    for sequence_move in restored_mcmc_moves:
                        # LangevinDynamicsMove doesn't have statistics.
                        for move_id in range(1, 2):
                            assert sequence_move.move_list[move_id].n_proposed == 4

    def test_checkpointing(self):
        """Test that checkpointing writes infrequently"""
        thermodynamic_states, sampler_states, unsampled_states = copy.deepcopy(self.alanine_test)

        with self.temporary_storage_path() as storage_path:
            # For this test, we simply check that the checkpointing writes on the interval
            # We don't care about the numbers, per se, but we do care about when things are written
            n_iterations = 3
            move = mmtools.mcmc.IntegratorMove(openmm.VerletIntegrator(1.0*unit.femtosecond), n_steps=1)
            reporter = Reporter(storage_path, checkpoint_interval=2)
            repex = ReplicaExchange(mcmc_moves=move, number_of_iterations=n_iterations)
            repex.create(thermodynamic_states, sampler_states, storage=reporter,
                         unsampled_thermodynamic_states=unsampled_states)
            # Propagate.
            repex.run()
            reporter.close()
            reporter = Reporter(storage_path, open_mode='r', checkpoint_interval=2)
            for i in range(n_iterations):
                energies, _ = reporter.read_energies(i)
                states = reporter.read_sampler_states(i)
                assert type(energies) is np.ndarray
                if reporter._calculate_checkpoint_iteration(i) is not None:
                    assert type(states[0].positions) is unit.Quantity
                else:
                    assert states is None

    def test_last_iteration_functions(self):
        """Test that the last_iteration functions work right"""
        """Test that checkpointing writes infrequently"""
        thermodynamic_states, sampler_states, unsampled_states = copy.deepcopy(self.alanine_test)
        with self.temporary_storage_path() as storage_path:
            # For this test, we simply check that the checkpointing writes on the interval
            # We don't care about the numbers, per se, but we do care about when things are written
            n_iterations = 10
            move = mmtools.mcmc.IntegratorMove(openmm.VerletIntegrator(1.0*unit.femtosecond), n_steps=1)
            repex = ReplicaExchange(mcmc_moves=move, number_of_iterations=n_iterations)
            reporter = Reporter(storage_path, checkpoint_interval=2)
            repex.create(thermodynamic_states, sampler_states, storage=reporter,
                         unsampled_thermodynamic_states=unsampled_states)
            # Propagate.
            repex.run()
            reporter.close()
            reporter = Reporter(storage_path, open_mode='a', checkpoint_interval=2)
            all_energies, _ = reporter.read_energies()
            # Break the checkpoint
            last_index = 4
            reporter.write_last_iteration(last_index)  # 5th iteration
            reporter.close()
            del reporter
            reporter = Reporter(storage_path, open_mode='r', checkpoint_interval=2)
            # Check single positive index within range
            energies , _= reporter.read_energies(1)
            assert np.all(energies == all_energies[1])
            # Check negative index was moved
            energies, _ = reporter.read_energies(-1)
            assert np.all(energies == all_energies[last_index])
            # Check slice
            energies, _ = reporter.read_energies()
            assert np.all(energies == all_energies[:last_index+1])  # +1 to make sure we get the last index
            # Check negative slicing
            energies, _ = reporter.read_energies(slice(-1, None, -1))
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
            reporter = Reporter(storage_path, checkpoint_storage_file=cp_file, open_mode='w')
            reporter.close()
            assert os.path.isfile(storage_path)
            assert os.path.isfile(cp_path)

    def test_checkpoint_uuid_matching(self):
        """Test that checkpoint and storage files have the same UUID"""
        with self.temporary_storage_path() as storage_path:
            cp_file = 'checkpoint_file.nc'
            reporter = Reporter(storage_path, checkpoint_storage_file=cp_file, open_mode='w')
            assert reporter._storage_checkpoint.UUID == reporter._storage_analysis.UUID

    def test_uuid_mismatch_errors(self):
        """Test that trying to use separate checkpoint file fails the UUID check"""
        with self.temporary_storage_path() as storage_path:
            file_base, ext = os.path.splitext(storage_path)
            storage_mod = file_base + '_mod' + ext
            cp_file_main = 'checkpoint_file.nc'
            cp_file_mod = 'checkpoint_mod.nc'
            reporter_main = Reporter(storage_path, checkpoint_storage_file=cp_file_main, open_mode='w')
            reporter_main.close()
            reporter_mod = Reporter(storage_mod, checkpoint_storage_file=cp_file_mod, open_mode='w')
            reporter_mod.close()
            del reporter_main, reporter_mod
            with assert_raises(IOError):
                Reporter(storage_path, checkpoint_storage_file=cp_file_mod, open_mode='r')

    def test_analysis_opens_without_checkpoint(self):
        """Test that the analysis file can open without the checkpoint file"""
        with self.temporary_storage_path() as storage_path:
            cp_file = 'checkpoint_file.nc'
            cp_file_mod = 'checkpoint_mod.nc'
            reporter = Reporter(storage_path, checkpoint_storage_file=cp_file, open_mode='w')
            reporter.close()
            del reporter
            Reporter(storage_path, checkpoint_storage_file=cp_file_mod, open_mode='r')

    def test_storage_reporter_and_string(self):
        """Test that creating a repex by storage string and reporter is the same"""
        thermodynamic_states, sampler_states, unsampled_states = copy.deepcopy(self.alanine_test)
        with self.temporary_storage_path() as storage_path:
            n_iterations = 5
            move = mmtools.mcmc.IntegratorMove(openmm.VerletIntegrator(1.0*unit.femtosecond), n_steps=1)
            repex = ReplicaExchange(mcmc_moves=move, number_of_iterations=n_iterations)
            repex.create(thermodynamic_states, sampler_states, storage=storage_path,
                         unsampled_thermodynamic_states=unsampled_states)
            # Propagate.
            repex.run()
            energies_str, _ = repex._reporter.read_energies()
            reporter = Reporter(storage_path)
            del repex
            repex = ReplicaExchange.from_storage(reporter)
            energies_rep, _ = repex._reporter.read_energies()
            assert np.all(energies_str == energies_rep)

# ==============================================================================
# MAIN AND TESTS
# ==============================================================================

if __name__ == "__main__":
    # Configure logger.
    utils.config_root_logger(False)

    # Try MPI, if possible.
    try:
        mpicomm = utils.initialize_mpi()
        if mpicomm.rank == 0:
            print("MPI initialized successfully.")
    except Exception as e:
        print(e)
        print("Could not start MPI. Using serial code instead.")
        mpicomm = None

    # Test simple system of harmonic oscillators.
    # Disabled until we fix the test
    # test_hamiltonian_exchange(mpicomm)
    test_replica_exchange(mpicomm)
