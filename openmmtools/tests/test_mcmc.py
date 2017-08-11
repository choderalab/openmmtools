#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test State classes in mcmc.py.

"""

# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import pickle
from functools import partial

import nose
from pymbar import timeseries

from openmmtools import testsystems
from openmmtools.states import SamplerState, ThermodynamicState
from openmmtools.mcmc import *


# =============================================================================
# GLOBAL TEST CONSTANTS
# =============================================================================

# Test various combinations of systems and MCMC schemes
analytical_testsystems = [
    ("HarmonicOscillator", testsystems.HarmonicOscillator(),
        GHMCMove(timestep=10.0*unit.femtoseconds, n_steps=100)),
    ("HarmonicOscillator", testsystems.HarmonicOscillator(),
        WeightedMove([(GHMCMove(timestep=10.0*unit.femtoseconds, n_steps=100), 0.5),
                      (HMCMove(timestep=10*unit.femtosecond, n_steps=10), 0.5)])),
    ("HarmonicOscillatorArray", testsystems.HarmonicOscillatorArray(N=4),
        LangevinDynamicsMove(timestep=10.0*unit.femtoseconds, n_steps=100)),
    ("IdealGas", testsystems.IdealGas(nparticles=216),
        SequenceMove([HMCMove(timestep=10*unit.femtosecond, n_steps=10),
                      MonteCarloBarostatMove()]))
    ]

NSIGMA_CUTOFF = 6.0  # cutoff for significance testing

debug = True  # set to True only for manual debugging of this nose test


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_minimizer_all_testsystems():
    # testsystem_classes = testsystems.TestSystem.__subclasses__()
    testsystem_classes = [testsystems.AlanineDipeptideVacuum]

    for testsystem_class in testsystem_classes:
        class_name = testsystem_class.__name__
        logging.info("Testing minimization with testsystem %s" % class_name)

        testsystem = testsystem_class()
        sampler_state = SamplerState(testsystem.positions)
        thermodynamic_state = ThermodynamicState(testsystem.system, 300*unit.kelvin)

        # Create sampler for minimization.
        sampler = MCMCSampler(thermodynamic_state, sampler_state, move=None)
        sampler.minimize(max_iterations=0)

        # Check if NaN.
        err_msg = 'Minimization of system {} yielded NaN'.format(class_name)
        assert not sampler_state.has_nan(), err_msg


def test_mcmc_expectations():
    # Select system:
    for [system_name, testsystem, move] in analytical_testsystems:
        f = partial(subtest_mcmc_expectation, testsystem, move)
        f.description = "Testing MCMC expectation for %s" % system_name
        logging.info(f.description)
        yield f


def subtest_mcmc_expectation(testsystem, move):
    if debug:
        print(testsystem.__class__.__name__)
        print(str(move))

    # Retrieve system and positions.
    [system, positions] = [testsystem.system, testsystem.positions]

    # Test settings.
    temperature = 298.0 * unit.kelvin
    niterations = 500  # number of production iterations
    if system.usesPeriodicBoundaryConditions():
        pressure = 1.0*unit.atmosphere
    else:
        pressure = None

    # Compute properties.
    kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
    kT = kB * temperature
    ndof = 3 * system.getNumParticles() - system.getNumConstraints()

    # Create sampler and thermodynamic state.
    sampler_state = SamplerState(positions=positions)
    thermodynamic_state = ThermodynamicState(system=system,
                                             temperature=temperature,
                                             pressure=pressure)

    # Create MCMC sampler
    sampler = MCMCSampler(thermodynamic_state, sampler_state, move=move)

    # Accumulate statistics.
    x_n = np.zeros([niterations], np.float64)  # x_n[i] is the x position of atom 1 after iteration i, in angstroms
    potential_n = np.zeros([niterations], np.float64)  # potential_n[i] is the potential energy after iteration i, in kT
    kinetic_n = np.zeros([niterations], np.float64)  # kinetic_n[i] is the kinetic energy after iteration i, in kT
    temperature_n = np.zeros([niterations], np.float64)  # temperature_n[i] is the instantaneous kinetic temperature from iteration i, in K
    volume_n = np.zeros([niterations], np.float64)  # volume_n[i] is the volume from iteration i, in K
    for iteration in range(niterations):
        # Update sampler state.
        sampler.run(1)

        # Get statistics.
        potential_energy = sampler.sampler_state.potential_energy
        kinetic_energy = sampler.sampler_state.kinetic_energy
        instantaneous_temperature = kinetic_energy * 2.0 / ndof / kB
        volume = sampler.sampler_state.volume

        # Accumulate statistics.
        x_n[iteration] = sampler_state.positions[0, 0] / unit.angstroms
        potential_n[iteration] = potential_energy / kT
        kinetic_n[iteration] = kinetic_energy / kT
        temperature_n[iteration] = instantaneous_temperature / unit.kelvin
        volume_n[iteration] = volume / (unit.nanometers**3)

    # Compute expected statistics.
    if (hasattr(testsystem, 'get_potential_expectation') and
            testsystem.get_potential_standard_deviation(thermodynamic_state) / kT.unit != 0.0):
        assert potential_n.std() != 0.0, 'Test {} shows no potential fluctuations'.format(
            testsystem.__class__.__name__)

        potential_expectation = testsystem.get_potential_expectation(thermodynamic_state) / kT
        [t0, g, Neff_max] = timeseries.detectEquilibration(potential_n)
        potential_mean = potential_n[t0:].mean()
        dpotential_mean = potential_n[t0:].std() / np.sqrt(Neff_max)
        potential_error = potential_mean - potential_expectation
        nsigma = abs(potential_error) / dpotential_mean

        err_msg = ('Potential energy expectation\n'
                   'observed {:10.5f} +- {:10.5f}kT | expected {:10.5f} | '
                   'error {:10.5f} +- {:10.5f} ({:.1f} sigma) | t0 {:5d} | g {:5.1f} | Neff {:8.1f}\n'
                   '----------------------------------------------------------------------------').format(
            potential_mean, dpotential_mean, potential_expectation, potential_error, dpotential_mean, nsigma, t0, g, Neff_max)
        assert nsigma <= NSIGMA_CUTOFF, err_msg.format()
        if debug:
            print(err_msg)
    elif debug:
        print('Skipping potential expectation test.')

    if (hasattr(testsystem, 'get_volume_expectation') and
            testsystem.get_volume_standard_deviation(thermodynamic_state) / (unit.nanometers**3) != 0.0):
        assert volume_n.std() != 0.0, 'Test {} shows no volume fluctuations'.format(
            testsystem.__class__.__name__)

        volume_expectation = testsystem.get_volume_expectation(thermodynamic_state) / (unit.nanometers**3)
        [t0, g, Neff_max] = timeseries.detectEquilibration(volume_n)
        volume_mean = volume_n[t0:].mean()
        dvolume_mean = volume_n[t0:].std() / np.sqrt(Neff_max)
        volume_error = volume_mean - volume_expectation
        nsigma = abs(volume_error) / dvolume_mean

        err_msg = ('Volume expectation\n'
                   'observed {:10.5f} +- {:10.5f}kT | expected {:10.5f} | '
                   'error {:10.5f} +- {:10.5f} ({:.1f} sigma) | t0 {:5d} | g {:5.1f} | Neff {:8.1f}\n'
                   '----------------------------------------------------------------------------').format(
            volume_mean, dvolume_mean, volume_expectation, volume_error, dvolume_mean, nsigma, t0, g, Neff_max)
        assert nsigma <= NSIGMA_CUTOFF, err_msg.format()
        if debug:
            print(err_msg)
    elif debug:
        print('Skipping volume expectation test.')


def test_barostat_move_frequency():
    """MonteCarloBarostatMove restore barostat's frequency afterwards."""
    # Get periodic test case.
    for test_case in analytical_testsystems:
        testsystem = test_case[1]
        if testsystem.system.usesPeriodicBoundaryConditions():
            break
    assert testsystem.system.usesPeriodicBoundaryConditions(), "Can't find periodic test case!"

    sampler_state = SamplerState(testsystem.positions)
    thermodynamic_state = ThermodynamicState(testsystem.system, 298*unit.kelvin,
                                             1*unit.atmosphere)
    move = MonteCarloBarostatMove(n_attempts=5)

    # Test-precondition: the frequency must be different than 1 or it
    # will never change during the application of the MCMC move.
    old_frequency = thermodynamic_state.barostat.getFrequency()
    assert old_frequency != 1

    move.apply(thermodynamic_state, sampler_state)
    assert thermodynamic_state.barostat.getFrequency() == old_frequency


def test_context_cache():
    """Test configuration of the context cache."""
    testsystem = testsystems.AlanineDipeptideImplicit()
    sampler_state = SamplerState(testsystem.positions)
    thermodynamic_state = ThermodynamicState(testsystem.system, 300*unit.kelvin)

    # By default the global context cache is used.
    cache.global_context_cache.empty()  # Clear cache from previous tests.
    move = SequenceMove([LangevinDynamicsMove(n_steps=5), GHMCMove(n_steps=5)])
    move.apply(thermodynamic_state, sampler_state)
    assert len(cache.global_context_cache) == 2

    # Configuring the global cache works correctly.
    cache.global_context_cache = cache.ContextCache(time_to_live=1)
    move.apply(thermodynamic_state, sampler_state)
    assert len(cache.global_context_cache) == 1

    # The ContextCache creates only one context with compatible moves.
    cache.global_context_cache = cache.ContextCache(capacity=10, time_to_live=None)
    move = SequenceMove([LangevinDynamicsMove(n_steps=1), LangevinDynamicsMove(n_steps=1),
                         LangevinDynamicsMove(n_steps=1), LangevinDynamicsMove(n_steps=1)])
    move.apply(thermodynamic_state, sampler_state)
    assert len(cache.global_context_cache) == 1

    # We can configure a local context cache instead of global.
    local_cache = cache.ContextCache()
    move = SequenceMove([LangevinDynamicsMove(n_steps=5), GHMCMove(n_steps=5)],
                        context_cache=local_cache)
    for m in move:
        assert m.context_cache == local_cache

    # Running with the local cache doesn't affect the global one.
    cache.global_context_cache = cache.ContextCache()  # empty global
    move.apply(thermodynamic_state, sampler_state)
    assert len(cache.global_context_cache) == 0
    assert len(local_cache) == 2

    # DummyContextCache works for all platforms.
    platforms = utils.get_available_platforms()
    dummy_cache = cache.DummyContextCache()
    for platform in platforms:
        dummy_cache.platform = platform
        move = LangevinDynamicsMove(n_steps=5, context_cache=dummy_cache)
        move.apply(thermodynamic_state, sampler_state)
    assert len(cache.global_context_cache) == 0


def test_moves_serialization():
    """Test serialization of various MCMCMoves."""
    # Test cases.
    platform = openmm.Platform.getPlatformByName('Reference')
    context_cache = cache.ContextCache(capacity=1, time_to_live=1)
    dummy_cache = cache.DummyContextCache(platform=platform)
    test_cases = [
        IntegratorMove(openmm.VerletIntegrator(1.0*unit.femtosecond), n_steps=10),
        LangevinDynamicsMove(),
        LangevinSplittingDynamicsMove(),
        GHMCMove(),
        HMCMove(context_cache=context_cache),
        MonteCarloBarostatMove(context_cache=dummy_cache),
        SequenceMove(move_list=[LangevinDynamicsMove(), GHMCMove()]),
        WeightedMove(move_set=[(HMCMove(), 0.5), (MonteCarloBarostatMove(), 0.5)])
    ]
    for move in test_cases:
        original_pickle = pickle.dumps(move)
        serialized_move = utils.serialize(move)
        deserialized_move = utils.deserialize(serialized_move)
        deserialized_pickle = pickle.dumps(deserialized_move)
        assert original_pickle == deserialized_pickle


def test_move_restart():
    """Test optional restart move if NaN is detected."""
    n_restart_attempts = 5

    # We define a Move that counts the times it is attempted.
    class MyMove(BaseIntegratorMove):
        def __init__(self, **kwargs):
            super(MyMove, self).__init__(n_steps=1, n_restart_attempts=n_restart_attempts, **kwargs)
            self.attempted_count = 0

        def _get_integrator(self, thermodynamic_state):
            return integrators.GHMCIntegrator(temperature=300*unit.kelvin)

        def _before_integration(self, context, thermodynamic_state):
            self.attempted_count += 1

    # Create a system with an extra NaN particle.
    testsystem = testsystems.AlanineDipeptideVacuum()
    system = testsystem.system
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            break

    # Add a non-interacting particle to the system at NaN position.
    system.addParticle(39.9 * unit.amu)
    force.addParticle(0.0, 1.0, 0.0)
    particle_position = np.array([np.nan, 0.2, 0.2])
    positions = unit.Quantity(np.vstack((testsystem.positions, particle_position)),
                              unit=testsystem.positions.unit)

    # Create and run move. An IntegratoMoveError is raised.
    sampler_state = SamplerState(positions)
    thermodynamic_state = ThermodynamicState(system, 300*unit.kelvin)

    # We use a local context cache with Reference platform since on the
    # CPU platform CustomIntegrators raises an error with NaN particles.
    reference_platform = openmm.Platform.getPlatformByName('Reference')
    move = MyMove(context_cache=cache.ContextCache(platform=reference_platform))
    with nose.tools.assert_raises(IntegratorMoveError) as cm:
        move.apply(thermodynamic_state, sampler_state)

    # We have counted the correct number of restart attempts.
    assert move.attempted_count == n_restart_attempts + 1

    # Test serialization of the error.
    with utils.temporary_directory() as tmp_dir:
        prefix = os.path.join(tmp_dir, 'prefix')
        cm.exception.serialize_error(prefix)
        assert os.path.exists(prefix + '-move.json')
        assert os.path.exists(prefix + '-system.xml')
        assert os.path.exists(prefix + '-integrator.xml')
        assert os.path.exists(prefix + '-state.xml')


def test_metropolized_moves():
    """Test Displacement and Rotation moves."""
    testsystem = testsystems.AlanineDipeptideVacuum()
    original_sampler_state = SamplerState(testsystem.positions)
    thermodynamic_state = ThermodynamicState(testsystem.system, 300*unit.kelvin)

    all_metropolized_moves = MetropolizedMove.__subclasses__()
    for move_class in all_metropolized_moves:
        move = move_class(atom_subset=range(thermodynamic_state.n_particles))
        sampler_state = copy.deepcopy(original_sampler_state)

        # Start applying the move and remove one at each iteration tyring
        # to generate both an accepted and rejected move.
        old_n_accepted, old_n_proposed = 0, 0
        while len(move.atom_subset) > 0:
            initial_positions = copy.deepcopy(sampler_state.positions)
            move.apply(thermodynamic_state, sampler_state)
            final_positions = copy.deepcopy(sampler_state.positions)

            # If the move was accepted the positions should be different.
            if move.n_accepted > old_n_accepted:
                assert not np.allclose(initial_positions, final_positions)

            # If we have generated a rejection and an acceptance, test next move.
            if move.n_accepted > 0 and move.n_accepted != move.n_proposed:
                break

            # Try with a smaller subset.
            move.atom_subset = move.atom_subset[:-1]
            old_n_accepted, old_n_proposed = move.n_accepted, move.n_proposed

        # Check that we were able to generate both an accepted and a rejected move.
        assert len(move.atom_subset) != 0, ('Could not generate an accepted and rejected '
                                            'move for class {}'.format(move_class.__name__))


def test_langevin_splitting_move():
    """Test that the langevin splitting mcmc move works with different splittings"""
    splittings = ["V R O R V", "V R R R O R R R V", "O { V R V } O"]
    testsystem = testsystems.AlanineDipeptideVacuum()
    sampler_state = SamplerState(testsystem.positions)
    thermodynamic_state = ThermodynamicState(testsystem.system, 300*unit.kelvin)
    for splitting in splittings:
        move = LangevinSplittingDynamicsMove(splitting=splitting)
        # Create MCMC sampler
        sampler = MCMCSampler(thermodynamic_state, sampler_state, move=move)
        sampler.run(1)



# =============================================================================
# MAIN AND TESTS
# =============================================================================

if __name__ == "__main__":
    test_minimizer_all_testsystems()
