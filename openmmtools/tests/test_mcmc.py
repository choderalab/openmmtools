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

from pymbar import timeseries
from functools import partial

from openmmtools import testsystems, utils
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
        WeightedMove({GHMCMove(timestep=10.0*unit.femtoseconds, n_steps=100): 0.5,
                      HMCMove(timestep=10*unit.femtosecond, n_steps=10): 0.5})),
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
    nequil = 10  # number of equilibration iterations
    niterations = 40  # number of production iterations
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

    # Create MCMC sampler and equilibrate.
    sampler = MCMCSampler(thermodynamic_state, sampler_state, move=move)
    sampler.run(nequil)

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
        potential_mean = potential_n.mean()
        g = timeseries.statisticalInefficiency(potential_n, fast=True)
        dpotential_mean = potential_n.std() / np.sqrt(niterations / g)
        potential_error = potential_mean - potential_expectation
        nsigma = abs(potential_error) / dpotential_mean

        err_msg = ('Potential energy expectation\n'
                   'observed {:10.5f} +- {:10.5f}kT | expected {:10.5f} | '
                   'error {:10.5f} +- {:10.5f} ({:.1f} sigma)\n'
                   '----------------------------------------------------------------------------').format(
            potential_mean, dpotential_mean, potential_expectation, potential_error, dpotential_mean, nsigma)
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
        volume_mean = volume_n.mean()
        g = timeseries.statisticalInefficiency(volume_n, fast=True)
        dvolume_mean = volume_n.std() / np.sqrt(niterations / g)
        volume_error = volume_mean - volume_expectation
        nsigma = abs(volume_error) / dvolume_mean

        err_msg = ('Volume expectation\n'
                   'observed {:10.5f} +- {:10.5f}kT | expected {:10.5f} | '
                   'error {:10.5f} +- {:10.5f} ({:.1f} sigma)\n'
                   '----------------------------------------------------------------------------').format(
            volume_mean, dvolume_mean, volume_expectation, volume_error, dvolume_mean, nsigma)
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


# =============================================================================
# MAIN AND TESTS
# =============================================================================

if __name__ == "__main__":
    test_minimizer_all_testsystems()
