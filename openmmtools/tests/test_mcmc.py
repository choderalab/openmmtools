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

from openmmtools import testsystems
from openmmtools.states import SamplerState, ThermodynamicState
from openmmtools.mcmc import *


# =============================================================================
# GLOBAL TEST CONSTANTS
# =============================================================================

# Test various combinations of systems and MCMC schemes
analytical_testsystems = [
    ("HarmonicOscillator", testsystems.HarmonicOscillator(),
        [GHMCMove(timestep=10.0*unit.femtoseconds, n_steps=100)]),
    ("HarmonicOscillator", testsystems.HarmonicOscillator(),
        {GHMCMove(timestep=10.0*unit.femtoseconds, n_steps=100): 0.5,
         HMCMove(timestep=10*unit.femtosecond, n_steps=10): 0.5}),
    ("HarmonicOscillatorArray", testsystems.HarmonicOscillatorArray(N=4),
        [LangevinDynamicsMove(timestep=10.0*unit.femtoseconds, n_steps=100)]),
    ("IdealGas", testsystems.IdealGas(nparticles=216),
        [HMCMove(timestep=10*unit.femtosecond, n_steps=10)])
    ]

NSIGMA_CUTOFF = 6.0  # cutoff for significance testing

debug = False  # set to True only for manual debugging of this nose test


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
        sampler = MCMCSampler(thermodynamic_state, sampler_state, move_set=[])
        sampler.minimize(max_iterations=0)

        # Check if NaN.
        err_msg = 'Minimization of system {} yielded NaN'.format(class_name)
        assert not sampler_state.has_nan(), err_msg


def test_mcmc_expectations():
    # Select system:
    for [system_name, testsystem, move_set] in analytical_testsystems:
        f = partial(subtest_mcmc_expectation, testsystem, move_set)
        f.description = "Testing MCMC expectation for %s" % system_name
        logging.info(f.description)
        yield f


def subtest_mcmc_expectation(testsystem, move_set):
    if debug:
        print(testsystem.__class__.__name__)
        print(str(move_set))

    # Test settings.
    temperature = 298.0 * unit.kelvin
    nequil = 10  # number of equilibration iterations
    niterations = 40  # number of production iterations

    # Retrieve system and positions.
    [system, positions] = [testsystem.system, testsystem.positions]

    # Compute properties.
    kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
    kT = kB * temperature
    ndof = 3 * system.getNumParticles() - system.getNumConstraints()

    # Create sampler and thermodynamic state.
    sampler_state = SamplerState(positions=positions)
    thermodynamic_state = ThermodynamicState(system=system,
                                             temperature=temperature)

    # Create MCMC sampler and equilibrate.
    sampler = MCMCSampler(thermodynamic_state, sampler_state, move_set=move_set)
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
    if hasattr(testsystem, 'get_potential_expectation'):
        # Skip this check if the std dev is zero.
        if potential_n.std() == 0.0:
            if debug:
                print("Skipping potential test since variance is zero.")
        else:
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

    if hasattr(testsystem, 'get_volume_expectation'):
        # Skip this check if the std dev is zero.
        if volume_n.std() == 0.0:
            if debug:
                print("Skipping volume test.")
        else:
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


# =============================================================================
# MAIN AND TESTS
# =============================================================================

if __name__ == "__main__":
    test_minimizer_all_testsystems()
