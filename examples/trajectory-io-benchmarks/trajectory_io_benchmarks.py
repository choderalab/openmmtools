#!/usr/bin/env python

"""
Benchmark the performance of several trajectory output strategies.

This script will test the performance of the strategy proposed in PR #434
against the default monolythic NetCDF file.
"""

from argparse import ArgumentParser
import time

import numpy as np
from openmmtools.tests.test_sampling import TestReporter
from openmmtools import testsystems, states
from simtk import unit

import warnings
warnings.simplefilter("ignore", UserWarning)


def run_simulation(
    test_system=testsystems.AlanineDipeptideVacuum,
    n_states=10,
    n_iterations=10,
    analysis_particle_indices=list(range(1, 10)),
    checkpoint_interval=2,
):
    """
    Writes n_states SamplerStates over n_iterations to disk
    and read them back to back.

    Returns a tuple of two timings: write and read, per iteration and state
    """
    with TestReporter.temporary_reporter(
        analysis_particle_indices=analysis_particle_indices,
        checkpoint_interval=checkpoint_interval,
    ) as reporter:
        test = test_system()
        positions = test.positions
        box_vectors = unit.Quantity(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], unit=unit.nanometer
        )
        sampler_states = [
            states.SamplerState(positions=positions, box_vectors=box_vectors)
            for _ in range(n_states)
        ]
        t0 = time.time()
        for iteration in range(n_iterations):
            reporter.write_sampler_states(sampler_states, iteration=iteration)
        reporter.write_last_iteration(iteration)
        t1 = time.time()
        for iteration in range(n_iterations):
            restored_sampler_states = reporter.read_sampler_states(iteration=iteration, analysis_particles_only=True)
        t2 = time.time()

    return (t1 - t0) / (n_states * n_iterations), (t2 - t1) / (n_states * n_iterations)

def main():
    tests = (
        {
            'test_system': testsystems.AlanineDipeptideVacuum,
            'n_states': 10,
            'n_iterations': 10,
            'analysis_particle_indices': list(range(10)),
            'checkpoint_interval': 20
        },
        {
            'test_system': testsystems.SrcImplicit,
            'n_states': 10,
            'n_iterations': 10,
            'analysis_particle_indices': list(range(2000)),
            'checkpoint_interval': 20
        },
        {
            'test_system': testsystems.DHFRExplicit,
            'n_states': 10,
            'n_iterations': 10,
            'analysis_particle_indices': list(range(10000)),
            'checkpoint_interval': 20
        }
    )
    print('Timings per iteration and state, averaged over three attempts')
    for options in tests:
        read_timings, write_timings = [], []
        for _ in range(3):
            write, read = run_simulation(**options)
            read_timings.append(read)
            write_timings.append(write)
        print(" ->", options['test_system'].__name__, f"({len(options['analysis_particle_indices'])}-atom subset)",
              "\n\tW:", np.mean(write_timings), "+-", np.std(write_timings),
              "\n\tR:", np.mean(read_timings), "+-", np.std(read_timings))


if __name__ == '__main__':
    main()