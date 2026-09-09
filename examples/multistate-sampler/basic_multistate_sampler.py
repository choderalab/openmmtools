#!/usr/bin/env python

"""
Minimal, somewhat realistic example of using MultiStateSampler.

Samples alanine dipeptide in implicit solvent at a ladder of temperatures,
with each replica confined to its own thermodynamic state (no exchanges).

Useful to test and showcase basic MultiStateSampler usage.
"""

import logging
import math

from openmm import unit
from openmmtools import testsystems, states, mcmc
from openmmtools.multistate import MultiStateSampler, MultiStateReporter

# Logging capabilities, default DEBUG
logging.basicConfig(level=logging.DEBUG)

testsystem = testsystems.AlanineDipeptideImplicit()

# Build a geometrically-spaced temperature ladder, one thermodynamic state per replica.
n_replicas = 4
T_min = 300.0 * unit.kelvin
T_max = 450.0 * unit.kelvin
temperatures = [
    T_min + (T_max - T_min) * (math.exp(float(i) / float(n_replicas - 1)) - 1.0) / (math.e - 1.0)
    for i in range(n_replicas)
]
thermodynamic_states = [
    states.ThermodynamicState(system=testsystem.system, temperature=T)
    for T in temperatures
]
sampler_states = [
    states.SamplerState(positions=testsystem.positions)
    for _ in range(n_replicas)
]

move = mcmc.LangevinDynamicsMove(timestep=2.0 * unit.femtoseconds, n_steps=500)
sampler = MultiStateSampler(mcmc_moves=move, number_of_iterations=100)

reporter = MultiStateReporter("multistate_sampler_example.nc", checkpoint_interval=10)
sampler.create(thermodynamic_states=thermodynamic_states,
                sampler_states=sampler_states,
                storage=reporter)

sampler.minimize()
sampler.run()

print(f"Ran {sampler.iteration} iterations across {n_replicas} temperatures: "
      f"{[T / unit.kelvin for T in temperatures]} K")
