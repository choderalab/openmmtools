#!/usr/bin/env python

"""
Benchmark various integrators provided in openmmtools on some test systems.

"""

from simtk import openmm
from simtk import unit
from simtk.openmm import app
from openmmtools import testsystems
from openmmtools import integrators
import numpy as np
import time

# Test systems to benchmark
testsystems_to_benchmark = ['LennardJonesFluid']

# Integrators to benchmark
integrators_to_benchmark = ['VerletIntegrator', 'VelocityVerletIntegrator', 'VVVRIntegrator', 'GHMCIntegrator']

# Parameters
timestep = 1.0 * unit.femtoseconds
collision_rate = 91.0 / unit.picoseconds
temperature = 300.0 * unit.kelvin
ntrials = 10 # number of timing trials
nsteps = 200 # number of steps per timing trial

# Cycle through test systems.
for testsystem_name in testsystems_to_benchmark:
    print(testsystem_name)

    # Create test system.
    testsystem = getattr(testsystems, testsystem_name)()

    # Minimize testsystem.
    integrator = openmm.VerletIntegrator(timestep)
    context = openmm.Context(testsystem.system, integrator)
    context.setPositions(testsystem.positions)
    openmm.LocalEnergyMinimizer.minimize(context)
    testsystem.positions = context.getState(getPositions=True).getPositions()
    del context, integrator

    # Benchmark integrators.
    for integrator_name in integrators_to_benchmark:
        if integrator_name == 'VerletIntegrator':
            integrator = openmm.VerletIntegrator(timestep)
        elif integrator_name == 'VelocityVerletIntegrator':
            integrator = integrators.VelocityVerletIntegrator(timestep)
        elif integrator_name == 'VVVRIntegrator':
            integrator = integrators.VVVRIntegrator(temperature=temperature, collision_rate=collision_rate, timestep=timestep)
        elif integrator_name == 'GHMCIntegrator':
            integrator = integrators.GHMCIntegrator(temperature=temperature, collision_rate=collision_rate, timestep=timestep)

        # Create system.
        context = openmm.Context(testsystem.system, integrator)
        context.setPositions(testsystem.positions)

        # Run one step.
        integrator.step(1)

        # Perform timing trials.
        elapsed_time = np.zeros([ntrials], np.float64)
        for trial in range(ntrials):
            initial_time = time.time()
            integrator.step(nsteps)
            final_time = time.time()
            elapsed_time[trial] = final_time - initial_time
        print("%32s : mean %8.3f ms / std %8.3f ms" % (integrator_name, elapsed_time.mean(), elapsed_time.std()))

        # Clean up.
        del context, integrator

    print("")
