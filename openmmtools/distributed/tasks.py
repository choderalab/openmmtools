from __future__ import absolute_import, unicode_literals
from openmmtools.distributed.celery import app
from simtk import openmm, unit
import numpy as np

@app.task
def propagate(system, positions):
    print('Propagate start')
    temperature = 300 * unit.kelvin
    collision_rate = 5.0 / unit.picoseconds
    timestep = 2.0 * unit.femtoseconds
    nsteps = 500
    integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
    context = openmm.Context(system, integrator)
    context.setPositions(positions)
    positions = context.getState(getPositions=True).getPositions(asNumpy=True)
    integrator.step(nsteps)
    del context, integrator
    print('Propagate end')
    return positions

@app.task
def compute_energy(positions, system=None):
    timestep = 2.0 * unit.femtoseconds
    integrator = openmm.VerletIntegrator(timestep)
    print('Calling openmm.Context with %s, %s' % (str(system), str(integrator)))
    context = openmm.Context(system, integrator)
    context.setPositions(positions)
    potential = context.getState(getEnergy=True).getPotentialEnergy()
    del context, integrator
    return potential

@app.task
def mix_replicas(energies):
    print(energies)
