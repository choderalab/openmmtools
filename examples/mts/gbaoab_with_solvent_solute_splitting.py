from openmmtools import testsystems
from openmmtools.forcefactories import split_nb_using_exceptions

testsystem = testsystems.AlanineDipeptideExplicit()  # using PME! should be starting with reaction field

new_system = split_nb_using_exceptions(testsystem.system, testsystem.mdtraj_topology)

from simtk.openmm.app import Simulation
from openmmtools.integrators import LangevinIntegrator
from simtk import unit

collision_rate = 1/unit.picoseconds
#langevin = LangevinIntegrator(splitting='V R O R V', timestep=6 * unit.femtosecond)
mts = LangevinIntegrator(splitting="V0 V1 R R O R R V1 V1 R R O R R V1 V0", collision_rate=collision_rate, timestep=8 * unit.femtosecond)
sim = Simulation(testsystem.topology, new_system, mts)
sim.context.setPositions(testsystem.positions)

print('minimizing energy')
sim.minimizeEnergy(maxIterations=5)

print('first step')
sim.step(1)
pos = sim.context.getState(getPositions=True).getPositions(asNumpy=True)
print(pos)

print('second step')
sim.step(2)
pos = sim.context.getState(getPositions=True).getPositions(asNumpy=True)
print(pos)

print('next 10 steps')
sim.step(10)
pos = sim.context.getState(getPositions=True).getPositions(asNumpy=True)
print(pos)

print('next 1000 steps')
sim.step(1000)
pos = sim.context.getState(getPositions=True).getPositions(asNumpy=True)
print(pos)
