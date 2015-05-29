from simtk import openmm
from simtk import unit

import re
import numpy
import time

from openmmtools import integrators, testsystems

def benchmark_hmc(testsystem):
   from simtk import openmm, unit
   from simtk.openmm import app
   from openmmtools import integrators, testsystems
   import numpy as np

   nsteps = 400

   # Create a bitwise-reversible velocity Verlet integrator.
   timestep = 1.0 * unit.femtoseconds
   integrator = integrators.HMCIntegrator(timestep, nsteps=1)
   # Demonstrate bitwise reversibility for a simple harmonic oscillator.
   platform = openmm.Platform.getPlatformByName('CPU')
   context = openmm.Context(testsystem.system, integrator, platform)
   context.setPositions(testsystem.positions)
   # Select velocity.
   context.setVelocitiesToTemperature(300*unit.kelvin)

   # Time dynamics.
   initial_time = time.time()
   integrator.step(nsteps)
   final_positions = context.getState(getPositions=True).getPositions(asNumpy=True)
   final_time = time.time()
   elapsed_time = final_time - initial_time
   print "%.3f s elapsed." % elapsed_time

if __name__ == '__main__':
   from openmmtools import testsystems
   from simtk.openmm import app

   #testsystem = testsystems.LennardJonesCluster()
   testsystem = testsystems.LennardJonesFluid()
   #testsystem = testsystems.UnconstrainedDiatomicFluid()
   #testsystem = testsystems.FlexibleWaterBox(nonbondedMethod=app.CutoffPeriodic)
   #testsystem = testsystems.AlanineDipeptideImplicit(constraints=None)

   benchmark_hmc(testsystem)


