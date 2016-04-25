from simtk import openmm
from simtk import unit

import re
import numpy

from openmmtools import integrators, testsystems

def bitrepr(f, length=64):
    #import struct
    #s = bin(struct.unpack('!I', struct.pack('!f', f))[0])[2:].zfill(32)
    #s = s[0] + ' ' + s[1:9] + ' ' + s[9:]

    from bitstring import BitArray
    s = BitArray(float=f, length=length).bin

    if length == 32:
        s = s[0] + ' ' + s[1:9] + ' ' + s[9:]
    elif length == 64:
        s = s[0] + ' ' + s[1:12] + ' ' + s[12:]
    else:
        raise Exception("length must be 32 or 64")

    return s


def check_bitwise_reversible_velocity_verlet(testsystem):
   from simtk import openmm, unit
   from simtk.openmm import app
   from openmmtools import integrators, testsystems
   import numpy as np

   nsteps = 10

   # Create a bitwise-reversible velocity Verlet integrator.
   timestep = 1.0 * unit.femtoseconds
   integrator = integrators.BitwiseReversibleVelocityVerletIntegrator(timestep, test=True)
   # Demonstrate bitwise reversibility for a simple harmonic oscillator.
   platform = openmm.Platform.getPlatformByName('Reference')
   context = openmm.Context(testsystem.system, integrator, platform)
   context.setPositions(testsystem.positions)
   # Select velocity.
   context.setVelocitiesToTemperature(300*unit.kelvin)
   # Truncate accuracy and store initial positions.
   integrator.truncatePrecision(context)
   initial_positions = context.getState(getPositions=True).getPositions(asNumpy=True)

   for step in range(nsteps):
       # Integrate forward in time.
       integrator.step(1)

       # Get intermediates
       v0 = integrator.getPerDofVariableByName('v0')
       x0 = integrator.getPerDofVariableByName('x0')
       v1 = integrator.getPerDofVariableByName('v1')
       x1 = integrator.getPerDofVariableByName('x1')
       v2 = integrator.getPerDofVariableByName('v2')
       vr = integrator.getPerDofVariableByName('vr')
       v1r = integrator.getPerDofVariableByName('v1r')
       x1r = integrator.getPerDofVariableByName('x1r')
       v2r = integrator.getPerDofVariableByName('v2r')

       for index in range(testsystem.system.getNumParticles()):
           for k in range(3):
               if (x1r[index][k] - x0[index][k]) == 0.0:
                   continue

               print "step %d" % step
               print ""

               print "index = %5d, k = %d" % (index, k)
               print ""

               print "x0   " + bitrepr(x0[index][k])
               print "v0   " + bitrepr(v0[index][k])
               print "v1   " + bitrepr(v1[index][k])
               print "x1   " + bitrepr(x1[index][k])
               print "v2   " + bitrepr(v2[index][k])
               print "vr   " + bitrepr(vr[index][k])
               print "v1r  " + bitrepr(v1r[index][k])
               print "v2r  " + bitrepr(v2r[index][k])
               print "x1r  " + bitrepr(x1r[index][k])
               print "-v2r " + bitrepr(-v2r[index][k])

               print ""

               print "-v1r - v1 : " + bitrepr(-v1r[index][k] - v1[index][k])

               print "-v2r - v0 : " + bitrepr(-v2r[index][k] - v0[index][k])

               print ""
               print "x1r  " + bitrepr(x1r[index][k])
               print "x0   " + bitrepr(x0[index][k])
               print "x1r - x0  : " + bitrepr(x1r[index][k] - x0[index][k])
               print "x1r - x0  : " + str(x1r[index][k] - x0[index][k])

               print ""
               print "------------------------------------"
               print ""
               stop

   return

if __name__ == '__main__':
   from openmmtools import testsystems
   from simtk.openmm import app

   #testsystem = testsystems.LennardJonesCluster()
   testsystem = testsystems.LennardJonesFluid()
   #testsystem = testsystems.UnconstrainedDiatomicFluid()
   #testsystem = testsystems.FlexibleWaterBox(nonbondedMethod=app.CutoffPeriodic)
   #testsystem = testsystems.AlanineDipeptideImplicit(constraints=None)

   check_bitwise_reversible_velocity_verlet(testsystem)


