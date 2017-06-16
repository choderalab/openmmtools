"""
Script to minimize and coarsely thermalize the Drew-Dickinson B-DNA dodecamer.
"""

from simtk import openmm, unit
from simtk.openmm import app

# Thermodynamic and simulation control parameters
temperature = 300.0 * unit.kelvin
collision_rate = 90.0 / unit.picoseconds
pressure = 1.0 * unit.atmospheres
timestep = 2.0 * unit.femtoseconds
equil_steps = 1500

# Load AMBER files
prmtop = app.AmberPrmtopFile('prmtop')
inpcrd = app.AmberInpcrdFile('inpcrd')

# Initialize system, including a barostat
system = prmtop.createSystem(nonbondedMethod=app.CutoffPeriodic, nonbondedCutoff=1.0*unit.nanometer, constraints=app.HBonds)
system.addForce(openmm.MonteCarloBarostat(pressure, temperature))
box_vectors = inpcrd.getBoxVectors(asNumpy=True)
system.setDefaultPeriodicBoxVectors(box_vectors[0], box_vectors[1], box_vectors[2])

# Create the integrator and context
integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
context = openmm.Context(system, integrator)
context.setPositions(inpcrd.positions)

# Minimize the system
openmm.LocalEnergyMinimizer.minimize(context)

# Briefly thermalize.
integrator.step(equil_steps)

# Minimize the system again to make it easier to run as is
openmm.LocalEnergyMinimizer.minimize(context)

# Record the positions
positions = context.getState(getPositions=True, enforcePeriodicBox=True).getPositions(asNumpy=True)
pdbfile = open('minimized_dna_dodecamer.pdb', 'w')
app.PDBFile.writeHeader(prmtop.topology, file=pdbfile)
app.PDBFile.writeModel(prmtop.topology, positions, file=pdbfile, modelIndex=0)
pdbfile.close()