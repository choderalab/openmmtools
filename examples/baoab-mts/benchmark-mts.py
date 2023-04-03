#!/usr/bin/env python

"""
Benchmark multiple timestep integration.
"""

import openmm
from openmm import unit
import numpy as np
import rich
import openmmtools
from openmmtools.constants import kB 

TEMPERATURE = 298*unit.kelvin
COLLISION_RATE = 1.0 / unit.picoseconds

class CustomLangevinMiddleIntegrator(openmm.CustomIntegrator):
    """
    CustomIntegrator implementation of LangevinMiddleIntegrator

    From http://docs.openmm.org/latest/api-python/generated/openmm.openmm.CustomIntegrator.html#openmm.openmm.CustomIntegrator
    """
    def __init__(self, temperature, collision_rate, timestep):
        super().__init__(timestep)

        self.addGlobalVariable("a", np.exp(-collision_rate*timestep));
        self.addGlobalVariable("b", np.sqrt(1-np.exp(-2*collision_rate*timestep)));
        self.addGlobalVariable("kT", kB*temperature);
        self.addPerDofVariable("x1", 0);
        self.addUpdateContextState();
        self.addComputePerDof("v", "v + dt*f/m");
        self.addConstrainVelocities();
        self.addComputePerDof("x", "x + 0.5*dt*v");
        self.addComputePerDof("v", "a*v + b*sqrt(kT/m)*gaussian");
        self.addComputePerDof("x", "x + 0.5*dt*v");
        self.addComputePerDof("x1", "x");
        self.addConstrainPositions();
        self.addComputePerDof("v", "v + (x-x1)/dt");


class CustomMTSLangevinMiddleIntegrator(openmm.CustomIntegrator):
    """
    CustomIntegrator implementation of LangevinMiddleIntegrator

    From http://docs.openmm.org/latest/api-python/generated/openmm.openmm.CustomIntegrator.html#openmm.openmm.CustomIntegrator
    """
    def __init__(self, temperature, collision_rate, timestep, n_inner_steps):
        super().__init__(timestep)

        # Initialize
        self.addGlobalVariable("a", np.exp(-collision_rate*timestep));
        self.addGlobalVariable("b", np.sqrt(1-np.exp(-2*collision_rate*timestep)));
        self.addGlobalVariable("kT", kB*temperature);
        self.addGlobalVariable("dti", (timestep / n_inner_steps));
        self.addPerDofVariable("x1", 0);
        self.addUpdateContextState();
        
        # V1 V1
        self.addComputePerDof("v", "v + dt*f1/m");
        
        for i in range(int((n_inner_steps-1)/2)):
            # V0 R R V0
            self.addComputePerDof("v", f"v + dti*f0/m");
            self.addConstrainVelocities();
            self.addComputePerDof("x", "x + dti*v");
            self.addComputePerDof("x1", "x");
            self.addConstrainPositions();
            self.addComputePerDof("v", "v + (x-x1)/dti");
            
        # V0 V0 0 R O R
        self.addComputePerDof("v", f"v + dti*f0/m");
        self.addConstrainVelocities();
        self.addComputePerDof("x", "x + 0.5*dti*v");
        self.addComputePerDof("v", "a*v + b*sqrt(kT/m)*gaussian");
        self.addComputePerDof("x", "x + 0.5*dti*v");
        self.addComputePerDof("x1", "x");
        self.addConstrainPositions();
        self.addComputePerDof("v", "v + (x-x1)/dti");


        for i in range(int((n_inner_steps-1)/2)):
            # V0 R R V0
            self.addComputePerDof("v", f"v + dti*f0/m");
            self.addConstrainVelocities();
            self.addComputePerDof("x", "x + dti*v");
            self.addComputePerDof("x1", "x");
            self.addConstrainPositions();
            self.addComputePerDof("v", "v + (x-x1)/dti");

        
# Integrators to regress over
def create_integrator(integrator_type, outer_timestep, n_inner_steps):
    """
    Create the appropriate integrator given the specified integrator name.

    For multiple timestep integrators, force group 0 is used for inner timesteps and 1 for outer timesteps.

    Parameters
    ----------
    integrator_type : str
        Integrator to create. One of ['openmm.LangevinMiddleIntegrator', 'openmmtools.LangevinIntegrator', 'MTSLangevinIntegrator']
    outer_timestep : openmm.Quantity with units compatible with femtoseconds
        Outer timestep
    n_inner_steps : int
        Number of inner steps (must be odd)

    Returns
    -------
    integrator : openmm.Integrator
        The requested integrator
    """
    temperature = TEMPERATURE
    collision_rate = COLLISION_RATE

    if integrator_type == 'openmm.LangevinMiddleIntegrator':
        from openmm import LangevinMiddleIntegrator
        if n_inner_steps != 1:
            return None # invalid combination
        return LangevinMiddleIntegrator(temperature, collision_rate, outer_timestep)
    elif integrator_type == 'CustomLangevinMiddleIntegrator':
        if n_inner_steps != 1:
            return None # invalid combination
        return CustomLangevinMiddleIntegrator(temperature, collision_rate, outer_timestep)
    elif integrator_type == 'CustomMTSLangevinMiddleIntegrator':
        return CustomMTSLangevinMiddleIntegrator(temperature, collision_rate, outer_timestep, n_inner_steps)
    elif integrator_type == 'openmmtools.LangevinIntegrator':
        from openmmtools.integrators import LangevinIntegrator
        if n_inner_steps == 1:
            return LangevinIntegrator(temperature, collision_rate, outer_timestep,
                                                  'V R O R V')
        else:
            if n_inner_steps % 2 == 0:
                return None # must be odd
            return LangevinIntegrator(temperature, collision_rate, outer_timestep,
                                                  'V1 ' +\
                                                  'V0 R R V0 '*((n_inner_steps-1)//2) +\
                                                  'V0 R O R V0 ' +\
                                                  'V0 R R V0 '*((n_inner_steps-1)//2) +\
                                                  'V1')
    elif integrator_type == 'ModifiedLangevinIntegrator':
        from newmtsintegrator import ModifiedLangevinIntegrator
        if n_inner_steps == 1:
            return ModifiedLangevinIntegrator(temperature, collision_rate, outer_timestep,
                                                  'V R O R V')
        else:
            if n_inner_steps % 2 == 0:
                return None # must be odd
            return ModifiedLangevinIntegrator(temperature, collision_rate, outer_timestep,
                                                  'V1 ' +\
                                                  'V0 R R V0 '*((n_inner_steps-1)//2) +\
                                                  'V0 R O R V0 ' +\
                                                  'V0 R R V0 '*((n_inner_steps-1)//2) +\
                                                  'V1', optimize=True)
    else:
        raise ParameterError(f'Integrator type "{integrator_type} unknown')

def create_splitting(system, splitting_type):
    """
    Create a modified copy of the specified System with the desired splitting type

    Parameters
    ----------
    system : openmm.System
        The System object to copy and modify
    splitting_type : str
        The type of splitting to perform
        ['none', 'NonbondedForce', 'NonbondedForce-ReciprocalSpace']

    Returns
    -------
    system : openmm.System
        The modified copy of the original system with appropriate force splitting.

    """

    import copy
    system = copy.deepcopy(system)
    if splitting_type == 'none':
        pass
    elif splitting_type == 'NonbondedForce':
        for force in system.getForces():
            if force.__class__.__name__ != 'NonbondedForce':
                force.setForceGroup( 0 )
            else:
                # Entire NonbondedForce is slow step
                force.setForceGroup( 1 )
                force.setReciprocalSpaceForceGroup( 1 )
    elif splitting_type == 'NonbondedForce-ReciprocalSpace':
        for force in system.getForces():
            if force.__class__.__name__ != 'NonbondedForce':
                force.setForceGroup( 0 )
            else:
                force.setForceGroup( 0 )
                # Only reciprocal space is slow step
                force.setReciprocalSpaceForceGroup( 1 )        
    else:
        raise ParameterError(f'Unknown splitting_type {splitting_type}')

    return system

def benchmark(system, positions, integrator, platform_name='CUDA', precision='mixed', nsteps=500):
    """
    Assess performance of specified system and integrator.
    
    """
    import openmm
    import time
    platform = openmm.Platform.getPlatformByName(platform_name)
    platform_properties = {'Precision' : precision}
    # TODO: PME properties
    context = openmm.Context(system, integrator, platform, platform_properties)
    context.setPositions(positions)
    
    from openmm import OpenMMException
    try:
        # Warm up integrator
        integrator.step(50)
        
        # TODO: Dynamically determine number of steps based on desired duration
        
        import time
        from openmm import unit
        initial_time = time.time()
        integrator.step(nsteps)
        final_time = time.time()
        elapsed_time = (final_time - initial_time) * unit.seconds
        ns_per_day = nsteps * integrator.getStepSize() / elapsed_time * unit.day / unit.nanoseconds
        
        # clean up
        del context, integrator
    except OpenMMException as e:
        del context, integrator
        ns_per_day = 0.0

    return ns_per_day

from openmmtools import testsystems
from openmmtools.testsystems import TestSystem, get_data_filename
from openmm import app
EWALD_ERROR_TOLERANCE = 1.0e-5

# TODO: Migrate hydrogenMass support back into openmmtools.testsystems
DEFAULT_EWALD_ERROR_TOLERANCE = 1.0e-5 # default Ewald error tolerance
DEFAULT_CUTOFF_DISTANCE = 10.0 * unit.angstroms # default cutoff distance
DEFAULT_SWITCH_WIDTH = 1.5 * unit.angstroms # default switch width
class MproExplicit(TestSystem):
    """SARS-CoV-2 Mpro

    """

    def __init__(self, nonbondedMethod=app.PME, nonbondedCutoff=DEFAULT_CUTOFF_DISTANCE, switch_width=DEFAULT_SWITCH_WIDTH, ewaldErrorTolerance=DEFAULT_EWALD_ERROR_TOLERANCE, hydrogenMass=None, use_dispersion_correction=True, **kwargs):

        TestSystem.__init__(self, **kwargs)

        def deserialize(filename):
            import bz2
            with bz2.open(filename, 'rt') as infile:
                from openmm import XmlSerializer
                return XmlSerializer.deserialize(infile.read())

        prefix = 'Mpro-x2646'
        import os

        system = deserialize(os.path.join(prefix, 'system.xml.bz2'))
        state = deserialize(os.path.join(prefix, 'state.xml.bz2'))
        pdbfile = app.PDBFile(os.path.join(prefix, 'equilibrated-all.pdb'))


        # Construct system.
        #forcefields_to_use = ['amber99sbildn.xml', 'tip3p.xml']  # list of forcefields to use in parameterization
        #forcefield = app.ForceField(*forcefields_to_use)
        #system = forcefield.createSystem(pdbfile.topology, nonbondedMethod=nonbondedMethod, constraints=app.HBonds, hydrogenMass=hydrogenMass,
        #                                 ewaldErrorTolerance=ewaldErrorTolerance, nonbondedCutoff=nonbondedCutoff)

        # Set dispersion correction use.
        forces = {system.getForce(index).__class__.__name__: system.getForce(index) for index in range(system.getNumForces())}
        forces['NonbondedForce'].setUseDispersionCorrection(use_dispersion_correction)
        forces['NonbondedForce'].setEwaldErrorTolerance(ewaldErrorTolerance)
        forces['NonbondedForce'].setCutoffDistance(nonbondedCutoff)
        if switch_width is not None:
            forces['NonbondedForce'].setUseSwitchingFunction(True)
            forces['NonbondedForce'].setSwitchingDistance(nonbondedCutoff - switch_width)

        # Get positions.
        positions = pdbfile.getPositions()

        self.system = system
        self.positions = state.getPositions()
        self.topology = pdbfile.getTopology()

class SrcExplicit(TestSystem):

    """Src kinase (AMBER 99sb-ildn) in explicit TIP3P solvent using PME electrostatics.
    Parameters
    ----------
    nonbondedMethod : openmm.app nonbonded method, optional, default=app.PME
       Sets the nonbonded method to use for the water box (CutoffPeriodic, app.Ewald, app.PME).
    Examples
    --------
    >>> src = SrcExplicit()
    >>> system, positions = src.system, src.positions
    """

    def __init__(self, nonbondedMethod=app.PME, nonbondedCutoff=DEFAULT_CUTOFF_DISTANCE, switch_width=DEFAULT_SWITCH_WIDTH, ewaldErrorTolerance=DEFAULT_EWALD_ERROR_TOLERANCE, hydrogenMass=None, use_dispersion_correction=True, **kwargs):

        TestSystem.__init__(self, **kwargs)

        pdb_filename = get_data_filename("data/src-explicit/1yi6-minimized.pdb")
        pdbfile = app.PDBFile(pdb_filename)

        # Construct system.
        forcefields_to_use = ['amber99sbildn.xml', 'tip3p.xml']  # list of forcefields to use in parameterization
        forcefield = app.ForceField(*forcefields_to_use)
        system = forcefield.createSystem(pdbfile.topology, nonbondedMethod=nonbondedMethod, constraints=app.HBonds, hydrogenMass=hydrogenMass,
                                         ewaldErrorTolerance=ewaldErrorTolerance, nonbondedCutoff=nonbondedCutoff)

        # Set dispersion correction use.
        if switch_width is not None:
            forces['NonbondedForce'].setUseSwitchingFunction(True)
            forces['NonbondedForce'].setSwitchingDistance(nonbondedCutoff - switch_width)



        # Get positions.
        positions = pdbfile.getPositions()

        self.system, self.positions, self.topology = system, positions, pdbfile.topology

class SrcExplicitReactionField(SrcExplicit):

    """
    Flexible water box.

    """

    def __init__(self, *args, **kwargs):
        """Src kinase (AMBER 99sb-ildn) in explicit TIP3P solvent using reaction field electrostatics.

        Parameters are inherited from SrcExplicit (except for 'nonbondedMethod').

        Examples
        --------

        >>> src = SrcExplicitReactionField()
        >>> system, positions = src.system, src.positions

        """
        super(SrcExplicitReactionField, self).__init__(nonbondedMethod=app.CutoffPeriodic, *args, **kwargs)
        import mdtraj
        solvent_indices = mdtraj.Topology.from_openmm(self.topology).select('water or (n_bonds == 0)') # water and ions
        all_indices = mdtraj.Topology.from_openmm(self.topology).select('all') # everything
        from openmmtools.forcefactories import replace_reaction_field_atomic_mts
        print(f'solvent_indices has {len(solvent_indices)} atoms out of {len(all_indices)} total atoms')
        self.system = replace_reaction_field_atomic_mts(self.system, method='riniker-AT-SHIFT-4-6', solvent_indices=solvent_indices)        

        # Minimize
        print('Minimizing...')
        import openmm
        integrator = openmm.VerletIntegrator(1.0*unit.femtoseconds)
        context = openmm.Context(self.system, integrator)
        context.setPositions(self.positions)
        openmm.LocalEnergyMinimizer.minimize(context)
        self.positions = context.getState(getPositions=True).getPositions(asNumpy=True)
        del context, integrator
        print('Done minimizing.')

class FAHTestSystem(TestSystem):
    """

    """

    def __init__(self, run='RUN0', nonbondedMethod=app.PME, nonbondedCutoff=DEFAULT_CUTOFF_DISTANCE, switch_width=DEFAULT_SWITCH_WIDTH, ewaldErrorTolerance=DEFAULT_EWALD_ERROR_TOLERANCE, hydrogenMass=None, use_dispersion_correction=True, **kwargs):

        TestSystem.__init__(self, **kwargs)

        def deserialize(filename):
            import bz2
            with bz2.open(filename, 'rt') as infile:
                from openmm import XmlSerializer
                return XmlSerializer.deserialize(infile.read())

        prefix = '17111-core22-0.0.20-openmm-performance-benchmarks/RUNS/RUN19'
        import os

        system = deserialize(os.path.join(prefix, 'system.xml.bz2'))
        state = deserialize(os.path.join(prefix, 'state.xml.bz2'))


        # Construct system.
        #forcefields_to_use = ['amber99sbildn.xml', 'tip3p.xml']  # list of forcefields to use in parameterization
        #forcefield = app.ForceField(*forcefields_to_use)
        #system = forcefield.createSystem(pdbfile.topology, nonbondedMethod=nonbondedMethod, constraints=app.HBonds, hydrogenMass=hydrogenMass,
        #                                 ewaldErrorTolerance=ewaldErrorTolerance, nonbondedCutoff=nonbondedCutoff)

        # Set dispersion correction use.
        forces = {system.getForce(index).__class__.__name__: system.getForce(index) for index in range(system.getNumForces())}
        forces['NonbondedForce'].setUseDispersionCorrection(use_dispersion_correction)
        forces['NonbondedForce'].setEwaldErrorTolerance(ewaldErrorTolerance)
        forces['NonbondedForce'].setCutoffDistance(nonbondedCutoff)
        if switch_width is not None:
            forces['NonbondedForce'].setUseSwitchingFunction(True)
            forces['NonbondedForce'].setSwitchingDistance(nonbondedCutoff - switch_width)

        self.system = system
        self.positions = state.getPositions()
        self.topology = None

HYDROGEN_MASS = 3.0*unit.amu

TESTSYSTEMS = {
#    'AlanineDipeptideExplicit' : testsystems.AlanineDipeptideExplicit(constraints=app.HBonds, hydrogenMass=HYDROGEN_MASS),
#    'WaterBox PME 8A cutoff' : testsystems.WaterBox(constrained=True, nonbondedMethhod=app.PME, box_edge=25*unit.angstroms, dispersion_correction=True, cutoff=8.0*unit.angstroms, switch_width=None, ewaldErrorTolerance=EWALD_ERROR_TOLERANCE),
#    'WaterBox PME 9A cutoff' : testsystems.WaterBox(constrained=True, nonbondedMethhod=app.PME, box_edge=25*unit.angstroms, dispersion_correction=True, cutoff=9.0*unit.angstroms, switch_width=None, ewaldErrorTolerance=EWALD_ERROR_TOLERANCE),
#    'WaterBox PME 10A cutoff' : testsystems.WaterBox(constrained=True, nonbondedMethhod=app.PME, box_edge=25*unit.angstroms, dispersion_correction=True, cutoff=10.0*unit.angstroms, switch_width=None, ewaldErrorTolerance=EWALD_ERROR_TOLERANCE),
#    'WaterBox PME 11A cutoff' : testsystems.WaterBox(constrained=True, nonbondedMethhod=app.PME, box_edge=25*unit.angstroms, dispersion_correction=True, cutoff=11.0*unit.angstroms, switch_width=None, ewaldErrorTolerance=EWALD_ERROR_TOLERANCE),
#    'WaterBox RF 12A cutoff' : testsystems.WaterBox(constrained=True, nonbondedMethhod=app.CutoffPeriodic, box_edge=25*unit.angstroms, dispersion_correction=True, cutoff=12.0*unit.angstroms, switch_width=None, ewaldErrorTolerance=EWALD_ERROR_TOLERANCE),
#    'Big WaterBox PME 9A cutoff' : testsystems.WaterBox(constrained=True, nonbondedMethhod=app.PME, box_edge=50*unit.angstroms, dispersion_correction=True, cutoff=9.0*unit.angstroms, switch_width=None, ewaldErrorTolerance=EWALD_ERROR_TOLERANCE),
#    'DHFRExplicit PME  9A cutoff 5e-4 tol' : testsystems.DHFRExplicit(hydrogenMass=HYDROGEN_MASS, ewaldErrorTolerance=5.0e-4, nonbondedCutoff=9.0*unit.angstroms),
#    'DHFRExplicit PME  9A cutoff 1e-4 tol' : testsystems.DHFRExplicit(hydrogenMass=HYDROGEN_MASS, ewaldErrorTolerance=1.0e-4, nonbondedCutoff=9.0*unit.angstroms),
#    'DHFRExplicit PME  9A cutoff 5e-5 tol' : testsystems.DHFRExplicit(hydrogenMass=HYDROGEN_MASS, ewaldErrorTolerance=5.0e-5, nonbondedCutoff=9.0*unit.angstroms),
#    'DHFRExplicit PME  9A cutoff 1e-5 tol' : testsystems.DHFRExplicit(hydrogenMass=HYDROGEN_MASS, ewaldErrorTolerance=1.0e-5, nonbondedCutoff=9.0*unit.angstroms),
#    'DHFRExplicit PME 12A cutoff 5e-4 tol' : testsystems.DHFRExplicit(hydrogenMass=HYDROGEN_MASS, ewaldErrorTolerance=5.0e-4, nonbondedCutoff=12.0*unit.angstroms),
#    'DHFRExplicit PME 12A cutoff 1e-4 tol' : testsystems.DHFRExplicit(hydrogenMass=HYDROGEN_MASS, ewaldErrorTolerance=1.0e-4, nonbondedCutoff=12.0*unit.angstroms),
#    'DHFRExplicit PME 12A cutoff 5e-5 tol' : testsystems.DHFRExplicit(hydrogenMass=HYDROGEN_MASS, ewaldErrorTolerance=5.0e-5, nonbondedCutoff=12.0*unit.angstroms),
#    'DHFRExplicit PME 12A cutoff 1e-5 tol' : testsystems.DHFRExplicit(hydrogenMass=HYDROGEN_MASS, ewaldErrorTolerance=1.0e-5, nonbondedCutoff=12.0*unit.angstroms),
#    'SrcExplicit PME  6A cutoff 5e-4 tol' : testsystems.SrcExplicit(hydrogenMass=HYDROGEN_MASS, ewaldErrorTolerance=5e-4, switch_width=None, nonbondedCutoff=6.0*unit.angstroms),
#    'SrcExplicit PME  7A cutoff 5e-4 tol' : testsystems.SrcExplicit(hydrogenMass=HYDROGEN_MASS, ewaldErrorTolerance=5e-4, switch_width=None, nonbondedCutoff=7.0*unit.angstroms),
#    'SrcExplicit PME  8A cutoff 5e-4 tol' : testsystems.SrcExplicit(hydrogenMass=HYDROGEN_MASS, ewaldErrorTolerance=5e-4, switch_width=None, nonbondedCutoff=8.0*unit.angstroms),
#    'SrcExplicit PME  9A cutoff 5e-4 tol' : testsystems.SrcExplicit(hydrogenMass=HYDROGEN_MASS, ewaldErrorTolerance=5e-4, switch_width=None, nonbondedCutoff=9.0*unit.angstroms),
#    'SrcExplicit PME  10A cutoff 5e-4 tol' : testsystems.SrcExplicit(hydrogenMass=HYDROGEN_MASS, ewaldErrorTolerance=5e-4, switch_width=None, nonbondedCutoff=10.0*unit.angstroms),
#    'SrcExplicit PME  9A cutoff 1e-4 tol' : testsystems.SrcExplicit(hydrogenMass=HYDROGEN_MASS, ewaldErrorTolerance=1e-4, switch_width=None, nonbondedCutoff=9.0*unit.angstroms),
#    'SrcExplicit PME  9A cutoff 5e-5 tol' : testsystems.SrcExplicit(hydrogenMass=HYDROGEN_MASS, ewaldErrorTolerance=5e-5, switch_width=None, nonbondedCutoff=9.0*unit.angstroms),
#    'SrcExplicit PME  9A cutoff 1e-5 tol' : testsystems.SrcExplicit(hydrogenMass=HYDROGEN_MASS, ewaldErrorTolerance=1e-5, switch_width=None, nonbondedCutoff=9.0*unit.angstroms),
#    'SrcExplicit PME 12A cutoff 5e-4 tol' : testsystems.SrcExplicit(hydrogenMass=HYDROGEN_MASS, ewaldErrorTolerance=5e-4, switch_width=None, nonbondedCutoff=12.0*unit.angstroms),
#    'SrcExplicit PME 12A cutoff 1e-4 tol' : testsystems.SrcExplicit(hydrogenMass=HYDROGEN_MASS, ewaldErrorTolerance=1e-4, switch_width=None, nonbondedCutoff=12.0*unit.angstroms),
#    'SrcExplicit PME 12A cutoff 5e-5 tol' : testsystems.SrcExplicit(hydrogenMass=HYDROGEN_MASS, ewaldErrorTolerance=5e-5, switch_width=None, nonbondedCutoff=12.0*unit.angstroms),
#    'SrcExplicit PME 12A cutoff 1e-5 tol' : testsystems.SrcExplicit(hydrogenMass=HYDROGEN_MASS, ewaldErrorTolerance=1e-5, switch_width=None, nonbondedCutoff=12.0*unit.angstroms),
    'SrcExplicit RF 12A' : SrcExplicitReactionField(hydrogenMass=HYDROGEN_MASS, switch_width=None, nonbondedCutoff=12.0*unit.angstroms),
#    'SrcExplicit RF 15A' : SrcExplicitReactionField(hydrogenMass=HYDROGEN_MASS, switch_width=None, nonbondedCutoff=15.0*unit.angstroms),
#    'MproExplicit  9A cutoff 5e-4' : MproExplicit(hydrogenMass=HYDROGEN_MASS, ewaldErrorTolerance=5e-4, switch_width=None, nonbondedCutoff=9.0*unit.angstroms),
#    'MproExplicit  9A cutoff 1e-4' : MproExplicit(hydrogenMass=HYDROGEN_MASS, ewaldErrorTolerance=1e-4, switch_width=None, nonbondedCutoff=9.0*unit.angstroms),
#    'MproExplicit  9A cutoff 5e-5' : MproExplicit(hydrogenMass=HYDROGEN_MASS, ewaldErrorTolerance=5e-5, switch_width=None, nonbondedCutoff=9.0*unit.angstroms),
#    'MproExplicit  9A cutoff 1e-5' : MproExplicit(hydrogenMass=HYDROGEN_MASS, ewaldErrorTolerance=1e-5, switch_width=None, nonbondedCutoff=9.0*unit.angstroms),
#    'MproExplicit 12A cutoff 5e-4' : MproExplicit(hydrogenMass=HYDROGEN_MASS, ewaldErrorTolerance=5e-4, switch_width=None, nonbondedCutoff=12.0*unit.angstroms),
#    'MproExplicit 12A cutoff 1e-4' : MproExplicit(hydrogenMass=HYDROGEN_MASS, ewaldErrorTolerance=1e-4, switch_width=None, nonbondedCutoff=12.0*unit.angstroms),
#    'MproExplicit 12A cutoff 5e-5' : MproExplicit(hydrogenMass=HYDROGEN_MASS, ewaldErrorTolerance=5e-5, switch_width=None, nonbondedCutoff=12.0*unit.angstroms),
#    'MproExplicit 12A cutoff 1e-5' : MproExplicit(hydrogenMass=HYDROGEN_MASS, ewaldErrorTolerance=1e-5, switch_width=None, nonbondedCutoff=12.0*unit.angstroms),
    #    'STMV  9A cutoff 1e-4' : FAHTestSystem(hydrogenMass=HYDROGEN_MASS, ewaldErrorTolerance=1e-4, switch_width=None, nonbondedCutoff=9.0*unit.angstroms, run='RUN19'),
    #    'STMV  9A cutoff 1e-5' : FAHTestSystem(hydrogenMass=HYDROGEN_MASS, ewaldErrorTolerance=1e-5, switch_width=None, nonbondedCutoff=9.0*unit.angstroms, run='RUN19'),
#    'STMV 12A cutoff 1e-4' : FAHTestSystem(hydrogenMass=HYDROGEN_MASS, ewaldErrorTolerance=1e-4, switch_width=None, nonbondedCutoff=12.0*unit.angstroms, run='RUN19'),
#    'STMV 12A cutoff 1e-5' : FAHTestSystem(hydrogenMass=HYDROGEN_MASS, ewaldErrorTolerance=1e-5, switch_width=None, nonbondedCutoff=12.0*unit.angstroms, run='RUN19'),
#    'cellulose  9A cutoff 1e-4' : FAHTestSystem(hydrogenMass=HYDROGEN_MASS, ewaldErrorTolerance=1e-4, switch_width=None, nonbondedCutoff=9.0*unit.angstroms, run='RUN17'),
#    'cellulose  9A cutoff 1e-5' : FAHTestSystem(hydrogenMass=HYDROGEN_MASS, ewaldErrorTolerance=1e-5, switch_width=None, nonbondedCutoff=9.0*unit.angstroms, run='RUN17'),
#    'cellulose 12A cutoff 1e-4' : FAHTestSystem(hydrogenMass=HYDROGEN_MASS, ewaldErrorTolerance=1e-4, switch_width=None, nonbondedCutoff=12.0*unit.angstroms, run='RUN17'),
#    'cellulose 12A cutoff 1e-5' : FAHTestSystem(hydrogenMass=HYDROGEN_MASS, ewaldErrorTolerance=1e-5, switch_width=None, nonbondedCutoff=12.0*unit.angstroms, run='RUN17'),
    }
    
# Show GPU
import subprocess
result = subprocess.run(['nvidia-smi', '-L'], stdout=subprocess.PIPE)
gpuinfo = result.stdout.decode('utf-8').strip()

# Show system sizes
from rich.console import Console
from rich.table import Table
table = Table(title="Systems")
table.add_column("Test system", justify="left", style='red1', no_wrap=True)
table.add_column("Number of atoms", justify="right", style='blue1', no_wrap=True)
for testsystem_name in TESTSYSTEMS.keys():
    testsystem = TESTSYSTEMS[testsystem_name]
    table.add_row(testsystem_name, f'{testsystem.system.getNumParticles()}')

console = Console()
console.print(table)    

# TODO: Minimize all test systems and warm them up
    

# Combinatorial options to explore
INTEGRATORS = [
    'openmm.LangevinMiddleIntegrator', 
#    'CustomLangevinMiddleIntegrator', 
    'CustomMTSLangevinMiddleIntegrator', 
    #'openmmtools.LangevinIntegrator', 
    #'ModifiedLangevinIntegrator'
]
INNER_STEPS = [
    1, 
    3, 
    5
]

TIMESTEPS = [unit.Quantity(x, unit.femtoseconds) for x in range(1, 28) ]
#TIMESTEPS = [unit.Quantity(4.0, unit.femtoseconds)] # DEBUG
CONSTRAINT_TOLERANCES = [1.0e-6]
FORCE_SPLITTINGS = [
    'none', 
    #'NonbondedForce', 
    #'NonbondedForce-ReciprocalSpace'
]

from rich.table import Table
table = Table(title=f"Benchmarks {gpuinfo}")

table.add_column("Test system", justify="left", style='red1', no_wrap=True)
table.add_column("Splitting", justify="left", style="orange4", no_wrap=True)
table.add_column("Integrator", justify="left", style="orange3", no_wrap=True)
table.add_column("Inner steps", justify="center", style="green1")
table.add_column("Timestep (fs)", justify="right", style="blue1")
table.add_column("Constraint tol", justify="right", style="violet")
table.add_column("ns/day", justify="right", style="dark_violet")

outfile = open('benchmark.csv', 'wt')

from itertools import product
for (testsystem_name, splitting_type, constraint_tolerance, n_inner_steps, outer_timestep, integrator_name) in product(TESTSYSTEMS.keys(), FORCE_SPLITTINGS, CONSTRAINT_TOLERANCES, INNER_STEPS, TIMESTEPS, INTEGRATORS):
    # Retrieve test system
    testsystem = TESTSYSTEMS[testsystem_name]
    # Create new integrator
    integrator = create_integrator(integrator_name, outer_timestep, n_inner_steps)
    # Reject combinations that do not make sense
    if integrator is None:
        continue    
#    if (n_inner_steps == 1) and (splitting_type != 'none'):
#        continue
#    if (n_inner_steps > 1) and (splitting_type == 'none'):
#        continue
    if (integrator == 'openmm.LangevinMiddleIntegrator') and (n_inner_steps != 1):
        continue
    # Set constraint tolerance
    integrator.setConstraintTolerance(constraint_tolerance)
    # Create splitting
    system = create_splitting(testsystem.system, splitting_type)
    # TODO: Add barostat
    # Benchmark
    ns_per_day = benchmark(system, testsystem.positions, integrator)
    # Report
    table.add_row(testsystem_name, splitting_type, integrator_name, f'{n_inner_steps:d}', f'{outer_timestep/unit.femtoseconds:5.1f}', f'{constraint_tolerance:5.0e}', f'{ns_per_day:8.1f}')
    outfile.write(f'{testsystem_name},{splitting_type},{integrator_name},{n_inner_steps:d},{outer_timestep/unit.femtoseconds:5.1f},{constraint_tolerance:5.0e},{ns_per_day:8.1f}\n')
    outfile.flush()
    
    console.print(table)

outfile.close()
