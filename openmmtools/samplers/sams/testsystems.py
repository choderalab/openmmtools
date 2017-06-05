"""
Test systems for expanded ensemble variants.

Examples
--------

Alanine dipeptide in various environments (vacuum, implicit, explicit):

>>> from openmmtools.samplers.sams.testsystems import AlanineDipeptideVacuumSimulatedTempering
>>> testsystem = AlanineDipeptideVacuumSimulatedTempering()
>>> exen_sampler = testsystem.exen_sampler['vacuum']
>>> exen_sampler.run(10)

"""

__author__ = 'John D. Chodera'

################################################################################
# IMPORTS
################################################################################

from simtk import openmm, unit
from simtk.openmm import app
import os, os.path
import sys, math
import numpy as np
import time
from functools import partial
from pkg_resources import resource_filename

from openmmtools import testsystems
from openmmtools.states import ThermodynamicState, SamplerState
from openmmtools.mcmc import MCMCSampler, GHMCMove, LangevinDynamicsMove
from openmmtools.samplers.sams import ExpandedEnsembleSampler, SAMSSampler

################################################################################
# CONSTANTS
################################################################################

################################################################################
# SUBROUTINES
################################################################################

def minimize(testsystem):
    """
    Minimize all structures in test system.

    Parameters
    ----------
    testystem : PersesTestSystem
        The testsystem to minimize.

    """
    print("Minimizing '%s'..." % testsystem.description)
    timestep = 1.0 * unit.femtoseconds
    collision_rate = 20.0 / unit.picoseconds
    temperature = 300 * unit.kelvin
    integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
    context = openmm.Context(testsystem.system, integrator)
    context.setPositions(testsystem.positions)
    print("Initial energy is %12.3f kcal/mol" % (context.getState(getEnergy=True).getPotentialEnergy() / unit.kilocalories_per_mole))
    TOL = 1.0
    MAX_STEPS = 500
    openmm.LocalEnergyMinimizer.minimize(context, TOL, MAX_STEPS)
    print("Final energy is   %12.3f kcal/mol" % (context.getState(getEnergy=True).getPotentialEnergy() / unit.kilocalories_per_mole))
    # Take some steps.
    nsteps = 5
    integrator.step(nsteps)
    print("After %d steps    %12.3f kcal/mol" % (nsteps, context.getState(getEnergy=True).getPotentialEnergy() / unit.kilocalories_per_mole))
    # Update positions.
    testsystem.positions = context.getState(getPositions=True).getPositions(asNumpy=True)
    testsystem.mcmc_sampler.sampler_state.positions = context.getState(getPositions=True).getPositions(asNumpy=True)
    # Clean up.
    del context, integrator

################################################################################
# TEST SYSTEMS
################################################################################

class SAMSTestSystem(object):
    """
    Create a consistent set of samplers useful for testing.

    Properties
    ----------
    environments : list of str
        Available environments
    topologies : dict of simtk.openmm.app.Topology
        Initial system Topology objects; topologies[environment] is the topology for `environment`
    positions : dict of simtk.unit.Quantity of [nparticles,3] with units compatible with nanometers
        Initial positions corresponding to initial Topology objects
    system_generators : dict of SystemGenerator objects
        SystemGenerator objects for environments
    proposal_engines : dict of ProposalEngine
        Proposal engines
    themodynamic_states : dict of thermodynamic_states
        Themodynamic states for each environment
    mcmc_samplers : dict of MCMCSampler objects
        MCMCSampler objects for environments
    exen_samplers : dict of ExpandedEnsembleSampler objects
        ExpandedEnsembleSampler objects for environments
    sams_samplers : dict of SAMSSampler objects
        SAMSSampler objects for environments

    """
    def __init__(self, netcdf_filename=None):
        import netCDF4
        self.ncfile = None
        if netcdf_filename is not None:
            self.ncfile = netCDF4.Dataset(netcdf_filename, mode='w')

class HarmonicOscillatorSimulatedTempering(SAMSTestSystem):
    """
    Similated tempering for 3D harmonic oscillator.

    Properties
    ----------
    topology : simtk.openmm.app.Topology
        The system Topology
    system : simtk.openmm.System
        The OpenMM System to simulate
    positions : simtk.unit.Quantity of [nparticles,3] with units compatible with nanometers
        Initial positions
    thermodynamic_states : list of ThermodynamicState
        List of thermodynamic states to be used in expanded ensemble sampling

    Examples
    --------

    >>> from openmmtools.samplers.sams.testsystems import HarmonicOscillatorSimulatedTempering
    >>> testsystem = HarmonicOscillatorSimulatedTempering()

    """
    def __init__(self, **kwargs):
        super(HarmonicOscillatorSimulatedTempering, self).__init__(**kwargs)
        self.description = 'Harmonic oscillator simulated tempering simulation'

        # Create topology, positions, and system.
        from openmmtools.testsystems import HarmonicOscillator
        K = 1.0 * unit.kilocalories_per_mole / unit.angstroms**2 # 3D harmonic oscillator spring constant
        mass = 39.948 * unit.amu # 3D harmonic oscillator particle mass
        period = 2.0 * np.pi * unit.sqrt(mass / K) # harmonic oscillator period
        timestep = 0.01 * period
        testsystem = HarmonicOscillator(K=K, mass=mass)
        self.topology = testsystem.topology
        self.positions = testsystem.positions
        self.system = testsystem.system

        # Create thermodynamic states.
        Tmin = 100 * unit.kelvin
        Tmax = 1000 * unit.kelvin
        ntemps = 8 # number of temperatures
        temperatures = unit.Quantity(np.logspace(np.log10(Tmin / unit.kelvin), np.log10(Tmax / unit.kelvin), ntemps), unit.kelvin)
        self.thermodynamic_states = [ ThermodynamicState(system=self.system, temperature=temperature) for temperature in temperatures ]

        # Compute analytical logZ for each thermodynamic state.
        self.logZ = np.zeros([ntemps], np.float64)
        for (index, thermodynamic_state) in enumerate(self.thermodynamic_states):
            beta = thermodynamic_state.beta
            self.logZ[index] = - 1.5 * np.log(beta * K * unit.angstrom**2)
        self.logZ[:] -= self.logZ[0]

        # Create SAMS samplers
        thermodynamic_state_index = 0 # initial thermodynamic state index
        thermodynamic_state = self.thermodynamic_states[thermodynamic_state_index]
        sampler_state = SamplerState(positions=self.positions)
        self.mcmc_sampler = MCMCSampler(sampler_state=sampler_state, thermodynamic_state=thermodynamic_state, ncfile=self.ncfile)
        self.mcmc_sampler.pdbfile = open('output.pdb', 'w')
        self.mcmc_sampler.topology = self.topology
        self.mcmc_sampler.timestep = timestep
        self.mcmc_sampler.collision_rate = 1.0 / (100 * timestep)
        self.mcmc_sampler.nsteps = 1000
        self.mcmc_sampler.verbose = True
        self.exen_sampler = ExpandedEnsembleSampler(self.mcmc_sampler, self.thermodynamic_states)
        self.exen_sampler.verbose = True
        self.sams_sampler = SAMSSampler(self.exen_sampler, update_stages='two-stage', update_method='optimal')
        self.sams_sampler.verbose = True


class AlanineDipeptideVacuumSimulatedTempering(SAMSTestSystem):
    """
    Similated tempering for alanine dipeptide in implicit solvent.

    Properties
    ----------
    topology : simtk.openmm.app.Topology
        The system Topology
    system : simtk.openmm.System
        The OpenMM System to simulate
    positions : simtk.unit.Quantity of [nparticles,3] with units compatible with nanometers
        Initial positions
    thermodynamic_states : list of ThermodynamicState
        List of thermodynamic states to be used in expanded ensemble sampling

    Examples
    --------

    >>> from openmmtools.samplers.sams.testsystems import AlanineDipeptideVacuumSimulatedTempering
    >>> testsystem = AlanineDipeptideVacuumSimulatedTempering()

    """
    def __init__(self, **kwargs):
        super(AlanineDipeptideVacuumSimulatedTempering, self).__init__(**kwargs)
        self.description = 'Alanine dipeptide in vacuum simulated tempering simulation'

        # Create topology, positions, and system.
        from openmmtools.testsystems import AlanineDipeptideVacuum
        testsystem = AlanineDipeptideVacuum()
        self.topology = testsystem.topology
        self.positions = testsystem.positions
        self.system = testsystem.system

        # Create thermodynamic states.
        Tmin = 270 * unit.kelvin
        Tmax = 600 * unit.kelvin
        ntemps = 8 # number of temperatures
        temperatures = unit.Quantity(np.logspace(np.log10(Tmin / unit.kelvin), np.log10(Tmax / unit.kelvin), ntemps), unit.kelvin)
        self.thermodynamic_states = [ ThermodynamicState(system=self.system, temperature=temperature) for temperature in temperatures ]

        # Create SAMS samplers
        thermodynamic_state_index = 0 # initial thermodynamic state index
        thermodynamic_state = self.thermodynamic_states[thermodynamic_state_index]
        sampler_state = SamplerState(positions=self.positions)
        self.mcmc_sampler = MCMCSampler(sampler_state=sampler_state, thermodynamic_state=thermodynamic_state, ncfile=self.ncfile)
        self.mcmc_sampler.pdbfile = open('output.pdb', 'w')
        self.mcmc_sampler.topology = self.topology
        self.mcmc_sampler.nsteps = 500 # reduce number of steps for testing
        self.mcmc_sampler.verbose = True
        self.exen_sampler = ExpandedEnsembleSampler(self.mcmc_sampler, self.thermodynamic_states)
        self.exen_sampler.verbose = True
        self.sams_sampler = SAMSSampler(self.exen_sampler)
        self.sams_sampler.verbose = True

class AlanineDipeptideExplicitSimulatedTempering(SAMSTestSystem):
    """
    Simulated tempering for alanine dipeptide in explicit solvent.

    Properties
    ----------
    topology : simtk.openmm.app.Topology
        The system Topology
    system : simtk.openmm.System
        The OpenMM System to simulate
    positions : simtk.unit.Quantity of [nparticles,3] with units compatible with nanometers
        Initial positions
    thermodynamic_states : list of ThermodynamicState
        List of thermodynamic states to be used in expanded ensemble sampling

    Examples
    --------

    >>> from openmmtools.samplers.sams.testsystems import AlanineDipeptideExplicitSimulatedTempering
    >>> testsystem = AlanineDipeptideExplicitSimulatedTempering()

    """
    def __init__(self, **kwargs):
        super(AlanineDipeptideExplicitSimulatedTempering, self).__init__(**kwargs)
        self.description = 'Alanine dipeptide in explicit solvent simulated tempering simulation'

        # Create topology, positions, and system.
        from openmmtools.testsystems import AlanineDipeptideExplicit
        testsystem = AlanineDipeptideExplicit(nonbondedMethod=app.CutoffPeriodic)
        self.topology = testsystem.topology
        self.positions = testsystem.positions
        self.system = testsystem.system

        # DEBUG: Write PDB
        from simtk.openmm.app import PDBFile
        outfile = open('initial.pdb', 'w')
        PDBFile.writeFile(self.topology, self.positions, outfile)
        outfile.close()

        # Add a MonteCarloBarostat
        temperature = 270 * unit.kelvin # will be replaced as thermodynamic state is updated
        pressure = 1.0 * unit.atmospheres
        barostat = openmm.MonteCarloBarostat(pressure, temperature)
        self.system.addForce(barostat)

        # Create thermodynamic states.
        Tmin = 270 * unit.kelvin
        Tmax = 600 * unit.kelvin
        ntemps = 256 # number of temperatures
        temperatures = unit.Quantity(np.logspace(np.log10(Tmin / unit.kelvin), np.log10(Tmax / unit.kelvin), ntemps), unit.kelvin)
        self.thermodynamic_states = [ ThermodynamicState(system=self.system, temperature=temperature, pressure=pressure) for temperature in temperatures ]

        # Create SAMS samplers
        thermodynamic_state_index = 0 # initial thermodynamic state index
        thermodynamic_state = self.thermodynamic_states[thermodynamic_state_index]
        sampler_state = SamplerState(positions=self.positions)
        self.mcmc_sampler = MCMCSampler(sampler_state=sampler_state, thermodynamic_state=thermodynamic_state, ncfile=self.ncfile)
        #self.mcmc_sampler.pdbfile = open('output.pdb', 'w')
        self.mcmc_sampler.topology = self.topology
        self.mcmc_sampler.nsteps = 500
        self.mcmc_sampler.timestep = 2.0 * unit.femtoseconds
        self.mcmc_sampler.verbose = True
        self.exen_sampler = ExpandedEnsembleSampler(self.mcmc_sampler, self.thermodynamic_states)
        self.exen_sampler.verbose = True
        self.sams_sampler = SAMSSampler(self.exen_sampler)
        self.sams_sampler.verbose = True

class AlchemicalSAMSTestSystem(SAMSTestSystem):
    """
    Alchemical free energy calculation SAMS test system base class.

    Properties
    ----------

    Assumes the following properties have been defined before calling the constructor:

    topology : simtk.openmm.app.Topology
        The system Topology
    system : simtk.openmm.System
        The OpenMM System to simulate
    positions : simtk.unit.Quantity of [nparticles,3] with units compatible with nanometers
        Initial positions
    alchemical_atoms : list of int
        The atoms to be alchemically annihilated.
    temperature : simtk.unit.Quantity with units compatible with kelvin
        Temperature
    pressure : simtk.unit.Quantity with units compatible with atmospheres, optional, default=None
        Pressure

    """
    def __init__(self, alchemical_protocol='two-phase', nlambda=50, **kwargs):
        """
        Create an alchemical free energy calculation SAMS test system from the provided system.

        Parameters
        ----------
        alchemical_protocol : str, optional, default='two-phase'
            Alchemical protocol scheme to use. ['two-phase', 'fused']
        nlambda : int, optional, default=50
            Number of alchemical states.

        """
        super(AlchemicalSAMSTestSystem, self).__init__(**kwargs)
        self.description = 'Alchemical SAMS test system'
        self.alchemical_protocol = alchemical_protocol

        if not (hasattr(self, 'topology') and hasattr(self, 'system') and hasattr(self, 'positions') and hasattr(self, 'alchemical_atoms')):
            raise Exception("%s: 'topology', 'system', 'positions', and 'alchemical_atoms' properties must be defined!" % self.__class__.__name__)
        if not hasattr(self, 'temperature'):
            self.temperature = 300 * unit.kelvin
        if not hasattr(self, 'temperature'):
            self.temperature = 300 * unit.kelvin
        if not hasattr(self, 'pressure'):
            self.pressure = None

        # Add a MonteCarloBarostat if system does not have one
        has_barostat = False
        for force in self.system.getForces():
            if force.__class__.__name__ in ['MonteCarloBarostat', 'MonteCarloAnisotropicBarostat']:
                has_barostat = True
        if (self.pressure is not None) and (not has_barostat):
            barostat = openmm.MonteCarloBarostat(self.pressure, self.temperature)
            self.system.addForce(barostat)

        # Create alchemically-modified system and populate thermodynamic states.
        from openmmtools.alchemy import AlchemicalRegion, AbsoluteAlchemicalFactory
        self.thermodynamic_states = list()
        if alchemical_protocol == 'fused':
            factory = AbsoluteAlchemicalFactory(consistent_exceptions=False)
            alchemical_region = AlchemicalRegion(alchemical_atoms=self.alchemical_atoms, annihilate_electrostatics=True, annihilate_sterics=False, softcore_beta=0.5)
            self.system = factory.create_alchemical_system(self.system, alchemical_region)
            alchemical_lambdas = np.linspace(1.0, 0.0, nlambda)
            for alchemical_lambda in alchemical_lambdas:
                parameters = {'lambda_sterics' : alchemical_lambda, 'lambda_electrostatics' : alchemical_lambda}
                self.thermodynamic_states.append( ThermodynamicState(system=self.system, temperature=self.temperature, pressure=self.pressure, parameters=parameters) )
        elif alchemical_protocol == 'two-phase':
            factory = AbsoluteAlchemicalFactory(consistent_exceptions=False)
            alchemical_region = AlchemicalRegion(alchemical_atoms=self.alchemical_atoms, annihilate_electrostatics=True, annihilate_sterics=False, softcore_beta=0.0) # turn off softcore
            self.system = factory.create_alchemical_system(self.system, alchemical_region)
            nelec = int(nlambda/2.0)
            nvdw = nlambda - nelec
            for state in range(nelec+1):
                parameters = {'lambda_sterics' : 1.0, 'lambda_electrostatics' : (1.0 - float(state)/float(nelec)) }
                self.thermodynamic_states.append( ThermodynamicState(system=self.system, temperature=self.temperature, pressure=self.pressure, parameters=parameters) )
            for state in range(1,nvdw+1):
                parameters = {'lambda_sterics' : (1.0 - float(state)/float(nvdw)), 'lambda_electrostatics' : 0.0 }
                self.thermodynamic_states.append( ThermodynamicState(system=self.system, temperature=self.temperature, pressure=self.pressure, parameters=parameters) )
        else:
            raise Exception("'alchemical_protocol' must be one of ['two-phase', 'fused']; scheme '%s' unknown." % alchemical_protocol)

        # Create SAMS samplers
        print('Setting up samplers...')
        thermodynamic_state_index = 0 # initial thermodynamic state index
        thermodynamic_state = self.thermodynamic_states[thermodynamic_state_index]
        sampler_state = SamplerState(positions=self.positions)
        self.mcmc_sampler = MCMCSampler(sampler_state=sampler_state, thermodynamic_state=thermodynamic_state, ncfile=self.ncfile)
        self.mcmc_sampler.timestep = 2.0 * unit.femtoseconds
        self.mcmc_sampler.nsteps = 500
        #self.mcmc_sampler.pdbfile = open('output.pdb', 'w')
        self.mcmc_sampler.topology = self.topology
        self.mcmc_sampler.verbose = True
        self.exen_sampler = ExpandedEnsembleSampler(self.mcmc_sampler, self.thermodynamic_states)
        self.exen_sampler.verbose = True
        self.sams_sampler = SAMSSampler(self.exen_sampler)
        self.sams_sampler.verbose = True

        # DEBUG: Write PDB of initial frame
        from simtk.openmm.app import PDBFile
        outfile = open('initial.pdb', 'w')
        PDBFile.writeFile(self.topology, self.positions, outfile)
        outfile.close()

class AlanineDipeptideVacuumAlchemical(AlchemicalSAMSTestSystem):
    """
    Alchemical free energy calculation for alanine dipeptide in vacuum.
    """
    def __init__(self, **kwargs):
        # Create topology, positions, and system.
        from openmmtools.testsystems import AlanineDipeptideVacuum
        testsystem = AlanineDipeptideVacuum()
        self.topology = testsystem.topology
        self.positions = testsystem.positions
        self.system = testsystem.system
        self.alchemical_atoms = range(0,22) # alanine dipeptide
        self.temperature = 300.0 * unit.kelvin

        super(AlanineDipeptideVacuumAlchemical, self).__init__(**kwargs)
        self.description = 'Alanine dipeptide in vacuum alchemical free energy calculation'

class AlanineDipeptideExplicitAlchemical(AlchemicalSAMSTestSystem):
    """
    Alchemical free energy calculation for alanine dipeptide in explicit solvent.
    """
    def __init__(self, **kwargs):
        # Create topology, positions, and system.
        from openmmtools.testsystems import AlanineDipeptideExplicit
        testsystem = AlanineDipeptideExplicit(nonbondedMethod=app.CutoffPeriodic)
        self.topology = testsystem.topology
        self.positions = testsystem.positions
        self.system = testsystem.system
        self.alchemical_atoms = range(0,22) # alanine dipeptide
        self.temperature = 300 * unit.kelvin
        self.pressure = 1.0 * unit.atmospheres

        super(AlanineDipeptideExplicitAlchemical, self).__init__(**kwargs)
        self.description = 'Alanine dipeptide in explicit solvent alchemical free energy calculation'

class WaterBoxAlchemical(AlchemicalSAMSTestSystem):
    """
    Alchemical free energy calculation for TIP3P water in TIP3P water.
    """
    def __init__(self, **kwargs):
        # Create topology, positions, and system.
        from openmmtools.testsystems import WaterBox
        testsystem = WaterBox(nonbondedMethod=app.CutoffPeriodic)
        self.topology = testsystem.topology
        self.positions = testsystem.positions
        self.system = testsystem.system
        self.temperature = 300 * unit.kelvin # will be replaced as thermodynamic state is updated
        self.pressure = 1.0 * unit.atmospheres
        self.alchemical_atoms = range(0,3) # water

        super(WaterBoxAlchemical, self).__init__(**kwargs)
        self.description = 'TIP3P water in TIP3P water NPT alchemical free energy calculation with %s protocol' % self.alchemical_protocol

class HostGuestAlchemical(AlchemicalSAMSTestSystem):
    """
    Alchemical free energy calculation for CB7:B2 host-guest system in TIP3P water.
    """
    def __init__(self, alchemical_protocol='fused', **kwargs):
        # Create topology, positions, and system.
        from openmmtools.testsystems import HostGuestExplicit
        testsystem = HostGuestExplicit(nonbondedMethod=app.CutoffPeriodic)
        self.topology = testsystem.topology
        self.positions = testsystem.positions
        self.system = testsystem.system
        self.temperature = 300 * unit.kelvin # will be replaced as thermodynamic state is updated
        self.pressure = 1.0 * unit.atmospheres
        self.receptor_atoms = range(0, 126)
        self.ligand_atoms = range(126, 156)
        self.alchemical_atoms = self.ligand_atoms

        super(HostGuestAlchemical, self).__init__(**kwargs)
        self.description = 'CB7:B2 host-guest alchemical free energy calculation with %s protocol' % self.alchemical_protocol

class AblImatinibVacuumAlchemical(AlchemicalSAMSTestSystem):
    """
    Alchemical free energy calculation for Abl:imatinib in vacuum
    """
    def __init__(self, **kwargs):
        self.temperature = 300 * unit.kelvin
        self.pressure = 1.0 * unit.atmospheres

        padding = 9.0*unit.angstrom
        setup_path = 'data/abl-imatinib'

        # Create topology, positions, and system.
        from pkg_resources import resource_filename
        gaff_xml_filename = resource_filename('openmmtools', 'data/gaff.xml')
        imatinib_xml_filename = resource_filename('openmmtools', 'data/abl-imatinib/imatinib.xml')
        system_generators = dict()
        ffxmls = [gaff_xml_filename, imatinib_xml_filename, 'amber99sbildn.xml', 'tip3p.xml']
        forcefield_kwargs={ 'nonbondedMethod' : app.NoCutoff, 'nonbondedCutoff' : 9.0 * unit.angstrom, 'implicitSolvent' : None, 'constraints' : app.HBonds, 'rigidWater' : True }

        # Load topologies and positions for all components
        print('Creating Abl:imatinib test system...')
        forcefield = app.ForceField(*ffxmls)
        from simtk.openmm.app import PDBFile, Modeller
        pdb_filename = resource_filename('openmmtools', os.path.join(setup_path, '%s.pdb' % 'complex'))
        pdbfile = PDBFile(pdb_filename)
        modeller = app.Modeller(pdbfile.topology, pdbfile.positions)
        self.topology = modeller.getTopology()
        self.positions = modeller.getPositions()
        print('Creating system...')
        self.system = forcefield.createSystem(self.topology, **forcefield_kwargs)
        self.alchemical_atoms = range(4266,4335) # Abl:imatinib

        super(AblImatinibVacuumAlchemical, self).__init__(**kwargs)
        self.description = 'Abl:imatinib in vacuum alchemical free energy calculation'

        # This test case requires minimization to not explode.
        minimize(self)

class AblImatinibExplicitAlchemical(AlchemicalSAMSTestSystem):
    """
    Alchemical free energy calculation for Abl:imatinib in explicit solvent.
    """
    def __init__(self, **kwargs):
        self.temperature = 300 * unit.kelvin
        self.pressure = 1.0 * unit.atmospheres

        padding = 9.0*unit.angstrom
        explicit_solvent_model = 'tip3p'
        setup_path = 'data/abl-imatinib'

        # Create topology, positions, and system.
        from pkg_resources import resource_filename
        gaff_xml_filename = resource_filename('openmmtools', 'data/gaff.xml')
        imatinib_xml_filename = resource_filename('openmmtools', 'data/abl-imatinib/imatinib.xml')
        system_generators = dict()
        ffxmls = [gaff_xml_filename, imatinib_xml_filename, 'amber99sbildn.xml', 'tip3p.xml']
        forcefield_kwargs={ 'nonbondedMethod' : app.PME, 'nonbondedCutoff' : 9.0 * unit.angstrom, 'implicitSolvent' : None, 'constraints' : app.HBonds, 'rigidWater' : True }

        # Load topologies and positions for all components
        print('Creating Abl:imatinib test system...')
        forcefield = app.ForceField(*ffxmls)
        from simtk.openmm.app import PDBFile, Modeller
        pdb_filename = resource_filename('openmmtools', os.path.join(setup_path, '%s.pdb' % 'complex'))
        pdbfile = PDBFile(pdb_filename)
        modeller = app.Modeller(pdbfile.topology, pdbfile.positions)
        print('Adding solvent...')
        initial_time = time.time()
        modeller.addSolvent(forcefield, model=explicit_solvent_model, padding=padding)
        final_time = time.time()
        elapsed_time = final_time - initial_time
        nadded = (len(modeller.positions) - len(pdbfile.positions)) / 3
        print('Adding solvent took %.3f s (%d molecules added)' % (elapsed_time, nadded))
        self.topology = modeller.getTopology()
        self.positions = modeller.getPositions()
        print('Creating system...')
        self.system = forcefield.createSystem(self.topology, **forcefield_kwargs)
        self.alchemical_atoms = range(4266,4335) # Abl:imatinib

        super(AblImatinibExplicitAlchemical, self).__init__(**kwargs)
        self.description = 'Abl:imatinib in explicit solvent alchemical free energy calculation'

        # This test case requires minimization to not explode.
        minimize(self)

def get_all_subclasses(cls):
    """
    Return all subclasses of a specified class.

    Parameters
    ----------
    cls : class
       The class for which all subclasses are to be returned.

    Returns
    -------
    all_subclasses : list of class
       List of all subclasses of `cls`.

    """

    all_subclasses = []

    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))

    return all_subclasses

def test_testsystems():
    np.set_printoptions(linewidth=130, precision=3)
    niterations = 2
    import sys
    current_module = sys.modules[__name__]
    for testsystem in get_all_subclasses(SAMSTestSystem):
        # Skip any classes that have subclasses
        if len(testsystem.__subclasses__()) > 0:
            continue

        test = testsystem()
        # Reduce number of steps for testing
        test.mcmc_sampler.nsteps = 2
        f = partial(test.mcmc_sampler.run, niterations)
        f.description = 'Testing ' + test.description + ' MCMC simulation'
        yield f
        f = partial(test.exen_sampler.run, niterations)
        f.description = 'Testing ' + test.description + ' expanded ensemble simulation'
        yield f
        f = partial(test.sams_sampler.run, niterations)
        f.description = 'Testing ' + test.description + ' SAMS simulation'
        yield f

def generate_ffxml(pdb_filename):
    from simtk.openmm.app import PDBFile, Modeller
    pdbfile = PDBFile(pdb_filename)
    residues = [ residue for residue in pdbfile.topology.residues() ]
    residue = residues[0]
    from openmoltools.forcefield_generators import generateForceFieldFromMolecules, generateOEMolFromTopologyResidue
    molecule = generateOEMolFromTopologyResidue(residue, geometry=False, tripos_atom_names=True)
    molecule.SetTitle('MOL')
    molecules = [molecule]
    ffxml = generateForceFieldFromMolecules(molecules)
    outfile = open('imatinib.xml', 'w')
    outfile.write(ffxml)
    outfile.close()

if __name__ == '__main__':
    netcdf_filename = 'output2.nc'

    #testsystem = HarmonicOscillatorSimulatedTempering(netcdf_filename=netcdf_filename)
    #testsystem = AblImatinibVacuumAlchemical(netcdf_filename=netcdf_filename)
    testsystem = AblImatinibExplicitAlchemical(netcdf_filename=netcdf_filename)
    #testsystem = HostGuestAlchemical(netcdf_filename=netcdf_filename)
    #testsystem = AlanineDipeptideExplicitAlchemical()
    #testsystem = AlanineDipeptideVacuumSimulatedTempering(netcdf_filename=netcdf_filename)
    #testsystem = AlanineDipeptideExplicitSimulatedTempering(netcdf_filename=netcdf_filename)
    #testsystem = WaterBoxAlchemical(netcdf_filename=netcdf_filename)

    testsystem.exen_sampler.update_scheme = 'restricted-range'
    testsystem.mcmc_sampler.nsteps = 2500
    testsystem.exen_sampler.locality = 5
    testsystem.sams_sampler.update_method = 'rao-blackwellized'
    niterations = 5000
    #testsystem.sams_sampler.run(niterations)

    # Test analysis
    from .analysis import analyze, write_trajectory
    netcdf_filename = 'output.nc'
    analyze(netcdf_filename, testsystem, 'analyze.pdf')
    reference_pdb_filename = 'output.pdb'
    dcd_trajectory_filename = 'output.dcd'
    trajectory_filename = 'output.xtc'
    write_trajectory(netcdf_filename, testsystem.topology, reference_pdb_filename, trajectory_filename)
