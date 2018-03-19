import os
import copy

import numpy as np

from simtk.openmm import app
from simtk import unit, openmm

import openeye.oechem as oechem # TODO: Eliminate OpenEye dependencies, or have tests skip if not found

from openmmtools import alchemy
from openmmtools.states import ThermodynamicState, SamplerState, CompoundThermodynamicState

from openmmtools.alchemy import TopologyProposal, HybridTopologyFactory

import openmmtools.mcmc as mcmc
import openmmtools.cache as cache

from unittest import skipIf

import pymbar
import pymbar.timeseries as timeseries

# TODO: What do do about openmoltools?
from openmoltools import forcefield_generators

# TODO: remove perses dependencies
from perses.tests.utils import createOEMolFromIUPAC, createSystemFromIUPAC
from perses.rjmc.topology_proposal import SmallMoleculeSetProposalEngine
from perses.tests.utils import createOEMolFromIUPAC, createSystemFromIUPAC


from openmmtools.constants import kB
temperature = 300.0 * unit.kelvin
kT = kB * temperature
beta = 1.0/kT

istravis = os.environ.get('TRAVIS', None) == 'true'

try:
    cache.global_context_cache.platform = openmm.Platform.getPlatformByName("OpenCL")
except Exception:
    cache.global_context_cache.platform = openmm.Platform.getPlatformByName("Reference")

ace = {
    'H1'  : [app.Element.getBySymbol('H'), (2.022 ,  0.992 ,  0.038)],#  1.00  0.00           H
    'CH3' : [app.Element.getBySymbol('C'), (1.990 ,  2.080 ,  0.002)],#  1.00  0.00           C
    'H2'  : [app.Element.getBySymbol('H'), (1.506 ,  2.554 ,  0.838)],#  1.00  0.00           H
    'H3'  : [app.Element.getBySymbol('H'), (1.466 ,  2.585 , -0.821)],#  1.00  0.00           H
    'C'   : [app.Element.getBySymbol('C'), (3.423 ,  2.632 ,  0.023)],#  1.00  0.00           C
    'O'   : [app.Element.getBySymbol('O'), (4.402 ,  1.911 , -0.004)],#  1.00  0.00           O
}

nme = {
    'N'   : [app.Element.getBySymbol('N'), (5.852 ,  6.852 ,  0.008)],#  1.00  0.00           N
    'H'   : [app.Element.getBySymbol('H'), (6.718 ,  6.342 , -0.055)],#  1.00  0.00           H
    'C'   : [app.Element.getBySymbol('C'), (5.827 ,  8.281 ,  0.014)],#  1.00  0.00           C
    'H1'  : [app.Element.getBySymbol('H'), (4.816 ,  8.703 , -0.069)],#  1.00  0.00           H
    'H2'  : [app.Element.getBySymbol('H'), (6.407 ,  8.745 ,  0.826)],#  1.00  0.00           H
    'H3'  : [app.Element.getBySymbol('H'), (6.321 ,  8.679 , -0.867)],#  1.00  0.00           H
}

core = {
    'N'  : [app.Element.getBySymbol('N'), (3.547 ,  3.932 , -0.028)],#  1.00  0.00           N
    'H'  : [app.Element.getBySymbol('H'), (2.712 ,  4.492 , -0.088)],#  1.00  0.00           H
    'CA' : [app.Element.getBySymbol('C'), (4.879 ,  4.603 ,  0.004)],#  1.00  0.00           C
    'HA' : [app.Element.getBySymbol('H'), (5.388 ,  4.297 ,  0.907)],#  1.00  0.00           H
    'C'  : [app.Element.getBySymbol('C'), (4.724 ,  6.133 , -0.020)],
    'O'  : [app.Element.getBySymbol('O'), (3.581 ,  6.640 ,  0.027)]
}

ala_unique = {
    'CB'  : [app.Element.getBySymbol('C'), (5.665 ,  4.222 , -1.237)],#  1.00  0.00           C
    'HB1' : [app.Element.getBySymbol('H'), (5.150 ,  4.540 , -2.116)],#  1.00  0.00           H
    'HB2' : [app.Element.getBySymbol('H'), (6.634 ,  4.705 , -1.224)],#  1.00  0.00           H
    'HB3' : [app.Element.getBySymbol('H'), (5.865 ,  3.182 , -1.341)],#  1.00  0.00           H
}

leu_unique = {
    'CB'  : [app.Element.getBySymbol('C'), (5.840 ,  4.228 , -1.172)],#  1.00  0.00           C
    'HB2' : [app.Element.getBySymbol('H'), (5.192 ,  3.909 , -1.991)],#  1.00  0.00           H
    'HB3' : [app.Element.getBySymbol('H'), (6.549 ,  3.478 , -0.826)],#  1.00  0.00           H
    'CG'  : [app.Element.getBySymbol('C'), (6.398 ,  5.525 , -1.826)],#  1.00  0.00           C
    'HG'  : [app.Element.getBySymbol('H'), (6.723 ,  5.312 , -2.877)],#  1.00  0.00           H
    'CD1' : [app.Element.getBySymbol('C'), (7.770 ,  5.753 , -1.221)],#  1.00  0.00           C
    'HD11': [app.Element.getBySymbol('H'), (8.170 ,  6.593 , -1.813)],#  1.00  0.00           H
    'HD12': [app.Element.getBySymbol('H'), (8.420 ,  4.862 , -1.288)],#  1.00  0.00           H
    'HD13': [app.Element.getBySymbol('H'), (7.793 ,  5.788 , -0.123)],#  1.00  0.00           H
    'CD2' : [app.Element.getBySymbol('C'), (5.182 ,  6.334 , -2.328)],#  1.00  0.00           C
    'HD21': [app.Element.getBySymbol('H'), (5.460 ,  7.247 , -2.790)],#  1.00  0.00           H
    'HD22': [app.Element.getBySymbol('H'), (4.353 ,  5.833 , -2.769)],#  1.00  0.00           H
    'HD23': [app.Element.getBySymbol('H'), (4.798 ,  6.958 , -1.550)],#  1.00  0.00           H
}

core_bonds = [
    ('ace-C','ace-O'),
    ('ace-C','ace-CH3'),
    ('ace-CH3','ace-H1'),
    ('ace-CH3','ace-H2'),
    ('ace-CH3','ace-H3'),
    ('ace-C','N'),
    ('N', 'H'),
    ('N', 'CA'),
    ('CA', 'HA'),
    ('CA', 'C'),
    ('C', 'O'),
    ('C','nme-N'),
    ('nme-N','nme-H'),
    ('nme-N','nme-C'),
    ('nme-C','nme-H1'),
    ('nme-C','nme-H2'),
    ('nme-C','nme-H3'),
]

ala_bonds = [
    ('CA', 'CB'),
    ('CB', 'HB1'),
    ('CB', 'HB2'),
    ('CB', 'HB3')
]

leu_bonds = [
    ('CA', 'CB'),
    ('CB', 'HB2'),
    ('CB', 'HB3'),
    ('CB', 'CG'),
    ('CG', 'HG'),
    ('CG', 'CD1'),
    ('CG', 'CD2'),
    ('CD1', 'HD11'),
    ('CD1', 'HD12'),
    ('CD1', 'HD13'),
    ('CD2', 'HD21'),
    ('CD2', 'HD22'),
    ('CD2', 'HD23')
]

forcefield = app.ForceField('amber99sbildn.xml')

def generate_vacuum_topology_proposal(mol_name="naphthalene", ref_mol_name="benzene"):
    """
    Generate a test vacuum topology proposal, current positions, and new positions triplet
    from two IUPAC molecule names.

    Parameters
    ----------
    mol_name : str, optional
        name of the first molecule
    ref_mol_name : str, optional
        name of the second molecule

    Returns
    -------
    topology_proposal : openmmtools.alchemy.TopologyProposal
        The topology proposal representing the transformation
    old_positions : np.array, unit-bearing
        The positions of the old system
    new_positions : np.array, unit-bearing
        The positions of the new system
    """
    m, unsolv_old_system, pos_old, top_old = createSystemFromIUPAC(mol_name)
    refmol = createOEMolFromIUPAC(ref_mol_name)

    initial_smiles = oechem.OEMolToSmiles(m)
    final_smiles = oechem.OEMolToSmiles(refmol)

    forcefield = app.ForceField('amber/gaff.xml', 'tip3p.xml') # 'amber/gaff.xml' requires openmm-forcefields to be installed
    forcefield.registerTemplateGenerator(forcefield_generators.gaffTemplateGenerator)

    solvated_system = forcefield.createSystem(top_old, removeCMMotion=False)

    system_generator = SystemGenerator(['amber/gaff.xml', 'amber99sbildn.xml', 'tip3p.xml'])
    geometry_engine = FFAllAngleGeometryEngine()
    proposal_engine = SmallMoleculeSetProposalEngine(
        [initial_smiles, final_smiles], system_generator, residue_name=mol_name)

    #generate topology proposal
    topology_proposal = proposal_engine.propose(solvated_system, top_old)

    #generate new positions with geometry engine
    new_positions, _ = geometry_engine.propose(topology_proposal, pos_old, beta)

    return topology_proposal, pos_old, new_positions

def generate_solvated_hybrid_test_topology(mol_name="naphthalene", ref_mol_name="benzene"):
    """
    Generate a test solvated topology proposal, current positions, and new positions triplet
    from two IUPAC molecule names.

    Parameters
    ----------
    mol_name : str, optional
        name of the first molecule
    ref_mol_name : str, optional
        name of the second molecule

    Returns
    -------
    topology_proposal : openmmtools.alchemy.TopologyProposal
        The topology proposal representing the transformation
    current_positions : np.array, unit-bearing
        The positions of the initial system
    new_positions : np.array, unit-bearing
        The positions of the new system
    """

    m, unsolv_old_system, pos_old, top_old = createSystemFromIUPAC(mol_name)
    refmol = createOEMolFromIUPAC(ref_mol_name)

    initial_smiles = oechem.OEMolToSmiles(m)
    final_smiles = oechem.OEMolToSmiles(refmol)

    forcefield = app.ForceField('amber/gaff.xml', 'tip3p.xml')
    forcefield.registerTemplateGenerator(forcefield_generators.gaffTemplateGenerator)

    modeller = app.Modeller(top_old, pos_old)
    modeller.addSolvent(forcefield, model='tip3p', padding=9.0*unit.angstrom)
    solvated_topology = modeller.getTopology()
    solvated_positions = modeller.getPositions()
    solvated_system = forcefield.createSystem(solvated_topology, nonbondedMethod=app.PME, removeCMMotion=False)
    barostat = openmm.MonteCarloBarostat(1.0*unit.atmosphere, temperature, 50)

    solvated_system.addForce(barostat)

    gaff_filename = get_data_filename('data/gaff.xml')

    system_generator = SystemGenerator([gaff_filename, 'amber99sbildn.xml', 'tip3p.xml'], barostat=barostat, forcefield_kwargs={'removeCMMotion': False, 'nonbondedMethod': app.PME})
    geometry_engine = FFAllAngleGeometryEngine()
    proposal_engine = SmallMoleculeSetProposalEngine(
        [initial_smiles, final_smiles], system_generator, residue_name=mol_name)

    #generate topology proposal
    topology_proposal = proposal_engine.propose(solvated_system, solvated_topology)

    #generate new positions with geometry engine
    new_positions, _ = geometry_engine.propose(topology_proposal, solvated_positions, beta)

    return topology_proposal, solvated_positions, new_positions

def run_hybrid_endpoint_overlap(topology_proposal, current_positions, new_positions):
    """
    Test that the variance of the perturbation from lambda={0,1} to the corresponding nonalchemical endpoint is not
    too large.

    Parameters
    ----------
    topology_proposal : openmmtools.alchemy.TopologyProposal
         TopologyProposal object describing the transformation
    current_positions : np.array, unit-bearing
         Positions of the initial system
    new_positions : np.array, unit-bearing
         Positions of the new system

    Returns
    -------
    hybrid_endpoint_results : list
       list of [df, ddf, N_eff] for 1 and 0
    """
    #create the hybrid system:
    hybrid_factory = HybridTopologyFactory(topology_proposal, current_positions, new_positions, use_dispersion_correction=True)

    #get the relevant thermodynamic states:
    nonalchemical_zero_thermodynamic_state, nonalchemical_one_thermodynamic_state, lambda_zero_thermodynamic_state, lambda_one_thermodynamic_state = generate_thermodynamic_states(
        hybrid_factory.hybrid_system, topology_proposal)

    nonalchemical_thermodynamic_states = [nonalchemical_zero_thermodynamic_state, nonalchemical_one_thermodynamic_state]

    alchemical_thermodynamic_states = [lambda_zero_thermodynamic_state, lambda_one_thermodynamic_state]

    #create an MCMCMove, BAOAB with default parameters
    mc_move = mcmc.LangevinDynamicsMove()

    initial_sampler_state = SamplerState(hybrid_factory.hybrid_positions, box_vectors=hybrid_factory.hybrid_system.getDefaultPeriodicBoxVectors())

    hybrid_endpoint_results = []
    for lambda_state in (0, 1):
        result = run_endpoint_perturbation(alchemical_thermodynamic_states[lambda_state],
                                        nonalchemical_thermodynamic_states[lambda_state], initial_sampler_state,
                                        mc_move, 100, hybrid_factory, lambda_index=lambda_state)
        print(result)

        hybrid_endpoint_results.append(result)

    return hybrid_endpoint_results

def check_result(results, threshold=3.0, neffmin=10):
    """
    Ensure results are within threshold standard deviations and Neff_max > neffmin

    Parameters
    ----------
    results : list
        list of [df, ddf, Neff_max]
    threshold : float, default 3.0
        the standard deviation threshold
    neff_min : float, default 10
        the minimum number of acceptable samples
    """
    df = results[0]
    ddf = results[1]
    N_eff = results[2]

    if N_eff < neffmin:
        raise Exception("Number of effective samples %f was below minimum of %f" % (N_eff, neffmin))

    if ddf > threshold:
        raise Exception("Standard deviation of %f exceeds threshold of %f" % (ddf, threshold))

def test_simple_overlap():
    """Test that the variance of the endpoint->nonalchemical perturbation is sufficiently small for pentane->butane in vacuum"""
    topology_proposal, current_positions, new_positions = generate_vacuum_topology_proposal()
    results = run_hybrid_endpoint_overlap(topology_proposal, current_positions, new_positions)

    for idx, lambda_result in enumerate(results):
        try:
            check_result(lambda_result)
        except Exception as e:
            message = "pentane->butane failed at lambda %d \n" % idx
            message += str(e)
            raise Exception(message)

@skipIf(istravis, "Skip expensive test on travis")
def test_difficult_overlap():
    """Test that the variance of the endpoint->nonalchemical perturbation is sufficiently small for imatinib->nilotinib in solvent"""
    topology_proposal, solvated_positions, new_positions = generate_solvated_hybrid_test_topology(mol_name='imatinib', ref_mol_name='nilotinib')
    results = run_hybrid_endpoint_overlap(topology_proposal, solvated_positions, new_positions)

    for idx, lambda_result in enumerate(results):
        try:
            check_result(lambda_result)
        except Exception as e:
            message = "solvated imatinib->nilotinib failed at lambda %d \n" % idx
            message += str(e)
            raise Exception(message)

def generate_thermodynamic_states(system: openmm.System, topology_proposal: TopologyProposal):
    """
    Generate endpoint thermodynamic states for the system

    Parameters
    ----------
    system : openmm.System
        System object corresponding to thermodynamic state
    topology_proposal : openmtools.alchemy.TopologyProposal
        TopologyProposal representing transformation

    Returns
    -------
    nonalchemical_zero_thermodynamic_state : ThermodynamicState
        Nonalchemical thermodynamic state for lambda zero endpoint
    nonalchemical_one_thermodynamic_state : ThermodynamicState
        Nonalchemical thermodynamic state for lambda one endpoint
    lambda_zero_thermodynamic_state : ThermodynamicState
        Alchemical (hybrid) thermodynamic state for lambda zero
    lambda_one_thermodynamic_State : ThermodynamicState
        Alchemical (hybrid) thermodynamic state for lambda one
    """
    #create the thermodynamic state
    lambda_zero_alchemical_state = alchemy.AlchemicalState.from_system(system)
    lambda_one_alchemical_state = copy.deepcopy(lambda_zero_alchemical_state)

    #ensure their states are set appropriately
    lambda_zero_alchemical_state.set_alchemical_parameters(0.0)
    lambda_one_alchemical_state.set_alchemical_parameters(0.0)

    #create the base thermodynamic state with the hybrid system
    thermodynamic_state = ThermodynamicState(system, temperature=temperature)

    #Create thermodynamic states for the nonalchemical endpoints
    nonalchemical_zero_thermodynamic_state = ThermodynamicState(topology_proposal.old_system, temperature=temperature)
    nonalchemical_one_thermodynamic_state = ThermodynamicState(topology_proposal.new_system, temperature=temperature)

    #Now create the compound states with different alchemical states
    lambda_zero_thermodynamic_state = CompoundThermodynamicState(thermodynamic_state, composable_states=[lambda_zero_alchemical_state])
    lambda_one_thermodynamic_state = CompoundThermodynamicState(thermodynamic_state, composable_states=[lambda_one_alchemical_state])

    return nonalchemical_zero_thermodynamic_state, nonalchemical_one_thermodynamic_state, lambda_zero_thermodynamic_state, lambda_one_thermodynamic_state

def run_endpoint_perturbation(lambda_thermodynamic_state, nonalchemical_thermodynamic_state, initial_hybrid_sampler_state, mc_move, n_iterations, factory, lambda_index=0):
    """

    Parameters
    ----------
    lambda_thermodynamic_state : ThermodynamicState
        The thermodynamic state corresponding to the hybrid system at a lambda endpoint
    nonalchemical_thermodynamic_state : ThermodynamicState
        The nonalchemical thermodynamic state for the relevant endpoint
    initial_hybrid_sampler_state : SamplerState
        Starting positions for the sampler. Must be compatible with lambda_thermodynamic_state
    mc_move : MCMCMove
        The MCMove that will be used for sampling at the lambda endpoint
    n_iterations : int
        The number of iterations
    factory : HybridTopologyFactory
        The hybrid topology factory
    lambda_index : int, optional default 0
        The index, 0 or 1, at which to retrieve nonalchemical positions

    Returns
    -------
    df : float
        Free energy difference between alchemical and nonalchemical systems, estimated with EXP
    ddf : float
        Standard deviation of estimate, corrected for correlation, from EXP estimator.
    """
    #run an initial minimization:
    mcmc_sampler = mcmc.MCMCSampler(lambda_thermodynamic_state, initial_hybrid_sampler_state, mc_move)
    mcmc_sampler.minimize(max_iterations=20)
    new_sampler_state = mcmc_sampler.sampler_state

    #initialize work array
    w = np.zeros([n_iterations])

    #run n_iterations of the endpoint perturbation:
    for iteration in range(n_iterations):
        mc_move.apply(lambda_thermodynamic_state, new_sampler_state)

        #compute the reduced potential at the new state
        hybrid_context, integrator = cache.global_context_cache.get_context(lambda_thermodynamic_state)
        new_sampler_state.apply_to_context(hybrid_context, ignore_velocities=True)
        hybrid_reduced_potential = lambda_thermodynamic_state.reduced_potential(hybrid_context)

        #generate a sampler state for the nonalchemical system
        if lambda_index == 0:
            nonalchemical_positions = factory.old_positions(new_sampler_state.positions)
        elif lambda_index == 1:
            nonalchemical_positions = factory.new_positions(new_sampler_state.positions)
        else:
            raise ValueError("The lambda index needs to be either one or zero for this to be meaningful")

        nonalchemical_sampler_state = SamplerState(nonalchemical_positions, box_vectors=new_sampler_state.box_vectors)

        #compute the reduced potential at the nonalchemical system as well:
        nonalchemical_context, integrator = cache.global_context_cache.get_context(nonalchemical_thermodynamic_state)
        nonalchemical_sampler_state.apply_to_context(nonalchemical_context, ignore_velocities=True)
        nonalchemical_reduced_potential = nonalchemical_thermodynamic_state.reduced_potential(nonalchemical_context)

        w[iteration] = nonalchemical_reduced_potential - hybrid_reduced_potential

    [t0, g, Neff_max] = timeseries.detectEquilibration(w)
    print(Neff_max)
    w_burned_in = w[t0:]

    [df, ddf] = pymbar.EXP(w_burned_in)
    ddf_corrected = ddf * np.sqrt(g)

    return [df, ddf_corrected, Neff_max]

def get_available_parameters(system, prefix='lambda'):
    parameters = list()
    for force_index in range(system.getNumForces()):
        force = system.getForce(force_index)
        if hasattr(force, 'getNumGlobalParameters'):
            for parameter_index in range(force.getNumGlobalParameters()):
                parameter_name = force.getGlobalParameterName(parameter_index)
                if parameter_name[0:(len(prefix)+1)] == (prefix + '_'):
                    parameters.append(parameter_name)
    return parameters

def build_two_residues():
    alanine_topology = app.Topology()
    alanine_positions = unit.Quantity(np.zeros((22,3)),unit.nanometer)
    leucine_topology = app.Topology()
    leucine_positions = unit.Quantity(np.zeros((31,3)),unit.nanometer)

    ala_chain = alanine_topology.addChain(id='A')
    leu_chain = leucine_topology.addChain(id='A')

    ala_ace = alanine_topology.addResidue('ACE', ala_chain)
    ala_res = alanine_topology.addResidue('ALA', ala_chain)
    ala_nme = alanine_topology.addResidue('NME', ala_chain)
    leu_ace = leucine_topology.addResidue('ACE', leu_chain)
    leu_res = leucine_topology.addResidue('LEU', leu_chain)
    leu_nme = leucine_topology.addResidue('NME', leu_chain)

    ala_atoms = dict()
    leu_atoms = dict()
    atom_map = dict()

    for core_atom_name, [element, position] in ace.items():
        position = np.asarray(position)
        alanine_positions[len(ala_atoms.keys())] = position*unit.angstrom
        leucine_positions[len(leu_atoms.keys())] = position*unit.angstrom

        ala_atom = alanine_topology.addAtom(core_atom_name, element, ala_ace)
        leu_atom = leucine_topology.addAtom(core_atom_name, element, leu_ace)

        ala_atoms['ace-'+core_atom_name] = ala_atom
        leu_atoms['ace-'+core_atom_name] = leu_atom
        atom_map[ala_atom.index] = leu_atom.index

    for core_atom_name, [element, position] in core.items():
        position = np.asarray(position)
        alanine_positions[len(ala_atoms.keys())] = position*unit.angstrom
        leucine_positions[len(leu_atoms.keys())] = position*unit.angstrom

        ala_atom = alanine_topology.addAtom(core_atom_name, element, ala_res)
        leu_atom = leucine_topology.addAtom(core_atom_name, element, leu_res)

        ala_atoms[core_atom_name] = ala_atom
        leu_atoms[core_atom_name] = leu_atom
        atom_map[ala_atom.index] = leu_atom.index

    for ala_atom_name, [element, position] in ala_unique.items():
        position = np.asarray(position)
        alanine_positions[len(ala_atoms.keys())] = position*unit.angstrom
        ala_atom = alanine_topology.addAtom(ala_atom_name, element, ala_res)
        ala_atoms[ala_atom_name] = ala_atom

    for leu_atom_name, [element, position] in leu_unique.items():
        position = np.asarray(position)
        leucine_positions[len(leu_atoms.keys())] = position*unit.angstrom
        leu_atom = leucine_topology.addAtom(leu_atom_name, element, leu_res)
        leu_atoms[leu_atom_name] = leu_atom

    for core_atom_name, [element, position] in nme.items():
        position = np.asarray(position)
        alanine_positions[len(ala_atoms.keys())] = position*unit.angstrom
        leucine_positions[len(leu_atoms.keys())] = position*unit.angstrom

        ala_atom = alanine_topology.addAtom(core_atom_name, element, ala_nme)
        leu_atom = leucine_topology.addAtom(core_atom_name, element, leu_nme)

        ala_atoms['nme-'+core_atom_name] = ala_atom
        leu_atoms['nme-'+core_atom_name] = leu_atom
        atom_map[ala_atom.index] = leu_atom.index

    for bond in core_bonds:
        alanine_topology.addBond(ala_atoms[bond[0]],ala_atoms[bond[1]])
        leucine_topology.addBond(leu_atoms[bond[0]],leu_atoms[bond[1]])
    for bond in ala_bonds:
        alanine_topology.addBond(ala_atoms[bond[0]],ala_atoms[bond[1]])
    for bond in leu_bonds:
        leucine_topology.addBond(leu_atoms[bond[0]],leu_atoms[bond[1]])

    return alanine_topology, alanine_positions, leucine_topology, leucine_positions, atom_map

def compute_alchemical_correction(unmodified_old_system, unmodified_new_system, alchemical_system, initial_positions, alchemical_positions, final_hybrid_positions, final_positions):

    def compute_logP(system, positions, parameter=None):
        integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
        context = openmm.Context(system, integrator)
        context.setPositions(positions)
        context.applyConstraints(integrator.getConstraintTolerance())
        if parameter is not None:
            available_parameters = get_available_parameters(system)
            for parameter_name in available_parameters:
                context.setParameter(parameter_name, parameter)
        potential = context.getState(getEnergy=True).getPotentialEnergy()
        print('Potential: %s' % potential)
        del context, integrator
        return potential


    forces_to_save = {
        'Bond and Nonbonded' : ['HarmonicBondForce', 'CustomBondForce', 'NonbondedForce', 'CustomNonbondedForce'],
        'Angle' : ['HarmonicAngleForce', 'CustomAngleForce'],
        'Torsion' : ['PeriodicTorsionForce', 'CustomTorsionForce'],
        'CMMotion' : ['CMMotionRemover'],
        'All' : []
    }

    for saved_force, force_names in forces_to_save.items():
        print('\nPotential using %s Force:' % saved_force)
        unmodified_old_sys = copy.deepcopy(unmodified_old_system)
        unmodified_new_sys = copy.deepcopy(unmodified_new_system)
        alchemical_sys = copy.deepcopy(alchemical_system)
        for unmodified_system in [unmodified_old_sys, unmodified_new_sys, alchemical_sys]:
            if unmodified_system == alchemical_sys and saved_force == 'Bond and Nonbonded': max_forces = 5
            elif saved_force == 'Bond and Nonbonded': max_forces = 2
            elif saved_force == 'All': max_forces = unmodified_system.getNumForces() + 10
            else: max_forces = 1
            while unmodified_system.getNumForces() > max_forces:
                for k, force in enumerate(unmodified_system.getForces()):
                    force_name = force.__class__.__name__
                    if not force_name in force_names:
                        unmodified_system.removeForce(k)
                        break
        # Compute correction from transforming real system to/from alchemical system
        print('Inital, hybrid - physical')
        initial_logP_correction = compute_logP(alchemical_sys, alchemical_positions, parameter=0) - compute_logP(unmodified_old_sys, initial_positions)
        print('Final, physical - hybrid')
        final_logP_correction = compute_logP(unmodified_new_sys, final_positions) - compute_logP(alchemical_sys, final_hybrid_positions, parameter=1)
        print('Difference in Initial potentials:')
        print(initial_logP_correction)
        print('Difference in Final potentials:')
        print(final_logP_correction)
        logP_alchemical_correction = initial_logP_correction + final_logP_correction


def compare_energies(mol_name="naphthalene", ref_mol_name="benzene"):
    """
    Make an atom map where the molecule at either lambda endpoint is identical, and check that the energies are also the same.
    """
    mol_name = "naphthalene"
    ref_mol_name = "benzene"

    mol = createOEMolFromIUPAC(mol_name)
    m, system, positions, topology = createSystemFromIUPAC(mol_name)

    refmol = createOEMolFromIUPAC(ref_mol_name)

    #map one of the rings
    atom_map = SmallMoleculeSetProposalEngine._get_mol_atom_map(mol, refmol)

    #now use the mapped atoms to generate a new and old system with identical atoms mapped. This will result in the
    #same molecule with the same positions for lambda=0 and 1, and ensures a contiguous atom map
    effective_atom_map = {value : value for value in atom_map.values()}

    #make a topology proposal with the appropriate data:
    top_proposal = TopologyProposal(new_topology=topology, new_system=system, old_topology=topology, old_system=system, new_to_old_atom_map=effective_atom_map, new_chemical_state_key="n1", old_chemical_state_key='n2')

    factory = HybridTopologyFactory(top_proposal, positions, positions)

    alchemical_system = factory.hybrid_system
    alchemical_positions = factory.hybrid_positions

    integrator = openmm.VerletIntegrator(1)
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(alchemical_system, integrator, platform)

    context.setPositions(alchemical_positions)

    functions = {
        'lambda_sterics' : '2*lambda * step(0.5 - lambda) + (1.0 - step(0.5 - lambda))',
        'lambda_electrostatics' : '2*(lambda - 0.5) * step(lambda - 0.5)',
        'lambda_bonds' : 'lambda',
        'lambda_angles' : 'lambda',
        'lambda_torsions' : 'lambda'
    }

    #set all to zero
    for parm in functions.keys():
        context.setParameter(parm, 0.0)

    initial_energy = context.getState(getEnergy=True).getPotentialEnergy()

    #set all to one
    for parm in functions.keys():
        context.setParameter(parm, 1.0)

    final_energy = context.getState(getEnergy=True).getPotentialEnergy()

    if np.abs(final_energy - initial_energy) > 1.0e-6*unit.kilojoule_per_mole:
        raise Exception("The energy at the endpoints was not equal for molecule %s" % mol_name)

def test_compare_energies():
    mols_and_refs = [['naphthalene', 'benzene'], ['pentane', 'propane'], ['biphenyl', 'benzene']]

    for mol_ref_pair in mols_and_refs:
        compare_energies(mol_name=mol_ref_pair[0], ref_mol_name=mol_ref_pair[1])

def generate_topology_proposal(old_mol_iupac="pentane", new_mol_iupac="butane"):
    """
    Utility function to generate a topologyproposal for tests

    Parameters
    ----------
    old_mol_iupac : str, optional
        name of old mol, default pentane
    new_mol_iupac : str, optional
        name of new mol, default butane

    Returns
    -------
    topology_proposal : openmmtools.alchemy.TopologyProposal
        the topology proposal corresponding to the given transformation
    old_positions : [n, 3] np.ndarray of float
        positions of old mol
    new_positions : [m, 3] np.ndarray of float
        positions of new mol
    """
    from perses.rjmc.topology_proposal import TwoMoleculeSetProposalEngine, SystemGenerator
    from perses.rjmc.geometry import FFAllAngleGeometryEngine
    from perses.tests.utils import createSystemFromIUPAC, get_data_filename
    import openmoltools.forcefield_generators as forcefield_generators
    from io import StringIO


    temperature = 300.0 * unit.kelvin
    kT = kB * temperature
    beta = 1.0/kT

    gaff_filename = get_data_filename("data/gaff.xml")
    forcefield_files = [gaff_filename, 'amber99sbildn.xml']

    #generate systems and topologies
    old_mol, old_system, old_positions, old_topology = createSystemFromIUPAC(old_mol_iupac)
    new_mol, new_system, new_positions, new_topology = createSystemFromIUPAC(new_mol_iupac)

    #set names
    old_mol.SetTitle("MOL")
    new_mol.SetTitle("MOL")

    #generate forcefield and ProposalEngine
    #ffxml=forcefield_generators.generateForceFieldFromMolecules([old_mol, new_mol])
    system_generator = SystemGenerator(forcefield_files, forcefield_kwargs={'removeCMMotion' : False})
    proposal_engine = TwoMoleculeSetProposalEngine(old_mol, new_mol, system_generator, residue_name="pentane")
    geometry_engine = FFAllAngleGeometryEngine()

    #create a TopologyProposal
    topology_proposal = proposal_engine.propose(old_system, old_topology)
    new_positions_geometry, _ = geometry_engine.propose(topology_proposal, old_positions, beta)

    return topology_proposal, old_positions, new_positions_geometry

def test_position_output():
    """
    Test that the hybrid returns the correct positions for the new and old systems after construction
    """

    #generate topology proposal
    topology_proposal, old_positions, new_positions = generate_topology_proposal()

    factory = HybridTopologyFactory(topology_proposal, old_positions, new_positions)

    old_positions_factory = factory.old_positions(factory.hybrid_positions)
    new_positions_factory = factory.new_positions(factory.hybrid_positions)

    assert np.all(np.isclose(old_positions.in_units_of(unit.nanometers), old_positions_factory.in_units_of(unit.nanometers)))
    assert np.all(np.isclose(new_positions.in_units_of(unit.nanometers), new_positions_factory.in_units_of(unit.nanometers)))
