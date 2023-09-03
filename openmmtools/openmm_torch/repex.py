"""utilities for building and running MM->NNP potential replica exchange free energy calculations"""
from openmmml.mlpotential import MLPotential
import openmm
from openmm import unit, app
import time
import numpy as np
from copy import deepcopy
from openmmtools import cache
from openmmtools import mcmc
from openmmtools.mcmc import LangevinSplittingDynamicsMove
from openmmtools.multistate import replicaexchange
from openmmtools.multistate.utils import NNPCompatibilityMixin
from openmmtools.alchemy import NNPAlchemicalState
from typing import Dict, Any, Iterable, Union, Optional, List

def deserialize_xml(filename):
    with open(filename, 'r') as infile:
        xml_readable = infile.read()
    xml_deserialized = openmm.XmlSerializer.deserialize(xml_readable)
    return xml_deserialized

class NNPRepexSampler(NNPCompatibilityMixin, replicaexchange.ReplicaExchangeSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
def get_atoms_from_resname(topology, resname) -> List:
    """get the atoms (in order) of the appropriate topology resname"""
    all_resnames = [res.name for res in topology.residues() if res.name == resname]
    assert len(all_resnames) == 1, f"did not find exactly 1 residue with the name {resname}; found {len(all_resnames)}"
    for residue in list(topology.residues()):
        if residue.name == resname:
            break
    atoms = []
    for atom in list(residue.atoms()):
        atoms.append(atom.index)
    assert sorted(atoms) == atoms, f"atom indices ({atoms}) are not in ascending order"
    return atoms

def assert_no_residue_constraints(system: openmm.System, atoms: Iterable[int]):
    """
    assert that there are no constraints within the nnp region before making mixed system
    """
    atom_set = set(atoms)
    all_constraints = []
    for idx in range(system.getNumConstraints()):
        p1, p2, _ = system.getConstraintParameters(idx)
        all_constraints.append(p1)
        all_constraints.append(p2)
    set_intersect = set(all_constraints) & atom_set
    if set_intersect:
        raise Exception(f"the intersection of system constraints and the specified atom set is not empty: {set_intersect}")
        
class MixedSystemConstructor():
    """simple handler to make vanilla `openmm.System` objects a mixedSystem with an `openmm.TorchForce`"""
    def __init__(self, 
                 system : openmm.System,  
                 topology : app.topology.Topology,
                 nnpify_resname: Optional[str]='MOL',
                 nnp_potential : Optional[str]='ani2x',
                 implementation: Optional[str]='nnpops', 
                 **createMixedSystem_kwargs):
        """
        initialize the constructor
        """
        self._system = system
        self._topology = topology
        self._nnpify_resname = nnpify_resname
        self._implementation = implementation
    
        self._atom_indices = get_atoms_from_resname(topology, nnpify_resname)
        assert_no_residue_constraints(system, self._atom_indices)
        self._nnp_potential_str = nnp_potential
        self._nnp_potential = MLPotential(self._nnp_potential_str)
        self._createMixedSystem_kwargs = createMixedSystem_kwargs
    
    @property
    def mixed_system(self):
        return self._nnp_potential.createMixedSystem(self._topology, 
                                                     system = self._system, 
                                                     atoms = self._atom_indices, 
                                                     implementation='nnpops', 
                                                     interpolate=True,
                                                     **self._createMixedSystem_kwargs
                                                     )

class RepexConstructor():
    """ 
    simple handler to build replica exchange sampler.
    """
    def __init__(self, 
                 mixed_system : openmm.System,
                 initial_positions: unit.Quantity,
                 n_states : int,
                 temperature : unit.Quantity,
                 storage_kwargs: Dict={'storage': 'repex.nc', 
                                       'checkpoint_interval': 10,
                                       'analysis_particle_indices': None},
                 mcmc_moves : Optional[mcmc.MCMCMove] = mcmc.LangevinDynamicsMove, # MiddleIntegrator
                 mcmc_moves_kwargs : Optional[Dict] = {'timestep': 1.0*unit.femtoseconds, 
                                                       'collision_rate': 1.0/unit.picoseconds,
                                                       'n_steps': 1000,
                                                       'reassign_velocities': True},
                 replica_exchange_sampler_kwargs : Optional[Dict] = {'number_of_iterations': 5000,
                                                                     'online_analysis_interval': 10,
                                                                     'online_analysis_minimum_iterations': 10,
                                                                    },
                 **kwargs):
        self._mixed_system = mixed_system
        self._storage_kwargs = storage_kwargs
        self._temperature = temperature
        self._mcmc_moves = mcmc_moves
        self._mcmc_moves_kwargs = mcmc_moves_kwargs
        self._replica_exchange_sampler_kwargs = replica_exchange_sampler_kwargs
        self._n_states = n_states
        self._extra_kwargs = kwargs
        
        # initial positions
        self._initial_positions = initial_positions
        
    @property
    def sampler(self):
        # set context cache
        from openmmtools.utils import get_fastest_platform
        from openmmtools import cache
        platform = get_fastest_platform(minimum_precision='mixed')
        context_cache = cache.ContextCache(capacity=None, time_to_live=None, platform=platform)

        mcmc_moves = self._mcmc_moves(**self._mcmc_moves_kwargs)
        _sampler = NNPRepexSampler(mcmc_moves=mcmc_moves, **self._replica_exchange_sampler_kwargs)
        _sampler.energy_context_cache = context_cache
        _sampler.sampler_context_cache = context_cache
        _sampler.setup(n_states=self._n_states, 
                      mixed_system = self._mixed_system, 
                      init_positions = self._initial_positions, 
                      temperature = self._temperature, 
                      storage_kwargs = self._storage_kwargs,
                      **self._extra_kwargs)  
        return _sampler
