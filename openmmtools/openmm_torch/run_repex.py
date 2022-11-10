from openmm import unit, app
from openmmtools.openmm_torch import repex
import mdtraj as md
from openeye import oechem
import argparse
import numpy as np
import os

parser = argparse.ArgumentParser(description="run free energy corrections")
parser.add_argument('--factory_path', type=str, help="full path to factory.npy.npz")
parser.add_argument('--phase', type=str, help="complex or solvent")
parser.add_argument('--endstate', type=str, help="old or new")
parser.add_argument(f"--out_dir", type=str, help=f"directory to out file")
parser.add_argument(f"--n_states", type=int, help=f"number of thermostates")
args = parser.parse_args()

assert args.phase in ['complex', 'solvent']
assert args.endstate in ['old', 'new']

def make_out_ncfile():
    split_original_path = args.factory_path.split('/')
    plensemble, plsystem = split_original_path[-2], split_original_path[-1] # plensemble is typically "tyk2_..."; plsystem is typically "out_{lig0}_{lig1}"
    plsystem_split = plsystem.split('_') # split "out", "lig0", "lig1"
    lig0, lig1 = int(plsystem_split[1]), int(plsystem_split[2])
    outdir = os.path.join(args.out_dir, plensemble)
    if not os.path.isdir(outdir): # if the plensemble directory doesnt exist, make it
        os.mkdir(outdir)
    _lig = lig0 if args.endstate == 'old' else lig1
    out_ncfile = f"{plsystem}.ligand_{_lig}.{args.phase}.{args.n_states}_states.nc"
    outfile = os.path.join(outdir, out_ncfile)
    return outfile


def load_from_path(_path, _phase, _endstate):
    factory = np.load(_path + '/out-hybrid_factory.npy.npz', allow_pickle=True)['arr_0'].item()[_phase]
    topology = getattr(factory._topology_proposal, f"_{_endstate}_topology")
    positions = getattr(factory, f"_{_endstate}_positions")
    system = getattr(factory._topology_proposal, f"_{_endstate}_system")
    return topology, positions, system

# loader
outnc = make_out_ncfile()
print(f"out_ncfile: {outnc}")
topology, positions, system = load_from_path(args.factory_path, args.phase, args.endstate)
md_top = md.Topology.from_openmm(topology)
pt_filename = outnc[:-2] + 'pt'
mixed_system = repex.MixedSystemConstructor(system = system, topology=topology, filename=pt_filename).mixed_system
print(mixed_system.getForces())

# kwarg dicts
storage_kwargs={
    'storage': outnc, 
    'checkpoint_interval': 10, 
    'analysis_particle_indices': md_top.select('not water')
}
mcmc_moves_kwargs={
    'timestep': 1.*unit.femtoseconds, 
    'collision_rate': 1.0/unit.picoseconds,
    'n_steps': 500,
    'reassign_velocities': True,
    'n_restart_attempts': 20
}
replica_exchange_sampler_kwargs={
    'number_of_iterations': 5000,
    'online_analysis_interval': 10,
    'online_analysis_minimum_iterations': 10,
}

sampler = repex.RepexConstructor(
    mixed_system, positions, 
    storage_kwargs=storage_kwargs, 
    mcmc_moves_kwargs=mcmc_moves_kwargs,
    temperature=300.*unit.kelvin, 
    replica_exchange_sampler_kwargs=replica_exchange_sampler_kwargs,
    n_states=args.n_states).sampler

sampler.minimize()
sampler.run()
