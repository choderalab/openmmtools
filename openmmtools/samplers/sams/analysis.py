"""
Analysis for self-adjusted mixture sampling (SAMS).

"""

__author__ = 'John D. Chodera'

################################################################################
# IMPORTS
################################################################################

import numpy as np
import netCDF4
import os, os.path
import sys, math
import copy
import time

from simtk import unit
import mdtraj

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
from matplotlib.backends.backend_pdf import PdfPages
import seaborn

from . import testsystems

################################################################################
# LOGGER
################################################################################

import logging
logger = logging.getLogger(__name__)

################################################################################
# Thermodynamic state description
################################################################################

def analyze(netcdf_filename, testsystem, pdf_filename):
    ncfile = netCDF4.Dataset(netcdf_filename, 'r')
    [nsamples, nstates] = ncfile.variables['logZ'].shape

    testsystem_name = testsystem.__class__.__name__
    nstates = len(testsystem.thermodynamic_states)

    with PdfPages(pdf_filename) as pdf:
        # PAGE 1
        plt.figure(figsize=(6, 6))

        if hasattr(testsystem, 'logZ'):
            plt.hold(True)
            plt.plot(testsystem.logZ, 'ro')
            print(testsystem.logZ)

        title_fontsize = 7

        logZ = ncfile.variables['logZ'][-2,:]
        plt.plot(logZ, 'ko')
        plt.title(testsystem.description, fontsize=title_fontsize)
        plt.xlabel('state index $j$')
        plt.ylabel('$\zeta^{(t)}$')
        plt.axis([0, nstates-1, min(logZ), max(logZ)])
        if hasattr(testsystem, 'logZ'):
            plt.axis([0, nstates-1, 0.0, max(testsystem.logZ)])
        pdf.savefig()  # saves the current figure into a pdf page

        # PAGE 2
        plt.figure(figsize=(6, 6))
        state_index = ncfile.variables['state_index'][:]
        plt.plot(state_index, '.')
        plt.title(testsystem.description, fontsize=title_fontsize)
        plt.xlabel('iteration $t$')
        plt.ylabel('state index')
        plt.axis([0, nsamples, 0, nstates-1])
        if hasattr(ncfile, 'second_stage_start'):
            t0 = getattr(ncfile, 'second_stage_start')
            plt.plot([t0, t0], [0, nstates-1], 'r-')
        pdf.savefig()  # saves the current figure into a pdf page

        # PAGE 3 : logZ estimates
        plt.figure(figsize=(6, 6))
        if hasattr(testsystem, 'logZ'):
            plt.hold(True)
            M = np.tile(testsystem.logZ, [nsamples,1])
            plt.plot(M, ':')
        logZ = ncfile.variables['logZ'][:,:]
        plt.plot(logZ[:,:], '-')
        plt.title(testsystem.description, fontsize=title_fontsize)
        plt.xlabel('iteration $t$')
        plt.ylabel('$\zeta^{(t)}$')
        plt.axis([0, nsamples, logZ.min(), logZ.max()])
        if hasattr(ncfile, 'second_stage_start'):
            t0 = getattr(ncfile, 'second_stage_start')
            plt.plot([t0, t0], [logZ.min(), logZ.max()], 'r-')
        pdf.savefig()  # saves the current figure into a pdf page

        # PAGE 4 : gamma
        plt.figure(figsize=(6, 6))
        plt.subplot(2,1,1)
        gamma = ncfile.variables['gamma'][:]
        plt.plot(gamma[:], '-')
        plt.hold(True)
        plt.axis([0, nsamples, gamma.min(), gamma.max()])
        if hasattr(ncfile, 'second_stage_start'):
            t0 = getattr(ncfile, 'second_stage_start')
            plt.plot([t0, t0], [gamma.min(), gamma.max()], 'r-')
        plt.title(testsystem.description, fontsize=title_fontsize)
        plt.xlabel('iteration $t$')
        plt.ylabel('$\gamma_t$')
        plt.subplot(2,1,2)
        log_gamma = np.log(ncfile.variables['gamma'][:])
        plt.plot(log_gamma[:], '-')
        plt.hold(True)
        plt.axis([0, nsamples, log_gamma.min(), log_gamma.max()])
        if hasattr(ncfile, 'second_stage_start'):
            t0 = getattr(ncfile, 'second_stage_start')
            plt.plot([t0, t0], [log_gamma.min(), log_gamma.max()], 'r-')
        plt.xlabel('iteration $t$')
        plt.ylabel('$\log \gamma_t$')
        pdf.savefig()  # saves the current figure into a pdf page

        # PAGE 4 : gamma
        plt.figure(figsize=(6, 6))
        log_target_probabilities = ncfile.variables['log_target_probabilities'][:,:]
        plt.plot(log_target_probabilities[:,:], '-')
        plt.hold(True)
        plt.axis([0, nsamples, log_target_probabilities.min(), log_target_probabilities.max()])
        if hasattr(ncfile, 'second_stage_start'):
            t0 = getattr(ncfile, 'second_stage_start')
            plt.plot([t0, t0], [log_target_probabilities.min(), log_target_probabilities.max()], 'r-')
        plt.title(testsystem.description, fontsize=title_fontsize)
        plt.xlabel('iteration $t$')
        plt.ylabel('log target probabilities')
        pdf.savefig()  # saves the current figure into a pdf page

        # FINISH
        plt.close()

def write_trajectory_dcd(netcdf_filename, topology, pdb_trajectory_filename, dcd_trajectory_filename):
    """
    Write trajectory.

    Parameters
    ----------
    netcdf_filename : str
        NetCDF filename.
    topology : Topology
        Topology object
    pdb_trajectory_filename : str
        PDB trajectory output filename
    dcd_trajectory_filename : str
        Output trajectory filename.

    """
    ncfile = netCDF4.Dataset(netcdf_filename, 'r')
    [nsamples, nstates] = ncfile.variables['logZ'].shape

    # Write reference.pdb file
    from simtk.openmm.app import PDBFile
    outfile = open(pdb_trajectory_filename, 'w')
    positions = unit.Quantity(ncfile.variables['positions'][0,:,:], unit.nanometers)
    PDBFile.writeFile(topology, positions, file=outfile)
    outfile.close()

    # TODO: Export as DCD trajectory with MDTraj
    from mdtraj.formats import DCDTrajectoryFile
    with DCDTrajectoryFile(dcd_trajectory_filename, 'w') as f:
        f.write(ncfile.variables['positions'][:,:,:] * 10.0) # angstroms

def write_trajectory(netcdf_filename, topology, reference_pdb_filename, trajectory_filename, strip_waters=True, image=False):
    """
    Write trajectory.

    Parameters
    ----------
    netcdf_filename : str
        NetCDF filename.
    topology : Topology
        OpenMM topology object
    reference_pdb_filename
        PDB trajectory output filename
    trajectory_filename : str
        Output trajectory filename. Type is autodetected by extension (.xtc, .dcd, .pdb) recognized by MDTraj
    strip_water : bool, optional, default=True
        If True, water will be stripped.
    image : bool, optional, default=False
        If True, image molecules before writing.

    """
    ncfile = netCDF4.Dataset(netcdf_filename, 'r')
    [nsamples, nstates] = ncfile.variables['logZ'].shape

    # Make selection
    mdtraj_topology = mdtraj.Topology.from_openmm(topology)
    dsl_selection = 'all'
    if strip_waters:
        dsl_selection = 'not water'
    atom_indices = mdtraj_topology.select(dsl_selection)
    selection_topology = mdtraj_topology.subset(atom_indices)

    # Convert to MDTraj trajectory.
    print('Creating MDTraj trajectory...')
    trajectory = mdtraj.Trajectory(ncfile.variables['positions'][:,atom_indices,:], selection_topology)
    trajectory.unitcell_vectors = ncfile.variables['box_vectors'][:,:,:]

    if image:
        print('Imaging molecules...')
        trajectory.image_molecules()

    # Write reference.pdb file
    print('Writing reference PDB file...')
    trajectory[0].save(reference_pdb_filename)

    print('Writing trajectory...')
    trajectory.save(trajectory_filename)
    print('Done.')
