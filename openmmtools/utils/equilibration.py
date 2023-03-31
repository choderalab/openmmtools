"""
Utility functions useful for equilibration.
"""

import logging
from openmm import unit, OpenMMException

# Set up logger
_logger = logging.getLogger(__name__)


def run_gentle_equilibration(topology, positions, system, stages, filename, platform_name='CUDA', save_box_vectors=True):
    """
    Run gentle equilibration. Serialize results in an external PDB or CIF file.

    Parameters
    ----------
    topology : openmm.app.Topology
        topology
    positions : np.array in unit.nanometer
        positions
    system : openmm.System
        system
    stages : list of dicts
        each dict corresponds to a stage of equilibration and contains the equilibration parameters for that stage

        equilibration parameters:
            EOM : str
                'minimize' or 'MD' or 'MD_interpolate' (the last one will allow interpolation between 'temperature' and 'temperature_end')
            n_steps : int
                number of steps of MD
            temperature : openmm.unit.kelvin
                temperature (kelvin)
            temperature_end : openmm.unit.kelvin, optional
                the temperature (kelvin) at which to finish interpolation, if 'EOM' is 'MD_interpolate'
            ensemble : str or None
                'NPT' or 'NVT'
            restraint_selection : str or None
                to be used by mdtraj to select atoms for which to apply restraints
            force_constant : openmm.unit.kilocalories_per_mole/openmm.unit.angstrom**2
                force constant (kcal/molA^2)
            collision_rate : 1/openmm.unit.picoseconds
                collision rate (1/picoseconds)
            timestep : openmm.unit.femtoseconds
                timestep (femtoseconds)
    filename : str
        path to save the equilibrated structure
    platform_name : str, default 'CUDA'
        name of platform to be used by OpenMM. If not specified, OpenMM will select the fastest available platform
    save_box_vectors : bool
        Whether to save the box vectors in a box_vectors.npy file in the working directory, after execution.
        Defaults to True.

    """
    import copy
    import openmm
    import time
    import numpy as np
    import mdtraj as md

    for stage_index, parameters in enumerate(stages):

        initial_time = time.time()
        print(f"Executing stage {stage_index + 1}")

        # Make a copy of the system
        system_copy = copy.deepcopy(system)

        # Add restraint
        if parameters['restraint_selection'] is not None:
            traj = md.Trajectory(positions, md.Topology.from_openmm(topology))
            selection_indices = traj.topology.select(parameters['restraint_selection'])

            custom_cv_force = openmm.CustomCVForce('(K_RMSD/2)*(RMSD)^2')
            custom_cv_force.addGlobalParameter('K_RMSD', parameters['force_constant'] * 2)
            rmsd_force = openmm.RMSDForce(positions, selection_indices)
            custom_cv_force.addCollectiveVariable('RMSD', rmsd_force)
            system_copy.addForce(custom_cv_force)

        # Set barostat update interval to 0 (for NVT)
        if parameters['ensemble'] == 'NVT':
            force_dict = {force.__class__.__name__: index for index, force in enumerate(system_copy.getForces())}
            try:
                system_copy.getForce(force_dict['MonteCarloBarostat']).setFrequency(0)  # This requires openmm 8
            except KeyError:
                # No MonteCarloBarostat found
                _logger.debug("No MonteCarloBarostat found in forces. Continuing.")
            except OpenMMException:
                # Must be Openmm<8
                system_copy.removeForce(force_dict['MonteCarloBarostat'])

        elif parameters['ensemble'] == 'NPT' or parameters['ensemble'] is None:
            pass

        else:
            raise ValueError("Invalid parameter supplied for 'ensemble'")

        # Set up integrator
        temperature = parameters['temperature']
        collision_rate = parameters['collision_rate']
        timestep = parameters['timestep']

        integrator = openmm.LangevinMiddleIntegrator(temperature, collision_rate, timestep)

        # Set up context
        platform = openmm.Platform.getPlatformByName(platform_name)
        if platform_name in ['CUDA', 'OpenCL']:
            platform.setPropertyDefaultValue('Precision', 'mixed')
        if platform_name in ['CUDA']:
            platform.setPropertyDefaultValue('DeterministicForces', 'true')

        context = openmm.Context(system_copy, integrator, platform)
        context.setPeriodicBoxVectors(*system_copy.getDefaultPeriodicBoxVectors())
        context.setPositions(positions)
        context.setVelocitiesToTemperature(temperature)

        # Run minimization or MD
        n_steps = parameters['n_steps']
        n_steps_per_iteration = 100

        if parameters['EOM'] == 'minimize':
            openmm.LocalEnergyMinimizer.minimize(context, maxIterations=n_steps)

        elif parameters['EOM'] == 'MD':
            for _ in range(int(n_steps / n_steps_per_iteration)):
                integrator.step(n_steps_per_iteration)

        elif parameters['EOM'] == 'MD_interpolate':
            temperature_end = parameters['temperature_end']
            temperature_unit = unit.kelvin
            temperatures = np.linspace(temperature / temperature_unit, temperature_end / temperature_unit,
                                       int(n_steps / n_steps_per_iteration)) * temperature_unit
            for temperature in temperatures:
                integrator.setTemperature(temperature)
                integrator.step(n_steps_per_iteration)

        else:
            raise ValueError("Invalid parameter supplied for 'EOM'")

        # Retrieve positions after this stage of equil
        state = context.getState(getPositions=True)
        positions = state.getPositions(asNumpy=True)

        # Update default box vectors for next iteration
        box_vectors = state.getPeriodicBoxVectors()
        system.setDefaultPeriodicBoxVectors(*box_vectors)

        # Delete context and integrator
        del context, integrator, system_copy

        elapsed_time = time.time() - initial_time
        print(f"\tStage {stage_index + 1} took {elapsed_time} seconds")

    # Save the final equilibrated positions
    if filename.endswith('pdb'):
        openmm.app.PDBFile.writeFile(topology, positions, open(filename, "w"), keepIds=True)
    elif filename.endswith('cif'):
        openmm.app.PDBxFile.writeFile(topology, positions, open(filename, "w"), keepIds=True)

    # Save the box vectors
    if save_box_vectors:
        with open(filename[:-4] + '_box_vectors.npy', 'wb') as f:
            np.save(f, box_vectors)

