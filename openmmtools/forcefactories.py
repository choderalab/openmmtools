#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Factories to manipulate OpenMM System forces.

"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import copy

import numpy as np
import mdtraj
try:
    import openmm
    from openmm import unit
except ImportError:  # OpenMM < 7.6
    from simtk import openmm, unit

from openmmtools import forces


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def replace_reaction_field(reference_system, switch_width=1.0*unit.angstrom,
                           return_copy=True, shifted=False):
    """Return a system converted to use a switched reaction-field electrostatics using :class:`openmmtools.forces.UnshiftedReactionField`.

    This will add an `UnshiftedReactionFieldForce` or `SwitchedReactionFieldForce`
    for each `NonbondedForce` that utilizes `CutoffPeriodic`.

    Note that `AbsoluteAlchemicalFactory.create_alchemical_system()` can NOT
    handle the resulting `System` object yet since the `CustomNonbondedForce`
    are not recognized and re-coded.

    Parameters
    ----------
    reference_system : openmm.System
        The system to use as a reference. This will be modified only if
        `return_copy` is `False`.
    switch_width : openmm.unit.Quantity, default=1.0*angstrom
        Switch width for electrostatics (units of distance).
    return_copy : bool, optional, default=True
        If `True`, `reference_system` is not modified, and a copy is returned.
        Setting it to `False` speeds up the function execution but modifies
        the `reference_system` object.
    shifted : bool, optional, default=False
        If `True`, a shifted reaction-field will be used.

    Returns
    -------
    system : openmm.System
        System with reaction-field converted to c_rf = 0

    """
    if return_copy:
        system = copy.deepcopy(reference_system)
    else:
        system = reference_system

    if shifted:
        force_constructor = getattr(forces, 'SwitchedReactionFieldForce')
    else:
        force_constructor = getattr(forces, 'UnshiftedReactionFieldForce')

    # Add an reaction field for each CutoffPeriodic NonbondedForce.
    for reference_force in forces.find_forces(system, openmm.NonbondedForce).values():
        if reference_force.getNonbondedMethod() == openmm.NonbondedForce.CutoffPeriodic:
            reaction_field_force = force_constructor.from_nonbonded_force(reference_force, switch_width=switch_width)
            system.addForce(reaction_field_force)

            # Remove particle electrostatics from reference force, but leave exceptions.
            for particle_index in range(reference_force.getNumParticles()):
                charge, sigma, epsilon = reference_force.getParticleParameters(particle_index)
                reference_force.setParticleParameters(particle_index, abs(0.0*charge), sigma, epsilon)

    return system


# =============================================================================
# RESTRAIN ATOMS
# =============================================================================

def restrain_atoms_by_dsl(thermodynamic_state, sampler_state, topology, atoms_dsl, **kwargs):
    # Make sure the topology is an MDTraj topology.
    if isinstance(topology, mdtraj.Topology):
        mdtraj_topology = topology
    else:
        mdtraj_topology = mdtraj.Topology.from_openmm(topology)

    # Determine indices of the atoms to restrain.
    restrained_atoms = mdtraj_topology.select(atoms_dsl).tolist()
    restrain_atoms(thermodynamic_state, sampler_state, restrained_atoms, **kwargs)


def restrain_atoms(thermodynamic_state, sampler_state, restrained_atoms, sigma=3.0*unit.angstroms):
    """Apply a soft harmonic restraint to the given atoms.

    This modifies the ``ThermodynamicState`` object.

    Parameters
    ----------
    thermodynamic_state : openmmtools.states.ThermodynamicState
        The thermodynamic state with the system. This will be modified.
    sampler_state : openmmtools.states.SamplerState
        The sampler state with the positions.
    topology : mdtraj.Topology or openmm.Topology
        The topology of the system.
    atoms_dsl : str
        The MDTraj DSL string for selecting the atoms to restrain.
    sigma : openmm.unit.Quantity, optional
        Controls the strength of the restrain. The smaller, the tighter
        (units of distance, default is 3.0*angstrom).

    """
    K = thermodynamic_state.kT / sigma**2  # Spring constant.
    system = thermodynamic_state.system  # This is a copy.

    # Check that there are atoms to restrain.
    if len(restrained_atoms) == 0:
        raise ValueError('No atoms to restrain.')

    # We need to translate the restrained molecule to the origin
    # to avoid MonteCarloBarostat rejections (see openmm#1854).
    if thermodynamic_state.pressure is not None:
        # First, determine all the molecule atoms. Reference platform is the cheapest to allocate?
        reference_platform = openmm.Platform.getPlatformByName('Reference')
        integrator = openmm.VerletIntegrator(1.0*unit.femtosecond)
        context = openmm.Context(system, integrator, reference_platform)
        molecules_atoms = context.getMolecules()
        del context, integrator

        # Make sure the atoms to restrain belong only to a single molecule.
        molecules_atoms = [set(molecule_atoms) for molecule_atoms in molecules_atoms]
        restrained_atoms_set = set(restrained_atoms)
        restrained_molecule_atoms = None
        for molecule_atoms in molecules_atoms:
            if restrained_atoms_set.issubset(molecule_atoms):
                # Convert set to list to use it as numpy array indices.
                restrained_molecule_atoms = list(molecule_atoms)
                break
        if restrained_molecule_atoms is None:
            raise ValueError('Cannot match the restrained atoms to any molecule. Restraining '
                             'two molecules is not supported when using a MonteCarloBarostat.')

        # Translate system so that the center of geometry is in
        # the origin to reduce the barostat rejections.
        distance_unit = sampler_state.positions.unit
        centroid = np.mean(sampler_state.positions[restrained_molecule_atoms,:] / distance_unit, axis=0)
        sampler_state.positions -= centroid * distance_unit

    # Create a CustomExternalForce to restrain all atoms.
    if thermodynamic_state.is_periodic:
        energy_expression = '(K/2)*periodicdistance(x, y, z, x0, y0, z0)^2' # periodic distance
    else:
        energy_expression = '(K/2)*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)' # non-periodic distance
    restraint_force = openmm.CustomExternalForce(energy_expression)
    # Adding the spring constant as a global parameter allows us to turn it off if desired
    restraint_force.addGlobalParameter('K', K)
    restraint_force.addPerParticleParameter('x0')
    restraint_force.addPerParticleParameter('y0')
    restraint_force.addPerParticleParameter('z0')
    for index in restrained_atoms:
        parameters = sampler_state.positions[index,:].value_in_unit_system(unit.md_unit_system)
        restraint_force.addParticle(index, parameters)

    # Update thermodynamic state.
    system.addForce(restraint_force)
    thermodynamic_state.system = system

class SystemFactory(object):
    """
    A factory for creating modified OpenMM Systems.
    """
    def __init__(self):
        """Configure the System factory and options"""
        pass
    
    def create_system(self, reference_system, **kwargs):
        """
        Create a modified OpenMM System.

        Parameters
        ----------
        reference_system : simtk.openmm.System
            The system to use as a reference.
        **kwargs
            Additional arguments to pass to the _create_system() method.

        Returns
        -------
        system : simtk.openmm.System
            The modified system.

        """
        system = self._create_system(reference_system, **kwargs)
        return system            

class ReactionFieldFactory(SystemFactory):
    """
    A factory for creating modified OpenMM Systems where electrostatics are replaced with reaction field electrostatics.
    """
    def __init__(self, method='Riniker'):
        """Configure the System factory and options
        
        Parameters
        ----------
        method : str, optional, default='Riniker'
            The method to use for reaction field electrostatics. Currently supported methods are:
                'Riniker' : A high-quality reaction field model intended for use with atomic cutoffs [1]
                'Riniker-MTS' : A high-quality reaction field model intended for use with atomic cutoffs [1] and multiple timestep methods 

        References
        ----------
        [1] Kubincová A, Riniker S, and Hünenberger P. Reaction-field electrostatics in molecular dynamics simulations: development of a 
        conservative scheme compatible with an atomic cutoff. Phys. Chem. Chem. Phys. 22:26419-26437, 2020.
        https://doi.org/10.1039/D0CP03835K 
        """
        self.method = method

    def create_system(self, reference_system, **kwargs):
        return super().create_system(reference_system, **kwargs)


def replace_reaction_field_atomic_mts(reference_system, method='riniker-AT-SHIFT-4-6', solvent_indices=None, cutoff=None, solvent_dielectric=None):
    """
    Replace electrostatics in the specified System with a high-quality reaction field model intended for use with atomic cutoffs [1]
    and multiple timestep methods.

    A copy of the System is returned where electrostatics and Lennard-Jones has been migrated from the NonbondedForce to a CustomNonbondedForce.
    Solvent-solvent interactions are split into a separate force group (group 1) to facilitate the use of multiple timestep methods with rigid solvents.

    A new System object will be created. The original will not be modified.

    NonbondedForce : 'Exceptions' : exceptions are retained, but charges and LJ parameters are zeroed (group 0)
    CustomNonbondedForce : 'Lennard-Jones solvent-solvent' (group 1)
    CustomNonbondedForce : 'electrostatics solvent-solvent' (group 1)
    CustomNonbondedForce : 'Lennard-Jones solute-solute and solute-solvent' (group 0)
    CustomNonbondedForce : 'electrostatics solute-solute and solute-solvent' (group 0)
    all other forces : left unmodified (group 0)

    Separate forces are used for Lennard-Jones and electrostatics because the Lennard-Jones forces may include long-range corrections.

    Note that `AbsoluteAlchemicalFactory` can NOT yet handle the resulting `System` object because the Custom*Forces cannot be automatically modified.

    .. warning :: This API is experimental and subject to change.

    TODO:
    * Can we allow the reaction field form to be selectable?
    * Should we refactor this to a factory?
    # Can we automatically identify all rigid solvent atoms in the OpenMM System and eliminate the need for manually specifying solvent atom indices?

    References
    ----------
    [1] Kubincová A, Riniker S, and Hünenberger P. Reaction-field electrostatics in molecular dynamics simulations: development of a 
    conservative scheme compatible with an atomic cutoff. Phys. Chem. Chem. Phys. 22:26419-26437, 2020.
    https://doi.org/10.1039/D0CP03835K 

    Parameters
    ----------
    reference_system : simtk.openmm.System
        The system to use as a reference.
    method : str, optional, default='riniker-AT-SHIFT-4-6'
        The reaction field method to use.
        One of ['riniker-AT-SHIFT-4-6', 'openmm-shifted', 'openmm-unshifted'] 
    solvent_indices : iterable of int, optional, default=None
        The indices of the solute atoms in the system. 
        If not None, interactions within the specified atom indices will be assigned to a separate CustomForce
        to allow force splitting.
        If None, all atoms will be included in a single CustomForce.
    cutoff : simtk.unit.Quantity with dimension of distance, optional, default=None
        If specified, the cutoff distance to use for the reaction field electrostatics.
    solvent_dielectric : float, optional, default=None
        If specified, the dielectric constant to use for the reaction field electrostatics.
        If not specified, will extract from NonbondedForce

    Returns
    -------
    system : simtk.openmm.System
        New System object with reaction field electrostatics.
    """
    
    # If not specified, solute will be empty list
    if solvent_indices is None:
        solvent_indices = list()

    # Create a copy of the reference system
    system = copy.deepcopy(reference_system)

    # Find the NonbondedForce
    nonbonded_forces = forces.find_forces(system, openmm.NonbondedForce)
    if len(nonbonded_forces) != 1:
        raise ValueError(f"Expected to find a single NonbondedForce in the System, but found {len(nonbonded_forces)}")
    nonbonded_force = list(nonbonded_forces.values())[0]

    # Set cutoff
    if cutoff is not None:
        nonbonded_force.setCutoffDistance(cutoff)    
    cutoff = nonbonded_force.getCutoffDistance()

    # Set solvent dielectric
    if solvent_dielectric is None:
        solvent_dielectric = nonbonded_force.getReactionFieldDielectric()

    def create_electrostatics_force(reference_force, method='riniker-AT-SHIFT-4-6', cutoff=None, solvent_dielectric=None, name=None, force_group=None):
        """
        Create a CustomNonbondedForce to replace electrostatics, integrating exceptions as exclusions.

        The atomic reaction field method from Ref [1] is used.

        TODO: Can we allow the reaction field form to be selectable?

        References
        ----------
        [1] Kubincová A, Riniker S, and Hünenberger P. Reaction-field electrostatics in molecular dynamics simulations: development of a 
        conservative scheme compatible with an atomic cutoff. Phys. Chem. Chem. Phys. 22:26419-26437, 2020.
        https://doi.org/10.1039/D0CP03835K 

        Parameters
        ----------
        reference_force : openmm.NonbondedForce
            The reference force used to look up exceptions/exclusions.
        method : str, optional, default='riniker-AT-SHIFT-4-6'
            The reaction field method to use.
            One of ['riniker-AT-SHIFT-4-6', 'openmm-shifted', 'openmm-unshifted'] 
        cutoff : simtk.unit.Quantity with dimension of distance, optional, default=None
            If specified, the cutoff distance to use for the reaction field electrostatics.
        solvent_dielectric : float, optional, default=None
            The dielectric constant to use for the reaction field electrostatics.
            This must be specified
        name : str, optional, default=None
            The name of the force.        
        force_group : int, optional, default=None
            The force group to assign to the force.
        """
        if cutoff is None:
            cutoff = nonbonded_force.getCutoffDistance()
        if solvent_dielectric is None:
            raise ValueError("solvent_dielectric must be specified")

        from openmmtools.constants import ONE_4PI_EPS0
        if method == 'riniker-AT-SHIFT-4-6':
            # Compute reaction field constants for AT reaction field method from Ref [1]
            krf = ((solvent_dielectric - 1) / (1 + 2 * solvent_dielectric))
            mrf = 4
            nrf = 6
            arfm = 3 / (mrf*(nrf - mrf)) * ((2*solvent_dielectric+nrf-1)/(1+2*solvent_dielectric))
            arfn = 3 / (nrf*(mrf - nrf)) * ((2*solvent_dielectric+mrf-1)/(1+2*solvent_dielectric))

            # Create a CustomNonbondedForce to replace the NonbondedForce.
            energy_expression = (""
            f"{ONE_4PI_EPS0}*(chargeprod/cutoff)*(1/x + krf*x^2 + arfm*x^{mrf} + arfn*x^{nrf} - crf);"
            f"x = r/cutoff;"
            f"chargeprod = charge1*charge2;"
            f"crf = (1 + krf + arfm + arfn);" # ensures that energy goes to zero at x = 1
            f"cutoff = {cutoff.value_in_unit_system(unit.md_unit_system)};"
            f"krf = {krf};"
            f"arfm = {arfm};"
            f"arfn = {arfn};"
            )
        elif method == 'openmm-shifted':        
            # Create OpenMM standard shifted reaction field force
            krf = ((solvent_dielectric - 1) / (1 + 2 * solvent_dielectric))
            energy_expression = (""
            f"{ONE_4PI_EPS0}*(chargeprod/cutoff)*(1/x + krf*x^2 - crf);"
            f"x = r/cutoff;"
            f"chargeprod = charge1*charge2;"
            f"crf = (1 + krf);" # ensures that energy goes to zero at x = 1
            f"cutoff = {cutoff.value_in_unit_system(unit.md_unit_system)};"
            f"krf = {krf};"
            )
        elif method == 'openmm-unshifted':
            # Create OpenMM standard shifted reaction field force
            krf = ((solvent_dielectric - 1) / (1 + 2 * solvent_dielectric))
            energy_expression = (""
            f"{ONE_4PI_EPS0}*(chargeprod/cutoff)*(1/x + krf*x^2);"
            f"x = r/cutoff;"
            f"chargeprod = charge1*charge2;"
            f"cutoff = {cutoff.value_in_unit_system(unit.md_unit_system)};"
            f"krf = {krf};"
            )
        else:
            raise ValueError(f'Unknown method {method}')

        custom_nonbonded_force = openmm.CustomNonbondedForce(energy_expression)
        custom_nonbonded_force.addPerParticleParameter("charge")
        custom_nonbonded_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
        custom_nonbonded_force.setUseLongRangeCorrection(False)
        if name is not None:
            custom_nonbonded_force.setName(name)
        if force_group is not None:
            custom_nonbonded_force.setForceGroup(force_group)
        custom_nonbonded_force.setCutoffDistance(cutoff)
        custom_nonbonded_force.setUseSwitchingFunction(False)
        custom_nonbonded_force.setSwitchingDistance(0.0)

        # Copy exclusions
        for exception_index in range(reference_force.getNumExceptions()):
            [iatom, jatom, chargeprod, sigma, epsilon] = reference_force.getExceptionParameters(exception_index)
            custom_nonbonded_force.addExclusion(iatom, jatom)            

        return custom_nonbonded_force

    def create_sterics_force(reference_force, cutoff=None, name=None, force_group=None, use_long_range_correction=False):
        """
        Create a CustomNonbondedForce to replace sterics, integrating exceptions as exclusions.

        Parameters
        ----------
        reference_force : openmm.NonbondedForce
            Reference NonbondedForce to use for exclusions/exceptions.
        cutoff : simtk.unit.Quantity with dimension of distance, optional, default=None
            If specified, the cutoff distance to use for the reaction field electrostatics.
        name : str, optional, default=None
            The name of the force.
        force_group : int, optional, default=None
            The force group to assign to the force.
        use_long_range_correction : bool, optional, default=False
            If True, the long-range correction will be included.
        """
        if cutoff is None:
            cutoff = nonbonded_force.getCutoffDistance()

        # Create a CustomNonbondedForce to replace the NonbondedForce.
        energy_expression = (""
        f"4*epsilon*x*(x-1);"        
        f"x = (sigma/r)^6;"
        f"epsilon = sqrt(epsilon1*epsilon2);"
        f"sigma = 0.5*(sigma1 + sigma2);"
        )
        
        custom_nonbonded_force = openmm.CustomNonbondedForce(energy_expression)
        custom_nonbonded_force.addPerParticleParameter("sigma")
        custom_nonbonded_force.addPerParticleParameter("epsilon")
        custom_nonbonded_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
        custom_nonbonded_force.setUseLongRangeCorrection(use_long_range_correction)
        if name is not None:
            custom_nonbonded_force.setName(name)
        if force_group is not None:
            custom_nonbonded_force.setForceGroup(force_group)
        custom_nonbonded_force.setCutoffDistance(cutoff)

        custom_nonbonded_force.setUseSwitchingFunction(reference_force.getUseSwitchingFunction())
        custom_nonbonded_force.setSwitchingDistance(reference_force.getSwitchingDistance())

        # Copy exclusions
        for exception_index in range(reference_force.getNumExceptions()):
            [iatom, jatom, chargeprod, sigma, epsilon] = reference_force.getExceptionParameters(exception_index)
            custom_nonbonded_force.addExclusion(iatom, jatom)

        return custom_nonbonded_force
    
    # Determine whether sterics will use long-range correction
    use_long_range_correction = nonbonded_force.getUseDispersionCorrection()

    #
    # Pre-process to make sure LJ sigma > 0, since sigma=0 will cause problems with our CustomNonbondedForces
    #
    SIGMA_MIN = 1.0e-4 * unit.angstroms # minimum allowed value of sigma
    for particle_index in range(nonbonded_force.getNumParticles()):
        charge, sigma, epsilon = nonbonded_force.getParticleParameters(particle_index)
        if (sigma < SIGMA_MIN):
            nonbonded_force.setParticleParameters(particle_index, charge, SIGMA_MIN, epsilon)

    #
    # Solvent-solvent electrostatics
    #

    electrostatics_force = create_electrostatics_force(nonbonded_force, name='electrostatics : solvent-solvent', method='riniker-AT-SHIFT-4-6', cutoff=cutoff, solvent_dielectric=solvent_dielectric, force_group=1)
    sterics_force = create_sterics_force(nonbonded_force, name='sterics : solvent-solvent', cutoff=cutoff, use_long_range_correction=use_long_range_correction, force_group=1)

    # Only add solvent interactions 
    # NOTE: We don't use interaction groups for solvent-solvent since this might not be able to take advantage of pairlists
    for particle_index in range(nonbonded_force.getNumParticles()):
        charge, sigma, epsilon = nonbonded_force.getParticleParameters(particle_index)
        if particle_index not in solvent_indices:
            charge = abs(0.0*charge)
            epsilon = abs(0.0*epsilon)
        electrostatics_force.addParticle([charge])
        sterics_force.addParticle([sigma, epsilon])

    system.addForce(electrostatics_force)
    system.addForce(sterics_force)

    #
    # Solute-solvent electrostatics
    #

    electrostatics_force = create_electrostatics_force(nonbonded_force, name='electrostatics : solute-solute and solute-solvent', method='riniker-AT-SHIFT-4-6', cutoff=cutoff, solvent_dielectric=solvent_dielectric)
    sterics_force = create_sterics_force(nonbonded_force, name='sterics : solute-solute and solute-solvent', cutoff=cutoff, use_long_range_correction=use_long_range_correction)

    # Add all charges to this force
    for particle_index in range(nonbonded_force.getNumParticles()):
        charge, sigma, epsilon = nonbonded_force.getParticleParameters(particle_index)
        electrostatics_force.addParticle([charge])
        sterics_force.addParticle([sigma, epsilon])
    
    # Only include interactions involving solute via interaction groups
    all_indices = [ particle_index for particle_index in range(nonbonded_force.getNumParticles()) ]
    solute_indices = list( set(all_indices).difference(solvent_indices) )

    if len(solute_indices) < len(all_indices):
        # This will only be efficient if solute_indices << all_indices
        electrostatics_force.addInteractionGroup(solute_indices, all_indices)
        sterics_force.addInteractionGroup(solute_indices, all_indices)

    # TODO: Warn this will be inefficient if (solute_indices < all_indices) but not (solute_indices << all_indices)

    system.addForce(electrostatics_force)
    system.addForce(sterics_force)

    #
    # Zero out particle sterics and electrostatics in NonbondedForce (retaining exceptions)
    #
    for particle_index in range(nonbonded_force.getNumParticles()):
        charge, sigma, epsilon = nonbonded_force.getParticleParameters(particle_index)
        nonbonded_force.setParticleParameters(particle_index, abs(0.0*charge), sigma, abs(0.0*epsilon))
    # Disable switching function (if enabled)
    nonbonded_force.setUseSwitchingFunction(False)
    
    return system


# =============================================================================
# RESTRAIN ATOMS
# =============================================================================

def restrain_atoms_by_dsl(thermodynamic_state, sampler_state, topology, atoms_dsl, **kwargs):
    # Make sure the topology is an MDTraj topology.
    if isinstance(topology, mdtraj.Topology):
        mdtraj_topology = topology
    else:
        mdtraj_topology = mdtraj.Topology.from_openmm(topology)

    # Determine indices of the atoms to restrain.
    restrained_atoms = mdtraj_topology.select(atoms_dsl).tolist()
    restrain_atoms(thermodynamic_state, sampler_state, restrained_atoms, **kwargs)


def restrain_atoms(thermodynamic_state, sampler_state, restrained_atoms, sigma=3.0*unit.angstroms):
    """Apply a soft harmonic restraint to the given atoms.

    This modifies the ``ThermodynamicState`` object.

    Parameters
    ----------
    thermodynamic_state : openmmtools.states.ThermodynamicState
        The thermodynamic state with the system. This will be modified.
    sampler_state : openmmtools.states.SamplerState
        The sampler state with the positions.
    topology : mdtraj.Topology or simtk.openmm.Topology
        The topology of the system.
    atoms_dsl : str
        The MDTraj DSL string for selecting the atoms to restrain.
    sigma : simtk.unit.Quantity, optional
        Controls the strength of the restrain. The smaller, the tighter
        (units of distance, default is 3.0*angstrom).

    """
    K = thermodynamic_state.kT / sigma**2  # Spring constant.
    system = thermodynamic_state.system  # This is a copy.

    # Check that there are atoms to restrain.
    if len(restrained_atoms) == 0:
        raise ValueError('No atoms to restrain.')

    # We need to translate the restrained molecule to the origin
    # to avoid MonteCarloBarostat rejections (see openmm#1854).
    if thermodynamic_state.pressure is not None:
        # First, determine all the molecule atoms. Reference platform is the cheapest to allocate?
        reference_platform = openmm.Platform.getPlatformByName('Reference')
        integrator = openmm.VerletIntegrator(1.0*unit.femtosecond)
        context = openmm.Context(system, integrator, reference_platform)
        molecules_atoms = context.getMolecules()
        del context, integrator

        # Make sure the atoms to restrain belong only to a single molecule.
        molecules_atoms = [set(molecule_atoms) for molecule_atoms in molecules_atoms]
        restrained_atoms_set = set(restrained_atoms)
        restrained_molecule_atoms = None
        for molecule_atoms in molecules_atoms:
            if restrained_atoms_set.issubset(molecule_atoms):
                # Convert set to list to use it as numpy array indices.
                restrained_molecule_atoms = list(molecule_atoms)
                break
        if restrained_molecule_atoms is None:
            raise ValueError('Cannot match the restrained atoms to any molecule. Restraining '
                             'two molecules is not supported when using a MonteCarloBarostat.')

        # Translate system so that the center of geometry is in
        # the origin to reduce the barostat rejections.
        distance_unit = sampler_state.positions.unit
        centroid = np.mean(sampler_state.positions[restrained_molecule_atoms,:] / distance_unit, axis=0)
        sampler_state.positions -= centroid * distance_unit

    # Create a CustomExternalForce to restrain all atoms.
    if thermodynamic_state.is_periodic:
        energy_expression = '(K/2)*periodicdistance(x, y, z, x0, y0, z0)^2' # periodic distance
    else:
        energy_expression = '(K/2)*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)' # non-periodic distance
    restraint_force = openmm.CustomExternalForce(energy_expression)
    # Adding the spring constant as a global parameter allows us to turn it off if desired
    restraint_force.addGlobalParameter('K', K)
    restraint_force.addPerParticleParameter('x0')
    restraint_force.addPerParticleParameter('y0')
    restraint_force.addPerParticleParameter('z0')
    for index in restrained_atoms:
        parameters = sampler_state.positions[index,:].value_in_unit_system(unit.md_unit_system)
        restraint_force.addParticle(index, parameters)

    # Update thermodynamic state.
    system.addForce(restraint_force)
    thermodynamic_state.system = system


if __name__ == '__main__':
    import doctest
    doctest.testmod()
