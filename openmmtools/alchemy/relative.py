import simtk.openmm as openmm
import simtk.openmm.app as app
import simtk.unit as unit
import mdtraj as md
import numpy as np
import copy
import enum
from io import StringIO
import lxml.etree as etree
from openmmtools.constants import ONE_4PI_EPS0

InteractionGroup = enum.Enum("InteractionGroup", ['unique_old', 'unique_new', 'core', 'environment'])

class TopologyProposal(object):
    """
    Container for storing transformation from one System to another System for relative alchemical factories.

    Arguments
    ---------
    new_topology : simtk.openmm.Topology object
        openmm Topology representing the proposed new system
    new_system : simtk.openmm.System object
        openmm System of the newly proposed state
    old_topology : simtk.openmm.Topology object
        openmm Topology of the current system
    old_system : simtk.openmm.System object
        openm System of the current state
    logp_proposal : float
        contribution from the chemical proposal to the log probability of acceptance (Eq. 36 for hybrid; Eq. 53 for two-stage)
    new_to_old_atom_map : dict
        {new_atom_idx : old_atom_idx} map for the two systems
    old_alchemical_atoms : list, optional, default=None
        List of all atoms in old system that are being transformed.
        If None, all atoms are assumed to be part of the alchemical region.
    chemical_state_key : str
        The current chemical state (unique)
    metadata : dict
        additional information of interest about the state

    Properties
    ----------
    new_topology : simtk.openmm.Topology object
        openmm Topology representing the proposed new system
    new_system : simtk.openmm.System object
        openmm System of the newly proposed state
    old_topology : simtk.openmm.Topology object
        openmm Topology of the current system
    old_system : simtk.openmm.System object
        openm System of the current state
    old_positions : [n, 3] np.array, Quantity
        positions of the old system
    logp_proposal : float
        contribution from the chemical proposal to the log probability of acceptance (Eq. 36 for hybrid; Eq. 53 for two-stage)
    new_to_old_atom_map : dict
        {new_atom_idx : old_atom_idx} map for the two systems
    old_to_new_atom_map : dict
        {old_atom_idx : new_atom_idx} map for the two systems
    new_alchemical_atoms : list
        List of all atoms in new system that are being transformed
    new_environment_atoms : list
        List of all atoms in new system that are not transformed, just mapped
    old_alchemical_atoms : list
        List of all atoms in old system that are being transformed
    old_environment_atoms : list
        List of all atoms in old system that are not transformed, just mapped
    unique_new_atoms : list of int
        List of indices of the unique new atoms
    unique_old_atoms : list of int
        List of indices of the unique old atoms
    natoms_new : int
        Number of atoms in the new system
    natoms_old : int
        Number of atoms in the old system
    old_chemical_state_key : str
        The previous chemical state key
    new_chemical_state_key : str
        The proposed chemical state key
    metadata : dict, optional, default=dict()
        additional information of interest about the state
    """

    def __init__(self,
                 new_topology=None, new_system=None,
                 old_topology=None, old_system=None,
                 new_to_old_atom_map=None,
                 old_alchemical_atoms=None
                 metadata=None):

        self._new_topology = new_topology
        self._new_system = new_system
        self._old_topology = old_topology
        self._old_system = old_system
        self._new_to_old_atom_map = new_to_old_atom_map
        self._old_to_new_atom_map = {old_atom : new_atom for new_atom, old_atom in new_to_old_atom_map.items()}
        self._unique_new_atoms = list(set(range(self._new_topology._numAtoms))-set(self._new_to_old_atom_map.keys()))
        self._unique_old_atoms = list(set(range(self._old_topology._numAtoms))-set(self._new_to_old_atom_map.values()))
        self._old_alchemical_atoms = set(old_alchemical_atoms) if (old_alchemical_atoms is not None) else {atom for atom in range(old_system.getNumParticles())}
        self._new_alchemical_atoms = set(self._old_to_new_atom_map.values()).union(self._unique_new_atoms)
        self._old_environment_atoms = set(range(old_system.getNumParticles())) - self._old_alchemical_atoms
        self._new_environment_atoms = set(range(new_system.getNumParticles())) - self._new_alchemical_atoms
        self._metadata = metadata or dict()

    @property
    def new_topology(self):
        return self._new_topology
    @property
    def new_system(self):
        return self._new_system
    @property
    def old_topology(self):
        return self._old_topology
    @property
    def old_system(self):
        return self._old_system
    @property
    def logp_proposal(self):
        return self._logp_proposal
    @property
    def new_to_old_atom_map(self):
        return self._new_to_old_atom_map
    @property
    def old_to_new_atom_map(self):
        return self._old_to_new_atom_map
    @property
    def unique_new_atoms(self):
        return self._unique_new_atoms
    @property
    def unique_old_atoms(self):
        return self._unique_old_atoms
    @property
    def new_alchemical_atoms(self):
        return list(self._new_alchemical_atoms)
    @property
    def old_alchemical_atoms(self):
        return list(self._old_alchemical_atoms)
    @property
    def new_environment_atoms(self):
        return list(self._new_environment_atoms)
    @property
    def old_environment_atoms(self):
        return list(self._old_environment_atoms)
    @property
    def n_atoms_new(self):
        return self._new_system.getNumParticles()
    @property
    def n_atoms_old(self):
        return self._old_system.getNumParticles()
    @property
    def new_chemical_state_key(self):
        return self._new_chemical_state_key
    @property
    def old_chemical_state_key(self):
        return self._old_chemical_state_key
    @property
    def metadata(self):
        return self._metadata

class HybridTopologyFactory(object):
    """
    This class generates a hybrid topology based on a perses topology proposal. This class treats atoms
    in the resulting hybrid system as being from one of four classes:

    unique_old_atom : these atoms are not mapped and only present in the old system. Their interactions will be on for
        lambda=0, off for lambda=1
    unique_new_atom : these atoms are not mapped and only present in the new system. Their interactions will be off
        for lambda=0, on for lambda=1
    core_atom : these atoms are mapped, and are part of a residue that is changing. Their interactions will be those
        corresponding to the old system at lambda=0, and those corresponding to the new system at lambda=1
    environment_atom : these atoms are mapped, and are not part of a changing residue. Their interactions are always
        on and are alchemically unmodified.

    """

    _known_forces = {'HarmonicBondForce', 'HarmonicAngleForce', 'PeriodicTorsionForce', 'NonbondedForce', 'MonteCarloBarostat'}

    def __init__(self, consistent_exceptions=False, switch_width=1.0*unit.angstroms,
                 alchemical_pme_treatment='direct-space', alchemical_rf_treatment='switched',
                 disable_alchemical_dispersion_correction=False, split_alchemical_forces=True
                 alchemical_functions=None):
        """
        consistent_exceptions : bool, optional, default = False
            If True, the same functional form of the System's nonbonded
            method will be use to determine the electrostatics contribution
            to the potential energy of 1,4 exceptions instead of the
            classical q1*q2/(4*epsilon*epsilon0*pi*r).
        switch_width : float, optional, default = 1.0 * angstroms
            Default switch width for electrostatics in periodic cutoff systems
            used in alchemical interactions only.
        alchemical_pme_treatment : str, optional, default = 'direct-space'
            Controls how alchemical region electrostatics are treated when PME is used.
            Options are ['direct-space', 'coulomb', 'exact'].
            - 'direct-space' only models the direct space contribution
            - 'coulomb' includes switched Coulomb interaction
            - 'exact' includes also the reciprocal space contribution, but it's
               only possible to annihilate the charges and the softcore parameters
               controlling the electrostatics are deactivated. Also, with this
               method, modifying the global variable `lambda_electrostatics` is
               not sufficient to control the charges. The recommended way to change
               them is through the `AlchemicalState` class.
        alchemical_rf_treatment : str, optional, default = 'switched'
            Controls how alchemical region electrostatics are treated when RF is used
            Options are ['switched', 'shifted']
            'switched' sets c_rf = 0 for all reaction-field interactions and ensures continuity with a switch
            'shifted' retains c_rf != 0 but can give erroneous results for hydration free energies
        disable_alchemical_dispersion_correction : bool, optional, default=False
            If True, the long-range dispersion correction will not be included for the alchemical
            region to avoid the need to recompute the correction (a CPU operation that takes ~ 0.5 s)
            every time 'lambda_sterics' is changed. If using nonequilibrium protocols, it is recommended
            that this be set to True since this can lead to enormous (100x) slowdowns if the correction
            must be recomputed every time step.
        split_alchemical_forces : bool, optional, default=True
            If True, forces that are altered to different alchemical variables
            will be split in different force groups. All non-alchemical forces
            will maintain their original force group. If more than 32 force
            groups are required, an error is thrown.
        alchemical_functions : dict, optional, default=None
            Alchemical functions that determine how each force is scaled with lambda. The keys must be strings with
            names beginning with lambda_ and ending with each of bonds, angles, torsions, sterics, electrostatics.
            If functions is none, then the integrator will need to set each of these and parameter derivatives will be unavailable.
            If functions is not None, all lambdas must be specified.

        """
        self.consistent_exceptions = consistent_exceptions
        self.switch_width = switch_width
        self.alchemical_pme_treatment = alchemical_pme_treatment
        self.alchemical_rf_treatment = alchemical_rf_treatment
        self.disable_alchemical_dispersion_correction = disable_alchemical_dispersion_correction
        self.split_alchemical_forces = split_alchemical_forces
        self.alchemical_functions = alchemical_functions

    def create_alchemical_system(self, topology_proposal, old_positions, new_positions):
        """Create an alchemically modified version of the reference system.

        To alter the alchemical state of the returned system use AlchemicalState.

        Parameters
        ----------
        topology_proposal : perses.rjmc.topology_proposal.TopologyProposal object
            TopologyProposal object rendered by the ProposalEngine
        old_positions : [n,3] np.ndarray of float
            The positions of the "old system"
        new_positions : [m,3] np.ndarray of float
            The positions of the "new system"

        Returns
        -------
        alchemical_system : openmm.System
            The hybrid system for simulation
        alchemical_positions : [n, 3] np.ndarray
            The positions of the hybrid system
        alchemical_topology : openmm.app.Topology
            The OpenMM topology object corresponding to the hybrid system
        new_to_hybrid_atom_map : dict of int : int
            The mapping of new system atoms to hybrid atoms
        old_to_hybrid_atom_map : dict of int : int
            The mapping of old system atoms to hybrid atoms
        """
        self._topology_proposal = topology_proposal
        self._old_system = copy.deepcopy(topology_proposal.old_system)
        self._new_system = copy.deepcopy(topology_proposal.new_system)
        self._old_to_hybrid_map = {}
        self._new_to_hybrid_map = {}
        self._hybrid_system_forces = dict()
        self._old_positions = old_positions
        self._new_positions = new_positions

        self.softcore_alpha=0.5
        self.softcore_beta=12*unit.angstrom**2 # TODO: Make this dimensionless

        if functions:
            self._functions = functions
            self._has_functions = True
        else:
            self._has_functions = False

        #prepare dicts of forces, which will be useful later
        self._old_system_forces = {type(force).__name__ : force for force in self._old_system.getForces()}
        self._new_system_forces = {type(force).__name__ : force for force in self._new_system.getForces()}

        #check that there are no unknown forces in the new and old systems:
        for force_name in self._old_system_forces.keys():
            if force_name not in self._known_forces:
                raise ValueError("Unkown force %s encountered in old system" % force_name)

        for force_name in self._new_system_forces.keys():
            if force_name not in self._known_forces:
                raise ValueError("Unkown force %s encountered in new system" % force_name)

        #get and store the nonbonded method from the system:
        self._nonbonded_method = self._old_system_forces['NonbondedForce'].getNonbondedMethod()

        #start by creating an empty system. This will become the hybrid system.
        self._hybrid_system = openmm.System()

        #begin by copying all particles in the old system to the hybrid system. Note that this does not copy the
        #interactions. It does, however, copy the particle masses. In general, hybrid index and old index should be
        #the same.
        for particle_idx in range(self._topology_proposal.n_atoms_old):
            particle_mass = self._old_system.getParticleMass(particle_idx)
            hybrid_idx = self._hybrid_system.addParticle(particle_mass)
            self._old_to_hybrid_map[particle_idx] = hybrid_idx

            #If the particle index in question is mapped, make sure to add it to the new to hybrid map as well.
            if particle_idx in self._topology_proposal.old_to_new_atom_map.keys():
                particle_index_in_new_system = self._topology_proposal.old_to_new_atom_map[particle_idx]
                self._new_to_hybrid_map[particle_index_in_new_system] = hybrid_idx

        #Next, add the remaining unique atoms from the new system to the hybrid system and map accordingly.
        #As before, this does not copy interactions, only particle indices and masses.
        for particle_idx in self._topology_proposal.unique_new_atoms:
            particle_mass = self._new_system.getParticleMass(particle_idx)
            hybrid_idx = self._hybrid_system.addParticle(particle_mass)
            self._new_to_hybrid_map[particle_idx] = hybrid_idx

        #check that if there is a barostat in the original system, it is added to the hybrid.
        #We copy the barostat from the old system.
        if "MonteCarloBarostat" in self._old_system_forces.keys():
            barostat = copy.deepcopy(self._old_system_forces["MonteCarloBarostat"])
            self._hybrid_system.addForce(barostat)

        #initialize unitless softcore beta
        self.softcore_beta = self.softcore_beta / self.softcore_beta.in_unit_system(unit.md_unit_system).unit

        #Copy over the box vectors:
        box_vectors = self._old_system.getDefaultPeriodicBoxVectors()
        self._hybrid_system.setDefaultPeriodicBoxVectors(*box_vectors)

        #assign atoms to one of the classes described in the class docstring
        self._atom_classes = self._determine_atom_classes()

        #verify that no constraints are changing over the course of the switching.

        #create the opposite atom maps for use in nonbonded force processing
        self._hybrid_to_old_map = {value : key for key, value in self._old_to_hybrid_map.items()}
        self._hybrid_to_new_map = {value : key for key, value in self._new_to_hybrid_map.items()}

        #verify that no constraints are changing over the course of the switching.
        self._constraint_check_fast()

        #construct dictionary of exceptions in old and new systems
        self._old_system_exceptions = self._generate_dict_from_exceptions(self._old_system_forces['NonbondedForce'])
        self._new_system_exceptions = self._generate_dict_from_exceptions(self._new_system_forces['NonbondedForce'])

        #copy over relevant virtual sites
        self._handle_virtual_sites()

        #call each of the methods to add the corresponding force terms and prepare the forces:
        self._add_bond_force_terms()
        self._add_angle_force_terms()
        self._add_torsion_force_terms()
        self._add_nonbonded_force_terms()
        self._handle_constraints()

        #call each force preparation method to generate the actual interactions that we need:
        self.handle_harmonic_bonds()
        self.handle_harmonic_angles()
        self.handle_periodic_torsion_force()
        self.handle_nonbonded()

        #get positions for the hybrid
        self._hybrid_positions = self._compute_hybrid_positions()

        #generate the topology representation
        self._hybrid_topology = self._create_topology()

        return self._hybrid_system, self._hybrid_positions, self._hybrid_topology, self._new_to_old_atom_map, self._old_to_new_atom_map

    def _force_sanity_check(self, force_name_list):
        """
        Make sure that there are no unknown forces in the system--these will not be handled by the hybrid topology
        engine.

        Parameters
        ----------
        force_name_list : list of str
            list of the force names for
        Returns
        -------
        unknown_forces_present : bool
            Whether unknown forces are present in the system
        """
        force_name_set = set(force_name_list)
        if len(force_name_set - self._known_forces) > 0:
            return True
        else:
            return False

    def _handle_virtual_sites(self):
        """
        Ensure that all virtual sites in old and new system are copied over to the hybrid system. Note that we do not
        support virtual sites in the changing region.
        """
        old_system = self._topology_proposal.old_system
        new_system = self._topology_proposal.new_system

        #first loop through the old system, looking for virtual sites
        for particle_idx in range(old_system.getNumParticles()):
            hybrid_idx = self._old_to_hybrid_map[particle_idx]

            #If it's a virtual site, make sure it is not in the unique or core atoms (unsupported).
            if old_system.isVirtualSite(particle_idx):
                if hybrid_idx not in self._atom_classes['environment']:
                    raise Exception("Virtual sites in changing residue are unsupported.")
                else:
                    virtual_site = old_system.getVirtualSite(particle_idx)
                    self._hybrid_system.setVirtualSite(hybrid_idx, virtual_site)

        #Since all supported virtual sites are in the environment, which are by definition common to both the new and
        #old systems, we only need to check that there are no virtual sites not in environment:
        for particle_idx in range(new_system.getNumParticles()):
            hybrid_idx = self._new_to_hybrid_map[particle_idx]

            if new_system.isVirtualSite(particle_idx):
                if hybrid_idx not in self._atom_classes['environment']:
                    raise Exception("Virtual sites in changing residue are unsupported.")

    def _get_core_atoms(self):
        """
        Determine which atoms in the old system are part of the "core" class. All necessary information is contained in
        the topology proposal passed to the constructor.

        Returns
        -------
        core_atoms : set of int
            The set of atoms (hybrid topology indexed) that are core atoms.
        environment_atoms : set of int
            The set of atoms (hybrid topology indexed) that are environment atoms.
        """

        #In order to be either a core or environment atom, the atom must be mapped.
        mapped_old_atoms_set = set(self._topology_proposal.old_to_new_atom_map.keys())
        mapped_new_atoms_set = set(self._topology_proposal.old_to_new_atom_map.values())
        mapped_hybrid_atoms_set = {self._old_to_hybrid_map[atom_idx] for atom_idx in mapped_old_atoms_set}

        #create sets for set arithmetic
        unique_old_set = set(self._topology_proposal.unique_old_atoms)
        unique_new_set = set(self._topology_proposal.unique_new_atoms)

        #we derive core atoms from the old topology:
        core_atoms_from_old = self._determine_core_atoms_in_topology(self._topology_proposal.old_topology,
                                                                     unique_old_set, mapped_old_atoms_set,
                                                                     self._old_to_hybrid_map)

        #we also derive core atoms from the new topology:
        core_atoms_from_new = self._determine_core_atoms_in_topology(self._topology_proposal.new_topology,
                                                                     unique_new_set, mapped_new_atoms_set,
                                                                     self._new_to_hybrid_map)

        #The union of the two will give the core atoms that can result from either new or old topology
        total_core_atoms = core_atoms_from_old.union(core_atoms_from_new)

        #as a side effect, we can now compute the environment atom indices too, by subtracting the core indices
        #from the mapped atom set (since any atom that is mapped but not core is environment)
        environment_atoms = mapped_hybrid_atoms_set.difference(total_core_atoms)

        return total_core_atoms, environment_atoms

    def _determine_core_atoms_in_topology(self, topology, unique_atoms, mapped_atoms, hybrid_map):
        """
        Given a topology and its corresponding unique and mapped atoms, return the set of atom indices in the
        hybrid system which would belong to the "core" atom class

        Parameters
        ----------
        topology : simtk.openmm.app.Topology
            An OpenMM topology representing a system of interest
        unique_atoms : set of int
            A set of atoms that are unique to this topology
        mapped_atoms : set of int
            A set of atoms that are mapped to another topology

        Returns
        -------
        core_atoms : set of int
            set of core atom indices in hybrid topology
        """
        core_atoms = set()

        #loop through the residues to look for ones with unique atoms
        for residue in topology.residues():
            atom_indices_old_system = {atom.index for atom in residue.atoms()}

            #if the residue contains an atom index that is unique, then the residue is changing.
            #We determine this by checking if the atom indices of the residue have any intersection with the unique atoms
            if len(atom_indices_old_system.intersection(unique_atoms)) > 0:
                #we can add the atoms in this residue which are mapped to the core_atoms set:
                for atom_index in atom_indices_old_system:
                    if atom_index in mapped_atoms:
                        #we specifically want to add the hybrid atom.
                        hybrid_index = hybrid_map[atom_index]
                        core_atoms.add(hybrid_index)

        return core_atoms

    def _determine_atom_classes(self):
        """
        This method determines whether each atom belongs to unique old, unique new, core, or environment, as defined above.
        All the information required is contained in the TopologyProposal passed to the constructor. All indices are
        indices in the hybrid system.

        Returns
        -------
        atom_classes : dict of list
            A dictionary of the form {'core' :core_list} etc.
        """
        atom_classes = {'unique_old_atoms' : set(), 'unique_new_atoms' : set(), 'core_atoms' : set(), 'environment_atoms' : set()}

        #first, find the unique old atoms, as this is the most straightforward:
        for atom_idx in self._topology_proposal.unique_old_atoms:
            hybrid_idx = self._old_to_hybrid_map[atom_idx]
            atom_classes['unique_old_atoms'].add(hybrid_idx)

        #Then the unique new atoms (this is substantially the same as above)
        for atom_idx in self._topology_proposal.unique_new_atoms:
            hybrid_idx = self._new_to_hybrid_map[atom_idx]
            atom_classes['unique_new_atoms'].add(hybrid_idx)

        core_atoms, environment_atoms = self._get_core_atoms()

        atom_classes['core_atoms'] = core_atoms
        atom_classes['environment_atoms'] = environment_atoms

        return atom_classes

    def _translate_nonbonded_method_to_custom(self, standard_nonbonded_method):
        """
        Utility function to translate the nonbonded method enum from the standard nonbonded force to the custom version
       `CutoffPeriodic`, `PME`, and `Ewald` all become `CutoffPeriodic`; `NoCutoff` becomes `NoCutoff`; `CutoffNonPeriodic` becomes `CutoffNonPeriodic`
        Parameters
        ----------
        standard_nonbonded_method : openmm.NonbondedForce.NonbondedMethod
            the nonbonded method of the standard force

        Returns
        -------
        custom_nonbonded_method : openmm.CustomNonbondedForce.NonbondedMethod
            the nonbonded method for the equivalent customnonbonded force
        """
        if standard_nonbonded_method in [openmm.NonbondedForce.CutoffPeriodic, openmm.NonbondedForce.PME, openmm.NonbondedForce.Ewald]:
            return openmm.CustomNonbondedForce.CutoffPeriodic
        elif standard_nonbonded_method == openmm.NonbondedForce.NoCutoff:
            return openmm.CustomNonbondedForce.NoCutoff
        elif standard_nonbonded_method == openmm.NonbondedForce.CutoffNonPeriodic:
            return openmm.CustomNonbondedForce.CutoffNonPeriodic
        else:
            raise NotImplementedError("This nonbonded method is not supported.")

    def _handle_constraints(self):
        """
        This method adds relevant constraints from the old and new systems. First, all constraints from the old system
        are added. Then, constraints to atoms unique to the new system are added.
        """
        #we add all constraints from the old system first.
        for constraint_idx in range(self._topology_proposal.old_system.getNumConstraints()):
            atom1, atom2, constraint = self._topology_proposal.old_system.getConstraintParameters(constraint_idx)
            atom1_hybrid = self._old_to_hybrid_map[atom1]
            atom2_hybrid = self._old_to_hybrid_map[atom2]
            self._hybrid_system.addConstraint(atom1_hybrid, atom2_hybrid, constraint)

        #Now we loop through constraints in the new system, but only add constraints involving new system atoms
        #since anything common to both was already added. Note that we do not have to worry about changing constraint
        #lengths because we already checked that that doesn't happen.
        for constraint_idx in range(self._topology_proposal.new_system.getNumConstraints()):
            atom1, atom2, constraint = self._topology_proposal.new_system.getConstraintParameters(constraint_idx)
            atom1_hybrid = self._new_to_hybrid_map[atom1]
            atom2_hybrid = self._new_to_hybrid_map[atom2]

            atom_set = {atom1_hybrid, atom2_hybrid}

            #If there's a nonempty intersection with unique new atoms, this constraint wasn't added.
            if len(atom_set.intersection(self._atom_classes['unique_new_atoms'])):
                self._hybrid_system.addConstraint(atom1_hybrid, atom2_hybrid, constraint)

    def _constraint_check(self):
        """
        This is a check to make sure that constraint lengths do not change over the course of the switching.
        In the future, we will determine a method to deal with this. Raises exception if a constraint length changes.
        """

        #this dict will be of the form {(atom1, atom2) : constraint_value}, with hybrid indices.
        constrained_atoms_dict = {}

        #first, loop through constraints in the old system and add them to the dict, with hybrid indices:
        for constraint_idx in range(self._topology_proposal.old_system.getNumConstraints()):
            atom1, atom2, constraint = self._topology_proposal.old_system.getConstraintParameters(constraint_idx)
            atom1_hybrid = self._old_to_hybrid_map[atom1]
            atom2_hybrid = self._old_to_hybrid_map[atom2]
            constrained_atoms_dict[(atom1_hybrid, atom2_hybrid)] = constraint

        #now, loop through constraints in the new system, and see if we are going to change a constraint length
        for constraint_idx in range(self._topology_proposal.new_system.getNumConstraints()):
            atom1, atom2, constraint = self._topology_proposal.new_system.getConstraintParameters(constraint_idx)
            atom1_hybrid = self._new_to_hybrid_map[atom1]
            atom2_hybrid = self._new_to_hybrid_map[atom2]

            #check if either permutation is in the keys
            if (atom1_hybrid, atom2_hybrid) in constrained_atoms_dict.keys():
                constraint_from_old_system = constrained_atoms_dict[(atom1_hybrid, atom2_hybrid)]
                if constraint != constraint_from_old_system:
                    raise ValueError("Constraints are changing during switching.")

            if (atom2_hybrid, atom1_hybrid) in constrained_atoms_dict.keys():
                constraint_from_old_system = constrained_atoms_dict[(atom2_hybrid, atom1_hybrid)]
                if constraint != constraint_from_old_system:
                    raise ValueError("Constraints are changing during switching.")

    def _constraint_check_fast(self):
        """
        This method will check for changing constraints by first serializing the new and old systems to xml, then using
        that xml to check for constraint changes. Using lxml and XPATH, this should be considerably faster than the
        OpenMM API. If a constraint is found to be changing, an exception will be raised, as this cannot currently be
        handled by the HybridTopologyFactory.
        """
        #set up an xpath string to find constraints
        constraint_string = '/System/Constraints/Constraint'

        #get a reference to maps with shorter names
        o_h_map = self._old_to_hybrid_map
        n_h_map = self._new_to_hybrid_map

        #serialize the systems
        old_system_xml = openmm.XmlSerializer.serialize(self._topology_proposal.old_system)
        new_system_xml = openmm.XmlSerializer.serialize(self._topology_proposal.new_system)

        #get the serialized systems into stringio form
        old_system_io = StringIO(old_system_xml)
        new_system_io = StringIO(new_system_xml)

        #parse the xml strings
        old_system_tree = etree.parse(old_system_io)
        new_system_tree = etree.parse(new_system_io)

        #get the list of constraints from new and old systems:
        old_system_constraint_list = old_system_tree.xpath(constraint_string)
        new_system_constraint_list = new_system_tree.xpath(constraint_string)

        #convert the list of constraint elements to dictionaries. By using frozenset, we can do this independent of the order of
        old_system_constraints = {frozenset((o_h_map[int(constraint.attrib['p1'])], o_h_map[int(constraint.attrib['p2'])])) : float(constraint.attrib['d']) for constraint in old_system_constraint_list}
        new_system_constraints = {frozenset((n_h_map[int(constraint.attrib['p1'])], n_h_map[int(constraint.attrib['p2'])])) : float(constraint.attrib['d']) for constraint in new_system_constraint_list}

        #find the set of constraints that are common to both:
        old_constraint_sets = set(old_system_constraints.keys())
        new_constraint_sets = set(new_system_constraints.keys())
        overlapping_constraints = old_constraint_sets.intersection(new_constraint_sets)

        #check that the constraints match in both cases:
        for constraint_pair in overlapping_constraints:
            if old_system_constraints[constraint_pair] != new_system_constraints[constraint_pair]:
                raise ValueError("There is a changing constraint length in this system.")

    def _determine_interaction_group(self, atoms_in_interaction):
        """
        This method determines which interaction group the interaction should fall under. There are four groups:

        Those involving unique old atoms: any interaction involving unique old atoms should be completely on at lambda=0
            and completely off at lambda=1

        Those involving unique new atoms: any interaction involving unique new atoms should be completely off at lambda=0
            and completely on at lambda=1

        Those involving core atoms and/or environment atoms: These interactions change their type, and should be the old
            character at lambda=0, and the new character at lambda=1

        Those involving only environment atoms: These interactions are unmodified.

        Parameters
        ----------
        atoms_in_interaction : list of int
            List of (hybrid) indices of the atoms in this interaction

        Returns
        -------
        interaction_group : InteractionGroup enum
            The group to which this interaction should be assigned
        """
        #make the interaction list a set to facilitate operations
        atom_interaction_set = set(atoms_in_interaction)

        #check if the interaction contains unique old atoms
        if len(atom_interaction_set.intersection(self._atom_classes['unique_old_atoms'])) > 0:
            return InteractionGroup.unique_old

        #Do the same for new atoms
        elif len(atom_interaction_set.intersection(self._atom_classes['unique_new_atoms'])) > 0:
            return InteractionGroup.unique_new

        #if the interaction set is a strict subset of the environment atoms, then it is in the environment group
        #and should not be alchemically modified at all.
        elif atom_interaction_set.issubset(self._atom_classes['environment_atoms']):
            return InteractionGroup.environment

        #having covered the cases of all-environment, unique old-containing, and unique-new-containing, anything else
        #should belong to the last class--contains core atoms but not any unique atoms.
        else:
            return InteractionGroup.core

    def _add_bond_force_terms(self):
        """
        This function adds the appropriate bond forces to the system (according to groups defined above). Note that it
        does _not_ add the particles to the force. It only adds the force to facilitate another method adding the
        particles to the force.
        """
        core_energy_expression = '(K/2)*(r-length)^2;'
        core_energy_expression += 'K = (1-lambda_bonds)*K1 + lambda_bonds*K2;' # linearly interpolate spring constant
        core_energy_expression += 'length = (1-lambda_bonds)*length1 + lambda_bonds*length2;' # linearly interpolate bond length
        if self._has_functions:
            try:
                core_energy_expression += 'lambda_bonds = ' + self._functions['lambda_bonds']
            except KeyError as e:
                print("Functions were provided, but no term was provided for the bonds")
                raise e

        #create the force and add the relevant parameters
        custom_core_force = openmm.CustomBondForce(core_energy_expression)
        custom_core_force.addPerBondParameter('length1') # old bond length
        custom_core_force.addPerBondParameter('K1') # old spring constant
        custom_core_force.addPerBondParameter('length2') # new bond length
        custom_core_force.addPerBondParameter('K2') #new spring constant

        if self._has_functions:
            custom_core_force.addGlobalParameter('lambda', 0.0)
            custom_core_force.addEnergyParameterDerivative('lambda')
        else:
            custom_core_force.addGlobalParameter('lambda_bonds', 0.0)

        self._hybrid_system.addForce(custom_core_force)
        self._hybrid_system_forces['core_bond_force'] = custom_core_force

        #add a bond force for environment and unique atoms (bonds are never scaled for these):
        standard_bond_force = openmm.HarmonicBondForce()
        self._hybrid_system.addForce(standard_bond_force)
        self._hybrid_system_forces['standard_bond_force'] = standard_bond_force

    def _add_angle_force_terms(self):
        """
        This function adds the appropriate angle force terms to the hybrid system. It does not add particles
        or parameters to the force; this is done elsewhere.
        """
        energy_expression  = '(K/2)*(theta-theta0)^2;'
        energy_expression += 'K = (1.0-lambda_angles)*K_1 + lambda_angles*K_2;' # linearly interpolate spring constant
        energy_expression += 'theta0 = (1.0-lambda_angles)*theta0_1 + lambda_angles*theta0_2;' # linearly interpolate equilibrium angle
        if self._has_functions:
            try:
                energy_expression += 'lambda_angles = ' + self._functions['lambda_angles']
            except KeyError as e:
                print("Functions were provided, but no term was provided for the angles")
                raise e

        #create the force and add relevant parameters
        custom_core_force = openmm.CustomAngleForce(energy_expression)
        custom_core_force.addPerAngleParameter('theta0_1') # molecule1 equilibrium angle
        custom_core_force.addPerAngleParameter('K_1') # molecule1 spring constant
        custom_core_force.addPerAngleParameter('theta0_2') # molecule2 equilibrium angle
        custom_core_force.addPerAngleParameter('K_2') # molecule2 spring constant

        if self._has_functions:
            custom_core_force.addGlobalParameter('lambda', 0.0)
            custom_core_force.addEnergyParameterDerivative('lambda')
        else:
            custom_core_force.addGlobalParameter('lambda_angles', 0.0)


        #add the force to the system and the force dict.
        self._hybrid_system.addForce(custom_core_force)
        self._hybrid_system_forces['core_angle_force'] = custom_core_force

        #add an angle term for environment/unique interactions--these are never scaled
        standard_angle_force = openmm.HarmonicAngleForce()
        self._hybrid_system.addForce(standard_angle_force)
        self._hybrid_system_forces['standard_angle_force'] = standard_angle_force

    def _add_torsion_force_terms(self):
        """
        This function adds the appropriate PeriodicTorsionForce terms to the system. Core torsions are interpolated,
        while environment and unique torsions are always on.
        """
        energy_expression  = '(1-lambda_torsions)*U1 + lambda_torsions*U2;'
        energy_expression += 'U1 = K1*(1+cos(periodicity1*theta-phase1));'
        energy_expression += 'U2 = K2*(1+cos(periodicity2*theta-phase2));'

        if self._has_functions:
            try:
                energy_expression += 'lambda_torsions = ' + self._functions['lambda_torsions']
            except KeyError as e:
                print("Functions were provided, but no term was provided for torsions")
                raise e


        #create the force and add the relevant parameters
        custom_core_force = openmm.CustomTorsionForce(energy_expression)
        custom_core_force.addPerTorsionParameter('periodicity1') # molecule1 periodicity
        custom_core_force.addPerTorsionParameter('phase1') # molecule1 phase
        custom_core_force.addPerTorsionParameter('K1') # molecule1 spring constant
        custom_core_force.addPerTorsionParameter('periodicity2') # molecule2 periodicity
        custom_core_force.addPerTorsionParameter('phase2') # molecule2 phase
        custom_core_force.addPerTorsionParameter('K2') # molecule2 spring constant

        if self._has_functions:
            custom_core_force.addGlobalParameter('lambda', 0.0)
            custom_core_force.addEnergyParameterDerivative('lambda')
        else:
            custom_core_force.addGlobalParameter('lambda_torsions', 0.0)

        #add the force to the system
        self._hybrid_system.addForce(custom_core_force)
        self._hybrid_system_forces['core_torsion_force'] = custom_core_force

        #create and add the torsion term for unique/environment atoms
        standard_torsion_force = openmm.PeriodicTorsionForce()
        self._hybrid_system.addForce(standard_torsion_force)
        self._hybrid_system_forces['standard_torsion_force'] = standard_torsion_force

    def _add_nonbonded_force_terms(self):
        """
        Add the nonbonded force terms to the hybrid system. Note that as with the other forces,
        this method does not add any interactions. It only sets up the forces.

        Parameters
        ----------
        nonbonded_method : int
            One of the openmm.NonbondedForce nonbonded methods.
        """

        #Add a regular nonbonded force for all interactions that are not changing.
        standard_nonbonded_force = openmm.NonbondedForce()
        self._hybrid_system.addForce(standard_nonbonded_force)
        self._hybrid_system_forces['standard_nonbonded_force'] = standard_nonbonded_force

        # Create a CustomNonbondedForce to handle alchemically interpolated nonbonded parameters.
        # Select functional form based on nonbonded method.
        if self._nonbonded_method in [openmm.NonbondedForce.NoCutoff]:
            sterics_energy_expression, electrostatics_energy_expression = self._nonbonded_custom_nocutoff()
        elif self._nonbonded_method in [openmm.NonbondedForce.CutoffPeriodic, openmm.NonbondedForce.CutoffNonPeriodic]:
            epsilon_solvent = self._old_system_forces['NonbondedForce'].getReactionFieldDielectric()
            r_cutoff = self._old_system_forces['NonbondedForce'].getCutoffDistance()
            sterics_energy_expression, electrostatics_energy_expression = self._nonbonded_custom_cutoff(epsilon_solvent, r_cutoff)
            standard_nonbonded_force.setReactionFieldDielectric(epsilon_solvent)
            standard_nonbonded_force.setCutoffDistance(r_cutoff)
        elif self._nonbonded_method in [openmm.NonbondedForce.PME, openmm.NonbondedForce.Ewald]:
            [alpha_ewald, nx, ny, nz] = self._old_system_forces['NonbondedForce'].getPMEParameters()
            delta = self._old_system_forces['NonbondedForce'].getEwaldErrorTolerance()
            r_cutoff = self._old_system_forces['NonbondedForce'].getCutoffDistance()
            sterics_energy_expression, electrostatics_energy_expression = self._nonbonded_custom_ewald(alpha_ewald, delta, r_cutoff)
            standard_nonbonded_force.setPMEParameters(alpha_ewald, nx, ny, nz)
            standard_nonbonded_force.setEwaldErrorTolerance(delta)
            standard_nonbonded_force.setCutoffDistance(r_cutoff)
        else:
            raise Exception("Nonbonded method %s not supported yet." % str(self._nonbonded_method))

        standard_nonbonded_force.setNonbondedMethod(self._nonbonded_method)
        sterics_energy_expression += self._nonbonded_custom_sterics_common()
        electrostatics_energy_expression += self._nonbonded_custom_electrostatics_common()

        sterics_mixing_rules, electrostatics_mixing_rules = self._nonbonded_custom_mixing_rules()

        custom_nonbonded_method = self._translate_nonbonded_method_to_custom(self._nonbonded_method)

        # Create CustomNonbondedForce to handle interactions between alchemically-modified atoms and rest of system.
        total_electrostatics_energy = "U_electrostatics;" + electrostatics_energy_expression + electrostatics_mixing_rules
        if self._has_functions:
            try:
                total_electrostatics_energy += 'lambda_electrostatics = ' + self._functions['lambda_electrostatics']
            except KeyError as e:
                print("Functions were provided, but there is no entry for electrostatics")
                raise e

        electrostatics_custom_nonbonded_force = openmm.CustomNonbondedForce(total_electrostatics_energy)
        electrostatics_custom_nonbonded_force.addGlobalParameter("softcore_beta", self.softcore_beta)
        electrostatics_custom_nonbonded_force.addPerParticleParameter("chargeA") # partial charge initial
        electrostatics_custom_nonbonded_force.addPerParticleParameter("chargeB") # partial charge final

        if self._has_functions:
            electrostatics_custom_nonbonded_force.addGlobalParameter("lambda", 0.0)
            electrostatics_custom_nonbonded_force.addEnergyParameterDerivative('lambda')
        else:
            electrostatics_custom_nonbonded_force.addGlobalParameter("lambda_electrostatics", 0.0)


        electrostatics_custom_nonbonded_force.setNonbondedMethod(custom_nonbonded_method)

        self._hybrid_system.addForce(electrostatics_custom_nonbonded_force)
        self._hybrid_system_forces['core_electrostatics_force'] = electrostatics_custom_nonbonded_force

        total_sterics_energy = "U_sterics;" + sterics_energy_expression + sterics_mixing_rules
        if self._has_functions:
            try:
                total_sterics_energy += 'lambda_sterics  = ' + self._functions['lambda_sterics']
            except KeyError as e:
                print("Functions were provided, but there is no entry for sterics")
                raise e

        sterics_custom_nonbonded_force = openmm.CustomNonbondedForce(total_sterics_energy)
        sterics_custom_nonbonded_force.addGlobalParameter("softcore_alpha", self.softcore_alpha)
        sterics_custom_nonbonded_force.addPerParticleParameter("sigmaA") # Lennard-Jones sigma initial
        sterics_custom_nonbonded_force.addPerParticleParameter("epsilonA") # Lennard-Jones epsilon initial
        sterics_custom_nonbonded_force.addPerParticleParameter("sigmaB") # Lennard-Jones sigma final
        sterics_custom_nonbonded_force.addPerParticleParameter("epsilonB") # Lennard-Jones epsilon final

        if self._has_functions:
            sterics_custom_nonbonded_force.addGlobalParameter('lambda', 0.0)
            sterics_custom_nonbonded_force.addEnergyParameterDerivative('lambda')
        else:
            sterics_custom_nonbonded_force.addGlobalParameter("lambda_sterics", 0.0)


        sterics_custom_nonbonded_force.setNonbondedMethod(custom_nonbonded_method)


        self._hybrid_system.addForce(sterics_custom_nonbonded_force)
        self._hybrid_system_forces['core_sterics_force'] = sterics_custom_nonbonded_force


        #set the use of dispersion correction to be the same between the new nonbonded force and the old one:
        if self._old_system_forces['NonbondedForce'].getUseDispersionCorrection():
            self._hybrid_system_forces['standard_nonbonded_force'].setUseDispersionCorrection(True)
            if not self.disable_alchemical_dispersion_correction:
                sterics_custom_nonbonded_force.setUseLongRangeCorrection(True)

        if self._old_system_forces['NonbondedForce'].getUseSwitchingFunction():
            switching_distance = self._old_system_forces['NonbondedForce'].getSwitchingDistance()
            standard_nonbonded_force.setUseSwitchingFunction(True)
            standard_nonbonded_force.setSwitchingDistance(switching_distance)
            sterics_custom_nonbonded_force.setUseSwitchingFunction(True)
            sterics_custom_nonbonded_force.setSwitchingDistance(switching_distance)
            electrostatics_custom_nonbonded_force.setUseSwitchingFunction(True)
            electrostatics_custom_nonbonded_force.setSwitchingDistance(switching_distance)
        else:
            standard_nonbonded_force.setUseSwitchingFunction(False)
            electrostatics_custom_nonbonded_force.setUseSwitchingFunction(False)
            sterics_custom_nonbonded_force.setUseSwitchingFunction(False)

        #Add a CustomBondForce for exceptions:
        custom_nonbonded_bond_force = self._nonbonded_custom_bond_force(sterics_energy_expression, electrostatics_energy_expression)
        self._hybrid_system.addForce(custom_nonbonded_bond_force)
        self._hybrid_system_forces['core_nonbonded_bond_force'] = custom_nonbonded_bond_force

    def _nonbonded_custom_sterics_common(self):
        """
        Get a custom sterics expression that is common to all nonbonded methods

        Returns
        -------
        sterics_addition : str
            The common softcore sterics energy expression
        """
        sterics_addition = "epsilon = (1-lambda_sterics)*epsilonA + lambda_sterics*epsilonB;" #interpolation
        sterics_addition += "reff_sterics = sigma*((softcore_alpha*lambda_alpha + (r/sigma)^6))^(1/6);" # effective softcore distance for sterics
        sterics_addition += "sigma = (1-lambda_sterics)*sigmaA + lambda_sterics*sigmaB;"
        sterics_addition += "lambda_alpha = lambda_sterics*(1-lambda_sterics);"
        return sterics_addition

    def _nonbonded_custom_electrostatics_common(self):
        """
        Get a custom electrostatics expression that is common to all nonbonded methods

        Returns
        -------
        electrostatics_addition : str
            The common electrostatics energy expression
        """
        electrostatics_addition = "chargeprod = (1-lambda_electrostatics)*chargeprodA + lambda_electrostatics*chargeprodB;" #interpolation
        electrostatics_addition += "reff_electrostatics = sqrt(softcore_beta*lambda_beta + r^2);" # effective softcore distance for electrostatics
        electrostatics_addition += "ONE_4PI_EPS0 = %f;" % ONE_4PI_EPS0 # already in OpenMM units
        electrostatics_addition += "lambda_beta = lambda_electrostatics*(1-lambda_electrostatics);"
        return electrostatics_addition

    def _nonbonded_custom_nocutoff(self):
        """
        Get a part of the nonbonded energy expression when there is no cutoff.

        Returns
        -------
        sterics_energy_expression : str
            The energy expression for U_sterics
        electrostatics_energy_expression : str
            The energy expression for electrostatics
        """
        # soft-core Lennard-Jones
        sterics_energy_expression = "U_sterics = 4*epsilon*x*(x-1.0); x = (sigma/reff_sterics)^6;"
        # soft-core Coulomb
        electrostatics_energy_expression = "U_electrostatics = ONE_4PI_EPS0*chargeprod/reff_electrostatics;"
        return sterics_energy_expression, electrostatics_energy_expression

    def _nonbonded_custom_cutoff(self, epsilon_solvent, r_cutoff):
        """
        Get the energy expressions for sterics and electrostatics under a reaction field assumption.

        Parameters
        ----------
        epsilon_solvent : float
            The reaction field dielectric
        r_cutoff : float
            The cutoff distance

        Returns
        -------
        sterics_energy_expression : str
            The energy expression for U_sterics
        electrostatics_energy_expression : str
            The energy expression for electrostatics
        """
        # soft-core Lennard-Jones
        sterics_energy_expression = "U_sterics = 4*epsilon*x*(x-1.0); x = (sigma/reff_sterics)^6;"

        electrostatics_energy_expression = "U_electrostatics = ONE_4PI_EPS0*chargeprod*(reff_electrostatics^(-1) + k_rf*reff_electrostatics^2 - c_rf);"
        k_rf = r_cutoff**(-3) * ((epsilon_solvent - 1) / (2*epsilon_solvent + 1))
        c_rf = r_cutoff**(-1) * ((3*epsilon_solvent) / (2*epsilon_solvent + 1))
        electrostatics_energy_expression += "k_rf = %f;" % (k_rf / k_rf.in_unit_system(unit.md_unit_system).unit)
        electrostatics_energy_expression += "c_rf = 0;"
        return sterics_energy_expression, electrostatics_energy_expression

    def _nonbonded_custom_ewald(self, alpha_ewald, delta, r_cutoff):
        """
        Get the energy expression for Ewald treatment.

        Parameters
        ----------
        alpha_ewald : float
            The Ewald alpha parameter
        delta : float
            The PME error tolerance
        r_cutoff : float
            The cutoff distance

        Returns
        -------
        sterics_energy_expression : str
            The energy expression for U_sterics
        electrostatics_energy_expression : str
            The energy expression for electrostatics
        """
        # soft-core Lennard-Jones
        sterics_energy_expression = "U_sterics = 4*epsilon*x*(x-1.0); x = (sigma/reff_sterics)^6;"
        if unit.is_quantity(alpha_ewald):
            alpha_ewald = alpha_ewald / alpha_ewald.in_unit_system(unit.md_unit_system).unit
        if alpha_ewald == 0.0:
            # If alpha is 0.0, alpha_ewald is computed by OpenMM from from the error tolerance.
            alpha_ewald = np.sqrt(-np.log(2*delta)) / r_cutoff
            alpha_ewald = alpha_ewald / alpha_ewald.in_unit_system(unit.md_unit_system).unit
        electrostatics_energy_expression = "U_electrostatics = ONE_4PI_EPS0*chargeprod*erfc(alpha_ewald*reff_electrostatics)/reff_electrostatics;"
        electrostatics_energy_expression += "alpha_ewald = %f;" % alpha_ewald
        return sterics_energy_expression, electrostatics_energy_expression

    def _nonbonded_custom_mixing_rules(self):
        """
        Mixing rules for the custom nonbonded force.

        Returns
        -------
        sterics_mixing_rules : str
            The mixing expression for sterics
        electrostatics_mixing_rules : str
            The mixiing rules for electrostatics
        """
        # Define mixing rules.
        sterics_mixing_rules = "epsilonA = sqrt(epsilonA1*epsilonA2);" # mixing rule for epsilon
        sterics_mixing_rules += "epsilonB = sqrt(epsilonB1*epsilonB2);" # mixing rule for epsilon
        sterics_mixing_rules += "sigmaA = 0.5*(sigmaA1 + sigmaA2);" # mixing rule for sigma
        sterics_mixing_rules += "sigmaB = 0.5*(sigmaB1 + sigmaB2);" # mixing rule for sigma
        electrostatics_mixing_rules = "chargeprodA = chargeA1*chargeA2;" # mixing rule for charges
        electrostatics_mixing_rules += "chargeprodB = chargeB1*chargeB2;" # mixing rule for charges
        return sterics_mixing_rules, electrostatics_mixing_rules

    def _nonbonded_custom_bond_force(self, sterics_energy_expression, electrostatics_energy_expression):
        """
        Add a CustomBondForce to represent the exceptions in the NonbondedForce

        Parameters
        ----------
        sterics_energy_expression : str
            The complete energy expression being used for sterics
        electrostatics_energy_expression : str
            The complete energy expression being used for electrostatics

        Returns
        -------
        custom_bond_force : openmm.CustomBondForce
            The custom bond force for the nonbonded exceptions
        """
        #Create the force and add its relevant parameters.
        #we don't need to check that the keys exist, since by the time this is called, these are already checked.
        if self._has_functions:
            sterics_energy_expression += 'lambda_sterics = ' + self._functions['lambda_sterics']
            electrostatics_energy_expression += 'lambda_electrostatics = ' + self._functions['lambda_electrostatics']
        custom_bond_force = openmm.CustomBondForce("U_sterics + U_electrostatics;" + sterics_energy_expression + electrostatics_energy_expression)
        custom_bond_force.addGlobalParameter("lambda_electrostatics", 0.0)
        custom_bond_force.addGlobalParameter("lambda_sterics", 0.0)
        custom_bond_force.addGlobalParameter("softcore_alpha", self.softcore_alpha)
        custom_bond_force.addGlobalParameter("softcore_beta", self.softcore_beta)
        custom_bond_force.addPerBondParameter("chargeprodA")
        custom_bond_force.addPerBondParameter("sigmaA")
        custom_bond_force.addPerBondParameter("epsilonA")
        custom_bond_force.addPerBondParameter("chargeprodB")
        custom_bond_force.addPerBondParameter("sigmaB")
        custom_bond_force.addPerBondParameter("epsilonB")

        if self._has_functions:
            custom_bond_force.addGlobalParameter('lambda', 0.0)
            custom_bond_force.addEnergyParameterDerivative('lambda')
        else:
            custom_bond_force.addGlobalParameter("lambda_electrostatics", 0.0)
            custom_bond_force.addGlobalParameter("lambda_sterics", 0.0)

        return custom_bond_force

    def _find_bond_parameters(self, bond_force, index1, index2):
        """
        This is a convenience function to find bond parameters in another system given the two indices.

        Parameters
        ----------
        bond_force : openmm.HarmonicBondForce
            The bond force where the parameters should be found
        index1 : int
           Index1 (order does not matter) of the bond atoms
        index2 : int
           Index2 (order does not matter) of the bond atoms

        Returns
        -------
        bond_parameters : list
            List of relevant bond parameters
        """
        index_set = {index1, index2}
        #loop through all the bonds:
        for bond_index in range(bond_force.getNumBonds()):
            [index1_term, index2_term, r0, k] = bond_force.getBondParameters(bond_index)
            if index_set=={index1_term, index2_term}:
                return [index1_term, index2_term, r0, k]

        raise ValueError("The requested bond was not found.")

    def handle_harmonic_bonds(self):
        """
        This method adds the appropriate interaction for all bonds in the hybrid system. The scheme used is:

        1) If the two atoms are both in the core, then we add to the CustomBondForce and interpolate between the two
            parameters
        2) Otherwise, we add the bond to a regular bond force.
        """
        old_system_bond_force = self._old_system_forces['HarmonicBondForce']
        new_system_bond_force = self._new_system_forces['HarmonicBondForce']

        #first, loop through the old system bond forces and add relevant terms
        for bond_index in range(old_system_bond_force.getNumBonds()):
            #get each set of bond parameters
            [index1_old, index2_old, r0_old, k_old] = old_system_bond_force.getBondParameters(bond_index)

            #map the indices to the hybrid system, for which our atom classes are defined.
            index1_hybrid = self._old_to_hybrid_map[index1_old]
            index2_hybrid = self._old_to_hybrid_map[index2_old]
            index_set = {index1_hybrid, index2_hybrid}

            #now check if it is a subset of the core atoms (that is, both atoms are in the core)
            #if it is, we need to find the parameters in the old system so that we can interpolate
            if index_set.issubset(self._atom_classes['core_atoms']):
                index1_new = self._topology_proposal.old_to_new_atom_map[index1_old]
                index2_new = self._topology_proposal.old_to_new_atom_map[index2_old]
                [index1, index2, r0_new, k_new] = self._find_bond_parameters(new_system_bond_force, index1_new, index2_new)
                self._hybrid_system_forces['core_bond_force'].addBond(index1_hybrid, index2_hybrid,[r0_old, k_old, r0_new, k_new])

            #otherwise, we just add the same parameters as those in the old system.
            else:
                self._hybrid_system_forces['standard_bond_force'].addBond(index1_hybrid, index2_hybrid, r0_old, k_old)

        #now loop through the new system to get the interactions that are unique to it.
        for bond_index in range(new_system_bond_force.getNumBonds()):
            #get each set of bond parameters
            [index1_new, index2_new, r0_new, k_new] = new_system_bond_force.getBondParameters(bond_index)

            #convert indices to hybrid, since that is how we represent atom classes:
            index1_hybrid = self._new_to_hybrid_map[index1_new]
            index2_hybrid = self._new_to_hybrid_map[index2_new]
            index_set = {index1_hybrid, index2_hybrid}

            #if the intersection of this set and unique new atoms contains anything, the bond is unique to the new system and must be added
            #all other bonds in the new system have been accounted for already.
            if len(index_set.intersection(self._atom_classes['unique_new_atoms'])) > 0:
                self._hybrid_system_forces['standard_bond_force'].addBond(index1_hybrid, index2_hybrid, r0_new, k_new)

    def _find_angle_parameters(self, angle_force, indices):
        """
        Convenience function to find the angle parameters corresponding to a particular set of indices

        Parameters
        ----------
        angle_force : openmm.HarmonicAngleForce
            The force where the angle of interest may be found.
        indices : list of int
            The indices (any order) of the angle atoms
        Returns
        -------
        angle_parameters : list
            list of angle parameters
        """
        index_set = set(indices)

        #now loop through and try to find the angle:
        for angle_index in range(angle_force.getNumAngles()):
            angle_parameters = angle_force.getAngleParameters(angle_index)

            #get a set representing the angle indices
            angle_parameter_indices = set(angle_parameters[:3])

            if index_set==angle_parameter_indices:
                return angle_parameters

        raise ValueError("The provided force does not contain the angle of interest.")

    def _find_torsion_parameters(self, torsion_force, indices):
        """
        Convenience function to find the torsion parameters corresponding to a particular set of indices.

        Parameters
        ----------
        torsion_force : openmm.PeriodicTorsionForce
            torsion force where the torsion of interest may be found
        indices : list of int
            The indices (any order) of the atoms of the torsion

        Returns
        -------
        torsion_parameters : list
            torsion parameters
        """
        index_set = set(indices)

        torsion_parameters_list = list()

        #now loop through and try to find the torsion:
        for torsion_index in range(torsion_force.getNumTorsions()):
            torsion_parameters = torsion_force.getTorsionParameters(torsion_index)

            #get a set representing the torsion indices:
            torsion_parameter_indices = set(torsion_parameters[:4])

            if index_set==torsion_parameter_indices:
                torsion_parameters_list.append(torsion_parameters)

        if len(torsion_parameters_list)==0:
            raise ValueError("No torsion found matching the indices specified")

        return torsion_parameters_list

    def handle_harmonic_angles(self):
        """
        This method adds the appropriate interaction for all angles in the hybrid system. The scheme used, as with bonds, is:

        1) If the three atoms are all in the core, then we add to the CustomAngleForce and interpolate between the two
            parameters
        2) Otherwise, we add the angle to a regular angle force.
        """
        old_system_angle_force = self._old_system_forces['HarmonicAngleForce']
        new_system_angle_force = self._new_system_forces['HarmonicAngleForce']

        #first, loop through all the angles in the old system to determine what to do with them. We will only use the
        #custom angle force if all atoms are part of "core." Otherwise, they are either unique to one system or never
        #change.
        for angle_index in range(old_system_angle_force.getNumAngles()):
            angle_parameters = old_system_angle_force.getAngleParameters(angle_index)

            #get the indices in the hybrid system
            hybrid_index_list = [self._old_to_hybrid_map[old_index] for old_index in angle_parameters[:3]]
            hybrid_index_set = set(hybrid_index_list)

            #if all atoms are in the core, we'll need to find the corresponding parameters in the old system and
            #interpolate
            if hybrid_index_set.issubset(self._atom_classes['core_atoms']):
                #get the new indices so we can get the new angle parameters
                new_indices = [self._topology_proposal.old_to_new_atom_map[old_index] for old_index in angle_parameters[:3]]
                new_angle_parameters = self._find_angle_parameters(new_system_angle_force, new_indices)

                #add to the hybrid force:
                #the parameters at indices 3 and 4 represent theta0 and k, respectively.
                hybrid_force_parameters = [angle_parameters[3], angle_parameters[4], new_angle_parameters[3], new_angle_parameters[4]]
                self._hybrid_system_forces['core_angle_force'].addAngle(hybrid_index_list[0], hybrid_index_list[1], hybrid_index_list[2], hybrid_force_parameters)

            #otherwise, just add the parameters to the regular force:
            else:
                self._hybrid_system_forces['standard_angle_force'].addAngle(hybrid_index_list[0], hybrid_index_list[1],
                                                                            hybrid_index_list[2], angle_parameters[3],
                                                                            angle_parameters[4])

        #finally, loop through the new system force to add any unique new angles
        for angle_index in range(new_system_angle_force.getNumAngles()):
            angle_parameters = new_system_angle_force.getAngleParameters(angle_index)

            #get the indices in the hybrid system
            hybrid_index_list = [self._new_to_hybrid_map[new_index] for new_index in angle_parameters[:3]]
            hybrid_index_set = set(hybrid_index_list)

            #if the intersection of this hybrid set with the unique new atoms is nonempty, it must be added:
            if len(hybrid_index_set.intersection(self._atom_classes['unique_new_atoms'])) > 0:
                self._hybrid_system_forces['standard_angle_force'].addAngle(hybrid_index_list[0], hybrid_index_list[1],
                                                                            hybrid_index_list[2], angle_parameters[3],
                                                                            angle_parameters[4])

    def handle_periodic_torsion_force(self):
        """
        Handle the torsions in the hybrid system in the same way as the angles and bonds.
        """
        old_system_torsion_force = self._old_system_forces['PeriodicTorsionForce']
        new_system_torsion_force = self._new_system_forces['PeriodicTorsionForce']

        #first, loop through all the torsions in the old system to determine what to do with them. We will only use the
        #custom torsion force if all atoms are part of "core." Otherwise, they are either unique to one system or never
        #change.

        #we need to keep track of what torsions we added so that we do not double count.
        added_torsions = []
        for torsion_index in range(old_system_torsion_force.getNumTorsions()):
            torsion_parameters = old_system_torsion_force.getTorsionParameters(torsion_index)

            #get the indices in the hybrid system
            hybrid_index_list = [self._old_to_hybrid_map[old_index] for old_index in torsion_parameters[:4]]
            hybrid_index_set = set(hybrid_index_list)

            #if all atoms are in the core, we'll need to find the corresponding parameters in the old system and
            #interpolate
            if hybrid_index_set.issubset(self._atom_classes['core_atoms']):
                torsion_indices = torsion_parameters[:4]

                #if we've already added these indices (they may appear >once for high periodicities)
                #then just continue to the next torsion.
                if torsion_indices in added_torsions:
                    continue
                #get the new indices so we can get the new angle parameters, as well as all old parameters of the old torsion
                #The reason we do it like this is to take care of varying periodicity between new and old system.
                torsion_parameters_list = self._find_torsion_parameters(old_system_torsion_force, torsion_indices)
                new_indices = [self._topology_proposal.old_to_new_atom_map[old_index] for old_index in torsion_indices]
                new_torsion_parameters_list = self._find_torsion_parameters(new_system_torsion_force, new_indices)

                #for old torsions, have the energy scale from full at lambda=0 to off at lambda=1
                for torsion_parameters in torsion_parameters_list:
                    hybrid_force_parameters = [torsion_parameters[4], torsion_parameters[5], torsion_parameters[6], 0.0, 0.0, 0.0]
                    self._hybrid_system_forces['core_torsion_force'].addTorsion(hybrid_index_list[0], hybrid_index_list[1], hybrid_index_list[2], hybrid_index_list[3], hybrid_force_parameters)

                #for new torsions, have the energy scale from 0 at lambda=0 to full at lambda=1
                for torsion_parameters in new_torsion_parameters_list:
                    #add to the hybrid force:
                    #the parameters at indices 3 and 4 represent theta0 and k, respectively.
                    hybrid_force_parameters = [0.0, 0.0, 0.0,torsion_parameters[4], torsion_parameters[5], torsion_parameters[6]]
                    self._hybrid_system_forces['core_torsion_force'].addTorsion(hybrid_index_list[0], hybrid_index_list[1], hybrid_index_list[2], hybrid_index_list[3], hybrid_force_parameters)

                added_torsions.append(torsion_indices)

            #otherwise, just add the parameters to the regular force:
            else:
                self._hybrid_system_forces['standard_torsion_force'].addTorsion(hybrid_index_list[0], hybrid_index_list[1],
                                                                            hybrid_index_list[2], hybrid_index_list[3], torsion_parameters[4],
                                                                            torsion_parameters[5], torsion_parameters[6])

        for torsion_index in range(new_system_torsion_force.getNumTorsions()):
            torsion_parameters = new_system_torsion_force.getTorsionParameters(torsion_index)

            #get the indices in the hybrid system:
            hybrid_index_list = [self._new_to_hybrid_map[new_index] for new_index in torsion_parameters[:4]]
            hybrid_index_set = set(hybrid_index_list)

            #if any are part of the unique new atoms, we will add them to the standard torsion force:
            if len(hybrid_index_set.intersection(self._atom_classes['unique_new_atoms'])) > 0:
                self._hybrid_system_forces['standard_torsion_force'].addTorsion(hybrid_index_list[0], hybrid_index_list[1],
                                                                            hybrid_index_list[2], hybrid_index_list[3], torsion_parameters[4],
                                                                            torsion_parameters[5], torsion_parameters[6])

    def handle_nonbonded(self):
        """

        """
        old_system_nonbonded_force = self._old_system_forces['NonbondedForce']
        new_system_nonbonded_force = self._new_system_forces['NonbondedForce']
        hybrid_to_old_map = self._hybrid_to_old_map
        hybrid_to_new_map = self._hybrid_to_new_map


        #We have to loop through the particles in the system, because nonbonded force does not accept index
        for particle_index in range(self._hybrid_system.getNumParticles()):

            if particle_index in self._atom_classes['unique_old_atoms']:
                #get the parameters in the old system
                old_index = hybrid_to_old_map[particle_index]
                [charge, sigma, epsilon] = old_system_nonbonded_force.getParticleParameters(old_index)

                #add the particle to the hybrid custom sterics and electrostatics.
                self._hybrid_system_forces['core_sterics_force'].addParticle([sigma, epsilon, 1.0, 0.0])
                self._hybrid_system_forces['core_electrostatics_force'].addParticle([charge, 0.0])

                #Add the particle to the regular nonbonded force as required, but zero out interaction
                #it will be handled by an exception
                self._hybrid_system_forces['standard_nonbonded_force'].addParticle(0.0, 1.0, 0.0)

            elif particle_index in self._atom_classes['unique_new_atoms']:
                #get the parameters in the new system
                new_index = hybrid_to_new_map[particle_index]
                [charge, sigma, epsilon] = new_system_nonbonded_force.getParticleParameters(new_index)

                #add the particle to the hybrid custom sterics and electrostatics
                self._hybrid_system_forces['core_sterics_force'].addParticle([1.0, 0.0, sigma, epsilon])
                self._hybrid_system_forces['core_electrostatics_force'].addParticle([0.0, charge])

                #Add the particle to the regular nonbonded force as required, but zero out interaction
                #it will be handled by an exception
                self._hybrid_system_forces['standard_nonbonded_force'].addParticle(0.0, 1.0, 0.0)


            elif particle_index in self._atom_classes['core_atoms']:
                #get the parameters in the new and old systems:
                old_index = hybrid_to_old_map[particle_index]
                [charge_old, sigma_old, epsilon_old] = old_system_nonbonded_force.getParticleParameters(old_index)
                new_index = hybrid_to_new_map[particle_index]
                [charge_new, sigma_new, epsilon_new] = new_system_nonbonded_force.getParticleParameters(new_index)

                #add the particle to the custom forces, interpolating between the two parameters
                self._hybrid_system_forces['core_sterics_force'].addParticle([sigma_old, epsilon_old, sigma_new, epsilon_new])
                self._hybrid_system_forces['core_electrostatics_force'].addParticle([charge_old, charge_new])

                #still add the particle to the regular nonbonded force, but with zeroed out parameters.
                self._hybrid_system_forces['standard_nonbonded_force'].addParticle(0.0, 1.0, 0.0)

            #otherwise, the particle is in the environment
            else:
                #the parameters will be the same in new and old system, so just take the old parameters
                old_index = hybrid_to_old_map[particle_index]
                [charge, sigma, epsilon] = old_system_nonbonded_force.getParticleParameters(old_index)

                #add the particle to the hybrid custom sterics and electrostatics, but they dont change
                self._hybrid_system_forces['core_sterics_force'].addParticle([sigma, epsilon, sigma, epsilon])
                self._hybrid_system_forces['core_electrostatics_force'].addParticle([charge, charge])

                #add the environment atoms to the regular nonbonded force as well:
                self._hybrid_system_forces['standard_nonbonded_force'].addParticle(charge, sigma, epsilon)

        self._handle_interaction_groups()
        self._handle_hybrid_exceptions()
        self._handle_original_exceptions()

    def _generate_dict_from_exceptions(self, force):
        """
        This is a utility function to generate a dictionary of the form
        (particle1_idx, particle2_idx) : [exception parameters]. This will facilitate access and search of exceptions

        Parameters
        ----------
        force : openmm.NonbondedForce object
            a force containing exceptions

        Returns
        -------
        exceptions_dict : dict
            Dictionary of exceptions
        """
        exceptions_dict = {}

        for exception_index in range(force.getNumExceptions()):
            [index1, index2, chargeProd, sigma, epsilon] = force.getExceptionParameters(exception_index)
            exceptions_dict[(index1, index2)] = [chargeProd, sigma, epsilon]

        return exceptions_dict

    def _handle_interaction_groups(self):
        """
        Create the appropriate interaction groups for the custom nonbonded forces. The groups are:

        1) Unique-old - core
        2) Unique-old - environment
        3) Unique-new - core
        4) Unique-new - environment
        5) Core - environment
        6) Core - core

        Unique-old and Unique new are prevented from interacting this way, and intra-unique interactions occur in an
        unmodified nonbonded force.

        Must be called after particles are added to the Nonbonded forces
        """
        #get the force objects for convenience:
        electrostatics_custom_force = self._hybrid_system_forces['core_electrostatics_force']
        sterics_custom_force = self._hybrid_system_forces['core_sterics_force']

        #also prepare the atom classes
        core_atoms = self._atom_classes['core_atoms']
        unique_old_atoms = self._atom_classes['unique_old_atoms']
        unique_new_atoms = self._atom_classes['unique_new_atoms']
        environment_atoms = self._atom_classes['environment_atoms']


        electrostatics_custom_force.addInteractionGroup(unique_old_atoms, core_atoms)
        sterics_custom_force.addInteractionGroup(unique_old_atoms, core_atoms)

        electrostatics_custom_force.addInteractionGroup(unique_old_atoms, environment_atoms)
        sterics_custom_force.addInteractionGroup(unique_old_atoms, environment_atoms)

        electrostatics_custom_force.addInteractionGroup(unique_new_atoms, core_atoms)
        sterics_custom_force.addInteractionGroup(unique_new_atoms, core_atoms)

        electrostatics_custom_force.addInteractionGroup(unique_new_atoms, environment_atoms)
        sterics_custom_force.addInteractionGroup(unique_new_atoms, environment_atoms)

        electrostatics_custom_force.addInteractionGroup(core_atoms, environment_atoms)
        sterics_custom_force.addInteractionGroup(core_atoms, environment_atoms)

        electrostatics_custom_force.addInteractionGroup(core_atoms, core_atoms)
        sterics_custom_force.addInteractionGroup(core_atoms, core_atoms)

    def _handle_hybrid_exceptions(self):
        """
        Instead of excluding interactions that shouldn't occur, we provide exceptions for interactions that were zeroed
        out but should occur.

        Returns
        -------

        """
        print("handling exceptions")

        old_system_nonbonded_force = self._old_system_forces['NonbondedForce']
        new_system_nonbonded_force = self._new_system_forces['NonbondedForce']

        import itertools
        #prepare the atom classes
        unique_old_atoms = self._atom_classes['unique_old_atoms']
        unique_new_atoms = self._atom_classes['unique_new_atoms']

        nonbonded_force = self._hybrid_system_forces['standard_nonbonded_force']

        #get the list of interaction pairs for which we need to set exceptions:
        unique_old_pairs = list(itertools.combinations(unique_old_atoms, 2))
        unique_new_pairs = list(itertools.combinations(unique_new_atoms, 2))

        #add back the interactions of the old unique atoms, unless there are exceptions
        for atom_pair in unique_old_pairs:
            #since the pairs are indexed in the dictionary by the old system indices, we need to convert
            old_index_atom_pair = (self._hybrid_to_old_map[atom_pair[0]], self._hybrid_to_old_map[atom_pair[1]])

            #now we check if the pair is in the exception dictionary
            if old_index_atom_pair in self._old_system_exceptions:
                [chargeProd, sigma, epsilon] = self._old_system_exceptions[old_index_atom_pair]
                nonbonded_force.addException(atom_pair[0], atom_pair[1], chargeProd, sigma, epsilon)

            #check if the pair is in the reverse order and use that if so
            elif old_index_atom_pair[::-1] in self._old_system_exceptions:
                [chargeProd, sigma, epsilon] = self._old_system_exceptions[old_index_atom_pair[::-1]]
                nonbonded_force.addException(atom_pair[0], atom_pair[1], chargeProd, sigma, epsilon)

            #If it's not handled by an exception in the original system, we just add the regular parameters as an exception
            else:
                [charge0, sigma0, epsilon0] = self._old_system_forces['NonbondedForce'].getParticleParameters(old_index_atom_pair[0])
                [charge1, sigma1, epsilon1] = self._old_system_forces['NonbondedForce'].getParticleParameters(old_index_atom_pair[1])
                chargeProd = charge0*charge1
                epsilon = unit.sqrt(epsilon0*epsilon1)
                sigma = 0.5*(sigma0+sigma1)
                nonbonded_force.addException(atom_pair[0], atom_pair[1], chargeProd, sigma, epsilon)

        #add back the interactions of the new unique atoms, unless there are exceptions
        for atom_pair in unique_new_pairs:
            #since the pairs are indexed in the dictionary by the new system indices, we need to convert
            new_index_atom_pair = (self._hybrid_to_new_map[atom_pair[0]], self._hybrid_to_new_map[atom_pair[1]])

            #now we check if the pair is in the exception dictionary
            if new_index_atom_pair in self._new_system_exceptions:
                [chargeProd, sigma, epsilon] = self._new_system_exceptions[new_index_atom_pair]
                nonbonded_force.addException(atom_pair[0], atom_pair[1], chargeProd, sigma, epsilon)

            #check if the pair is present in the reverse order and use that if so
            elif new_index_atom_pair[::-1] in self._new_system_exceptions:
                [chargeProd, sigma, epsilon] = self._new_system_exceptions[new_index_atom_pair[::-1]]
                nonbonded_force.addException(atom_pair[0], atom_pair[1], chargeProd, sigma, epsilon)

            #If it's not handled by an exception in the original system, we just add the regular parameters as an exception
            else:
                [charge0, sigma0, epsilon0] = self._new_system_forces['NonbondedForce'].getParticleParameters(new_index_atom_pair[0])
                [charge1, sigma1, epsilon1] = self._new_system_forces['NonbondedForce'].getParticleParameters(new_index_atom_pair[1])
                chargeProd = charge0*charge1
                epsilon = unit.sqrt(epsilon0*epsilon1)
                sigma = 0.5*(sigma0+sigma1)
                nonbonded_force.addException(atom_pair[0], atom_pair[1], chargeProd, sigma, epsilon)
        print("done handling exceptions")

    def _handle_original_exceptions(self):
        """
        This method ensures that exceptions present in the original systems are present in the hybrid appropriately.
        """
        #get what we need to find the exceptions from the new and old systems:
        old_system_nonbonded_force = self._old_system_forces['NonbondedForce']
        new_system_nonbonded_force = self._new_system_forces['NonbondedForce']
        hybrid_to_old_map = {value: key for key, value in self._old_to_hybrid_map.items()}
        hybrid_to_new_map = {value: key for key, value in self._new_to_hybrid_map.items()}

        #first, loop through the old system's exceptions and add them to the hybrid appropriately:
        for exception_pair, exception_parameters in self._old_system_exceptions.items():

            [index1_old, index2_old] = exception_pair

            [chargeProd_old, sigma_old, epsilon_old] = exception_parameters

            #get hybrid indices:
            index1_hybrid = self._old_to_hybrid_map[index1_old]
            index2_hybrid = self._old_to_hybrid_map[index2_old]
            index_set = {index1_hybrid, index2_hybrid}

            #in this case, the interaction is only covered by the regular nonbonded force, and as such will be copied to that force
            #in the unique-old case, it is handled elsewhere due to internal peculiarities regarding exceptions
            if index_set.issubset(self._atom_classes['environment_atoms']):
                self._hybrid_system_forces['standard_nonbonded_force'].addException(index1_hybrid, index2_hybrid, chargeProd_old, sigma_old, epsilon_old)

            #otherwise, check if one of the atoms in the set is in the unique_old_group:
            elif len(index_set.intersection(self._atom_classes['unique_old_atoms'])) > 0:
                #if it is, we should add it to the CustomBondForce for the nonbonded exceptions, and have it remain on
                #by having the two endpoints with the same parameters.
                #Currently, we keep sigma at the same value
                self._hybrid_system_forces['core_nonbonded_bond_force'].addBond(index1_hybrid, index2_hybrid,
                                                                                 [chargeProd_old, sigma_old,
                                                                                  epsilon_old, chargeProd_old,
                                                                                  sigma_old, epsilon_old])

                #We also need to exclude this interaction from the custom nonbonded forces, otherwise we'll be double counting
                self._hybrid_system_forces['core_sterics_force'].addExclusion(index1_hybrid, index2_hybrid)
                self._hybrid_system_forces['core_electrostatics_force'].addExclusion(index1_hybrid, index2_hybrid)

            #If the exception particles are neither solely old unique, solely environment, nor contain any unique old atoms, they are either core/environment or core/core
            #In this case, we need to get the parameters from the exception in the other (new) system, and interpolate between the two
            else:
                #first get the new indices.
                index1_new = hybrid_to_new_map[index1_hybrid]
                index2_new = hybrid_to_new_map[index2_hybrid]

                #get the exception parameters:
                [index1_new, index2_new, chargeProd_new, sigma_new, epsilon_new] = self._find_exception(
                    new_system_nonbonded_force, index1_new, index2_new)

                #Now add a term to the CustomBondForce to interpolate between the new and old systems:
                self._hybrid_system_forces['core_nonbonded_bond_force'].addBond(index1_hybrid, index2_hybrid,
                                                                                 [chargeProd_old, sigma_old,
                                                                                  epsilon_old, chargeProd_new,
                                                                                  sigma_new, epsilon_new])

                #We also need to exclude this interaction from the custom nonbonded forces, otherwise we'll be double counting
                self._hybrid_system_forces['core_sterics_force'].addExclusion(index1_hybrid, index2_hybrid)
                self._hybrid_system_forces['core_electrostatics_force'].addExclusion(index1_hybrid, index2_hybrid)

        #now, loop through the new system to collect remaining interactions. The only that remain here are
        #uniquenew-uniquenew, uniquenew-core, and uniquenew-environment.
        for exception_pair, exception_parameters in self._new_system_exceptions.items():
            [index1_new, index2_new] = exception_pair
            [chargeProd_new, sigma_new, epsilon_new] = exception_parameters

            #get hybrid indices:
            index1_hybrid = self._new_to_hybrid_map[index1_new]
            index2_hybrid = self._new_to_hybrid_map[index2_new]
            index_set = {index1_hybrid, index2_hybrid}

            #if it's a subset of unique_new_atoms, then this is an intra-unique interaction and should have its exceptions
            #specified in the regular nonbonded force. However, this is handled elsewhere as above due to pecularities with exception handling
            if index_set.issubset(self._atom_classes['unique_new_atoms']):
                continue

            #look for the final class- interactions between uniquenew-core and uniquenew-environment. They are treated
            #similarly: they are simply on and constant the entire time (as a valence term)
            elif len(index_set.intersection(self._atom_classes['unique_new_atoms'])) > 0:
                self._hybrid_system_forces['core_nonbonded_bond_force'].addBond(index1_hybrid, index2_hybrid,
                                                                                 [chargeProd_new, sigma_new,
                                                                                  epsilon_new, chargeProd_new,
                                                                                  sigma_new, epsilon_new])

                #We also need to exclude this interaction from the custom nonbonded forces, otherwise we'll be double counting
                self._hybrid_system_forces['core_sterics_force'].addExclusion(index1_hybrid, index2_hybrid)
                self._hybrid_system_forces['core_electrostatics_force'].addExclusion(index1_hybrid, index2_hybrid)

    def _find_exception(self, force, index1, index2):
        """
        Find the exception that corresponds to the given indices in the given system

        Parameters
        ----------
        force : openmm.NonbondedForce object
            System containing the exceptions
        index1 : int
            The index of the first atom (order is unimportant)
        index2 : int
            The index of the second atom (order is unimportant)

        Returns
        -------
        exception_parameters : list
            List of exception parameters
        """
        index_set = {index1, index2}

        #loop through the exceptions and try to find one matching the criteria
        for exception_idx in range(force.getNumExceptions()):
            exception_parameters = force.getExceptionParameters(exception_idx)
            if index_set==set(exception_parameters[:2]):
                return exception_parameters

        raise ValueError("The provided force does not have an exception between those particles.")

    def _compute_hybrid_positions(self):
        """
        The positions of the hybrid system. Dimensionality is (n_environment + n_core + n_old_unique + n_new_unique)
        The positions are assigned by first copying all the mapped positions from the old system in, then copying the
        mapped positions from the new system. This means that there is an assumption that the positions common to old
        and new are the same (which is the case for perses as-is).

        Returns
        -------
        hybrid_positions : np.ndarray [n, 3]
            Positions of the hybrid system, in nm
        """
        #get unitless positions
        old_positions_without_units = np.array(self._old_positions.value_in_unit(unit.nanometer))
        new_positions_without_units = np.array(self._new_positions.value_in_unit(unit.nanometer))

        #determine the number of particles in the system
        n_atoms_hybrid = self._hybrid_system.getNumParticles()

        #initialize an array for hybrid positions
        hybrid_positions_array = np.zeros([n_atoms_hybrid, 3])

        #loop through the old system indices, and assign positions.
        for old_index, hybrid_index in self._old_to_hybrid_map.items():
            hybrid_positions_array[hybrid_index, :] = old_positions_without_units[old_index, :]

        #Do the same for new indices. Note that this overwrites some coordinates, but as stated above, the assumption
        #is that these are the same.
        for new_index, hybrid_index in self._new_to_hybrid_map.items():
            hybrid_positions_array[hybrid_index, :] = new_positions_without_units[new_index, :]

        return unit.Quantity(hybrid_positions_array, unit=unit.nanometers)

    def _create_topology(self):
        """
        Create an mdtraj topology corresponding to the hybrid system.
        This is purely for writing out trajectories--it is not expected to be parameterized.

        Returns
        -------
        hybrid_topology : mdtraj.Topology
        """
        #first, make an md.Topology of the old system:
        old_topology = md.Topology.from_openmm(self._topology_proposal.old_topology)

        #now make a copy for the hybrid:
        hybrid_topology = copy.deepcopy(old_topology)

        #next, make a topology of the new system:
        new_topology = md.Topology.from_openmm(self._topology_proposal.new_topology)

        added_atoms = dict()

        #get the core atoms in the new index system (as opposed to the hybrid index system). We will need this later
        core_atoms_new_indices = {self._hybrid_to_new_map[core_atom] for core_atom in self._atom_classes['core_atoms']}

        #now, add each unique new atom to the topology (this is the same order as the system)
        for particle_idx in self._topology_proposal.unique_new_atoms:
            new_system_atom = new_topology.atom(particle_idx)

            #first, we get the residue in the new system associated with this atom
            new_system_residue = new_system_atom.residue

            #next, we have to enumerate the other atoms in that residue to find mapped atoms
            new_system_atom_set = {atom.index for atom in new_system_residue.atoms}

            #Now, we find the subset of atoms that are mapped. These must be in the "core" category, since they are mapped
            #and part of a changing residue
            mapped_new_atom_indices = core_atoms_new_indices.intersection(new_system_atom_set)

            #Now get the old indices of the above atoms so that we can find the appropriate residue in the old system
            #for this we can use the new to old atom map
            mapped_old_atom_indices = [self._topology_proposal.new_to_old_atom_map[atom_idx] for atom_idx in mapped_new_atom_indices]

            #we can just take the first one--they all have the same residue
            first_mapped_old_atom_index = mapped_old_atom_indices[0]

            #get the atom object corresponding to this index from the hybrid (which is a deepcopy of the old)
            mapped_hybrid_system_atom = hybrid_topology.atom(first_mapped_old_atom_index)

            #get the residue that is relevant to this atom
            mapped_residue = mapped_hybrid_system_atom.residue

            #add the atom using the mapped residue
            added_atoms[particle_idx] = hybrid_topology.add_atom(new_system_atom.name, new_system_atom.element, mapped_residue)

        #now loop through the bonds in the new system, and if the bond contains a unique new atom, then add it to the hybrid topology
        for (atom1, atom2) in new_topology.bonds:
            atom1_index_in_hybrid = self._new_to_hybrid_map[atom1.index]
            atom2_index_in_hybrid = self._new_to_hybrid_map[atom2.index]

            #if at least one atom is in the unique new class, we need to add it to the hybrid system
            if atom1_index_in_hybrid in self._atom_classes['unique_new_atoms'] or atom2_index_in_hybrid in self._atom_classes['unique_new_atoms']:
                if atom1.index in self._atom_classes['unique_new_atoms']:
                    atom1_to_bond = added_atoms[atom1.index]
                else:
                    atom1_to_bond = atom1

                if atom2.index in self._atom_classes['unique_new_atoms']:
                    atom2_to_bond = added_atoms[atom2.index]
                else:
                    atom2_to_bond = atom2

                hybrid_topology.add_bond(atom1_to_bond, atom2_to_bond)

        return hybrid_topology

    def old_positions(self, hybrid_positions):
        """
        Get the positions corresponding to the old system

        Parameters
        ----------
        hybrid_positions : [n, 3] np.ndarray with unit
            The positions of the hybrid system

        Returns
        -------
        old_positions : [m, 3] np.ndarray with unit
            The positions of the old system
        """
        n_atoms_old = self._topology_proposal.n_atoms_old
        old_positions = unit.Quantity(np.zeros([n_atoms_old, 3]), unit=unit.nanometer)
        for idx in range(n_atoms_old):
            old_positions[idx, :] = hybrid_positions[idx, :]
        return old_positions

    def new_positions(self, hybrid_positions):
        """
        Get the positions corresponding to the new system.

        Parameters
        ----------
        hybrid_positions : [n, 3] np.ndarray with unit
            The positions of the hybrid system

        Returns
        -------
        new_positions : [m, 3] np.ndarray with unit
            The positions of the new system
        """
        n_atoms_new = self._topology_proposal.n_atoms_new
        new_positions = unit.Quantity(np.zeros([n_atoms_new, 3]), unit=unit.nanometer)
        for idx in range(n_atoms_new):
            new_positions[idx, :] = hybrid_positions[self._new_to_hybrid_map[idx], :]
        return new_positions

    @property
    def hybrid_system(self):
        """
        The hybrid system.

        Returns
        -------
        hybrid_system : openmm.System
            The system representing a hybrid between old and new topologies
        """
        return self._hybrid_system

    @property
    def new_to_hybrid_atom_map(self):
        """
        Give a dictionary that maps new system atoms to the hybrid system.

        Returns
        -------
        new_to_hybrid_atom_map : dict of {int, int}
            The mapping of atoms from the new system to the hybrid
        """
        return self._new_to_hybrid_map

    @property
    def old_to_hybrid_atom_map(self):
        """
        Give a dictionary that maps old system atoms to the hybrid system.

        Returns
        -------
        old_to_hybrid_atom_map : dict of {int, int}
            The mapping of atoms from the old system to the hybrid
        """
        return self._old_to_hybrid_map

    @property
    def hybrid_positions(self):
        """
        The positions of the hybrid system. Dimensionality is (n_environment + n_core + n_old_unique + n_new_unique)
        The positions are assigned by first copying all the mapped positions from the old system in, then copying the
        mapped positions from the new system.

        Returns
        -------
        hybrid_positions : [n, 3] Quantity nanometers
        """
        return self._hybrid_positions

    @property
    def hybrid_topology(self):
        """
        An MDTraj hybrid topology for the purpose of writing out trajectories. Note that we do not expect this to be
        able to be parameterized by the openmm forcefield class.

        Returns
        -------
        hybrid_topology : mdtraj.Topology
        """
        return self._hybrid_topology

    @property
    def omm_hybrid_topology(self):
        """
        An OpenMM format of the hybrid topology. Also cannot be used to parameterize system, only to write out trajectories.

        Returns
        -------
        hybrid_topology : simtk.openmm.app.Topology
        """
        return md.Topology.to_openmm(self._hybrid_topology)
