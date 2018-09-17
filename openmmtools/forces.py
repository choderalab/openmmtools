#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Custom OpenMM Forces classes and utilities.

"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import abc
import collections
import copy
import inspect
import logging
import math
import re

import scipy
import numpy as np
from simtk import openmm, unit

from openmmtools import utils
from openmmtools.constants import ONE_4PI_EPS0, STANDARD_STATE_VOLUME


logger = logging.getLogger(__name__)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

class MultipleForcesError(Exception):
    """Error raised when multiple forces of the same class are found."""
    pass


class NoForceFoundError(Exception):
    """Error raised when no forces matching the given criteria are found."""
    pass


def iterate_forces(system):
    """Iterate over and restore the Python interface of the forces in the system."""
    for force in system.getForces():
        utils.RestorableOpenMMObject.restore_interface(force)
        yield force
    # Yield empty generator if the system has no forces.
    return


def find_forces(system, force_type, only_one=False, include_subclasses=False):
    """Find all the ``Force`` object of a given type in an OpenMM system.

    Parameters
    ----------
    system : simtk.openmm.System
        The system to search.
    force_type : str, or type
        The class of the force to search, or a regular expression that
        is used to match its name. Note that ``re.match()`` is used in
        this case, not ``re.search()``. The ``iter_subclasses`` argument
        must be False when this is a string.
    only_one : bool
        If True, an exception is raised when multiple forces of the same
        type are found in the system, and only a single force is returned.
    include_subclasses : bool, optional
        If True, all forces inheriting from ``force_type`` are returned
        as well (default is False). This can't be enabled if `force_type``
        is not a class.

    Returns
    -------
    forces : OrderedDict or tuple
        If ``only_one`` is False, a dictionary force_index: force is returned
        with all the forces matching the criteria. Otherwise,, a single pair
        ``(force_idx, force)`` is returned.

    Raises
    ------
    NoForceFoundError
        If ``only_one`` is True and no forces matching the criteria are found.
    MultipleForcesError
        If ``only_one`` is True and multiple forces matching the criteria
        are found

    Examples
    --------
    The ``only_one`` flag can be used to retrieve a single force.

    >>> from openmmtools import testsystems
    >>> system = testsystems.TolueneVacuum().system
    >>> force_index, force = find_forces(system, openmm.NonbondedForce, only_one=True)
    >>> force.__class__.__name__
    'NonbondedForce'

    It is possible to search for force subclasses.

    >>> class MyHarmonicForce(utils.RestorableOpenMMObject, openmm.CustomBondForce):
    ...     pass
    >>> force_idx = system.addForce(openmm.CustomBondForce('0.0'))
    >>> force_idx = system.addForce(MyHarmonicForce('0.0'))
    >>> forces = find_forces(system, openmm.CustomBondForce, include_subclasses=True)
    >>> [(force_idx, force.__class__.__name__) for force_idx, force in forces.items()]
    [(5, 'CustomBondForce'), (6, 'MyHarmonicForce')]

    A regular expression can be used instead of a class.

    >>> forces = find_forces(system, 'HarmonicAngleForce')
    >>> [(force_idx, force.__class__.__name__) for force_idx, force in forces.items()]
    [(1, 'HarmonicAngleForce')]

    >>> forces = find_forces(system, '.*Harmonic.*')
    >>> [(force_idx, force.__class__.__name__) for force_idx, force in forces.items()]
    [(0, 'HarmonicBondForce'), (1, 'HarmonicAngleForce'), (6, 'MyHarmonicForce')]

    """
    # Handle force_type argument when it's not a class.
    re_pattern = None
    if not inspect.isclass(force_type):
        re_pattern = re.compile(force_type)

    # Find all forces matching the force_type.
    forces = {}
    for force_idx, force in enumerate(iterate_forces(system)):
        # Check force name.
        if re_pattern is not None:
            if re_pattern.match(force.__class__.__name__):
                forces[force_idx] = force
        # Check if the force class matches the requirements.
        elif type(force) is force_type or (include_subclasses and isinstance(force, force_type)):
            forces[force_idx] = force

    # Second pass to find all subclasses of the matching forces.
    if include_subclasses and re_pattern is not None:
        matched_force_classes = [force.__class__ for force in forces.values()]
        for force_idx, force in enumerate(iterate_forces(system)):
            if force_idx in forces:
                continue
            for matched_force_class in matched_force_classes:
                if isinstance(force, matched_force_class):
                    forces[force_idx] = force

    # Reorder forces by index.
    forces = collections.OrderedDict(sorted(forces.items()))

    # Handle only_one.
    if only_one is True:
        if len(forces) == 0:
            raise NoForceFoundError('No force of type {} could be found.'.format(force_type))
        if len(forces) > 1:
            raise MultipleForcesError('Found multiple forces of type {}'.format(force_type))
        return forces.popitem(last=False)

    return forces


def _compute_sphere_volume(radius):
    """Compute the volume of a square well restraint."""
    return 4.0 / 3 * np.pi * radius**3


def _compute_harmonic_volume(radius, spring_constant, beta):
    """Compute the volume of an harmonic potential from 0 to radius.

    Parameters
    ----------
    radius : simtk.unit.Quantity
        The upper limit on the distance (units of length).
    spring_constant : simtk.unit.Quantity
        The spring constant of the harmonic potential (units of
        energy/mole/length^2).
    beta : simtk.unit.Quantity
        Thermodynamic beta (units of mole/energy).

    Returns
    -------
    volume : simtk.unit.Quantity
        The volume of the harmonic potential (units of length^3).

    """
    # Turn everything to consistent dimension-less units.
    length_unit = unit.nanometers
    energy_unit = unit.kilojoules_per_mole
    radius /= length_unit
    beta *= energy_unit
    spring_constant /= energy_unit/length_unit**2

    bk = beta * spring_constant
    bk_2 = bk / 2
    bkr2_2 = bk_2 * radius**2
    volume = math.sqrt(math.pi/2) * math.erf(math.sqrt(bkr2_2)) / bk**(3.0/2)
    volume -= math.exp(-bkr2_2) * radius / bk
    return 4 * math.pi * volume * length_unit**3


def _compute_harmonic_radius(spring_constant, potential_energy):
    """Find the radius at which the harmonic potential is energy.

    Parameters
    ----------
    spring_constant : simtk.unit.Quantity
        The spring constant of the harmonic potential (units of
        energy/mole/length^2).
    potential_energy : simtk.unit.Quantity
        The energy of the harmonic restraint (units of energy/mole).

    Returns
    -------
    radius : simtk.unit.Quantity
        The radius at which the harmonic potential is energy.

    """
    length_unit = unit.nanometers
    spring_constant *= length_unit**2
    return math.sqrt(2 * potential_energy / spring_constant) * length_unit


# =============================================================================
# GENERIC CLASSES FOR RADIALLY SYMMETRIC RECEPTOR-LIGAND RESTRAINTS
# =============================================================================

class RadiallySymmetricRestraintForce(utils.RestorableOpenMMObject):
    """Base class for radially-symmetric restraint force.

    Provide facility functions to compute the standard state correction
    of a receptor-ligand restraint.

    To create a subclass, implement the properties :func:`restrained_atom_indices1`
    and :func:`restrained_atom_indices2` (with their setters) that return
    the indices of the restrained atoms.

    You will also have to implement :func:`_create_bond`, which should add
    the bond using the correct function/signature.

    Optionally, you can implement :func:`distance_at_energy` if an
    analytical expression for distance(potential_energy) exists.

    If you subclass this, and plan on adding additional global parameters,
    you need to invoke this class ``super().__init__`` first as the
    ``controlling_parameter_name`` must be the first global variable.

    Parameters
    ----------
    restraint_parameters : OrderedDict
        An ordered dictionary containing the bond parameters in the form
        parameter_name: parameter_value. The order is important to make
        sure that parameters can be retrieved from the bond force with
        the correct force index.
    restrained_atom_indices1 : iterable of int
        The indices of the first group of atoms to restrain.
    restrained_atom_indices2 : iterable of int
        The indices of the second group of atoms to restrain.
    controlling_parameter_name : str
        The name of the global parameter controlling the energy function.
    *args, **kwargs
        Parameters to pass to the super constructor.

    Attributes
    ----------
    controlling_parameter_name

    """

    def __init__(self, restraint_parameters, restrained_atom_indices1,
                 restrained_atom_indices2, controlling_parameter_name,
                 *args, **kwargs):
        super(RadiallySymmetricRestraintForce, self).__init__(*args, **kwargs)

        # Unzip bond parameters names and values from dict.
        assert len(restraint_parameters) == 1 or isinstance(restraint_parameters, collections.OrderedDict)
        parameter_names, parameter_values = zip(*restraint_parameters.items())

        # Let the subclass initialize its bond.
        self._create_bond(parameter_values, restrained_atom_indices1, restrained_atom_indices2)

        # Add parameters. First global parameter is _restorable_force__class_hash
        # from the RestorableOpenMMObject class.
        err_msg = ('The force should have a single global parameter at this point. '
                   'This is likely because the subclass called addGlobalParameter '
                   'before calling super().__init__')
        assert self.getNumGlobalParameters() == 1, err_msg
        self.addGlobalParameter(controlling_parameter_name, 1.0)
        for parameter in parameter_names:
            self.addPerBondParameter(parameter)

    # -------------------------------------------------------------------------
    # Abstract methods.
    # -------------------------------------------------------------------------

    @abc.abstractmethod
    def _create_bond(self, bond_parameter_values, restrained_atom_indices1,
                     restrained_atom_indices2):
        """Create the bond modelling the restraint.

        Parameters
        ----------
        bond_parameter_values : list of floats
            The list of the parameter values of the bond.
        restrained_atom_indices1 : list of int
            The indices of the first group of atoms to restrain.
        restrained_atom_indices2 : list of int
            The indices of the second group of atoms to restrain.

        """
        pass

    # -------------------------------------------------------------------------
    # Properties.
    # -------------------------------------------------------------------------

    @abc.abstractproperty
    def restrained_atom_indices1(self):
        """list: The indices of the first group of restrained atoms."""
        pass

    @abc.abstractproperty
    def restrained_atom_indices2(self):
        """list: The indices of the first group of restrained atoms."""
        pass

    @property
    def restraint_parameters(self):
        """OrderedDict: The restraint parameters in dictionary form."""
        parameter_values = self.getBondParameters(0)[-1]
        restraint_parameters = [(self.getPerBondParameterName(parameter_idx), parameter_value)
                                for parameter_idx, parameter_value in enumerate(parameter_values)]
        return collections.OrderedDict(restraint_parameters)

    @property
    def controlling_parameter_name(self):
        """str: The name of the global parameter controlling the energy function (read-only)."""
        return self.getGlobalParameterName(1)

    def distance_at_energy(self, potential_energy):
        """Compute the distance at which the potential energy is ``potential_energy``.

        Parameters
        ----------
        potential_energy : simtk.unit.Quantity
            The potential energy of the restraint (units of energy/mole).

        Returns
        -------
        distance : simtk.unit.Quantity
            The distance at which the potential energy is ``potential_energy``
            (units of length).

        """
        raise NotImplementedError()

    # -------------------------------------------------------------------------
    # Methods to compute the standard state correction.
    # -------------------------------------------------------------------------

    def compute_standard_state_correction(self, thermodynamic_state, square_well=False,
                                          radius_cutoff=None, energy_cutoff=None,
                                          max_volume=None):
        """Return the standard state correction of the restraint.

        The standard state correction is computed as

            - log(V_standard / V_restraint)

        where V_standard is the volume at standard state concentration and
        V_restraint is the restraint volume. V_restraint is bounded by the
        volume of the periodic box.

        The ``square_well`` parameter, can be used to re-compute the standard
        state correction when removing the bias introduced by the restraint.

        Parameters
        ----------
        thermodynamic_state : states.ThermodynamicState
            The thermodynamic state at which to compute the standard state
            correction.
        square_well : bool, optional
            If True, this computes the standard state correction assuming
            the restraint to obey a square well potential. The energy
            cutoff is still applied to the original energy potential.
        radius_cutoff : simtk.unit.Quantity, optional
            The maximum distance achievable by the restraint (units
            compatible with nanometers). This is equivalent to placing
            a hard wall potential at this distance.
        energy_cutoff : float, optional
            The maximum potential energy achievable by the restraint in kT.
            This is equivalent to placing a hard wall potential at a
            distance such that ``potential_energy(distance) == energy_cutoff``.
        max_volume : simtk.unit.Quantity or 'system', optional
            The volume of the periodic box (units compatible with nanometer**3).
            This must be provided the thermodynamic state is in NPT. If the
            string 'system' is passed, the maximum volume is computed from
            the system box vectors (this has no effect if the system is not
            periodic).

        Returns
        -------
        correction : float
           The unit-less standard state correction in kT at the given
           thermodynamic state.

        Raises
        ------
        TypeError
            If the thermodynamic state is in the NPT ensemble, and
            ``max_volume`` is not provided, or if the system is non-periodic
            and no cutoff is given.

        """
        # Determine restraint bound volume.
        is_npt = thermodynamic_state.pressure is not None
        if max_volume == 'system':
            # ThermodynamicState.volume is None in the NPT ensemble.
            # max_volume will still be None if the system is not periodic.
            max_volume = thermodynamic_state.get_volume(ignore_ensemble=True)
        elif max_volume is None and not is_npt:
            max_volume = thermodynamic_state.volume
        elif max_volume is None:
            raise TypeError('max_volume must be provided with NPT ensemble')

        # Non periodic systems reweighted to a square-well restraint must always have a cutoff.
        if (not thermodynamic_state.is_periodic and square_well is True and
                    radius_cutoff is None and energy_cutoff is None and max_volume is None):
            raise TypeError('One between radius_cutoff, energy_cutoff, or max_volume '
                            'must be provided when reweighting non-periodic thermodynamic '
                            'states to a square-well restraint.')

        # If we evaluate the square well potential with no cutoffs,
        # just use the volume of the periodic box.
        if square_well is True and energy_cutoff is None and radius_cutoff is None:
            restraint_volume = max_volume
        # If we evaluate the square well potential with no energy cutoff,
        # this can easily be solved analytically.
        elif square_well is True and radius_cutoff is not None:
            restraint_volume = _compute_sphere_volume(radius_cutoff)
        # Use numerical integration.
        else:
            restraint_volume = self._compute_restraint_volume(
                thermodynamic_state, square_well, radius_cutoff, energy_cutoff)

        # Bound the restraint volume to the periodic box volume.
        if max_volume is not None and restraint_volume > max_volume:
            debug_msg = 'Limiting the restraint volume to {} nm^3 (original was {} nm^3)'
            logger.debug(debug_msg.format(max_volume / unit.nanometers**3,
                                          restraint_volume / unit.nanometers**3))
            restraint_volume = max_volume

        return -math.log(STANDARD_STATE_VOLUME / restraint_volume)

    def _compute_restraint_volume(self, thermodynamic_state, square_well,
                                  radius_cutoff, energy_cutoff):
        """Compute the volume of the restraint.

        This function is called by ``compute_standard_state_correction()`` when
        the standard state correction depends on the restraint potential.

        Parameters
        ----------
        thermodynamic_state : states.ThermodynamicState
            The thermodynamic state at which to compute the standard state
            correction.
        square_well : bool, optional
            If True, this computes the standard state correction assuming
            the restraint to obey a square well potential. The energy
            cutoff is still applied to the original energy potential.
        radius_cutoff : simtk.unit.Quantity, optional
            The maximum distance achievable by the restraint (units
            compatible with nanometers). This is equivalent to placing
            a hard wall potential at this distance.
        energy_cutoff : float, optional
            The maximum potential energy achievable by the restraint in kT.
            This is equivalent to placing a hard wall potential at a
            distance such that ``potential_energy(distance) == energy_cutoff``.

        Returns
        -------
        restraint_volume : simtk.unit.Quantity
            The volume of the restraint (units of length^3).

        """
        # By default, use numerical integration.
        return self._integrate_restraint_volume(thermodynamic_state, square_well,
                                                radius_cutoff, energy_cutoff)

    def _integrate_restraint_volume(self, thermodynamic_state, square_well,
                                    radius_cutoff, energy_cutoff):
        """Compute the restraint volume through numerical integration.

        Parameters
        ----------
        thermodynamic_state : states.ThermodynamicState
            The thermodynamic state at which to compute the standard state
            correction.
        square_well : bool, optional
            If True, this computes the standard state correction assuming
            the restraint to obey a square well potential. The energy
            cutoff is still applied to the original energy potential.
        radius_cutoff : simtk.unit.Quantity, optional
            The maximum distance achievable by the restraint (units
            compatible with nanometers). This is equivalent to placing
            a hard wall potential at this distance.
        energy_cutoff : float, optional
            The maximum potential energy achievable by the restraint in kT.
            This is equivalent to placing a hard wall potential at a
            distance such that ``potential_energy(distance) == energy_cutoff``.

        Returns
        -------
        restraint_volume : simtk.unit.Quantity
            The volume of the restraint (units of length^3).

        """
        distance_unit = unit.nanometer

        # Create a System object containing two particles
        # connected by the restraint force.
        system = openmm.System()
        system.addParticle(1.0 * unit.amu)
        system.addParticle(1.0 * unit.amu)
        force = copy.deepcopy(self)
        force.restrained_atom_indices1 = [0]
        force.restrained_atom_indices2 = [1]
        # Disable the PBC for this approximation of the analytical solution.
        force.setUsesPeriodicBoundaryConditions(False)
        system.addForce(force)

        # Create a Reference context to evaluate energies on the CPU.
        integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
        platform = openmm.Platform.getPlatformByName('Reference')
        context = openmm.Context(system, integrator, platform)

        # Set default positions.
        positions = unit.Quantity(np.zeros([2,3]), distance_unit)
        context.setPositions(positions)

        # Create a function to compute integrand as a function of interparticle separation.
        beta = thermodynamic_state.beta

        def restraint_potential_func(r):
            """Return the potential energy in kT from the distance in nanometers."""
            positions[1, 0] = r * distance_unit
            context.setPositions(positions)
            state = context.getState(getEnergy=True)
            return beta * state.getPotentialEnergy()

        def integrand(r):
            """
            Parameters
            ----------
            r : float
                Inter-particle separation in nanometers

            Returns
            -------
            dI : float
               Contribution to the integral (in nm^2).

            """
            potential = restraint_potential_func(r)
            # If above the energy cutoff, this doesn't contribute to the integral.
            if energy_cutoff is not None and potential > energy_cutoff:
                return 0.0
            # Check if we're reweighting to a square well potential.
            if square_well:
                potential = 0.0
            dI = 4.0 * math.pi * r**2 * math.exp(-potential)
            return dI

        # Determine integration limits.
        r_min, r_max, analytical_volume = self._determine_integral_limits(
            thermodynamic_state, radius_cutoff, energy_cutoff, restraint_potential_func)

        # Integrate restraint volume.
        restraint_volume, restraint_volume_error = scipy.integrate.quad(
            lambda r: integrand(r), r_min / distance_unit, r_max / distance_unit)
        restraint_volume = restraint_volume * distance_unit**3 + analytical_volume
        logger.debug("restraint_volume = {} nm^3".format(restraint_volume / distance_unit**3))

        return restraint_volume

    def _determine_integral_limits(self, thermodynamic_state, radius_cutoff,
                                   energy_cutoff, potential_energy_func):
        """Determine integration limits for the standard state correction calculation.

        This is called by ``_integrate_restraint_volume()`` to determine
        the limits for numerical integration. This is important if we have
        a cutoff as the points evaluated by scipy.integrate.quad are adaptively
        chosen, and the hard wall can create numerical problems.

        If part of the potential energy function can be computed analytically
        you can reduce the integration interval and return a non-zero constant
        to be added to the result of the integration.

        Parameters
        ----------
        thermodynamic_state : states.ThermodynamicState
            The thermodynamic state at which to compute the standard state
            correction.
        square_well : bool, optional
            If True, this computes the standard state correction assuming
            the restraint to obey a square well potential. The energy
            cutoff is still applied to the original energy potential.
        radius_cutoff : simtk.unit.Quantity, optional
            The maximum distance achievable by the restraint (units
            compatible with nanometers). This is equivalent to placing
            a hard wall potential at this distance.
        energy_cutoff : float, optional
            The maximum potential energy achievable by the restraint in kT.
            This is equivalent to placing a hard wall potential at a
            distance such that ``potential_energy(distance) == energy_cutoff``.

        Returns
        -------
        r_min : simtk.unit.Quantity
            The lower limit for numerical integration.
        r_max : simtk.unit.Quantity
            The upper limit for numerical integration.
        analytical_volume : simtk.unit.Quantity
            Volume excluded from the numerical integration that has been
            computed analytically. This will be summed to the volume
            computed through numerical integration.

        """
        distance_unit = unit.nanometers

        # The lower limit is always 0. Find the upper limit.
        r_min = 0.0 * distance_unit
        r_max = float('inf')
        analytical_volume = 0.0 * distance_unit**3

        if radius_cutoff is not None:
            r_max = min(r_max, radius_cutoff / distance_unit)

        if energy_cutoff is not None:
            # First check if an analytical solution is available.
            try:
                energy_cutoff_distance = self.distance_at_energy(energy_cutoff*thermodynamic_state.kT)
            except NotImplementedError:
                # Find the first distance that exceeds the cutoff.
                potential = 0.0
                energy_cutoff_distance = 0.0  # In nanometers.
                while potential <= energy_cutoff and energy_cutoff_distance < r_max:
                    energy_cutoff_distance += 0.1  # 1 Angstrom.
                    potential = potential_energy_func(energy_cutoff_distance)
            r_max = min(r_max, energy_cutoff_distance)

        # Handle the case where there are no distance or energy cutoff.
        if r_max == float('inf'):
            # For periodic systems, take thrice the maximum dimension of the system.
            if thermodynamic_state.is_periodic:
                box_vectors = thermodynamic_state.default_box_vectors
                max_dimension = np.max(unit.Quantity(box_vectors) / distance_unit)
                r_max = 3.0 * max_dimension
            else:
                r_max = 100.0  # distance_unit

        r_max *= distance_unit
        return r_min, r_max, analytical_volume


class RadiallySymmetricCentroidRestraintForce(RadiallySymmetricRestraintForce,
                                              openmm.CustomCentroidBondForce):
    """Base class for radially-symmetric restraints between the centroids of two groups of atoms.

    The restraint is applied between the centers of mass of two groups
    of atoms. The restraint strength is controlled by a global context
    parameter whose name is passed on construction through the optional
    argument ``controlling_parameter_name``.

    With OpenCL, only on 64bit platforms are supported.

    Parameters
    ----------
    energy_function : str
        The energy function to pass to ``CustomCentroidBondForce``. The
        name of the controlling global parameter  will be prepended to
        this expression.
    restraint_parameters : OrderedDict
        An ordered dictionary containing the bond parameters in the form
        parameter_name: parameter_value. The order is important to make
        sure that parameters can be retrieved from the bond force with
        the correct force index.
    restrained_atom_indices1 : iterable of int
        The indices of the first group of atoms to restrain.
    restrained_atom_indices2 : iterable of int
        The indices of the second group of atoms to restrain.
    controlling_parameter_name : str, optional
        The name of the global parameter controlling the energy function.
        The default value is 'lambda_restraints'.

    Attributes
    ----------
    restraint_parameters
    restrained_atom_indices1
    restrained_atom_indices2
    controlling_parameter_name

    """

    def __init__(self, energy_function, restraint_parameters,
                 restrained_atom_indices1, restrained_atom_indices2,
                 controlling_parameter_name='lambda_restraints'):
        # Initialize CustomCentroidBondForce.
        energy_function = controlling_parameter_name + ' * (' + energy_function + ')'
        custom_centroid_bond_force_args = [2, energy_function]
        super(RadiallySymmetricCentroidRestraintForce, self).__init__(
            restraint_parameters, restrained_atom_indices1, restrained_atom_indices2,
            controlling_parameter_name, *custom_centroid_bond_force_args)

    @property
    def restrained_atom_indices1(self):
        """The indices of the first group of restrained atoms."""
        restrained_atom_indices1, weights_group1 = self.getGroupParameters(0)
        return list(restrained_atom_indices1)

    @restrained_atom_indices1.setter
    def restrained_atom_indices1(self, atom_indices):
        self.setGroupParameters(0, atom_indices)

    @property
    def restrained_atom_indices2(self):
        """The indices of the first group of restrained atoms."""
        restrained_atom_indices2, weights_group2 = self.getGroupParameters(1)
        return list(restrained_atom_indices2)

    @restrained_atom_indices2.setter
    def restrained_atom_indices2(self, atom_indices):
        self.setGroupParameters(1, atom_indices)

    def _create_bond(self, bond_parameter_values, restrained_atom_indices1,
                     restrained_atom_indices2):
        """Create the bond modelling the restraint."""
        self.addGroup(restrained_atom_indices1)
        self.addGroup(restrained_atom_indices2)
        self.addBond([0, 1], bond_parameter_values)


class RadiallySymmetricBondRestraintForce(RadiallySymmetricRestraintForce,
                                          openmm.CustomBondForce):
    """Base class for radially-symmetric restraints between two atoms.

    This is a version of ``RadiallySymmetricCentroidRestraintForce`` that can
    be used with OpenCL 32-bit platforms. It supports atom groups with only a
    single atom.

    """

    def __init__(self, energy_function, restraint_parameters,
                 restrained_atom_index1, restrained_atom_index2,
                 controlling_parameter_name='lambda_restraints'):
        # Initialize CustomBondForce.
        energy_function = energy_function.replace('distance(g1,g2)', 'r')
        energy_function = controlling_parameter_name + ' * (' + energy_function + ')'
        super(RadiallySymmetricBondRestraintForce, self).__init__(
            restraint_parameters, [restrained_atom_index1], [restrained_atom_index2],
            controlling_parameter_name, energy_function)

    # -------------------------------------------------------------------------
    # Public properties.
    # -------------------------------------------------------------------------

    @property
    def restrained_atom_indices1(self):
        """The indices of the first group of restrained atoms."""
        atom1, atom2, parameters = self.getBondParameters(0)
        return [atom1]

    @restrained_atom_indices1.setter
    def restrained_atom_indices1(self, atom_indices):
        assert len(atom_indices) == 1
        atom1, atom2, parameters = self.getBondParameters(0)
        self.setBondParameters(0, atom_indices[0], atom2, parameters)

    @property
    def restrained_atom_indices2(self):
        """The indices of the first group of restrained atoms."""
        atom1, atom2, parameters = self.getBondParameters(0)
        return [atom2]

    @restrained_atom_indices2.setter
    def restrained_atom_indices2(self, atom_indices):
        assert len(atom_indices) == 1
        atom1, atom2, parameters = self.getBondParameters(0)
        self.setBondParameters(0, atom1, atom_indices[0], parameters)

    def _create_bond(self, bond_parameter_values, restrained_atom_indices1, restrained_atom_indices2):
        """Create the bond modelling the restraint."""
        self.addBond(restrained_atom_indices1[0], restrained_atom_indices2[0], bond_parameter_values)


# =============================================================================
# HARMONIC RESTRAINTS
# =============================================================================

class HarmonicRestraintForceMixIn(object):
    """A mix-in providing the interface for harmonic restraints."""

    def __init__(self, spring_constant, *args, **kwargs):
        energy_function = '(K/2)*distance(g1,g2)^2'
        restraint_parameters = collections.OrderedDict([('K', spring_constant)])
        super(HarmonicRestraintForceMixIn, self).__init__(energy_function, restraint_parameters,
                                                          *args, **kwargs)

    @property
    def spring_constant(self):
        """unit.simtk.Quantity: The spring constant K (units of energy/mole/distance^2)."""
        # This works for both CustomBondForce and CustomCentroidBondForce.
        parameters = self.getBondParameters(0)[-1]
        return parameters[0] * unit.kilojoule_per_mole/unit.nanometers**2

    def distance_at_energy(self, potential_energy):
        """Compute the distance at which the potential energy is ``potential_energy``.

        Parameters
        ----------
        potential_energy : simtk.unit.Quantity
            The potential energy of the restraint (units of energy/mole).

        Returns
        -------
        distance : simtk.unit.Quantity
            The distance at which the potential energy is ``potential_energy``
            (units of length).

        """
        return _compute_harmonic_radius(self.spring_constant, potential_energy)

    def _compute_restraint_volume(self, thermodynamic_state, square_well,
                                  radius_cutoff, energy_cutoff):
        """Compute the restraint volume analytically."""
        # If there is not a cutoff, integrate up to 100kT
        if energy_cutoff is None:
            energy_cutoff = 100.0  # kT
        radius = self.distance_at_energy(energy_cutoff * thermodynamic_state.kT)
        if radius_cutoff is not None:
            radius = min(radius, radius_cutoff)
        if square_well:
            return _compute_sphere_volume(radius)
        return _compute_harmonic_volume(radius, self.spring_constant,
                                        thermodynamic_state.beta)


class HarmonicRestraintForce(HarmonicRestraintForceMixIn,
                             RadiallySymmetricCentroidRestraintForce):
    """Impose a single harmonic restraint between the centroids of two groups of atoms.

    This can be used to prevent the ligand from drifting too far from the
    protein in implicit solvent calculations or to keep the ligand close
    to the binding pocket in the decoupled states to increase mixing.

    The restraint is applied between the centroids of two groups of atoms
    that belong to the receptor and the ligand respectively. The centroids
    are determined by a mass-weighted average of the group particles positions.

    The energy expression of the restraint is given by

       ``E = controlling_parameter * (K/2)*r^2``

    where `K` is the spring constant, `r` is the distance between the
    two group centroids, and `controlling_parameter` is a scale factor that
    can be used to control the strength of the restraint.

    With OpenCL, only on 64bit platforms are supported.

    Parameters
    ----------
    spring_constant : simtk.unit.Quantity
        The spring constant K (see energy expression above) in units
        compatible with joule/nanometer**2/mole.
    restrained_atom_indices1 : iterable of int
        The indices of the first group of atoms to restrain.
    restrained_atom_indices2 : iterable of int
        The indices of the second group of atoms to restrain.
    controlling_parameter_name : str, optional
        The name of the global parameter controlling the energy function.
        The default value is 'lambda_restraints'.

    Attributes
    ----------
    spring_constant
    restrained_atom_indices1
    restrained_atom_indices2
    restraint_parameters
    controlling_parameter_name

    """
    # All the methods are provided by the mix-ins.
    pass


class HarmonicRestraintBondForce(HarmonicRestraintForceMixIn,
                                 RadiallySymmetricBondRestraintForce):
    """Impose a single harmonic restraint between two atoms.

    This is a version of ``HarmonicRestraintForce`` that can be used with
    OpenCL 32-bit platforms. It supports atom groups with only a single atom.

    Parameters
    ----------
    spring_constant : simtk.unit.Quantity
        The spring constant K (see energy expression above) in units
        compatible with joule/nanometer**2/mole.
    restrained_atom_index1 : int
        The index of the first atom to restrain.
    restrained_atom_index2 : int
        The index of the second atom to restrain.
    controlling_parameter_name : str, optional
        The name of the global parameter controlling the energy function.
        The default value is 'lambda_restraints'.

    Attributes
    ----------
    spring_constant
    restrained_atom_indices1
    restrained_atom_indices2
    restraint_parameters
    controlling_parameter_name

    """
    # All the methods are provided by the mix-ins.
    pass


# =============================================================================
# FLAT-BOTTOM RESTRAINTS
# =============================================================================

class FlatBottomRestraintForceMixIn(object):
    """A mix-in providing the interface for flat-bottom restraints."""

    def __init__(self, spring_constant, well_radius, *args, **kwargs):
        energy_function = 'step(distance(g1,g2)-r0) * (K/2)*(distance(g1,g2)-r0)^2'
        restraint_parameters = collections.OrderedDict([
            ('K', spring_constant),
            ('r0', well_radius)
        ])
        super(FlatBottomRestraintForceMixIn, self).__init__(energy_function, restraint_parameters,
                                                            *args, **kwargs)

    @property
    def spring_constant(self):
        """unit.simtk.Quantity: The spring constant K (units of energy/mole/length^2)."""
        # This works for both CustomBondForce and CustomCentroidBondForce.
        parameters = self.getBondParameters(0)[-1]
        return parameters[0] * unit.kilojoule_per_mole/unit.nanometers**2

    @property
    def well_radius(self):
        """unit.simtk.Quantity: The distance at which the harmonic restraint is imposed (units of length)."""
        # This works for both CustomBondForce and CustomCentroidBondForce.
        parameters = self.getBondParameters(0)[-1]
        return parameters[1] * unit.nanometers

    def distance_at_energy(self, potential_energy):
        """Compute the distance at which the potential energy is ``potential_energy``.

        Parameters
        ----------
        potential_energy : simtk.unit.Quantity
            The potential energy of the restraint (units of energy/mole).

        Returns
        -------
        distance : simtk.unit.Quantity
            The distance at which the potential energy is ``potential_energy``
            (units of length).

        """
        if potential_energy == 0.0*unit.kilojoules_per_mole:
            raise ValueError('Cannot compute the distance at this potential energy.')
        harmonic_radius = _compute_harmonic_radius(self.spring_constant, potential_energy)
        return self.well_radius + harmonic_radius

    def _compute_restraint_volume(self, thermodynamic_state, square_well,
                                  radius_cutoff, energy_cutoff):
        """Compute the restraint volume analytically."""
        # Check if we are using square well and we can avoid numerical integration.
        if square_well:
            _, r_max, _ = self._determine_integral_limits(
                thermodynamic_state, radius_cutoff, energy_cutoff)
            return _compute_sphere_volume(r_max)
        return self._integrate_restraint_volume(thermodynamic_state, square_well,
                                                radius_cutoff, energy_cutoff)

    def _determine_integral_limits(self, thermodynamic_state, radius_cutoff,
                                   energy_cutoff, potential_energy_func=None):
        # If there is not a cutoff, integrate up to 100kT.
        if energy_cutoff is None:
            energy_cutoff = 100.0  # kT
        energy_cutoff = energy_cutoff * thermodynamic_state.kT
        r_max = _compute_harmonic_radius(self.spring_constant, energy_cutoff)
        r_max += self.well_radius
        if radius_cutoff is not None:
            r_max = min(r_max, radius_cutoff)

        # Compute the volume from the flat-bottom part of the potential.
        r_min = min(r_max, self.well_radius)
        analytical_volume = _compute_sphere_volume(r_min)
        return r_min, r_max, analytical_volume


class FlatBottomRestraintForce(FlatBottomRestraintForceMixIn,
                               RadiallySymmetricCentroidRestraintForce):
    """A restraint between the centroids of two groups of atoms using a flat potential well with harmonic walls.

    An alternative choice to receptor-ligand restraints that uses a flat
    potential inside most of the protein volume with harmonic restraining
    walls outside of this. It can be used to prevent the ligand from
    drifting too far from protein in implicit solvent calculations while
    still exploring the surface of the protein for putative binding sites.

    The restraint is applied between the centroids of two groups of atoms
    that belong to the receptor and the ligand respectively. The centroids
    are determined by a mass-weighted average of the group particles positions.

    More precisely, the energy expression of the restraint is given by

        ``E = controlling_parameter * step(r-r0) * (K/2)*(r-r0)^2``

    where ``K`` is the spring constant, ``r`` is the distance between the
    restrained atoms, ``r0`` is another parameter defining the distance
    at which the restraint is imposed, and ``controlling_parameter``
    is a scale factor that can be used to control the strength of the
    restraint.

    With OpenCL, only on 64bit platforms are supported.

    Parameters
    ----------
    spring_constant : simtk.unit.Quantity
        The spring constant K (see energy expression above) in units
        compatible with joule/nanometer**2/mole.
    well_radius : simtk.unit.Quantity
        The distance r0 (see energy expression above) at which the harmonic
        restraint is imposed in units of distance.
    restrained_atom_indices1 : iterable of int
        The indices of the first group of atoms to restrain.
    restrained_atom_indices2 : iterable of int
        The indices of the second group of atoms to restrain.
    controlling_parameter_name : str, optional
        The name of the global parameter controlling the energy function.
        The default value is 'lambda_restraints'.

    Attributes
    ----------
    spring_constant
    well_radius
    restrained_atom_indices1
    restrained_atom_indices2
    restraint_parameters
    controlling_parameter_name

    """
    # All the methods are provided by the mix-ins.
    pass


class FlatBottomRestraintBondForce(FlatBottomRestraintForceMixIn,
                                   RadiallySymmetricBondRestraintForce):
    """A restraint between two atoms using a flat potential well with harmonic walls.

    This is a version of ``FlatBottomRestraintForce`` that can be used with
    OpenCL 32-bit platforms. It supports atom groups with only a single atom.

    Parameters
    ----------
    spring_constant : simtk.unit.Quantity
        The spring constant K (see energy expression above) in units
        compatible with joule/nanometer**2/mole.
    well_radius : simtk.unit.Quantity
        The distance r0 (see energy expression above) at which the harmonic
        restraint is imposed in units of distance.
    restrained_atom_index1 : int
        The index of the first group of atoms to restrain.
    restrained_atom_index2 : int
        The index of the second group of atoms to restrain.
    controlling_parameter_name : str, optional
        The name of the global parameter controlling the energy function.
        The default value is 'lambda_restraints'.

    Attributes
    ----------
    spring_constant
    well_radius
    restrained_atom_indices1
    restrained_atom_indices2
    restraint_parameters
    controlling_parameter_name

    """
    # All the methods are provided by the mix-ins.
    pass


# =============================================================================
# REACTION FIELD
# =============================================================================

class UnshiftedReactionFieldForce(openmm.CustomNonbondedForce):
    """A force modelling switched reaction-field electrostatics.

    Contrarily to a normal `NonbondedForce` with `CutoffPeriodic` nonbonded
    method, this force sets the `c_rf` to 0.0 and uses a switching function
    to avoid forces discontinuities at the cutoff distance.

    Parameters
    ----------
    cutoff_distance : simtk.unit.Quantity, default 15*angstroms
        The cutoff distance (units of distance).
    switch_width : simtk.unit.Quantity, default 1.0*angstrom
        Switch width for electrostatics (units of distance).
    reaction_field_dielectric : float
        The dielectric constant used for the solvent.

    """

    def __init__(self, cutoff_distance=15*unit.angstroms, switch_width=1.0*unit.angstrom,
                 reaction_field_dielectric=78.3):
        k_rf = cutoff_distance**(-3) * (reaction_field_dielectric - 1.0) / (2.0*reaction_field_dielectric + 1.0)

        # Energy expression omits c_rf constant term.
        energy_expression = "ONE_4PI_EPS0*chargeprod*(r^(-1) + k_rf*r^2);"
        energy_expression += "chargeprod = charge1*charge2;"
        energy_expression += "k_rf = {:f};".format(k_rf.value_in_unit_system(unit.md_unit_system))
        energy_expression += "ONE_4PI_EPS0 = {:f};".format(ONE_4PI_EPS0)  # already in OpenMM units

        # Create CustomNonbondedForce.
        super(UnshiftedReactionFieldForce, self).__init__(energy_expression)

        # Add parameters.
        self.addPerParticleParameter("charge")

        # Configure force.
        self.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
        self.setCutoffDistance(cutoff_distance)
        self.setUseLongRangeCorrection(False)
        if switch_width is not None:
            self.setUseSwitchingFunction(True)
            self.setSwitchingDistance(cutoff_distance - switch_width)
        else:  # Truncated
            self.setUseSwitchingFunction(False)

    @classmethod
    def from_nonbonded_force(cls, nonbonded_force, switch_width=1.0*unit.angstrom):
        """Copy constructor from an OpenMM `NonbondedForce`.

        The returned force has same cutoff distance and dielectric, and
        its particles have the same charges. Exclusions corresponding to
        `nonbonded_force` exceptions are also added.

        .. warning
            This only creates the force object. The electrostatics in
            `nonbonded_force` remains unmodified. Use the function
            `replace_reaction_field` to correctly convert a system to
            use an unshifted reaction field potential.

        Parameters
        ----------
        nonbonded_force : simtk.openmm.NonbondedForce
            The nonbonded force to copy.
        switch_width : simtk.unit.Quantity
            Switch width for electrostatics (units of distance).

        Returns
        -------
        reaction_field_force : UnshiftedReactionFieldForce
            The reaction field force with copied particles.

        """
        # OpenMM gives unitless values.
        cutoff_distance = nonbonded_force.getCutoffDistance()
        reaction_field_dielectric = nonbonded_force.getReactionFieldDielectric()
        reaction_field_force = cls(cutoff_distance, switch_width, reaction_field_dielectric)

        # Set particle charges.
        for particle_index in range(nonbonded_force.getNumParticles()):
            charge, sigma, epsilon = nonbonded_force.getParticleParameters(particle_index)
            reaction_field_force.addParticle([charge])

        # Add exclusions to CustomNonbondedForce.
        for exception_index in range(nonbonded_force.getNumExceptions()):
            iatom, jatom, chargeprod, sigma, epsilon = nonbonded_force.getExceptionParameters(exception_index)
            reaction_field_force.addExclusion(iatom, jatom)

        return reaction_field_force

    @classmethod
    def from_system(cls, system, switch_width=1.0*unit.angstrom):
        """Copy constructor from the first OpenMM `NonbondedForce` in `system`.

        If multiple `NonbondedForce`s are found, an exception is raised.

        .. warning
            This only creates the force object. The electrostatics in
            `nonbonded_force` remains unmodified. Use the function
            `replace_reaction_field` to correctly convert a system to
            use an unshifted reaction field potential.

        Parameters
        ----------
        system : simtk.openmm.System
            The system containing the nonbonded force to copy.
        switch_width : simtk.unit.Quantity
            Switch width for electrostatics (units of distance).

        Returns
        -------
        reaction_field_force : UnshiftedReactionFieldForce
            The reaction field force.

        See Also
        --------
        UnshiftedReactionField.from_nonbonded_force

        """
        force_idx, nonbonded_force = find_forces(system, openmm.NonbondedForce, only_one=True)
        return cls.from_nonbonded_force(nonbonded_force, switch_width)


class SwitchedReactionFieldForce(openmm.CustomNonbondedForce):
    """A force modelling switched reaction-field electrostatics.

    Parameters
    ----------
    cutoff_distance : simtk.unit.Quantity, default 15*angstroms
        The cutoff distance (units of distance).
    switch_width : simtk.unit.Quantity, default 1.0*angstrom
        Switch width for electrostatics (units of distance).
    reaction_field_dielectric : float
        The dielectric constant used for the solvent.

    """

    def __init__(self, cutoff_distance=15*unit.angstroms, switch_width=1.0*unit.angstrom,
                 reaction_field_dielectric=78.3):
        k_rf = cutoff_distance**(-3) * (reaction_field_dielectric - 1.0) / (2.0*reaction_field_dielectric + 1.0)
        c_rf = cutoff_distance**(-1) * (3*reaction_field_dielectric) / (2.0*reaction_field_dielectric + 1.0)

        # Energy expression omits c_rf constant term.
        energy_expression = "ONE_4PI_EPS0*chargeprod*(r^(-1) + k_rf*r^2 - c_rf);"
        energy_expression += "chargeprod = charge1*charge2;"
        energy_expression += "k_rf = {:f};".format(k_rf.value_in_unit_system(unit.md_unit_system))
        energy_expression += "c_rf = {:f};".format(c_rf.value_in_unit_system(unit.md_unit_system))
        energy_expression += "ONE_4PI_EPS0 = {:f};".format(ONE_4PI_EPS0)  # already in OpenMM units

        # Create CustomNonbondedForce.
        super(SwitchedReactionFieldForce, self).__init__(energy_expression)

        # Add parameters.
        self.addPerParticleParameter("charge")

        # Configure force.
        self.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
        self.setCutoffDistance(cutoff_distance)
        self.setUseLongRangeCorrection(False)
        if switch_width is not None:
            self.setUseSwitchingFunction(True)
            self.setSwitchingDistance(cutoff_distance - switch_width)
        else:  # Truncated
            self.setUseSwitchingFunction(False)

    @classmethod
    def from_nonbonded_force(cls, nonbonded_force, switch_width=1.0*unit.angstrom):
        """Copy constructor from an OpenMM `NonbondedForce`.

        The returned force has same cutoff distance and dielectric, and
        its particles have the same charges. Exclusions corresponding to
        `nonbonded_force` exceptions are also added.

        .. warning
            This only creates the force object. The electrostatics in
            `nonbonded_force` remains unmodified. Use the function
            `replace_reaction_field` to correctly convert a system to
            use an unshifted reaction field potential.

        Parameters
        ----------
        nonbonded_force : simtk.openmm.NonbondedForce
            The nonbonded force to copy.
        switch_width : simtk.unit.Quantity
            Switch width for electrostatics (units of distance).

        Returns
        -------
        reaction_field_force : UnshiftedReactionFieldForce
            The reaction field force with copied particles.

        """
        # OpenMM gives unitless values.
        cutoff_distance = nonbonded_force.getCutoffDistance()
        reaction_field_dielectric = nonbonded_force.getReactionFieldDielectric()
        reaction_field_force = cls(cutoff_distance, switch_width, reaction_field_dielectric)

        # Set particle charges.
        for particle_index in range(nonbonded_force.getNumParticles()):
            charge, sigma, epsilon = nonbonded_force.getParticleParameters(particle_index)
            reaction_field_force.addParticle([charge])

        # Add exclusions to CustomNonbondedForce.
        for exception_index in range(nonbonded_force.getNumExceptions()):
            iatom, jatom, chargeprod, sigma, epsilon = nonbonded_force.getExceptionParameters(exception_index)
            reaction_field_force.addExclusion(iatom, jatom)

        return reaction_field_force

    @classmethod
    def from_system(cls, system, switch_width=1.0*unit.angstrom):
        """Copy constructor from the first OpenMM `NonbondedForce` in `system`.

        If multiple `NonbondedForce`s are found, an exception is raised.

        .. warning
            This only creates the force object. The electrostatics in
            `nonbonded_force` remains unmodified. Use the function
            `replace_reaction_field` to correctly convert a system to
            use an unshifted reaction field potential.

        Parameters
        ----------
        system : simtk.openmm.System
            The system containing the nonbonded force to copy.
        switch_width : simtk.unit.Quantity
            Switch width for electrostatics (units of distance).

        Returns
        -------
        reaction_field_force : UnshiftedReactionFieldForce
            The reaction field force.

        See Also
        --------
        UnshiftedReactionField.from_nonbonded_force

        """
        force_idx, nonbonded_force = find_forces(system, openmm.NonbondedForce, only_one=True)
        return cls.from_nonbonded_force(nonbonded_force, switch_width)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
