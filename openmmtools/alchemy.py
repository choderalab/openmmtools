#!/usr/bin/python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Alchemical factory for free energy calculations that operates directly on OpenMM System objects.

DESCRIPTION

This module contains enumerative factories for generating alchemically-modified System objects
usable for the calculation of free energy differences of hydration or ligand binding.

* `AlchemicalFactory` uses fused elecrostatic and steric alchemical modifications.

"""

# TODO
# - Generalize treatment of nonbonded sterics/electrostatics intra-alchemical
#   forces to support arbitrary mixing rules. Can we eliminate decoupling to something simpler?
# - Add support for other GBSA models.
# - Add functions for the automatic optimization of alchemical states?
# - Can we store serialized form of Force objects so that we can save time in reconstituting
#   Force objects when we make copies?  We can even manipulate the XML representation directly.
# - Allow protocols to automatically be resized to arbitrary number of states, to
#   allow number of states to be enlarged to be an integral multiple of number of GPUs.
# - Finish AMOEBA support.
# - Can alchemically-modified System objects share unmodified Force objects to avoid overhead
#   of duplicating Forces that are not modified?


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import copy
import inspect
import logging
import collections

import numpy as np
from simtk import openmm, unit

from openmmtools import states, utils
from openmmtools.constants import ONE_4PI_EPS0

logger = logging.getLogger(__name__)


# =============================================================================
# ALCHEMICAL STATE
# =============================================================================

class AlchemicalStateError(states.ComposableStateError):
    """Error raised by an AlchemicalState."""
    pass


class AlchemicalFunction(object):
    """A function of alchemical variables.

    Parameters
    ----------
    expression : str
        A mathematical expression involving alchemical variables.

    Examples
    --------
    >>> alchemical_state = AlchemicalState(lambda_sterics=1.0, lambda_angles=1.0)
    >>> alchemical_state.set_alchemical_variable('lambda', 0.5)
    >>> alchemical_state.set_alchemical_variable('lambda2', 1.0)
    >>> alchemical_state.lambda_sterics = AlchemicalFunction('lambda**2')
    >>> alchemical_state.lambda_sterics
    0.25
    >>> alchemical_state.lambda_angles = AlchemicalFunction('(lambda + lambda2) / 2')
    >>> alchemical_state.lambda_angles
    0.75

    """
    def __init__(self, expression):
        self._expression = expression

    def __call__(self, variables):
        return utils.math_eval(self._expression, variables)


class AlchemicalState(object):
    """Represent an alchemical state.

    The alchemical parameters modify the Hamiltonian and affect the
    computation of the energy. Alchemical parameters that have value
    None are considered undefined, which means that applying this
    state to System and Context that have that parameter as a global
    variable will raise an AlchemicalStateError.

    Parameters
    ----------
    lambda_sterics : float, optional
        Scaling factor for ligand sterics (Lennard-Jones and Halgren)
        interactions (default is 1.0).
    lambda_electrostatics : float, optional
        Scaling factor for ligand charges, intrinsic Born radii, and surface
        area term (default is 1.0).
    lambda_bonds : float, optional
        Scaling factor for alchemically-softened bonds (default is 1.0).
    lambda_angles : float, optional
        Scaling factor for alchemically-softened angles (default is 1.0).
    lambda_torsions : float, optional
        Scaling factor for alchemically-softened torsions (default is 1.0).

    Attributes
    ----------
    lambda_sterics
    lambda_electrostatics
    lambda_bonds
    lambda_angles
    lambda_torsions

    Examples
    --------
    Create an alchemically modified system.

    >>> from openmmtools import testsystems
    >>> factory = AlchemicalFactory(consistent_exceptions=False)
    >>> alanine_vacuum = testsystems.AlanineDipeptideVacuum().system
    >>> alchemical_region = AlchemicalRegion(alchemical_atoms=range(22))
    >>> alanine_alchemical_system = factory.create_alchemical_system(reference_system=alanine_vacuum,
    ...                                                              alchemical_regions=alchemical_region)

    Create a completely undefined alchemical state.

    >>> alchemical_state = AlchemicalState()
    >>> print(alchemical_state.lambda_sterics)
    None
    >>> alchemical_state.apply_to_system(alanine_alchemical_system)
    Traceback (most recent call last):
    ...
    AlchemicalStateError: The system parameter lambda_sterics is not defined in this state.

    Create an AlchemicalState that matches the parameters defined in
    the System.

    >>> alchemical_state = AlchemicalState.from_system(alanine_alchemical_system)
    >>> alchemical_state.lambda_sterics
    1.0
    >>> alchemical_state.lambda_electrostatics
    1.0
    >>> print(alchemical_state.lambda_angles)
    None

    AlchemicalState implement the IComposableState interface, so it can be
    used with CompoundThermodynamicState. All the alchemical parameters are
    accessible through the compound state.

    >>> from simtk import openmm, unit
    >>> thermodynamic_state = states.ThermodynamicState(system=alanine_alchemical_system,
    ...                                                 temperature=300*unit.kelvin)
    >>> compound_state = states.CompoundThermodynamicState(thermodynamic_state=thermodynamic_state,
    ...                                                    composable_states=[alchemical_state])
    >>> compound_state.lambda_sterics
    1.0

    You can control the parameters in the OpenMM Context in this state by
    setting the state attributes.

    >>> compound_state.lambda_sterics = 0.5
    >>> integrator = openmm.VerletIntegrator(1.0*unit.femtosecond)
    >>> context = compound_state.create_context(integrator)
    >>> context.getParameter('lambda_sterics')
    0.5
    >>> compound_state.lambda_sterics = 1.0
    >>> compound_state.apply_to_context(context)
    >>> context.getParameter('lambda_sterics')
    1.0

    You can express the alchemical parameters as a mathematical expression
    involving alchemical variables. Here is an example for a two-stage function.

    >>> compound_state.set_alchemical_variable('lambda', 1.0)
    >>> compound_state.lambda_sterics = AlchemicalFunction('step_hm(lambda - 0.5) + 2*lambda * step_hm(0.5 - lambda)')
    >>> compound_state.lambda_electrostatics = AlchemicalFunction('2*(lambda - 0.5) * step(lambda - 0.5)')
    >>> for l in [0.0, 0.25, 0.5, 0.75, 1.0]:
    ...     compound_state.set_alchemical_variable('lambda', l)
    ...     print(compound_state.lambda_sterics)
    0.0
    0.5
    1.0
    1.0
    1.0


    """

    # -------------------------------------------------------------------------
    # Constructors
    # -------------------------------------------------------------------------

    def __init__(self, **kwargs):
        self._initialize(**kwargs)

    @classmethod
    def from_system(cls, system):
        """Constructor reading the state from an alchemical system.

        Parameters
        ----------
        system : simtk.openmm.System
            An alchemically modified system in a defined alchemical state.

        Returns
        -------
        The AlchemicalState object representing the alchemical state of
        the system.

        Raises
        ------
        AlchemicalStateError
            If the same parameter has different values in the system, or
            if the system has no lambda parameters.

        """
        alchemical_parameters = {}
        for force, parameter_name, parameter_id in cls._get_system_lambda_parameters(system):
            parameter_value = force.getGlobalParameterDefaultValue(parameter_id)

            # Check that we haven't already found
            # the parameter with a different value.
            if parameter_name in alchemical_parameters:
                if alchemical_parameters[parameter_name] != parameter_value:
                    err_msg = ('Parameter {} has been found in the force {} with two values: '
                               '{} and {}').format(parameter_name, force.__class__.__name__,
                                                   parameter_value, alchemical_parameters[parameter_name])
                    raise AlchemicalStateError(err_msg)
            else:
                alchemical_parameters[parameter_name] = parameter_value

        # Check that the system is alchemical.
        if len(alchemical_parameters) == 0:
            raise AlchemicalStateError('System has no lambda parameters.')

        # Create and return the AlchemicalState.
        return AlchemicalState(**alchemical_parameters)

    # -------------------------------------------------------------------------
    # Lambda properties
    # -------------------------------------------------------------------------

    # Lambda properties. The set of supported parameters is dynamically
    # discovered by _get_supported_parameters() based on this list. We
    # list them explicitly to preserve auto-completion and prevent silent
    # bugs due to monkey-patching.
    class _LambdaProperty(object):
        """Descriptor of a lambda parameter."""
        def __init__(self, parameter_name):
            self._parameter_name = parameter_name

        def __get__(self, instance, owner_class=None):
            parameter_value = instance._parameters[self._parameter_name]
            if isinstance(parameter_value, AlchemicalFunction):
                parameter_value = parameter_value(instance._alchemical_variables)
            assert parameter_value is None or 0.0 <= parameter_value <= 1.0, '{}: {}'.format(
                self._parameter_name, parameter_value)
            return parameter_value

        def __set__(self, instance, new_value):
            assert (new_value is None or isinstance(new_value, AlchemicalFunction) or
                    0.0 <= new_value <= 1.0)
            instance._parameters[self._parameter_name] = new_value

    lambda_sterics = _LambdaProperty('lambda_sterics')
    lambda_electrostatics = _LambdaProperty('lambda_electrostatics')
    lambda_bonds = _LambdaProperty('lambda_bonds')
    lambda_angles = _LambdaProperty('lambda_angles')
    lambda_torsions = _LambdaProperty('lambda_torsions')

    def set_alchemical_parameters(self, new_value):
        """Set all defined parameters to the given value.

        The undefined parameters (i.e. those being set to None) remain
        undefined.

        Parameters
        ----------
        new_value : float
            The new value for all defined parameters.

        """
        for parameter_name in self._parameters:
            if self._parameters[parameter_name] is not None:
                setattr(self, parameter_name, new_value)

    # -------------------------------------------------------------------------
    # Alchemical variables
    # -------------------------------------------------------------------------

    def get_alchemical_variable(self, variable_name):
        """Return the value of the alchemical parameter.

        Parameters
        ----------
        variable_name : str
            The name of the alchemical variable.

        Returns
        -------
        variable_value : float
            The value of the alchemical variable.

        """
        try:
            variable_value = self._alchemical_variables[variable_name]
        except KeyError:
            raise AlchemicalStateError('Unknown alchemical variable {}'.format(variable_name))
        return variable_value

    def set_alchemical_variable(self, variable_name, new_value):
        """Set the value of the alchemical variable.

        Parameters
        ----------
        variable_name : str
            The name of the alchemical variable.
        new_value : float
            The new value for the variable.

        """
        if variable_name in self._parameters:
            raise AlchemicalStateError('Cannot have an alchemical variable with the same name '
                                       'of the predefined alchemical parameter {}.'.format(variable_name))
        self._alchemical_variables[variable_name] = new_value

    # -------------------------------------------------------------------------
    # Operators
    # -------------------------------------------------------------------------

    def __eq__(self, other):
        is_equal = True
        for parameter_name in self._parameters:
            self_value = getattr(self, parameter_name)
            other_value = getattr(other, parameter_name)
            is_equal = is_equal and self_value == other_value
        return is_equal

    def __ne__(self, other):
        # TODO: we can safely remove this when dropping support for Python 2
        return not self == other

    def __str__(self):
        return str(self._parameters)

    def __getstate__(self):
        """Return a dictionary representation of the state."""
        serialization = dict(parameters={}, alchemical_variables={})

        # Copy parameters and convert AlchemicalFunctions to string expressions.
        for parameter_class in ['parameters', 'alchemical_variables']:
            parameters = getattr(self, '_' + parameter_class)
            for parameter, value in parameters.items():
                if isinstance(value, AlchemicalFunction):
                    serialization[parameter_class][parameter] = value._expression
                else:
                    serialization[parameter_class][parameter] = value
        return serialization

    def __setstate__(self, serialization):
        """Set the state from a dictionary representation."""
        parameters = serialization['parameters']
        alchemical_variables = serialization['alchemical_variables']
        alchemical_functions = dict()

        # Temporarily store alchemical functions.
        for parameter_name, value in parameters.items():
            if isinstance(value, str):
                alchemical_functions[parameter_name] = value
                parameters[parameter_name] = None

        # Initialize parameters and add all alchemical variables.
        self._initialize(**parameters)
        for variable_name, value in alchemical_variables.items():
            self.set_alchemical_variable(variable_name, value)

        # Add back alchemical functions.
        for parameter_name, expression in alchemical_functions.items():
            setattr(self, parameter_name, AlchemicalFunction(expression))

    # -------------------------------------------------------------------------
    # IComposableState interface
    # -------------------------------------------------------------------------

    def apply_to_system(self, system):
        """Set the alchemical state of the system to this.

        Parameters
        ----------
        system : simtk.openmm.System
            The system to modify.

        Raises
        ------
        AlchemicalStateError
            If the system does not have the required lambda global variables.

        """
        parameters_applied = set()
        for force, parameter_name, parameter_id in self._get_system_lambda_parameters(system):
            parameter_value = getattr(self, parameter_name)
            if parameter_value is None:
                err_msg = 'The system parameter {} is not defined in this state.'
                raise AlchemicalStateError(err_msg.format(parameter_name))
            else:
                parameters_applied.add(parameter_name)
                force.setGlobalParameterDefaultValue(parameter_id, parameter_value)

        # Check that we set all the defined parameters.
        for parameter_name in self._get_supported_parameters():
            if (self._parameters[parameter_name] is not None and
                    parameter_name not in parameters_applied):
                err_msg = 'Could not find parameter {} in the system'
                raise AlchemicalStateError(err_msg.format(parameter_name))

    def check_system_consistency(self, system):
        """Check if the system is consistent with the alchemical state.

        It raises a AlchemicalStateError if the system is not consistent
        with the alchemical state.

        Parameters
        ----------
        system : simtk.openmm.System
            The system to test.

        Raises
        ------
        AlchemicalStateError
            If the system is not consistent with this state.

        """
        system_alchemical_state = AlchemicalState.from_system(system)

        # Check if parameters are all the same.
        if self != system_alchemical_state:
            err_msg = ('Consistency check failed:\n'
                       '\tSystem parameters          {}\n'
                       '\tAlchemicalState parameters {}')
            raise AlchemicalStateError(err_msg.format(self, system_alchemical_state))

    def apply_to_context(self, context):
        """Put the Context into this AlchemicalState.

        Parameters
        ----------
        context : simtk.openmm.Context
            The context to set.

        Raises
        ------
        AlchemicalStateError
            If the context does not have the required lambda global variables.

        """
        context_parameters = context.getParameters()

        # Set parameters in Context.
        for parameter_name in self._parameters:
            parameter_value = getattr(self, parameter_name)
            if parameter_value is None:
                # Check that Context does not have this parameter.
                if parameter_name in context_parameters:
                    err_msg = 'Context has parameter {} which is undefined in this state'
                    raise AlchemicalStateError(err_msg.format(parameter_name))
                continue
            try:
                context.setParameter(parameter_name, parameter_value)
            except Exception:
                err_msg = 'Could not find parameter {} in context'
                raise AlchemicalStateError(err_msg.format(parameter_name))

    @classmethod
    def _standardize_system(cls, system):
        """Standardize the given system.

        Set all global lambda parameters of the system to 1.0.

        Parameters
        ----------
        system : simtk.openmm.System
            The system to standardize.

        Raises
        ------
        AlchemicalStateError
            If the system is not consistent with this state.

        """
        alchemical_state = AlchemicalState.from_system(system)
        alchemical_state.set_alchemical_parameters(1.0)
        alchemical_state.apply_to_system(system)

    # -------------------------------------------------------------------------
    # Internal-usage
    # -------------------------------------------------------------------------

    def _initialize(self, **kwargs):
        """Initialize the alchemical state."""
        self._alchemical_variables = {}

        # Get supported parameters from properties introspection.
        supported_parameters = self._get_supported_parameters()

        # Check for unknown parameters
        unknown_parameters = set(kwargs) - supported_parameters
        if len(unknown_parameters) > 0:
            err_msg = "Unknown parameters {}".format(unknown_parameters)
            raise AlchemicalStateError(err_msg)

        # Default value for all parameters is None.
        self._parameters = dict.fromkeys(supported_parameters, None)

        # Update parameters with constructor arguments. Calling
        # the properties perform type check on the values.
        for parameter_name, value in kwargs.items():
            setattr(self, parameter_name, value)

    @classmethod
    def _get_supported_parameters(cls):
        """Return a set of the supported alchemical parameters.

        This is based on the exposed properties. This ways we keep autocompletion
        working and avoid silent bugs due to possible monkey patching caused by
        a typo in the name of the variable.

        """
        # TODO just use inspect.getmembers when dropping Python 2
        supported_parameters = {name for name, value in cls.__dict__.items()
                                if isinstance(value, cls._LambdaProperty)}
        return supported_parameters

    @classmethod
    def _get_system_lambda_parameters(cls, system):
        """Yields the supported lambda parameters in the system.

        Yields
        ------
        A tuple force, parameter_name, parameter_index for each supported
        lambda parameter.

        """
        supported_parameters = cls._get_supported_parameters()

        # Retrieve all the forces with global supported parameters.
        for force_index in range(system.getNumForces()):
            force = system.getForce(force_index)
            try:
                n_global_parameters = force.getNumGlobalParameters()
            except AttributeError:
                continue
            for parameter_id in range(n_global_parameters):
                parameter_name = force.getGlobalParameterName(parameter_id)
                if parameter_name in supported_parameters:
                    yield force, parameter_name, parameter_id


# =============================================================================
# ALCHEMICAL REGION
# =============================================================================

_ALCHEMICAL_REGION_ARGS = collections.OrderedDict([
    ('alchemical_atoms', None),
    ('alchemical_bonds', None),
    ('alchemical_angles', None),
    ('alchemical_torsions', None),
    ('annihilate_electrostatics', True),
    ('annihilate_sterics', False),
    ('softcore_alpha', 0.5), ('softcore_a', 1), ('softcore_b', 1), ('softcore_c', 6),
    ('softcore_beta', 0.0), ('softcore_d', 1), ('softcore_e', 1), ('softcore_f', 2)
])


# The class is just a way to document the namedtuple.
class AlchemicalRegion(collections.namedtuple('AlchemicalRegion', _ALCHEMICAL_REGION_ARGS.keys())):
    """Alchemical region.

    This is a namedtuple used to tell the AlchemicalFactory which region
    of the system to alchemically modify and how.

    Parameters
    ----------
    alchemical_atoms : list of int, optional
        List of atoms to be designated for which the nonbonded forces (both
        sterics and electrostatics components) have to be alchemically
        modified (default is None).
    alchemical_bonds : bool or list of int, optional
        If a list of bond indices are specified, these HarmonicBondForce
        entries are softened with 'lambda_bonds'. If set to True, this list
        is auto-generated to include all bonds involving any alchemical
        atoms (default is None).
    alchemical_angles : bool or list of int, optional
        If a list of angle indices are specified, these HarmonicAngleForce
        entries are softened with 'lambda_angles'. If set to True, this
        list is auto-generated to include all angles involving any alchemical
        atoms (default is None).
    alchemical_torsions : bool or list of int, optional
        If a list of torsion indices are specified, these PeriodicTorsionForce
        entries are softened with 'lambda_torsions'. If set to True, this list
        is auto-generated to include al proper torsions involving any alchemical
        atoms. Improper torsions are not softened (default is None).
    annihilate_electrostatics : bool, optional
        If True, electrostatics should be annihilated, rather than decoupled
        (default is True).
    annihilate_sterics : bool, optional
        If True, sterics (Lennard-Jones or Halgren potential) will be annihilated,
        rather than decoupled (default is False).
    softcore_alpha : float, optional
        Alchemical softcore parameter for Lennard-Jones (default is 0.5).
    softcore_a, softcore_b, softcore_c : float, optional
        Parameters modifying softcore Lennard-Jones form. Introduced in
        Eq. 13 of Ref. [1] (default is 1).
    softcore_beta : float, optional
        Alchemical softcore parameter for electrostatics. Set this to zero
        to recover standard electrostatic scaling (default is 0.0).
    softcore_d, softcore_e, softcore_f : float, optional
        Parameters modifying softcore electrostatics form (default is 1).

    Notes
    -----
    The parameters softcore_e and softcore_f determine the effective distance
    between point charges according to

    r_eff = sigma*((softcore_beta*(lambda_electrostatics-1)^softcore_e + (r/sigma)^softcore_f))^(1/softcore_f)

    References
    ----------
    [1] Pham TT and Shirts MR. Identifying low variance pathways for free
    energy calculations of molecular transformations in solution phase.
    JCP 135:034114, 2011. http://dx.doi.org/10.1063/1.3607597

    """
AlchemicalRegion.__new__.__defaults__ = tuple(_ALCHEMICAL_REGION_ARGS.values())


# =============================================================================
# ABSOLUTE ALCHEMICAL FACTORY
# =============================================================================

class AlchemicalFactory(object):
    """Factory of alchemically modified OpenMM Systems.

    The context parameters created are:
    - softcore_alpha: factor controlling softcore lengthscale for Lennard-Jones
    - softcore_beta: factor controlling softcore lengthscale for Coulomb
    - softcore_a: softcore Lennard-Jones parameter from Eq. 13 of Ref [1]
    - softcore_b: softcore Lennard-Jones parameter from Eq. 13 of Ref [1]
    - softcore_c: softcore Lennard-Jones parameter from Eq. 13 of Ref [1]
    - softcore_d: softcore electrostatics parameter
    - softcore_e: softcore electrostatics parameter
    - softcore_f: softcore electrostatics parameter

    Parameters
    ----------
    consistent_exceptions : bool, optional, default = False
        If True, the same functional form of the System's nonbonded
        method will be use to determine the electrostatics contribution
        to the potential energy of 1,4 exceptions instead of the
        classical q1*q2/(4*epsilon*epsilon0*pi*r).

    Examples
    --------

    Create an alchemical factory to alchemically modify OpenMM System objects.

    >>> factory = AlchemicalFactory(consistent_exceptions=False)

    Create an alchemically modified version of p-xylene in T4 lysozyme L99A in GBSA.

    >>> # Create a reference system.
    >>> from openmmtools import testsystems
    >>> reference_system = testsystems.LysozymeImplicit().system
    >>> # Alchemically soften the pxylene atoms
    >>> pxylene_atoms = range(2603,2621) # p-xylene
    >>> alchemical_region = AlchemicalRegion(alchemical_atoms=pxylene_atoms)
    >>> alchemical_system = factory.create_alchemical_system(reference_system, alchemical_region)

    Alchemically modify one water in a water box.

    >>> reference_system = testsystems.WaterBox().system
    >>> alchemical_region = AlchemicalRegion(alchemical_atoms=[0, 1, 2])
    >>> alchemical_system = factory.create_alchemical_system(reference_system, alchemical_region)

    Alchemically modify some angles and torsions in alanine dipeptide and
    annihilate both sterics and electrostatics.

    >>> reference_system = testsystems.AlanineDipeptideVacuum().system
    >>> alchemical_region = AlchemicalRegion(alchemical_atoms=[0], alchemical_torsions=[0,1,2],
    ...                                      alchemical_angles=[0,1,2], annihilate_sterics=True,
    ...                                      annihilate_electrostatics=True)
    >>> alchemical_system = factory.create_alchemical_system(reference_system, alchemical_region)

    Alchemically modify a bond, angles, and torsions in toluene by automatically
    selecting bonds involving alchemical atoms.

    >>> toluene_implicit = testsystems.TolueneImplicit()
    >>> alchemical_region = AlchemicalRegion(alchemical_atoms=[0,1], alchemical_torsions=True,
    ...                                      alchemical_angles=True, annihilate_sterics=True)
    >>> alchemical_system = factory.create_alchemical_system(reference_system, alchemical_region)

    Once the alchemical system is created, you can modify its Hamiltonian
    through AlchemicalState

    >>> alchemical_state = AlchemicalState.from_system(alchemical_system)
    >>> alchemical_state.lambda_sterics
    1.0
    >>> alchemical_state.lambda_electrostatics = 0.5
    >>> alchemical_state.apply_to_system(alchemical_system)

    You can also modify its Hamiltonian directly into a context

    >>> from simtk import openmm, unit
    >>> integrator = openmm.VerletIntegrator(1.0*unit.femtosecond)
    >>> context = openmm.Context(alchemical_system, integrator)
    >>> alchemical_state.set_alchemical_parameters(0.0)  # Set all lambda to 0
    >>> alchemical_state.apply_to_context(context)

    References
    ----------
    [1] Pham TT and Shirts MR. Identifying low variance pathways for free
    energy calculations of molecular transformations in solution phase.
    JCP 135:034114, 2011. http://dx.doi.org/10.1063/1.3607597

    """

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def __init__(self, consistent_exceptions=False):
        self.consistent_exceptions = consistent_exceptions

    def create_alchemical_system(self, reference_system, alchemical_regions):
        """Create an alchemically modified version of the reference system.

        To alter the alchemical state of the returned system use AlchemicalState.

        Parameters
        ----------
        reference_system : simtk.openmm.System
            The system to use as a reference for the creation of the
            alchemical system. This will not be modified.
        alchemical_regions : AlchemicalRegion
            The region of the reference system to alchemically soften.

        Returns
        -------
        alchemical_system : simtk.openmm.System
            Alchemically-modified version of reference_system.

        """
        # TODO implement multiple alchemical regions support.
        if not isinstance(alchemical_regions, AlchemicalRegion):
            raise NotImplemented('There is no support for multiple alchemical regions yet.')
        alchemical_region = alchemical_regions

        # Resolve alchemical region.
        alchemical_region = self._resolve_alchemical_region(reference_system, alchemical_region)

        # Record timing statistics.
        timer = utils.Timer()
        timer.start('Create alchemically modified system')

        # Build alchemical system to modify.
        alchemical_system = openmm.System()

        # Set periodic box vectors.
        box_vectors = reference_system.getDefaultPeriodicBoxVectors()
        alchemical_system.setDefaultPeriodicBoxVectors(*box_vectors)

        # Add particles.
        for particle_index in range(reference_system.getNumParticles()):
            mass = reference_system.getParticleMass(particle_index)
            alchemical_system.addParticle(mass)

        # Add constraints.
        for constraint_index in range(reference_system.getNumConstraints()):
            atom_i, atom_j, r0 = reference_system.getConstraintParameters(constraint_index)
            alchemical_system.addConstraint(atom_i, atom_j, r0)

        # Modify forces as appropriate, copying other forces without modification.
        for reference_force in reference_system.getForces():
            # TODO switch to functools.singledispatch when drop Python2
            reference_force_name = reference_force.__class__.__name__
            alchemical_force_creator_name = '_alchemically_modify_{}'.format(reference_force_name)
            try:
                alchemical_force_creator_func = getattr(self, alchemical_force_creator_name)
            except AttributeError:
                alchemical_forces = [copy.deepcopy(reference_force)]
            else:
                alchemical_forces = alchemical_force_creator_func(reference_force, alchemical_region)
            for alchemical_force in alchemical_forces:
                alchemical_system.addForce(alchemical_force)

        # Record timing statistics.
        timer.stop('Create alchemically modified system')
        timer.report_timing()

        return alchemical_system

    @classmethod
    def get_energy_components(cls, alchemical_system, alchemical_state, positions, platform=None):
        """Compute potential energy of the alchemical system by Force.

        This can be useful for debug and analysis.

        Parameters
        ----------
        alchemical_system : simtk.openmm.AlchemicalSystem
            An alchemically modified system.
        alchemical_state : AlchemicalState
            The alchemical state to set the Context to.
        positions : simtk.unit.Quantity of dimension (natoms, 3)
            Coordinates to use for energy test (units of distance).
        platform : simtk.openmm.Platform, optional
            The OpenMM platform to use to compute the energy. If None,
            OpenMM tries to select the fastest available.

        Returns
        -------
        energy_components : dict str: simtk.unit.Quantity
            A string label describing the role of the force associated to
            its contribution to the potential energy.

        """
        # Find and label all forces.
        force_labels = cls._find_force_components(alchemical_system)
        assert len(force_labels) <= 32, ("The alchemical system has more than 32 force groups; "
                                         "can't compute individual force component energies.")

        # Create deep copy of alchemical system.
        system = copy.deepcopy(alchemical_system)

        # Separate all forces into separate force groups.
        for force_index, force in enumerate(system.getForces()):
            force.setForceGroup(force_index)

        # Create a Context in the given state.
        integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
        if platform is None:
            context = openmm.Context(system, integrator)
        else:
            context = openmm.Context(system, integrator, platform)
        context.setPositions(positions)
        alchemical_state.apply_to_context(context)

        # Get energy components
        energy_components = collections.OrderedDict()
        for force_label, force_index in force_labels.items():
            energy_components[force_label] = context.getState(getEnergy=True,
                                                              groups=2**force_index).getPotentialEnergy()
        # Clean up
        del context, integrator

        return energy_components

    # -------------------------------------------------------------------------
    # Internal usage: AlchemicalRegion
    # -------------------------------------------------------------------------

    @classmethod
    def _resolve_alchemical_region(cls, system, alchemical_region):
        """Return a new AlchemicalRegion with sets of bonds/angles/torsions resolved.

        Also transform any list of indices into a frozenset.

        Parameters
        ----------
        system : simtk.openmm.System
            The system.
        alchemical_region : AlchemicalRegion
            The alchemical region of the system.

        Returns
        -------
        AlchemicalRegion
            A new AlchemicalRegion object in which all alchemical_X (where X is
            atoms, bonds, angles, or torsions) have been converted to a frozenset
            of indices that belong to the System.

        Raises
        ------
        ValueError
            If the indices in the AlchemicalRegion can't be found in the system.

        """
        # TODO move to AlchemicalRegion?
        # TODO process also custom forces? (also in _build_alchemical_X_list methods)
        # Find and cache the reference forces.
        reference_forces = {force.__class__.__name__: force for force in system.getForces()}

        # Count number of particles, angles, etc. in system. Atoms
        # must be processed first since the others build on that.
        reference_counts = collections.OrderedDict([
            ('atom', system.getNumParticles()),
            ('bond', reference_forces['HarmonicBondForce'].getNumBonds()
             if 'HarmonicBondForce' in reference_forces else 0),
            ('angle', reference_forces['HarmonicAngleForce'].getNumAngles()
             if 'HarmonicAngleForce' in reference_forces else 0),
            ('torsion', reference_forces['PeriodicTorsionForce'].getNumTorsions()
             if 'PeriodicTorsionForce' in reference_forces else 0)
        ])

        # Transform named tuple to dict for easy modification.
        alchemical_region = alchemical_region._asdict()

        for region in reference_counts:
            region_name = 'alchemical_' + region + 's'
            region_indices = alchemical_region[region_name]

            # Convert None and False to empty lists.
            if region_indices is None or region_indices is False:
                region_indices = set()

            # Automatically build indices list if True.
            elif region_indices is True:
                if reference_counts[region] == 0:
                    region_indices = set()
                else:
                    # TODO switch to functools.singledispatch when drop Python2
                    builder_function = getattr(cls, '_build_alchemical_{}_list'.format(region))
                    region_indices = builder_function(alchemical_region['alchemical_atoms'],
                                                      reference_forces, system)

            # Convert numpy arrays to Python lists since SWIG
            # have problems with np.int (see openmm#1650).
            elif isinstance(region_indices, np.ndarray):
                region_indices = region_indices.tolist()

            # Convert to set and update alchemical region.
            region_indices = frozenset(region_indices)
            alchemical_region[region_name] = region_indices

            # Check that the given indices are in the system.
            indices_diff = region_indices - set(range(reference_counts[region]))
            if len(indices_diff) > 0:
                err_msg = 'Indices {} in {} cannot be found in the system'
                raise ValueError(err_msg.format(indices_diff, region_name))

        # Check that an alchemical region is defined.
        total_alchemically_modified = 0
        for region in reference_counts:
            total_alchemically_modified += len(alchemical_region['alchemical_' + region + 's'])
        if total_alchemically_modified == 0:
            raise ValueError('The AlchemicalRegion is empty.')

        # Return a new AlchemicalRegion with the resolved indices lists.
        return AlchemicalRegion(**alchemical_region)

    @staticmethod
    def _tabulate_bonds(system):
        """
        Tabulate bonds for the specified system.

        Parameters
        ----------
        system : simtk.openmm.System
            The system for which bonds are to be tabulated.

        Returns
        -------
        bonds : list of set
            bonds[i] is the set of bonds to atom i

        TODO:
        * Could we use a Topology object to simplify this?

        """
        bonds = [set() for _ in range(system.getNumParticles())]

        forces = {system.getForce(index).__class__.__name__: system.getForce(index)
                  for index in range(system.getNumForces())}

        # Process HarmonicBondForce
        bond_force = forces['HarmonicBondForce']
        for bond_index in range(bond_force.getNumBonds()):
            [particle1, particle2, r, K] = bond_force.getBondParameters(bond_index)
            bonds[particle1].add(particle2)
            bonds[particle2].add(particle1)
        # Process constraints.
        for constraint_index in range(system.getNumConstraints()):
            [particle1, particle2, r] = system.getConstraintParameters(constraint_index)
            bonds[particle1].add(particle2)
            bonds[particle2].add(particle1)

        # TODO: Process CustomBondForce?

        return bonds

    @classmethod
    def _build_alchemical_torsion_list(cls, alchemical_atoms, reference_forces, system):
        """
        Build a list of proper torsion indices that involve any alchemical atom.

        Parameters
        ----------
        alchemical_atoms : set of int
            The set of alchemically modified atoms.
        reference_forces : dict str: force
            A dictionary of cached forces in the system accessible by names.
        system : simtk.openmm.System
            The system.

        Returns
        -------
        torsion_list : list of int
            The list of torsion indices that should be alchemically softened

        """

        # Tabulate all bonds
        bonds = cls._tabulate_bonds(system)

        def is_bonded(i, j):
            if j in bonds[i]:
                return True
            return False

        def is_proper_torsion(i, j, k, l):
            if is_bonded(i, j) and is_bonded(j, k) and is_bonded(k, l):
                return True
            return False

        # Create a list of proper torsions that involve any alchemical atom.
        torsion_list = list()
        force = reference_forces['PeriodicTorsionForce']
        for torsion_index in range(force.getNumTorsions()):
            particle1, particle2, particle3, particle4, periodicity, phase, k = force.getTorsionParameters(torsion_index)
            if set([particle1, particle2, particle3, particle4]).intersection(alchemical_atoms):
                if is_proper_torsion(particle1, particle2, particle3, particle4):
                    torsion_list.append(torsion_index)

        return torsion_list

    @staticmethod
    def _build_alchemical_angle_list(alchemical_atoms, reference_forces, system):
        """
        Build a list of angle indices that involve any alchemical atom.

        Parameters
        ----------
        alchemical_atoms : set of int
            The set of alchemically modified atoms.
        reference_forces : dict str: force
            A dictionary of cached forces in the system accessible by names.
        system : simtk.openmm.System
            The system (unused).

        Returns
        -------
        angle_list : list of int
            The list of angle indices that should be alchemically softened

        """
        angle_list = list()
        force = reference_forces['HarmonicAngleForce']
        for angle_index in range(force.getNumAngles()):
            [particle1, particle2, particle3, theta0, K] = force.getAngleParameters(angle_index)
            if set([particle1, particle2, particle3]).intersection(alchemical_atoms):
                angle_list.append(angle_index)

        return angle_list

    @staticmethod
    def _build_alchemical_bond_list(alchemical_atoms, reference_forces, system):
        """
        Build a list of bond indices that involve any alchemical atom, allowing a list of bonds to override.

        Parameters
        ----------
        alchemical_atoms : set of int
            The set of alchemically modified atoms.
        reference_forces : dict str: force
            A dictionary of cached forces in the system accessible by names.
        system : simtk.openmm.System
            The system (unused).

        Returns
        -------
        bond_list : list of int
            The list of bond indices that should be alchemically softened

        """
        bond_list = list()
        force = reference_forces['HarmonicBondForce']
        for bond_index in range(force.getNumBonds()):
            [particle1, particle2, r, K] = force.getBondParameters(bond_index)
            if set([particle1, particle2]).intersection(alchemical_atoms):
                bond_list.append(bond_index)

        return bond_list

    # -------------------------------------------------------------------------
    # Internal usage: Alchemical forces
    # -------------------------------------------------------------------------

    @staticmethod
    def _alchemically_modify_PeriodicTorsionForce(reference_force, alchemical_region):
        """Create alchemically-modified version of PeriodicTorsionForce.

        Parameters
        ----------
        reference_force : simtk.openmm.PeriodicTorsionForce
            The reference PeriodicTorsionForce to be alchemically modify.
        alchemical_region : AlchemicalRegion
            The alchemical region containing the indices of the torsions to
            alchemically modify.

        Returns
        -------
        force : simtk.openmm.PeriodicTorsionForce
            The force responsible for the non-alchemical torsions.
        custom_force : simtk.openmm.CustomTorsionForce
            The force responsible for the alchemically-softened torsions.
            This will not be present if there are not alchemical torsions
            in alchemical_region.

        """
        # Don't create a force if there are no alchemical torsions.
        if len(alchemical_region.alchemical_torsions) == 0:
            return [copy.deepcopy(reference_force)]

        # Create PeriodicTorsionForce to handle unmodified torsions.
        force = openmm.PeriodicTorsionForce()

        # Create CustomTorsionForce to handle alchemically modified torsions.
        energy_function = "lambda_torsions*k*(1+cos(periodicity*theta-phase))"
        custom_force = openmm.CustomTorsionForce(energy_function)
        custom_force.addGlobalParameter('lambda_torsions', 1.0)
        custom_force.addPerTorsionParameter('periodicity')
        custom_force.addPerTorsionParameter('phase')
        custom_force.addPerTorsionParameter('k')
        # Process reference torsions.
        for torsion_index in range(reference_force.getNumTorsions()):
            # Retrieve parameters.
            particle1, particle2, particle3, particle4, periodicity, phase, k = reference_force.getTorsionParameters(torsion_index)
            # Create torsions.
            if torsion_index in alchemical_region.alchemical_torsions:
                # Alchemically modified torsion.
                custom_force.addTorsion(particle1, particle2, particle3, particle4, [periodicity, phase, k])
            else:
                # Standard torsion.
                force.addTorsion(particle1, particle2, particle3, particle4, periodicity, phase, k)

        return [force, custom_force]

    @staticmethod
    def _alchemically_modify_HarmonicAngleForce(reference_force, alchemical_region):
        """Create alchemically-modified version of HarmonicAngleForce

        Parameters
        ----------
        reference_force : simtk.openmm.HarmonicAngleForce
            The reference HarmonicAngleForce to be alchemically modify.
        alchemical_region : AlchemicalRegion
            The alchemical region containing the indices of the angles to
            alchemically modify.

        Returns
        -------
        force : simtk.openmm.HarmonicAngleForce
            The force responsible for the non-alchemical angles.
        custom_force : simtk.openmm.CustomAngleForce
            The force responsible for the alchemically-softened angles.
            This will not be present if there are not alchemical angles
            in alchemical_region.

        """
        # Don't create a force if there are no alchemical angles.
        if len(alchemical_region.alchemical_angles) == 0:
            return [copy.deepcopy(reference_force)]

        # Create standard HarmonicAngleForce to handle unmodified angles.
        force = openmm.HarmonicAngleForce()

        # Create CustomAngleForce to handle alchemically modified angles.
        energy_function = "lambda_angles*(K/2)*(theta-theta0)^2;"
        custom_force = openmm.CustomAngleForce(energy_function)
        custom_force.addGlobalParameter('lambda_angles', 1.0)
        custom_force.addPerAngleParameter('theta0')
        custom_force.addPerAngleParameter('K')
        # Process reference angles.
        for angle_index in range(reference_force.getNumAngles()):
            # Retrieve parameters.
            [particle1, particle2, particle3, theta0, K] = reference_force.getAngleParameters(angle_index)
            if angle_index in alchemical_region.alchemical_angles:
                # Alchemically modified angle.
                custom_force.addAngle(particle1, particle2, particle3, [theta0, K])
            else:
                # Standard angle.
                force.addAngle(particle1, particle2, particle3, theta0, K)

        return [force, custom_force]

    @staticmethod
    def _alchemically_modify_HarmonicBondForce(reference_force, alchemical_region):
        """Create alchemically-modified version of HarmonicBondForce

        Parameters
        ----------
        reference_force : simtk.openmm.HarmonicBondForce
            The reference HarmonicBondForce to be alchemically modify.
        alchemical_region : AlchemicalRegion
            The alchemical region containing the indices of the bonds to
            alchemically modify.

        Returns
        -------
        force : simtk.openmm.HarmonicBondForce
            The force responsible for the non-alchemical bonds.
        custom_force : simtk.openmm.CustomBondForce
            The force responsible for the alchemically-softened bonds.
            This will not be present if there are not alchemical bonds
            in alchemical_region.

        """
        # Don't create a force if there are no alchemical bonds.
        if len(alchemical_region.alchemical_bonds) == 0:
            return [copy.deepcopy(reference_force)]

        # Create standard HarmonicBondForce to handle unmodified bonds.
        force = openmm.HarmonicBondForce()

        # Create CustomBondForce to handle alchemically modified bonds.
        energy_function = "lambda_bonds*(K/2)*(r-r0)^2;"
        custom_force = openmm.CustomBondForce(energy_function)
        custom_force.addGlobalParameter('lambda_bonds', 1.0)
        custom_force.addPerBondParameter('r0')
        custom_force.addPerBondParameter('K')
        # Process reference bonds.
        for bond_index in range(reference_force.getNumBonds()):
            # Retrieve parameters.
            [particle1, particle2, theta0, K] = reference_force.getBondParameters(bond_index)
            if bond_index in alchemical_region.alchemical_bonds:
                # Alchemically modified bond.
                custom_force.addBond(particle1, particle2, [theta0, K])
            else:
                # Standard bond.
                force.addBond(particle1, particle2, theta0, K)

        return [force, custom_force]

    def _alchemically_modify_NonbondedForce(self, reference_force, alchemical_region):
        """Create alchemically-modified version of NonbondedForce.

        Parameters
        ----------
        reference_force : simtk.openmm.NonbondedForce
            The reference NonbondedForce to be alchemically modify.
        alchemical_region : AlchemicalRegion
            The alchemical region containing the indices of the atoms to
            alchemically modify.

        Returns
        -------
        nonbonded_force : simtk.openmm.NonbondedForce
            The force responsible for interactions and exceptions of non-alchemical atoms.
        aa_sterics_custom_nonbonded_force : simtk.openmm.CustomNonbondedForce
            The force responsible for sterics interactions of alchemical/alchemical atoms.
        aa_electrostatics_custom_nonbonded_force : simtk.openmm.CustomNonbondedForce
            The force responsible for electrostatics interactions of alchemical/alchemical
            atoms.
        na_sterics_custom_nonbonded_force : simtk.openmm.CustomNonbondedForce
            The force responsible for sterics interactions of non-alchemical/alchemical atoms.
        na_electrostatics_custom_nonbonded_force : simtk.openmm.CustomNonbondedForce
            The force responsible for electrostatics interactions of non-alchemical/alchemical
            atoms.
        aa_sterics_custom_bond_force : simtk.openmm.CustomBondForce
            The force responsible for sterics exceptions of alchemical/alchemical atoms.
        aa_electrostatics_custom_bond_force : simtk.openmm.CustomBondForce
            The force responsible for electrostatics exceptions of alchemical/alchemical
            atoms.
        na_sterics_custom_bond_force : simtk.openmm.CustomBondForce
            The force responsible for sterics exceptions of non-alchemical/alchemical atoms.
        na_electrostatics_custom_bond_force : simtk.openmm.CustomBondForce
            The force responsible for electrostatics exceptions of non-alchemical/alchemical
            atoms.

        References
        ----------
        [1] Pham TT and Shirts MR. Identifying low variance pathways for free
        energy calculations of molecular transformations in solution phase.
        JCP 135:034114, 2011. http://dx.doi.org/10.1063/1.3607597

        """
        # TODO Change softcore_beta to a dimensionless scalar to multiply some intrinsic length-scale, like Lennard-Jones alpha.
        # TODO Try using a single, common "reff" effective softcore distance for both Lennard-Jones and Coulomb.

        # Don't create a force if there are no alchemical atoms.
        if len(alchemical_region.alchemical_atoms) == 0:
            return [copy.deepcopy(reference_force)]

        nonbonded_method = reference_force.getNonbondedMethod()

        # --------------------------------------------------
        # Determine energy expression for all custom forces
        # --------------------------------------------------

        # Soft-core Lennard-Jones
        sterics_energy_expression = "U_sterics = (lambda_sterics^softcore_a)*4*epsilon*x*(x-1.0); x = (sigma/reff_sterics)^6;"

        # Electrostatics energy expression for NoCutoff nonbonded method
        nocutoff_electrostatics_energy_expression = "U_electrostatics = (lambda_electrostatics^softcore_d)*ONE_4PI_EPS0*chargeprod/reff_electrostatics;"

        # Electrostatics energy expression will change according to the nonbonded method.
        electrostatics_energy_expression = ""

        # Energy expression for 1,4 electrostatics exceptions can be either electrostatics_energy_expression
        # or nocutoff_electrostatics_energy_expression if self.consistent_exceptions is True or False respectively
        exceptions_electrostatics_energy_expression = ""

        # Select electrostatics functional form based on nonbonded method.
        if nonbonded_method in [openmm.NonbondedForce.NoCutoff]:
            # soft-core Coulomb
            electrostatics_energy_expression += nocutoff_electrostatics_energy_expression
        elif nonbonded_method in [openmm.NonbondedForce.CutoffPeriodic, openmm.NonbondedForce.CutoffNonPeriodic]:
            # reaction-field electrostatics
            epsilon_solvent = reference_force.getReactionFieldDielectric()
            r_cutoff = reference_force.getCutoffDistance()
            electrostatics_energy_expression += "U_electrostatics = (lambda_electrostatics^softcore_d)*ONE_4PI_EPS0*chargeprod*(reff_electrostatics^(-1) + k_rf*reff_electrostatics^2 - c_rf);"
            k_rf = r_cutoff**(-3) * ((epsilon_solvent - 1) / (2*epsilon_solvent + 1))
            c_rf = r_cutoff**(-1) * ((3*epsilon_solvent) / (2*epsilon_solvent + 1))
            electrostatics_energy_expression += "k_rf = %f;" % (k_rf.value_in_unit_system(unit.md_unit_system))
            electrostatics_energy_expression += "c_rf = %f;" % (c_rf.value_in_unit_system(unit.md_unit_system))
        elif nonbonded_method in [openmm.NonbondedForce.PME, openmm.NonbondedForce.Ewald]:
            # Ewald direct-space electrostatics
            [alpha_ewald, nx, ny, nz] = reference_force.getPMEParameters()
            if (alpha_ewald/alpha_ewald.unit) == 0.0:
                # If alpha is 0.0, alpha_ewald is computed by OpenMM from from the error tolerance.
                tol = reference_force.getEwaldErrorTolerance()
                alpha_ewald = (1.0/reference_force.getCutoffDistance()) * np.sqrt(-np.log(2.0*tol))
            electrostatics_energy_expression += "U_electrostatics = (lambda_electrostatics^softcore_d)*ONE_4PI_EPS0*chargeprod*erfc(alpha_ewald*reff_electrostatics)/reff_electrostatics;"
            electrostatics_energy_expression += "alpha_ewald = %f;" % (alpha_ewald.value_in_unit_system(unit.md_unit_system))
            # TODO: Handle reciprocal-space electrostatics for alchemically-modified particles.  These are otherwise neglected.
            # NOTE: There is currently no way to do this in OpenMM.
        else:
            raise Exception("Nonbonded method %s not supported yet." % str(nonbonded_method))

        # Add additional definitions common to all methods.
        sterics_energy_expression += "reff_sterics = sigma*((softcore_alpha*(1.0-lambda_sterics)^softcore_b + (r/sigma)^softcore_c))^(1/softcore_c);" # effective softcore distance for sterics
        reff_electrostatics_expression = "reff_electrostatics = sigma*((softcore_beta*(1.0-lambda_electrostatics)^softcore_e + (r/sigma)^softcore_f))^(1/softcore_f);" # effective softcore distance for electrostatics
        reff_electrostatics_expression += "ONE_4PI_EPS0 = %f;" % ONE_4PI_EPS0 # already in OpenMM units
        electrostatics_energy_expression += reff_electrostatics_expression

        # Define functional form of 1,4 electrostatic exceptions
        if self.consistent_exceptions:
            exceptions_electrostatics_energy_expression += electrostatics_energy_expression
        else:
            exceptions_electrostatics_energy_expression += nocutoff_electrostatics_energy_expression
            exceptions_electrostatics_energy_expression += reff_electrostatics_expression

        # Define mixing rules.
        sterics_mixing_rules = ""
        sterics_mixing_rules += "epsilon = sqrt(epsilon1*epsilon2);" # mixing rule for epsilon
        sterics_mixing_rules += "sigma = 0.5*(sigma1 + sigma2);" # mixing rule for sigma
        electrostatics_mixing_rules = ""
        electrostatics_mixing_rules += "chargeprod = charge1*charge2;" # mixing rule for charges
        electrostatics_mixing_rules += "sigma = 0.5*(sigma1 + sigma2);" # mixing rule for sigma

        # ------------------------------------------------------------
        # Create and configure all forces to add to alchemical system
        # ------------------------------------------------------------

        # Interactions and exceptions will be distributed according to the following table

        # --------------------------------------------------------------------------------------------------
        # FORCE                                    | INTERACTION GROUP                                     |
        # --------------------------------------------------------------------------------------------------
        # nonbonded_force (unmodified)             | all interactions nonalchemical/nonalchemical          |
        #                                          | all exceptions nonalchemical/nonalchemical            |
        # --------------------------------------------------------------------------------------------------
        # aa_sterics_custom_nonbonded_force        | sterics interactions alchemical/alchemical            |
        # --------------------------------------------------------------------------------------------------
        # aa_electrostatics_custom_nonbonded_force | electrostatics interactions alchemical/alchemical     |
        # --------------------------------------------------------------------------------------------------
        # na_sterics_custom_nonbonded_force        | sterics interactions non-alchemical/alchemical        |
        # --------------------------------------------------------------------------------------------------
        # na_electrostatics_custom_nonbonded_force | electrostatics interactions non-alchemical/alchemical |
        # --------------------------------------------------------------------------------------------------
        # aa_sterics_custom_bond_force             | sterics exceptions alchemical/alchemical              |
        # --------------------------------------------------------------------------------------------------
        # aa_electrostatics_custom_bond_force      | electrostatics exceptions alchemical/alchemical       |
        # --------------------------------------------------------------------------------------------------
        # na_sterics_custom_bond_force             | sterics exceptions non-alchemical/alchemical          |
        # --------------------------------------------------------------------------------------------------
        # na_electrostatics_custom_bond_force      | electrostatics exceptions non-alchemical/alchemical   |
        # --------------------------------------------------------------------------------------------------

        # Create a copy of the NonbondedForce to handle particle interactions and
        # 1,4 exceptions between non-alchemical/non-alchemical atoms (nn).
        nonbonded_force = copy.deepcopy(reference_force)

        # Create CustomNonbondedForces to handle sterics particle interactions between
        # non-alchemical/alchemical atoms (na) and alchemical/alchemical atoms (aa). Fix lambda
        # to 1.0 for decoupled interactions
        basic_sterics_expression = "U_sterics;" + sterics_energy_expression + sterics_mixing_rules

        na_sterics_custom_nonbonded_force = openmm.CustomNonbondedForce(basic_sterics_expression)
        na_sterics_custom_nonbonded_force.addGlobalParameter("lambda_sterics", 1.0)
        if alchemical_region.annihilate_sterics:
            aa_sterics_custom_nonbonded_force = openmm.CustomNonbondedForce(basic_sterics_expression)
            aa_sterics_custom_nonbonded_force.addGlobalParameter("lambda_sterics", 1.0)
        else:  # for decoupling fix lambda_sterics to 1.0
            aa_sterics_custom_nonbonded_force = openmm.CustomNonbondedForce(basic_sterics_expression +
                                                                            'lambda_sterics=1.0;')

        # Add parameters and configure CustomNonbondedForces to match reference force
        is_method_periodic = nonbonded_method in [openmm.NonbondedForce.Ewald, openmm.NonbondedForce.PME,
                                                  openmm.NonbondedForce.CutoffPeriodic]

        for force in [na_sterics_custom_nonbonded_force, aa_sterics_custom_nonbonded_force]:
            force.addPerParticleParameter("sigma")  # Lennard-Jones sigma
            force.addPerParticleParameter("epsilon")  # Lennard-Jones epsilon
            force.setUseSwitchingFunction(nonbonded_force.getUseSwitchingFunction())
            force.setCutoffDistance(nonbonded_force.getCutoffDistance())
            force.setSwitchingDistance(nonbonded_force.getSwitchingDistance())
            force.setUseLongRangeCorrection(nonbonded_force.getUseDispersionCorrection())

            if is_method_periodic:
                force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
            else:
                force.setNonbondedMethod(nonbonded_force.getNonbondedMethod())

        # Create CustomNonbondedForces to handle electrostatics particle interactions between
        # non-alchemical/alchemical atoms (na) and alchemical/alchemical atoms (aa). Fix lambda
        # to 1.0 for decoupled interactions
        basic_electrostatics_expression = "U_electrostatics;" + electrostatics_energy_expression +\
                                          electrostatics_mixing_rules

        na_electrostatics_custom_nonbonded_force = openmm.CustomNonbondedForce(basic_electrostatics_expression)
        na_electrostatics_custom_nonbonded_force.addGlobalParameter("lambda_electrostatics", 1.0)
        if alchemical_region.annihilate_electrostatics:
            aa_electrostatics_custom_nonbonded_force = openmm.CustomNonbondedForce(basic_electrostatics_expression)
            aa_electrostatics_custom_nonbonded_force.addGlobalParameter("lambda_electrostatics", 1.0)
        else:  # for decoupling fix lambda_electrostatics to 1.0
            aa_electrostatics_custom_nonbonded_force = openmm.CustomNonbondedForce(basic_electrostatics_expression +
                                                                                   'lambda_electrostatics=1.0;')

        # Add parameters and configure CustomNonbondedForces to match reference force
        for force in [na_electrostatics_custom_nonbonded_force, aa_electrostatics_custom_nonbonded_force]:
            force.addPerParticleParameter("charge")  # partial charge
            force.addPerParticleParameter("sigma")  # Lennard-Jones sigma
            force.setUseSwitchingFunction(False)  # no switch for electrostatics, since NonbondedForce doesn't use it
            force.setCutoffDistance(nonbonded_force.getCutoffDistance())
            force.setUseLongRangeCorrection(False)  # long-range dispersion correction is meaningless for electrostatics

            if is_method_periodic:
                force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
            else:
                force.setNonbondedMethod(nonbonded_force.getNonbondedMethod())

        # Create CustomBondForces to handle sterics 1,4 exceptions interactions between
        # non-alchemical/alchemical atoms (na) and alchemical/alchemical atoms (aa). Fix lambda
        # to 1.0 for decoupled interactions
        basic_sterics_expression = "U_sterics;" + sterics_energy_expression

        na_sterics_custom_bond_force = openmm.CustomBondForce(basic_sterics_expression)
        na_sterics_custom_bond_force.addGlobalParameter("lambda_sterics", 1.0)
        if alchemical_region.annihilate_sterics:
            aa_sterics_custom_bond_force = openmm.CustomBondForce(basic_sterics_expression)
            aa_sterics_custom_bond_force.addGlobalParameter("lambda_sterics", 1.0)
        else:  # for decoupling fix lambda_sterics to 1.0
            aa_sterics_custom_bond_force = openmm.CustomBondForce(basic_sterics_expression + 'lambda_sterics=1.0;')

        for force in [na_sterics_custom_bond_force, aa_sterics_custom_bond_force]:
            force.addPerBondParameter("sigma")  # Lennard-Jones effective sigma
            force.addPerBondParameter("epsilon")  # Lennard-Jones effective epsilon

        # Create CustomBondForces to handle electrostatics 1,4 exceptions interactions between
        # non-alchemical/alchemical atoms (na) and alchemical/alchemical atoms (aa). Fix lambda
        # to 1.0 for decoupled interactions
        if self.consistent_exceptions:
            basic_electrostatics_expression = "U_electrostatics;" + electrostatics_energy_expression
        else:
            basic_electrostatics_expression = "U_electrostatics;" + exceptions_electrostatics_energy_expression

        na_electrostatics_custom_bond_force = openmm.CustomBondForce(basic_electrostatics_expression)
        na_electrostatics_custom_bond_force.addGlobalParameter("lambda_electrostatics", 1.0)
        if alchemical_region.annihilate_electrostatics:
            aa_electrostatics_custom_bond_force = openmm.CustomBondForce(basic_electrostatics_expression)
            aa_electrostatics_custom_bond_force.addGlobalParameter("lambda_electrostatics", 1.0)
        else:  # for decoupling fix lambda_electrostatics to 1.0
            aa_electrostatics_custom_bond_force = openmm.CustomBondForce(basic_electrostatics_expression +
                                                                         'lambda_electrostatics=1.0;')

        # Create CustomBondForce to handle exceptions for electrostatics
        for force in [na_electrostatics_custom_bond_force, aa_electrostatics_custom_bond_force]:
            force.addPerBondParameter("chargeprod")  # charge product
            force.addPerBondParameter("sigma")  # Lennard-Jones effective sigma

        # -------------------------------------------------------------------------------
        # Distribute particle interactions contributions in appropriate nonbonded forces
        # -------------------------------------------------------------------------------

        # Fix any NonbondedForce issues with Lennard-Jones sigma = 0 (epsilon = 0), which should have sigma > 0.
        for particle_index in range(nonbonded_force.getNumParticles()):
            # Retrieve parameters.
            [charge, sigma, epsilon] = nonbonded_force.getParticleParameters(particle_index)
            # Check particle sigma is not zero.
            if (sigma == 0.0 * unit.angstrom):
                logger.warning("particle %d has Lennard-Jones sigma = 0 (charge=%s, sigma=%s, epsilon=%s); setting sigma=1A" % (particle_index, str(charge), str(sigma), str(epsilon)))
                sigma = 1.0 * unit.angstrom
                # Fix it.
                nonbonded_force.setParticleParameters(particle_index, charge, sigma, epsilon)
        for exception_index in range(nonbonded_force.getNumExceptions()):
            # Retrieve parameters.
            [iatom, jatom, chargeprod, sigma, epsilon] = nonbonded_force.getExceptionParameters(exception_index)
            # Check particle sigma is not zero.
            if (sigma == 0.0 * unit.angstrom):
                logger.warning("exception %d has Lennard-Jones sigma = 0 (iatom=%d, jatom=%d, chargeprod=%s, sigma=%s, epsilon=%s); setting sigma=1A" % (exception_index, iatom, jatom, str(chargeprod), str(sigma), str(epsilon)))
                sigma = 1.0 * unit.angstrom
                # Fix it.
                nonbonded_force.setExceptionParameters(exception_index, iatom, jatom, chargeprod, sigma, epsilon)

        # Copy NonbondedForce particle terms for alchemically-modified particles to CustomNonbondedForces.
        # On CUDA, for efficiency reasons, all nonbonded forces (custom and not) must have the same particles.
        for particle_index in range(nonbonded_force.getNumParticles()):
            # Retrieve parameters.
            [charge, sigma, epsilon] = nonbonded_force.getParticleParameters(particle_index)
            if (sigma / unit.angstroms) == 0.0:
                raise Exception('sigma is %s for particle %d; sigma must be positive' % (str(sigma), particle_index))
            na_sterics_custom_nonbonded_force.addParticle([sigma, epsilon])
            aa_sterics_custom_nonbonded_force.addParticle([sigma, epsilon])
            na_electrostatics_custom_nonbonded_force.addParticle([charge, sigma])
            aa_electrostatics_custom_nonbonded_force.addParticle([charge, sigma])

        # Create atom groups.
        alchemical_atomset = alchemical_region.alchemical_atoms
        all_atomset = set(range(reference_force.getNumParticles()))  # all atoms, including alchemical region
        nonalchemical_atomset = all_atomset.difference(alchemical_atomset)

        # Turn off interactions contribution from alchemically-modified particles in unmodified
        # NonbondedForce that will be handled by all other forces
        for particle_index in range(nonbonded_force.getNumParticles()):
            # Retrieve parameters.
            [charge, sigma, epsilon] = nonbonded_force.getParticleParameters(particle_index)
            if particle_index in alchemical_atomset:
                nonbonded_force.setParticleParameters(particle_index, abs(0*charge), sigma, abs(0*epsilon))

        # Restrict interaction evaluation of CustomNonbondedForces to their respective atom groups.
        na_sterics_custom_nonbonded_force.addInteractionGroup(nonalchemical_atomset, alchemical_atomset)
        na_electrostatics_custom_nonbonded_force.addInteractionGroup(nonalchemical_atomset, alchemical_atomset)
        aa_sterics_custom_nonbonded_force.addInteractionGroup(alchemical_atomset, alchemical_atomset)
        aa_electrostatics_custom_nonbonded_force.addInteractionGroup(alchemical_atomset, alchemical_atomset)

        # ---------------------------------------------------------------
        # Distribute exceptions contributions in appropriate bond forces
        # ---------------------------------------------------------------

        # Move all NonbondedForce exception terms for alchemically-modified particles to CustomBondForces.
        for exception_index in range(nonbonded_force.getNumExceptions()):
            # Retrieve parameters.
            [iatom, jatom, chargeprod, sigma, epsilon] = nonbonded_force.getExceptionParameters(exception_index)

            # Exclude this atom pair in CustomNonbondedForces. All nonbonded forces
            # must have the same number of exceptions/exclusions on CUDA platform.
            na_sterics_custom_nonbonded_force.addExclusion(iatom, jatom)
            aa_sterics_custom_nonbonded_force.addExclusion(iatom, jatom)
            na_electrostatics_custom_nonbonded_force.addExclusion(iatom, jatom)
            aa_electrostatics_custom_nonbonded_force.addExclusion(iatom, jatom)

            # Check how many alchemical atoms we have
            both_alchemical = iatom in alchemical_atomset and jatom in alchemical_atomset
            only_one_alchemical = (iatom in alchemical_atomset) != (jatom in alchemical_atomset)

            # Check if this is an exception or an exclusion
            is_exception_epsilon = abs(epsilon.value_in_unit_system(unit.md_unit_system)) > 0.0
            is_exception_chargeprod = abs(chargeprod.value_in_unit_system(unit.md_unit_system)) > 0.0

            # If exception (and not exclusion), add special CustomBondForce terms to
            # handle alchemically-modified Lennard-Jones and electrostatics exceptions
            if both_alchemical:
                if is_exception_epsilon:
                    aa_sterics_custom_bond_force.addBond(iatom, jatom, [sigma, epsilon])
                if is_exception_chargeprod:
                    aa_electrostatics_custom_bond_force.addBond(iatom, jatom, [chargeprod, sigma])
            elif only_one_alchemical:
                if is_exception_epsilon:
                    na_sterics_custom_bond_force.addBond(iatom, jatom, [sigma, epsilon])
                if is_exception_chargeprod:
                    na_electrostatics_custom_bond_force.addBond(iatom, jatom, [chargeprod, sigma])
            # else: both particles are non-alchemical, leave them in the unmodified NonbondedForce

        # Turn off all exception contributions from alchemical atoms in the NonbondedForce
        # modelling non-alchemical atoms only
        for exception_index in range(nonbonded_force.getNumExceptions()):
            [iatom, jatom, chargeprod, sigma, epsilon] = nonbonded_force.getExceptionParameters(exception_index)
            if iatom in alchemical_atomset or jatom in alchemical_atomset:
                nonbonded_force.setExceptionParameters(exception_index, iatom, jatom,
                                                       abs(0.0*chargeprod), sigma, abs(0.0*epsilon))

        # Add global parameters to forces.
        def add_global_parameters(force):
            force.addGlobalParameter('softcore_alpha', alchemical_region.softcore_alpha)
            force.addGlobalParameter('softcore_beta', alchemical_region.softcore_beta)
            force.addGlobalParameter('softcore_a', alchemical_region.softcore_a)
            force.addGlobalParameter('softcore_b', alchemical_region.softcore_b)
            force.addGlobalParameter('softcore_c', alchemical_region.softcore_c)
            force.addGlobalParameter('softcore_d', alchemical_region.softcore_d)
            force.addGlobalParameter('softcore_e', alchemical_region.softcore_e)
            force.addGlobalParameter('softcore_f', alchemical_region.softcore_f)

        all_custom_forces = [na_sterics_custom_nonbonded_force, na_electrostatics_custom_nonbonded_force,
                             aa_sterics_custom_nonbonded_force, aa_electrostatics_custom_nonbonded_force,
                             na_sterics_custom_bond_force, na_electrostatics_custom_bond_force,
                             aa_sterics_custom_bond_force, aa_electrostatics_custom_bond_force]
        for force in all_custom_forces:
            add_global_parameters(force)

        return [nonbonded_force] + all_custom_forces

    def _alchemically_modify_AmoebaMultipoleForce(self, reference_force, alchemical_region):
        raise Exception("Not implemented; needs CustomMultipleForce")

    def _alchemically_modify_AmoebaVdwForce(self, reference_force, alchemical_region):
        """Create alchemically-modified version of AmoebaVdwForce.

        Not implemented.

        TODO
        ----
        * Supported periodic boundary conditions need to be handled correctly.
        * Exceptions/exclusions need to be dealt with.

        """
        # This feature is incompletely implemented, so raise an exception.
        raise NotImplemented('Alchemical modification of Amoeba VdW Forces is not supported.')

        # Softcore Halgren potential from Eq. 3 of
        # Shi, Y., Jiao, D., Schnieders, M.J., and Ren, P. (2009). Trypsin-ligand binding free
        # energy calculation with AMOEBA. Conf Proc IEEE Eng Med Biol Soc 2009, 2328-2331.
        energy_expression = 'lambda^5 * epsilon * (1.07^7 / (0.7*(1-lambda)^2+(rho+0.07)^7)) * (1.12 / (0.7*(1-lambda)^2 + rho^7 + 0.12) - 2);'
        energy_expression += 'epsilon = 4*epsilon1*epsilon2 / (sqrt(epsilon1) + sqrt(epsilon2))^2;'
        energy_expression += 'rho = r / R0;'
        energy_expression += 'R0 = (R01^3 + R02^3) / (R01^2 + R02^2);'
        energy_expression += 'lambda = vdw_lambda * (ligand1*(1-ligand2) + ligand2*(1-ligand1)) + ligand1*ligand2;'
        energy_expression += 'vdw_lambda = %f;' % vdw_lambda

        softcore_force = openmm.CustomNonbondedForce(energy_expression)
        softcore_force.addPerParticleParameter('epsilon')
        softcore_force.addPerParticleParameter('R0')
        softcore_force.addPerParticleParameter('ligand')

        for particle_index in range(system.getNumParticles()):
            # Retrieve parameters from vdW force.
            [parentIndex, sigma, epsilon, reductionFactor] = force.getParticleParameters(particle_index)
            # Add parameters to CustomNonbondedForce.
            if particle_index in alchemical_region.alchemical_atoms:
                softcore_force.addParticle([epsilon, sigma, 1])
            else:
                softcore_force.addParticle([epsilon, sigma, 0])

            # Deal with exclusions.
            excluded_atoms = force.getParticleExclusions(particle_index)
            for jatom in excluded_atoms:
                if (particle_index < jatom):
                    softcore_force.addExclusion(particle_index, jatom)

        # Make sure periodic boundary conditions are treated the same way.
        # TODO: Handle PBC correctly.
        softcore_force.setNonbondedMethod( openmm.CustomNonbondedForce.CutoffPeriodic )
        softcore_force.setCutoffDistance( force.getCutoff() )

        # Turn off vdW interactions for alchemically-modified atoms.
        for particle_index in ligand_atoms:
            # Retrieve parameters.
            [parentIndex, sigma, epsilon, reductionFactor] = force.getParticleParameters(particle_index)
            epsilon = 1.0e-6 * epsilon # TODO: For some reason, we cannot set epsilon to 0.
            force.setParticleParameters(particle_index, parentIndex, sigma, epsilon, reductionFactor)

        # Deal with exceptions here.
        # TODO

        return [softcore_force]

    @staticmethod
    def _alchemically_modify_GBSAOBCForce(reference_force, alchemical_region, sasa_model='ACE'):
        """Create alchemically-modified version of GBSAOBCForce.

        Parameters
        ----------
        reference_force : simtk.openmm.GBSAOBCForce
            The reference GBSAOBCForce to be alchemically modify.
        alchemical_region : AlchemicalRegion
            The alchemical region containing the indices of the atoms to
            alchemically modify.
        sasa_model : str, optional, default='ACE'
            Solvent accessible surface area model.

        Returns
        -------
        custom_force : simtk.openmm.CustomGBForce
            The alchemical version of the reference force.

        TODO
        ----
        * Add support for all types of GBSA forces supported by OpenMM.
        * Can we more generally modify any CustomGBSAForce?

        """
        # TODO make sasa_model a Factory attribute?
        custom_force = openmm.CustomGBForce()

        # Add per-particle parameters.
        custom_force.addGlobalParameter("lambda_electrostatics", 1.0);
        custom_force.addPerParticleParameter("charge");
        custom_force.addPerParticleParameter("radius");
        custom_force.addPerParticleParameter("scale");
        custom_force.addPerParticleParameter("alchemical");

        # Set nonbonded method.
        custom_force.setNonbondedMethod(reference_force.getNonbondedMethod())
        custom_force.setCutoffDistance(reference_force.getCutoffDistance())

        # Add global parameters.
        custom_force.addGlobalParameter("solventDielectric", reference_force.getSolventDielectric())
        custom_force.addGlobalParameter("soluteDielectric", reference_force.getSoluteDielectric())
        custom_force.addGlobalParameter("offset", 0.009)

        custom_force.addComputedValue("I",  "(lambda_electrostatics*alchemical2 + (1-alchemical2))*step(r+sr2-or1)*0.5*(1/L-1/U+0.25*(r-sr2^2/r)*(1/(U^2)-1/(L^2))+0.5*log(L/U)/r);"
                                "U=r+sr2;"
                                "L=max(or1, D);"
                                "D=abs(r-sr2);"
                                "sr2 = scale2*or2;"
                                "or1 = radius1-offset; or2 = radius2-offset", openmm.CustomGBForce.ParticlePairNoExclusions)

        custom_force.addComputedValue("B", "1/(1/or-tanh(psi-0.8*psi^2+4.85*psi^3)/radius);"
                                  "psi=I*or; or=radius-offset", openmm.CustomGBForce.SingleParticle)

        custom_force.addEnergyTerm("-0.5*138.935485*(1/soluteDielectric-1/solventDielectric)*(lambda_electrostatics*alchemical+(1-alchemical))*charge^2/B", openmm.CustomGBForce.SingleParticle)
        if sasa_model == 'ACE':
            custom_force.addEnergyTerm("(lambda_electrostatics*alchemical+(1-alchemical))*28.3919551*(radius+0.14)^2*(radius/B)^6", openmm.CustomGBForce.SingleParticle)

        custom_force.addEnergyTerm("-138.935485*(1/soluteDielectric-1/solventDielectric)*(lambda_electrostatics*alchemical1+(1-alchemical1))*charge1*(lambda_electrostatics*alchemical2+(1-alchemical2))*charge2/f;"
                             "f=sqrt(r^2+B1*B2*exp(-r^2/(4*B1*B2)))", openmm.CustomGBForce.ParticlePairNoExclusions);

        # Add particle parameters.
        for particle_index in range(reference_force.getNumParticles()):
            # Retrieve parameters.
            [charge, radius, scaling_factor] = reference_force.getParticleParameters(particle_index)
            # Set particle parameters.
            if particle_index in alchemical_region.alchemical_atoms:
                parameters = [charge, radius, scaling_factor, 1.0]
            else:
                parameters = [charge, radius, scaling_factor, 0.0]
            custom_force.addParticle(parameters)

        return [custom_force]

    # -------------------------------------------------------------------------
    # Internal usage: Infer force labels
    # -------------------------------------------------------------------------

    @staticmethod
    def _find_force_components(alchemical_system):
        """Return force labels and indices for each force."""
        def add_label(label, index):
            assert label not in force_labels
            force_labels[label] = index

        def check_parameter(custom_force, parameter):
            for parameter_id in range(custom_force.getNumGlobalParameters()):
                parameter_name = custom_force.getGlobalParameterName(parameter_id)
                if parameter == parameter_name:
                    return True
            return False

        def check_function(custom_force, parameter):
            energy_function = custom_force.getEnergyFunction()
            return parameter in energy_function

        force_labels = {}
        nonbonded_forces = []
        sterics_bond_forces = []
        electro_bond_forces = []

        # We save CustomBondForces and CustomNonbondedForces used for nonbonded
        # forces and exceptions to distinguish them later
        for force_index, force in enumerate(alchemical_system.getForces()):
            if isinstance(force, openmm.CustomAngleForce) and check_parameter(force, 'lambda_angles'):
                add_label('alchemically modified HarmonicAngleForce', force_index)
            elif isinstance(force, openmm.CustomBondForce) and check_parameter(force, 'lambda_bonds'):
                add_label('alchemically modified HarmonicBondForce', force_index)
            elif isinstance(force, openmm.CustomTorsionForce) and check_parameter(force, 'lambda_torsions'):
                add_label('alchemically modified PeriodicTorsionForce', force_index)
            elif isinstance(force, openmm.CustomGBForce) and check_parameter(force, 'lambda_electrostatics'):
                add_label('alchemically modified GBSAOBCForce', force_index)
            elif isinstance(force, openmm.CustomBondForce) and check_function(force, 'lambda'):
                if check_function(force, 'lambda_sterics'):
                    sterics_bond_forces.append([force_index, force])
                else:
                    electro_bond_forces.append([force_index, force])
            elif isinstance(force, openmm.CustomNonbondedForce) and check_function(force, 'lambda'):
                if check_function(force, 'lambda_sterics'):
                    nonbonded_forces.append(['sterics', force_index, force])
                else:
                    nonbonded_forces.append(['electrostatics', force_index, force])
            else:
                add_label('unmodified ' + force.__class__.__name__, force_index)

        # Differentiate between na/aa nonbonded forces.
        for force_type, force_index, force in nonbonded_forces:
            label = 'alchemically modified NonbondedForce for {}alchemical/alchemical ' + force_type
            interacting_atoms, alchemical_atoms = force.getInteractionGroupParameters(0)
            if interacting_atoms == alchemical_atoms:  # alchemical-alchemical atoms
                add_label(label.format(''), force_index)
            else:
                add_label(label.format('non-'), force_index)

        # Differentiate between na/aa bond forces for exceptions.
        for force_type, bond_forces in [('sterics', sterics_bond_forces), ('electrostatics', electro_bond_forces)]:
            assert len(bond_forces) == 2
            label = 'alchemically modified BondForce for {}alchemical/alchemical ' + force_type + ' exceptions'

            # Sort forces by number of bonds.
            bond_forces = sorted(bond_forces, key=lambda x: x[1].getNumBonds())
            (force_index1, force1), (force_index2, force2) = bond_forces

            # Check if both define their parameters (with decoupling the lambda
            # parameter doesn't exist in the alchemical-alchemical force)
            parameter_name = 'lambda_' + force_type
            if check_parameter(force1, parameter_name) != check_parameter(force2, parameter_name):
                if check_parameter(force1, parameter_name):
                    add_label(label.format('non-'), force_index1)
                    add_label(label.format(''), force_index2)
                else:
                    add_label(label.format(''), force_index1)
                    add_label(label.format('non-'), force_index2)

            # If they are both empty they are identical and any label works.
            elif force1.getNumBonds() == 0:
                add_label(label.format(''), force_index1)
                add_label(label.format('non-'), force_index2)

            # We check that the bond atoms are both alchemical or not.
            else:
                atom_i, atom_j = force1.getBondParameters(0)
                both_alchemical = atom_i in alchemical_atoms and atom_j in alchemical_atoms
                if both_alchemical:
                    add_label(label.format(''), force_index1)
                    add_label(label.format('non-'), force_index2)
                else:
                    add_label(label.format('non-'), force_index1)
                    add_label(label.format(''), force_index2)

        return force_labels


if __name__ == '__main__':
    import doctest
    doctest.testmod()
    # doctest.run_docstring_examples(AlchemicalFunction, globals())
