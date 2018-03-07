#!/usr/bin/python

"""
Alchemical state storage objects

"""

# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import copy
import logging
import collections

import numpy as np
from simtk import openmm, unit

from openmmtools import states, forcefactories, utils
from openmmtools.constants import ONE_4PI_EPS0

logger = logging.getLogger(__name__)

# =============================================================================
# INTERNAL-USAGE CONSTANTS
# =============================================================================

_UPDATE_ALCHEMICAL_CHARGES_PARAMETER = '_update_alchemical_charges'
_UPDATE_ALCHEMICAL_CHARGES_PARAMETER_IDX = 1

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
    update_alchemical_charges : bool, optional
        If True, ``lambda_electrostatics`` changes in alchemical systems
        that use exact treatment of PME electrostatics will be considered
        incompatible. This means that a new ``Context`` will be required
        for each `lambda_electrostatics`` state.

    Attributes
    ----------
    lambda_sterics
    lambda_electrostatics
    lambda_bonds
    lambda_angles
    lambda_torsions
    update_alchemical_charges

    Examples
    --------
    Create an alchemically modified system.

    >>> from openmmtools import testsystems
    >>> factory = AbsoluteAlchemicalFactory(consistent_exceptions=False)
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
        for force, parameter_name, parameter_id in cls._get_system_lambda_parameters(
                system, other_parameters={_UPDATE_ALCHEMICAL_CHARGES_PARAMETER}):
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

        # Handle the update parameters flag.
        update_alchemical_charges = bool(alchemical_parameters.pop(_UPDATE_ALCHEMICAL_CHARGES_PARAMETER,
                                                                   cls._UPDATE_ALCHEMICAL_CHARGES_DEFAULT))

        # Check that the system is alchemical.
        if len(alchemical_parameters) == 0:
            raise AlchemicalStateError('System has no lambda parameters.')

        # Create and return the AlchemicalState.
        return AlchemicalState(update_alchemical_charges=update_alchemical_charges,
                               **alchemical_parameters)

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
        self._set_alchemical_parameters(new_value, exclusions=frozenset())

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
        forbidden_variable_names = set(self._parameters)
        forbidden_variable_names.add(_UPDATE_ALCHEMICAL_CHARGES_PARAMETER)
        if variable_name in forbidden_variable_names:
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
        serialization = dict(
            parameters={},
            alchemical_variables={},
            update_alchemical_charges=self.update_alchemical_charges
        )

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
        update_alchemical_charges = serialization.get('update_alchemical_charges', True)
        alchemical_functions = dict()

        # Temporarily store alchemical functions.
        for parameter_name, value in parameters.items():
            if isinstance(value, str):
                alchemical_functions[parameter_name] = value
                parameters[parameter_name] = None

        # Initialize parameters and add all alchemical variables.
        self._initialize(update_alchemical_charges=update_alchemical_charges,
                         **parameters)
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
        self._apply_to_system(system, set_update_charges_flag=True)

    def check_system_consistency(self, system):
        """Check if the system is in this alchemical state.

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
        has_lambda_electrostatics_changed = False
        context_parameters = context.getParameters()

        # Set lambda parameters in Context.
        for parameter_name in self._parameters:
            parameter_value = getattr(self, parameter_name)
            if parameter_value is None:
                # Check that Context does not have this parameter.
                if parameter_name in context_parameters:
                    err_msg = 'Context has parameter {} which is undefined in this state'
                    raise AlchemicalStateError(err_msg.format(parameter_name))
                continue
            try:
                # If lambda_electrostatics, first check if we're changing it for later.
                # This avoids us to loop through the System forces if we don't need to
                # set the NonbondedForce charges.
                if parameter_name == 'lambda_electrostatics':
                    old_parameter_value = context_parameters[parameter_name]
                    has_lambda_electrostatics_changed = (has_lambda_electrostatics_changed or
                                                         parameter_value != old_parameter_value)
                context.setParameter(parameter_name, parameter_value)
            except Exception:
                err_msg = 'Could not find parameter {} in context'
                raise AlchemicalStateError(err_msg.format(parameter_name))

        # Handle lambda_electrostatics changes with exact PME electrostatic treatment.
        # If the context doesn't use exact PME electrostatics, or if lambda_electrostatics
        # hasn't changed, we don't need to do anything.
        if (_UPDATE_ALCHEMICAL_CHARGES_PARAMETER not in context_parameters or
                not has_lambda_electrostatics_changed):
            return

        # Find exact PME treatment key force objects.
        original_charges_force, nonbonded_force = self._find_exact_pme_forces(context.getSystem())

        # Quick checks for compatibility.
        context_charge_update = bool(context_parameters[_UPDATE_ALCHEMICAL_CHARGES_PARAMETER])
        if not (context_charge_update and self.update_alchemical_charges):
            err_msg = 'Attempted to set the alchemical state of an incompatible Context.'
            raise AlchemicalStateError(err_msg)

        # Write NonbondedForce charges
        self._set_exact_pme_charges(original_charges_force, nonbonded_force)
        nonbonded_force.updateParametersInContext(context)

    def _standardize_system(self, system, set_lambda_electrostatics=False):
        """Standardize the given system.

        Set all global lambda parameters of the system to 1.0.

        Parameters
        ----------
        system : simtk.openmm.System
            The system to standardize.
        set_lambda_electrostatics : bool, optional
            Whether to set the lambda electrostatics of this system or not.

        Raises
        ------
        AlchemicalStateError
            If the system is not consistent with this state.

        """
        alchemical_state = AlchemicalState.from_system(system)
        # If this system uses exact PME treatment and update_alchemical_charges
        # is enabled, we don't want to set lambda_electrostatics.
        if self.update_alchemical_charges:
            exclusions = frozenset()
        else:
            original_charges_force = alchemical_state._find_exact_pme_forces(system, original_charges_only=True)
            if original_charges_force is not None:
                exclusions = {'lambda_electrostatics'}
            else:
                exclusions = frozenset()
        alchemical_state._set_alchemical_parameters(1.0, exclusions=exclusions)

        if set_lambda_electrostatics:
            alchemical_state.lambda_electrostatics = self.lambda_electrostatics

        # We don't want to overwrite the update_alchemical_charges flag as
        # states with different settings must be incompatible.
        alchemical_state._apply_to_system(system, set_update_charges_flag=False)

    def _on_setattr(self, standard_system, attribute_name):
        """Check if the standard system needs changes after a state attribute is set.

        Parameters
        ----------
        standard_system : simtk.openmm.System
            The standard system before setting the attribute.
        attribute_name : str
            The name of the attribute that has just been set or retrieved.

        Returns
        -------
        need_changes : bool
            True if the standard system has to be updated, False if no change
            occurred.

        """
        need_changes = False

        # The standard_system changes with update_alchemical_charges
        # if the system uses exact PME treatment.
        if attribute_name == 'update_alchemical_charges':
            original_charges_force = self._find_exact_pme_forces(standard_system, original_charges_only=True)
            if original_charges_force is not None:
                old_update_charge_parameter = bool(original_charges_force.getGlobalParameterDefaultValue(
                    _UPDATE_ALCHEMICAL_CHARGES_PARAMETER_IDX))
                need_changes = old_update_charge_parameter != self.update_alchemical_charges

        # If we are not allowed to update_alchemical_charges is off and
        # we change lambda_electrostatics we also change the compatibility.
        elif self.update_alchemical_charges is False and attribute_name == 'lambda_electrostatics':
            # Look for old value of lambda_electrostatics.
            for force, parameter_name, parameter_idx in self._get_system_lambda_parameters(standard_system):
                if parameter_name == 'lambda_electrostatics':
                    break
            old_lambda_electrostatics = force.getGlobalParameterDefaultValue(parameter_idx)
            need_changes = old_lambda_electrostatics != self.lambda_electrostatics

        return need_changes

    def _find_force_groups_to_update(self, context, current_context_state, memo):
        """Find the force groups whose energy must be recomputed after applying self.

        Parameters
        ----------
        context : Context
            The context, currently in `current_context_state`, that will
            be moved to this state.
        current_context_state : ThermodynamicState
            The full thermodynamic state of the given context. This is
            guaranteed to be compatible with self.
        memo : dict
            A dictionary that can be used by the state for memoization
            to speed up consecutive calls on the same context.

        Returns
        -------
        force_groups_to_update : set of int
            The indices of the force groups whose energy must be computed
            again after applying this state, assuming the context to be in
            `current_context_state`.
        """
        # Cache information about system force groups.
        if len(memo) == 0:
            parameters_found = set()
            system = context.getSystem()
            for force, parameter_name, _ in self._get_system_lambda_parameters(system):
                if parameter_name not in parameters_found:
                    parameters_found.add(parameter_name)
                    # Keep track of valid lambdas only.
                    if self._parameters[parameter_name] is not None:
                        memo[parameter_name] = force.getForceGroup()
                    # Break the loop if we have found all the parameters.
                    if len(parameters_found) == len(self._parameters):
                        break

        # Find lambda parameters that will change.
        force_groups_to_update = set()
        for parameter_name, force_group in memo.items():
            self_parameter_value = getattr(self, parameter_name)
            current_parameter_value = getattr(current_context_state, parameter_name)
            if self_parameter_value != current_parameter_value:
                force_groups_to_update.add(force_group)
        return force_groups_to_update

    # -------------------------------------------------------------------------
    # Internal-usage
    # -------------------------------------------------------------------------

    _UPDATE_ALCHEMICAL_CHARGES_DEFAULT = True

    def _initialize(self, update_alchemical_charges=_UPDATE_ALCHEMICAL_CHARGES_DEFAULT,
                    **kwargs):
        """Initialize the alchemical state."""
        self._alchemical_variables = {}
        self.update_alchemical_charges = update_alchemical_charges

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

    def _apply_to_system(self, system, set_update_charges_flag):
        """Set the alchemical state of the system to this.

        Raises
        ------
        AlchemicalStateError
            If the system does not have the required lambda global variables.

        """
        has_lambda_electrostatics_changed = False
        parameters_applied = set()
        for force, parameter_name, parameter_id in self._get_system_lambda_parameters(system):
            parameter_value = getattr(self, parameter_name)
            if parameter_value is None:
                err_msg = 'The system parameter {} is not defined in this state.'
                raise AlchemicalStateError(err_msg.format(parameter_name))
            else:
                # If lambda_electrostatics, first check if we're changing it for later.
                # This avoids us to loop through the System forces if we don't need to
                # set the NonbondedForce charges.
                if parameter_name == 'lambda_electrostatics':
                    old_parameter_value = force.getGlobalParameterDefaultValue(parameter_id)
                    has_lambda_electrostatics_changed = (has_lambda_electrostatics_changed or
                                                         parameter_value != old_parameter_value)
                parameters_applied.add(parameter_name)
                force.setGlobalParameterDefaultValue(parameter_id, parameter_value)

        # Check that we set all the defined parameters.
        for parameter_name in self._get_supported_parameters():
            if (self._parameters[parameter_name] is not None and
                    parameter_name not in parameters_applied):
                err_msg = 'Could not find parameter {} in the system'
                raise AlchemicalStateError(err_msg.format(parameter_name))

        # Nothing else to do if we don't need to modify the exact PME forces.
        if not (has_lambda_electrostatics_changed or set_update_charges_flag):
            return

        # Loop through system and retrieve exact PME forces.
        original_charges_force, nonbonded_force = self._find_exact_pme_forces(system)

        # Write NonbondedForce charges if PME is treated exactly.
        if has_lambda_electrostatics_changed:
            self._set_exact_pme_charges(original_charges_force, nonbonded_force)

        # Flag if updateParametersInContext is allowed.
        if set_update_charges_flag:
            self._set_force_update_charge_parameter(original_charges_force)

    def _set_force_update_charge_parameter(self, original_charges_force):
        """Set the global parameter that controls the charges updates."""
        if original_charges_force is None:
            return

        parameter_idx = _UPDATE_ALCHEMICAL_CHARGES_PARAMETER_IDX  # Shortcut.
        if self.update_alchemical_charges:
            original_charges_force.setGlobalParameterDefaultValue(parameter_idx, 1)
        else:
            original_charges_force.setGlobalParameterDefaultValue(parameter_idx, 0)

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
    def _get_system_lambda_parameters(cls, system, other_parameters=frozenset()):
        """Yields the supported lambda parameters in the system.

        Yields
        ------
        A tuple force, parameter_name, parameter_index for each supported
        lambda parameter.

        """
        supported_parameters = cls._get_supported_parameters()
        searched_parameters = supported_parameters.union(other_parameters)

        # Retrieve all the forces with global supported parameters.
        for force_index in range(system.getNumForces()):
            force = system.getForce(force_index)
            try:
                n_global_parameters = force.getNumGlobalParameters()
            except AttributeError:
                continue
            for parameter_id in range(n_global_parameters):
                parameter_name = force.getGlobalParameterName(parameter_id)
                if parameter_name in searched_parameters:
                    yield force, parameter_name, parameter_id

    def _set_alchemical_parameters(self, new_value, exclusions):
        """Set all defined parameters to the given value.

        The undefined parameters (i.e. those being set to None) remain
        undefined.

        Parameters
        ----------
        new_value : float
            The new value for all defined parameters.
        exclusions : set
            The lambda parameters not to set.

        """
        for parameter_name in self._parameters:
            if parameter_name not in exclusions and self._parameters[parameter_name] is not None:
                setattr(self, parameter_name, new_value)

    @classmethod
    def _find_exact_pme_forces(cls, system, original_charges_only=False):
        """Return the NonbondedForce and the CustomNonbondedForce with the original charges."""
        original_charges_force = None
        nonbonded_force = None
        n_found = 0
        for force_idx, force in enumerate(system.getForces()):
            if (isinstance(force, openmm.CustomNonbondedForce) and
                        force.getEnergyFunction() == '0.0;' and
                        force.getGlobalParameterName(0) == 'lambda_electrostatics'):
                original_charges_force = force
                if original_charges_only:
                    break
                n_found += 1
            elif isinstance(force, openmm.NonbondedForce):
                nonbonded_force = force
                n_found += 1
            if n_found == 2:
                break
        if original_charges_only:
            return original_charges_force
        return original_charges_force, nonbonded_force

    def _set_exact_pme_charges(self, original_charges_force, nonbonded_force):
        """Set the NonbondedForce charges from the original value and lambda_electrostatics."""
        # If we don't treat PME exactly, we don't need to set the charges.
        if original_charges_force is None:
            return

        # Set alchemical atoms charges.
        lambda_electrostatics = self.lambda_electrostatics
        _, alchemical_atoms = original_charges_force.getInteractionGroupParameters(0)
        for atom_idx in alchemical_atoms:
            charge, sigma, epsilon = nonbonded_force.getParticleParameters(atom_idx)
            original_charge = original_charges_force.getParticleParameters(atom_idx)[0]
            charge = lambda_electrostatics * original_charge
            nonbonded_force.setParticleParameters(atom_idx, charge, sigma, epsilon)
