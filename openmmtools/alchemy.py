#!/usr/bin/python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Alchemical factory for free energy calculations that operates directly on OpenMM System objects.

DESCRIPTION

This module contains enumerative factories for generating alchemically-modified System objects
usable for the calculation of free energy differences of hydration or ligand binding.

* `AbsoluteAlchemicalFactory` uses fused elecrostatic and steric alchemical modifications.

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
# - Add support for arbitrary softcore reprogramming of all Custom*Force classes


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

    def __init__(self, region_name = None, **kwargs):
        #self.RegionName = region_name
        self._initialize(region_name, **kwargs)

    @classmethod
    def from_system(cls, system, region_name = None):
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
        #if region_name == None:
        #    region_name = 'zero'
        alchemical_parameters = {}
        for force, parameter_name, parameter_id in cls._get_system_lambda_parameters(
                system, other_parameters={_UPDATE_ALCHEMICAL_CHARGES_PARAMETER}):
            parameter_value = force.getGlobalParameterDefaultValue(parameter_id)
            #name = parameter_name.split('_')[2]
            #print(force, parameter_name, parameter_id, parameter_value)

            # Check that we haven't already found
            # the parameter with a different value.
            if region_name in parameter_name:
                if parameter_name in alchemical_parameters:
                    if alchemical_parameters[parameter_name] != parameter_value:
                        err_msg = ('Parameter {} has been found in the force {} with two values: '
                                   '{} and {}').format(parameter_name, force.__class__.__name__,
                                                       parameter_value, alchemical_parameters[parameter_name])
                        raise AlchemicalStateError(err_msg)
                else:
                    alchemical_parameters[parameter_name] = parameter_value
            else:
                pass

        # Handle the update parameters flag.
        update_alchemical_charges = bool(alchemical_parameters.pop(_UPDATE_ALCHEMICAL_CHARGES_PARAMETER,
                                                                   cls._UPDATE_ALCHEMICAL_CHARGES_DEFAULT))

        # Check that the system is alchemical.
        if len(alchemical_parameters) == 0:
            raise AlchemicalStateError('System has no lambda parameters.')

        # Create and return the AlchemicalState.
        return AlchemicalState(region_name = region_name, update_alchemical_charges=update_alchemical_charges,
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

    lambda_sterics_zero = _LambdaProperty('lambda_sterics_zero')
    lambda_electrostatics_zero = _LambdaProperty('lambda_electrostatics_zero')
    lambda_sterics_one = _LambdaProperty('lambda_sterics_one')
    lambda_electrostatics_one = _LambdaProperty('lambda_electrostatics_one')
    lambda_sterics_two = _LambdaProperty('lambda_sterics_two')
    lambda_electrostatics_two = _LambdaProperty('lambda_electrostatics_two')

    lambda_torsions_zero = _LambdaProperty('lambda_torsions_zero')
    lambda_torsions_one = _LambdaProperty('lambda_torsions_one')
    lambda_torsions_two = _LambdaProperty('lambda_torsions_two')

    lambda_bonds_zero = _LambdaProperty('lambda_bonds_zero')
    lambda_bonds_one = _LambdaProperty('lambda_bonds_one')
    lambda_bonds_two = _LambdaProperty('lambda_bonds_two')

    lambda_angles_zero = _LambdaProperty('lambda_angles_zero')
    lambda_angles_one = _LambdaProperty('lambda_angles_one')
    lambda_angles_two = _LambdaProperty('lambda_angles_two')

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
            update_alchemical_charges=self.update_alchemical_charges,
            region_name = None
        )
        # Copy parameters and convert AlchemicalFunctions to string expressions.
        serialization['region_name'] = getattr(self, 'region_name')
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
        region_name = serialization['region_name']
        # New attribute in OpenMMTools 0.14.0.
        update_alchemical_charges = serialization.get('update_alchemical_charges', True)
        alchemical_functions = dict()

        # Temporarily store alchemical functions.
        for parameter_name, value in parameters.items():
            if isinstance(value, str):
                alchemical_functions[parameter_name] = value
                parameters[parameter_name] = None

        # Initialize parameters and add all alchemical variables.
        self._initialize(region_name = region_name, update_alchemical_charges=update_alchemical_charges,
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
        system_alchemical_state = AlchemicalState.from_system(system, self.region_name)

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
        region_name = self.region_name
        #if region_name == None:
        #    region_name = 'zero'
        # Set lambda parameters in Context.
        for parameter_name in self._parameters:
            if region_name in parameter_name:
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
                    if 'electrostatics' in parameter_name:
                        elec_region = parameter_name
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
        original_charges_force, nonbonded_force = self._find_exact_pme_forces(context.getSystem(), region_name)
        if original_charges_force != None:
            lambda_electrostatics = getattr(self, elec_region)
        else:
            lambda_electrostatics = None

        # Quick checks for compatibility.
        context_charge_update = bool(context_parameters[_UPDATE_ALCHEMICAL_CHARGES_PARAMETER])
        if not (context_charge_update and self.update_alchemical_charges):
            err_msg = 'Attempted to set the alchemical state of an incompatible Context.'
            raise AlchemicalStateError(err_msg)

        # Write NonbondedForce charges
        self._set_exact_pme_charges(original_charges_force, nonbonded_force, lambda_electrostatics)
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
        alchemical_state = AlchemicalState.from_system(system, self.region_name)
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
            for force, parameter_name in self._get_system_lambda_parameters(system):
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

    def _initialize(self, region_name = None , update_alchemical_charges=_UPDATE_ALCHEMICAL_CHARGES_DEFAULT,
                    **kwargs):
        self.region_name = region_name
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
        setattr(self, 'region_name', region_name)
        for parameter_name, value in kwargs.items():
            setattr(self, parameter_name, value)

    def _apply_to_system(self, system, set_update_charges_flag):
        """Set the alchemical state of the system to this.

        Raises
        ------
        AlchemicalStateError
            If the system does not have the required lambda global variables.

        """
        region_name = self.region_name
        #if region_name == None:
        #    region_name = 'zero'
        has_lambda_electrostatics_changed = False
        parameters_applied = set()
        for force, parameter_name, parameter_id in self._get_system_lambda_parameters(system):
            parameter_value = getattr(self, parameter_name)
            if region_name in parameter_name:
                if parameter_value == None:
                    err_msg = 'The system parameter {} is not defined in this state.'
                    raise AlchemicalStateError(err_msg.format(parameter_name))
                else:
                    # If lambda_electrostatics, first check if we're changing it for later.
                    # This avoids us to loop through the System forces if we don't need to
                    # set the NonbondedForce charges.
                    if 'electrostatics' in parameter_name:
                        elec_region = parameter_name
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
        original_charges_force, nonbonded_force = self._find_exact_pme_forces(system, region_name)
        if original_charges_force != None:
            lambda_electrostatics = getattr(self, elec_region)
        else:
            lambda_electrostatics = None

        # Write NonbondedForce charges if PME is treated exactly.
        if has_lambda_electrostatics_changed:
            self._set_exact_pme_charges(original_charges_force, nonbonded_force, lambda_electrostatics)

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
    def _find_exact_pme_forces(cls, system, region_name, original_charges_only=False):
        """Return the NonbondedForce and the CustomNonbondedForce with the original charges."""
        original_charges_force = None
        nonbonded_force = None
        n_found = 0
        for force_idx, force in enumerate(system.getForces()):
            if (isinstance(force, openmm.CustomNonbondedForce) and
                        force.getEnergyFunction() == '0.0;' and
                        force.getGlobalParameterName(0) == 'lambda_electrostatics_{}'.format(region_name)):
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

    def _set_exact_pme_charges(self, original_charges_force, nonbonded_force, lambda_electrostatics):
        """Set the NonbondedForce charges from the original value and lambda_electrostatics."""
        # If we don't treat PME exactly, we don't need to set the charges.
        if original_charges_force is None:
            return
        # Set alchemical atoms charges.
        #lambda_electrostatics = self.lambda_electrostatics_one
        _, alchemical_atoms = original_charges_force.getInteractionGroupParameters(0)
        for atom_idx in alchemical_atoms:
            charge, sigma, epsilon = nonbonded_force.getParticleParameters(atom_idx)
            original_charge = original_charges_force.getParticleParameters(atom_idx)[0]
            charge = lambda_electrostatics * original_charge
            nonbonded_force.setParticleParameters(atom_idx, charge, sigma, epsilon)


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
    ('softcore_beta', 0.0), ('softcore_d', 1), ('softcore_e', 1), ('softcore_f', 2),
    ('name', None)
])


# The class is just a way to document the namedtuple.
class AlchemicalRegion(collections.namedtuple('AlchemicalRegion', _ALCHEMICAL_REGION_ARGS.keys())):
    """Alchemical region.

    This is a namedtuple used to tell the AbsoluteAlchemicalFactory which
    region of the system to alchemically modify and how.

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

class AbsoluteAlchemicalFactory(object):
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

    Examples
    --------

    Create an alchemical factory to alchemically modify OpenMM System objects.

    >>> factory = AbsoluteAlchemicalFactory(consistent_exceptions=False)

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

    Neglecting the long-range dispersion correction for the alchemical region
    (for nonequilibrium switching, for example) requires instantiating a factory
    with the appropriate options:

    >>> new_factory = AbsoluteAlchemicalFactory(consistent_exceptions=False, disable_alchemical_dispersion_correction=True)
    >>> reference_system = testsystems.WaterBox().system
    >>> alchemical_region = AlchemicalRegion(alchemical_atoms=[0, 1, 2])
    >>> alchemical_system = new_factory.create_alchemical_system(reference_system, alchemical_region)

    References
    ----------
    [1] Pham TT and Shirts MR. Identifying low variance pathways for free
    energy calculations of molecular transformations in solution phase.
    JCP 135:034114, 2011. http://dx.doi.org/10.1063/1.3607597

    """

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def __init__(self, consistent_exceptions=False, switch_width=1.0*unit.angstroms,
                 alchemical_pme_treatment='direct-space', alchemical_rf_treatment='switched',
                 disable_alchemical_dispersion_correction=False, split_alchemical_forces=True):

        self.consistent_exceptions = consistent_exceptions
        self.switch_width = switch_width
        self.alchemical_pme_treatment = alchemical_pme_treatment
        self.alchemical_rf_treatment = alchemical_rf_treatment
        self.disable_alchemical_dispersion_correction = disable_alchemical_dispersion_correction
        self.split_alchemical_forces = split_alchemical_forces

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
            print('Using %s alchemical regions' % (len(alchemical_regions)))
            if len(alchemical_regions) > 3:
                raise ValueError('Maximum alchemical regions is 3')
        else:
            alchemical_regions = [alchemical_regions]

        # Resolve alchemical region.
        AlchemicalRegions = []
        allalchemical = {'Atoms': [], 'Bonds': [], 'Angles': [], 'Torsions': []}
        CheckDuplicateNames = []
        for alchemical_region in alchemical_regions:
            AlchemicalRegions.append(self._resolve_alchemical_region(reference_system, alchemical_region))
            allalchemical['Atoms'].append(alchemical_region.alchemical_atoms)
            allalchemical['Bonds'].append(alchemical_region.alchemical_bonds)
            allalchemical['Angles'].append(alchemical_region.alchemical_angles)
            allalchemical['Torsions'].append(alchemical_region.alchemical_torsions)
            CheckDuplicateNames.append(alchemical_region.name)

        if len(CheckDuplicateNames) != len(set(CheckDuplicateNames)):
            raise ValueError('Two regions alchemical regions have the identical name')
        del CheckDuplicateNames

        #remove all Nones and check for duplicates
        for k, v in allalchemical.items():
            v = [x for x in v if x is not None]
            for i, region in enumerate(v):
                if i == len(v)-1:
                    break
                a = region
                b = v[i+1]
                if not set(a).isdisjoint(b):
                    raise ValueError('Two alchemical regions share same entries, please remove duplicates from {}'.format(k))

        allalchemicalatoms = allalchemical['Atoms']

        # Record timing statistics.
        timer = utils.Timer()
        timer.start('Create alchemically modified system')

        # Build alchemical system to modify. This copies particles, vsites,
        # constraints, box vectors and all the forces. We'll later remove
        # the forces that we remodel to be alchemically modified.
        alchemical_system = copy.deepcopy(reference_system)

        # Check that there are no virtual sites to alchemically modify.
        for alchemical_region in AlchemicalRegions:
            for particle_index in range(reference_system.getNumParticles()):
                if (reference_system.isVirtualSite(particle_index) and
                        particle_index in alchemical_region.alchemical_atoms):
                    raise ValueError('Alchemically modified virtual sites are not supported')

        # Modify forces as appropriate. We delete the forces that
        # have been processed modified at the end of the for loop.
        forces_to_remove = []
        alchemical_forces_by_lambda = {}

        for force_index, reference_force in enumerate(reference_system.getForces()):
            cumulative_force = None
            for alchemical_region in AlchemicalRegions:
                # TODO switch to functools.singledispatch when we drop Python2 support
                reference_force_name = reference_force.__class__.__name__
                alchemical_force_creator_name = '_alchemically_modify_{}'.format(reference_force_name)
                try:
                    alchemical_force_creator_func = getattr(self, alchemical_force_creator_name)
                except AttributeError:
                    pass
                else:
                    # The reference system force will be deleted.
                    if force_index not in forces_to_remove:
                        forces_to_remove.append(force_index)
                    # Collect all the Force objects modeling the reference force.
                    alchemical_forces = alchemical_force_creator_func(reference_force, alchemical_region, allalchemicalatoms, cumulative_force)
                    for lambda_variable_name, lambda_forces in alchemical_forces.items():
                        try:
                            alchemical_forces_by_lambda[lambda_variable_name].extend(lambda_forces)
                        except KeyError:
                            alchemical_forces_by_lambda[lambda_variable_name] = lambda_forces
                    # Want to pass NonBondedForce to next alchemical region and remove it from current
                    # alchemical region unless we are in the last alchemical region.
                    if not alchemical_region == AlchemicalRegions[-1]:
                        if reference_force_name ==  'NonbondedForce':
                            #print(alchemical_forces_by_lambda)
                            cumulative_force = alchemical_forces_by_lambda[''][-1]
                            #If using PME the NonbondedForce will be found as last entry of 'lambda_electrostatics'
                            if not isinstance(cumulative_force, openmm.NonbondedForce):
                                cumulative_force = alchemical_forces_by_lambda['lambda_electrostatics_{}'.format(alchemical_region.name)][-1]
                                del alchemical_forces_by_lambda['lambda_electrostatics_{}'.format(alchemical_region.name)][-1]
                            else:
                                del alchemical_forces_by_lambda[''][-1]
                        if reference_force_name in ['HarmonicBondForce', 'HarmonicAngleForce', 'PeriodicTorsionForce']:
                            cumulative_force = alchemical_forces_by_lambda[''][-1]
                            del alchemical_forces_by_lambda[''][-1]
                        else:
                            pass

        #Do I need to check the NonbondedForce is where it is supposed to be. Might be needed if last alchemical region contains no alchemical
        # atoms and using PME because in this case NonbondedForce will appear at the end of '' not the end of lambda_electrostatics
        #Does this matter?

        #Could clean up 'HarmonicBondForce', 'HarmonicAngleForce', 'PeriodicTorsionForce' here as these forces are calculating
        #potneials in the alchemical region twice once in customforce then again in standard force but with K = 0

        #Is treatment of aa steric interaction correct? By what combination rule do two different alchemical regions see each other?

        #DEBUG
        for x in alchemical_forces_by_lambda:
            print(x, alchemical_forces_by_lambda[x])
            print('/////////////////')


        # Remove original forces that have been alchemically modified.
        #print(forces_to_remove)
        for force_index in reversed(forces_to_remove):
            #print(force_index)
            alchemical_system.removeForce(force_index)

        # Add forces and split groups if necessary.
        self._add_alchemical_forces(alchemical_system, alchemical_forces_by_lambda)

        # Record timing statistics.
        timer.stop('Create alchemically modified system')
        timer.report_timing()

        # If the System uses a NonbondedForce, replace its NonbondedForce implementation of reaction field
        # with a Custom*Force implementation that uses c_rf = 0.
        # NOTE: This adds an additional CustomNonbondedForce
        if self.alchemical_rf_treatment == 'switched':
            forcefactories.replace_reaction_field(alchemical_system, return_copy=False,
                                                  switch_width=self.switch_width)

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

    def _add_alchemical_forces(self, alchemical_system, alchemical_forces_by_lambda):
        """Add the forces to the alchemical system and eventually split the force groups."""
        # OpenMM can have a maximum of 32 groups.
        available_force_groups = set(range(32))

        # Add non-alchemical groups. New forces will have force group 0, and we don't
        # want to modify the force group of forces that have been copied from the reference.
        non_alchemical_forces = alchemical_forces_by_lambda.pop('', [])
        for non_alchemical_force in non_alchemical_forces:
            alchemical_system.addForce(non_alchemical_force)

        # Find which force groups are still available for alchemical forces.
        for force in alchemical_system.getForces():
            available_force_groups.discard(force.getForceGroup())

        # Check if there are enough force groups to split alchemical forces.
        if (self.split_alchemical_forces and
                    len(available_force_groups) < len(alchemical_forces_by_lambda)):
            raise RuntimeError('There are not enough force groups to split alchemical forces.\n'
                               'Consider merging some non-alchemical forces in a single group '
                               'or set split_alchemical_forces to False.')

        # Add the alchemical forces in a deterministic way (just to be safe).
        for lambda_variable in sorted(alchemical_forces_by_lambda):
            if self.split_alchemical_forces:
                # Assign to these forces the smallest force group index available.
                force_group = min(available_force_groups)
                available_force_groups.remove(force_group)
            for force in alchemical_forces_by_lambda[lambda_variable]:
                if self.split_alchemical_forces:
                    force.setForceGroup(force_group)
                alchemical_system.addForce(force)

    @staticmethod
    def _alchemically_modify_PeriodicTorsionForce(reference_force, alchemical_region, _, cumulative_torsion_force):
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
        if cumulative_torsion_force == None:
            force = copy.deepcopy(reference_force)
        else:
            force = copy.deepcopy(cumulative_torsion_force)

        if len(alchemical_region.alchemical_torsions) == 0:
            return {'': [copy.deepcopy(force)]}

        # Create CustomTorsionForce to handle alchemically modified torsions.
        energy_function = "lambda_torsions_{}*k*(1+cos(periodicity*theta-phase))".format(alchemical_region.name)
        custom_force = openmm.CustomTorsionForce(energy_function)
        custom_force.addGlobalParameter('lambda_torsions_{}'.format(alchemical_region.name), 1.0)
        custom_force.addPerTorsionParameter('periodicity')
        custom_force.addPerTorsionParameter('phase')
        custom_force.addPerTorsionParameter('k')
        # Process reference torsions.
        for torsion_index in range(force.getNumTorsions()):
            # Retrieve parameters.
            particle1, particle2, particle3, particle4, periodicity, phase, k = force.getTorsionParameters(torsion_index)
            #print(torsion_index, particle1, particle2, particle3, particle4, periodicity, phase, k)
            # Create torsions.
            if torsion_index in alchemical_region.alchemical_torsions:
                #Add to alchemical
                custom_force.addTorsion(particle1, particle2, particle3, particle4, [periodicity, phase, k])
                #Remove from std
                force.setTorsionParameters(torsion_index, particle1, particle2, particle3, particle4, periodicity, phase, k*0.0)
        return {'': [force], 'lambda_torsions_{}'.format(alchemical_region.name): [custom_force]}

    @staticmethod
    def _alchemically_modify_HarmonicAngleForce(reference_force, alchemical_region, _, cumulative_angle_force):
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
        if cumulative_angle_force == None:
            force = copy.deepcopy(reference_force)
        else:
            force = copy.deepcopy(cumulative_angle_force)

        if len(alchemical_region.alchemical_angles) == 0:
            return {'': [copy.deepcopy(force)]}

        # Create CustomAngleForce to handle alchemically modified angles.
        energy_function = "lambda_angles_{}*(K/2)*(theta-theta0)^2;".format(alchemical_region.name)
        custom_force = openmm.CustomAngleForce(energy_function)
        custom_force.addGlobalParameter('lambda_angles_{}'.format(alchemical_region.name), 1.0)
        custom_force.addPerAngleParameter('theta0')
        custom_force.addPerAngleParameter('K')
        # Process reference angles.
        for angle_index in range(force.getNumAngles()):
            # Retrieve parameters.
            [particle1, particle2, particle3, theta0, K] = force.getAngleParameters(angle_index)
            if angle_index in alchemical_region.alchemical_angles:
                # Add alchemical
                custom_force.addAngle(particle1, particle2, particle3, [theta0, K])
                #remove std
                force.setAngleParameters(angle_index, particle1, particle2, particle3, theta0, K*0.0)

        return {'': [force], 'lambda_angles_{}'.format(alchemical_region.name): [custom_force]}

    @staticmethod
    def _alchemically_modify_HarmonicBondForce(reference_force, alchemical_region, _, cumulative_bond_force):
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
        if cumulative_bond_force == None:
            force = copy.deepcopy(reference_force)
        else:
            force = copy.deepcopy(cumulative_bond_force)

        if len(alchemical_region.alchemical_bonds) == 0:
            return {'': [copy.deepcopy(force)]}

        # Create CustomBondForce to handle alchemically modified bonds.
        energy_function = "lambda_bonds_{}*(K/2)*(r-r0)^2;".format(alchemical_region.name)
        custom_force = openmm.CustomBondForce(energy_function)
        custom_force.addGlobalParameter('lambda_bonds_{}'.format(alchemical_region.name), 1.0)
        custom_force.addPerBondParameter('r0')
        custom_force.addPerBondParameter('K')
        # Process reference bonds.
        for bond_index in range(force.getNumBonds()):
            # Retrieve parameters.
            [particle1, particle2, theta0, K] = force.getBondParameters(bond_index)
            if bond_index in alchemical_region.alchemical_bonds:
                # add alchemical
                custom_force.addBond(particle1, particle2, [theta0, K])
                #remove from std
                force.setBondParameters(bond_index, particle1, particle2, theta0, K*0.0)

        return {'': [force], 'lambda_bonds_{}'.format(alchemical_region.name): [custom_force]}

    def _get_nonbonded_energy_exspressions(self, alchemical_region_idx):
        # Sterics mixing rules.
        sterics_mixing_rules = ('epsilon = sqrt(epsilon1*epsilon2);'  # Mixing rule for epsilon.
                                'sigma = 0.5*(sigma1 + sigma2);')  # Mixing rule for sigma.

        # Soft-core Lennard-Jones.
        exceptions_sterics_energy_expression = ('U_{0}_sterics;'
                                                'U_{0}_sterics = (lambda_sterics_{0}^softcore_a)*4*epsilon*x*(x-1.0);'
                                                'x = (sigma/reff_sterics)^6;'
                                                # Effective softcore distance for sterics.
                                                'reff_sterics = sigma*((softcore_alpha*(1.0-lambda_sterics_{0})^softcore_b + (r/sigma)^softcore_c))^(1/softcore_c);').format(
                                                alchemical_region_idx)


        # Define energy expression for electrostatics.
        return sterics_mixing_rules, exceptions_sterics_energy_expression

    def _get_electrostatics_energy_expressions(self, reference_force, alchemical_region_idx):
        """Return the energy expressions for electrostatics."""

        # The final expression will be prefix + method + suffix.
        electrostatics_prefix = ('U_{0}_electrostatics;'
                                 'U_{0}_electrostatics=(lambda_electrostatics_{0}^softcore_d)*ONE_4PI_EPS0*chargeprod').format(alchemical_region_idx)

        # Effective softcore distance for electrostatics (common to all methods).
        electrostatics_suffix = ('reff_electrostatics = sigma*((softcore_beta*(1.0-lambda_electrostatics_{0})^softcore_e + (r/sigma)^softcore_f))^(1/softcore_f);'
                                 'ONE_4PI_EPS0 = {1};').format(alchemical_region_idx, ONE_4PI_EPS0)  # Already in OpenMM units.

        # Define mixing rules.
        electrostatics_mixing_rules = ('chargeprod = charge1*charge2;'  # Mixing rule for charges.
                                       'sigma = 0.5*(sigma1 + sigma2);')  # Mixing rule for sigma.

        # Standard Coulomb expression with softened core. This is used
        #   - When the nonbonded method of the reference force is NoCutoff.
        #   - When alchemical_pme_treatment is set to 'coulomb'.
        #   - With 1-4 exceptions, unless self.consistent_exceptions is True.
        coulomb_expression = '/reff_electrostatics;'

        # Select electrostatics functional form based on nonbonded method.
        nonbonded_method = reference_force.getNonbondedMethod()
        is_ewald_method = nonbonded_method in [openmm.NonbondedForce.PME, openmm.NonbondedForce.Ewald]

        # With exact treatment, the energy expression is irrelevant as CustomNonbondedForces
        # only keep the original charges, but they don't contribute to the energy of the system.
        if is_ewald_method and self.alchemical_pme_treatment == 'exact':
            electrostatics_energy_expression = '0.0;'
            exceptions_electrostatics_energy_expression = (electrostatics_prefix + coulomb_expression +
                                                           electrostatics_suffix)
            return electrostatics_energy_expression, exceptions_electrostatics_energy_expression

        # Soft-core Coulomb.
        if nonbonded_method in [openmm.NonbondedForce.NoCutoff]:
            electrostatics_method_expression = coulomb_expression
        # Reaction-field electrostatics.
        elif nonbonded_method in [openmm.NonbondedForce.CutoffPeriodic, openmm.NonbondedForce.CutoffNonPeriodic]:
            electrostatics_method_expression = self._get_reaction_field_unique_expression(reference_force)
        # PME electrostatics.
        elif nonbonded_method in [openmm.NonbondedForce.PME, openmm.NonbondedForce.Ewald]:
            # Ewald direct-space electrostatics.
            if self.alchemical_pme_treatment == 'direct-space':
                electrostatics_method_expression = self._get_pme_direct_space_unique_expression(reference_force)
            # Use switched standard Coulomb potential, following MTS scheme described in
            # http://dx.doi.org/10.1063/1.1385159
            elif self.alchemical_pme_treatment == 'coulomb':
                electrostatics_method_expression = coulomb_expression
            else:
                raise ValueError("Unknown alchemical_pme_treatment scheme '{}'".format(self.alchemical_pme_treatment))
        else:
            raise ValueError("Nonbonded method {} not supported yet.".format(nonbonded_method))

        # Define energy expression for 1,4 electrostatic exceptions.
        exceptions_electrostatics_energy_expression = electrostatics_prefix
        if self.consistent_exceptions:
            exceptions_electrostatics_energy_expression += electrostatics_method_expression
        else:
            exceptions_electrostatics_energy_expression += coulomb_expression
        exceptions_electrostatics_energy_expression += electrostatics_suffix

        # Define energy expression for electrostatics.
        electrostatics_energy_expression = (electrostatics_prefix + electrostatics_method_expression +
                                            electrostatics_suffix + electrostatics_mixing_rules)

        return electrostatics_energy_expression, exceptions_electrostatics_energy_expression

    def _get_reaction_field_unique_expression(self, reference_force):
        """Unique part of the expression for reaction-field electrostatics.

        Parameters
        ----------
        reference_force : openmm.NonbondedForce
            The reference force including the reaction-field parameters.

        Returns
        -------
        rf_expression : str
            The unique expression for reaction-field electrostatics.

        See Also
        --------
        _get_nonbonded_energy_expressions
        """
        epsilon_solvent = reference_force.getReactionFieldDielectric()
        r_cutoff = reference_force.getCutoffDistance()

        # Determine reaction fields parameters.
        k_rf = r_cutoff**(-3) * ((epsilon_solvent - 1) / (2*epsilon_solvent + 1))
        if self.alchemical_rf_treatment == 'switched':
            c_rf = 0.0 / unit.nanometers
        elif self.alchemical_rf_treatment == 'shifted':
            # WARNING: Setting c_rf != 0 can cause large errors in DeltaG for hydration free energies
            c_rf = r_cutoff**(-1) * ((3*epsilon_solvent) / (2*epsilon_solvent + 1))
        else:
            raise ValueError("Unknown alchemical_rf_treatment scheme '{}'".format(self.alchemical_rf_treatment))

        k_rf = k_rf.value_in_unit_system(unit.md_unit_system)
        c_rf = c_rf.value_in_unit_system(unit.md_unit_system)
        rf_expression = ('*(reff_electrostatics^(-1) + k_rf*reff_electrostatics^2 - c_rf);'
                         'k_rf = {k_rf};'
                         'c_rf = {c_rf};').format(k_rf=k_rf, c_rf=c_rf)
        return rf_expression

    def _get_pme_direct_space_unique_expression(self, reference_force):
        """Unique part of the expression for Ewald direct-space electrostatics.

        Parameters
        ----------
        reference_force : openmm.NonbondedForce
            The reference force including the Ewald parameters.

        Returns
        -------
        rf_expression : str
            The unique expression for Ewald direct-space electrostatics.

        See Also
        --------
        _get_nonbonded_energy_expressions
        """
        # Determine PME parameters.
        [alpha_ewald, nx, ny, nz] = reference_force.getPMEParameters()
        if (alpha_ewald/alpha_ewald.unit) == 0.0:
            # If alpha is 0.0, alpha_ewald is computed by OpenMM from from the error tolerance.
            tol = reference_force.getEwaldErrorTolerance()
            alpha_ewald = (1.0/reference_force.getCutoffDistance()) * np.sqrt(-np.log(2.0*tol))

        alpha_ewald = alpha_ewald.value_in_unit_system(unit.md_unit_system)
        pme_expression = ("*erfc(alpha_ewald*reff_electrostatics)/reff_electrostatics;"
                          "alpha_ewald = {};").format(alpha_ewald)
        return pme_expression

    def _alchemically_modify_NonbondedForce(self, reference_force, alchemical_region, all_alchemical_atoms, cumulative_nonbonded_force):
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

        # Create a copy of the NonbondedForce to handle particle interactions and
        # 1,4 exceptions between non-alchemical/non-alchemical atoms (nn).
        if cumulative_nonbonded_force == None:
            nonbonded_force = copy.deepcopy(reference_force)
        else:
            nonbonded_force = copy.deepcopy(cumulative_nonbonded_force)

        # Don't create a force if there are no alchemical atoms.
        if len(alchemical_region.alchemical_atoms) == 0:
            return {'': [copy.deepcopy(nonbonded_force)]}

        # --------------------------------------------------
        # Determine energy expression for all custom forces
        # --------------------------------------------------
        RegionName = alchemical_region.name
        #if RegionName == None:
        #    RegionName = 'zero'
        sterics_mixing_rules, exceptions_sterics_energy_expression = self._get_nonbonded_energy_exspressions(RegionName)
        sterics_energy_expression = exceptions_sterics_energy_expression + sterics_mixing_rules


        # Define energy expression for electrostatics based on nonbonded method.
        nonbonded_method = reference_force.getNonbondedMethod()
        is_ewald_method = nonbonded_method in [openmm.NonbondedForce.Ewald,
                                               openmm.NonbondedForce.PME]
        is_rf_method = nonbonded_method in [openmm.NonbondedForce.CutoffPeriodic,
                                            openmm.NonbondedForce.CutoffNonPeriodic]
        is_periodic_method = is_ewald_method or nonbonded_method == openmm.NonbondedForce.CutoffPeriodic
        use_exact_pme_treatment = is_ewald_method and self.alchemical_pme_treatment == 'exact'

        # Warn about reaction field.
        if is_rf_method:
            logger.warning('Reaction field support is still experimental. For free energy '
                           'calculations in explicit solvent, we suggest using PME for now.')

        # Check that PME treatment is supported with the region's parameters.
        if use_exact_pme_treatment:
            err_msg = ' not supported with exact treatment of Ewald electrostatics.'
            if not alchemical_region.annihilate_electrostatics:
                raise ValueError('Decoupled electrostatics is' + err_msg)
            if self.consistent_exceptions:
                raise ValueError('Consistent exceptions are' + err_msg)
            if (alchemical_region.softcore_beta, alchemical_region.softcore_d, alchemical_region.softcore_e) != (0, 1, 1):
                raise ValueError('Softcore electrostatics is' + err_msg)

        energy_expressions = self._get_electrostatics_energy_expressions(reference_force, RegionName)
        (electrostatics_energy_expression,
         exceptions_electrostatics_energy_expression) = energy_expressions  # Unpack tuple.

        # ------------------------------------------------------------
        # Create and configure all forces to add to alchemical system
        # ------------------------------------------------------------

        # Interactions and exceptions will be distributed according to the following table.

        # --------------------------------------------------------------------------------------------------
        # FORCE                                    | INTERACTION GROUP                                     |
        # --------------------------------------------------------------------------------------------------
        # nonbonded_force (unmodified)             | all interactions nonalchemical/nonalchemical          |
        #                                          | all exceptions nonalchemical/nonalchemical            |
        # --------------------------------------------------------------------------------------------------
        # aa_sterics_custom_nonbonded_force        | sterics interactions alchemical/alchemical            |
        # --------------------------------------------------------------------------------------------------
        # aa_electrostatics_custom_nonbonded_force | electrostatics interactions alchemical/alchemical     |
        #                                          | (only without exact PME treatment)                    |
        # --------------------------------------------------------------------------------------------------
        # na_sterics_custom_nonbonded_force        | sterics interactions non-alchemical/alchemical        |
        # --------------------------------------------------------------------------------------------------
        # na_electrostatics_custom_nonbonded_force | electrostatics interactions non-alchemical/alchemical |
        #                                          | (only without exact PME treatment)                    |
        # --------------------------------------------------------------------------------------------------
        # aa_sterics_custom_bond_force             | sterics exceptions alchemical/alchemical              |
        # --------------------------------------------------------------------------------------------------
        # aa_electrostatics_custom_bond_force      | electrostatics exceptions alchemical/alchemical       |
        # --------------------------------------------------------------------------------------------------
        # na_sterics_custom_bond_force             | sterics exceptions non-alchemical/alchemical          |
        # --------------------------------------------------------------------------------------------------
        # na_electrostatics_custom_bond_force      | electrostatics exceptions non-alchemical/alchemical   |
        # --------------------------------------------------------------------------------------------------

        def create_force(force_cls, energy_expression, lambda_variable_name, is_lambda_controlled):
            """Shortcut to create a lambda-controlled custom forces."""
            if is_lambda_controlled:
                force = force_cls(energy_expression)
                force.addGlobalParameter(lambda_variable_name, 1.0)
            else:  # fix lambda variable to 1.0
                energy_expression = energy_expression + lambda_variable_name + '=1.0;'
                force = force_cls(energy_expression)
            return force

        # Create CustomNonbondedForces to handle sterics particle interactions between
        # non-alchemical/alchemical atoms (na) and alchemical/alchemical atoms (aa). Fix lambda
        # to 1.0 for decoupled interactions in alchemical/alchemical force.
        na_sterics_custom_nonbonded_force = create_force(openmm.CustomNonbondedForce, sterics_energy_expression,
                                                         'lambda_sterics_{}'.format(RegionName), is_lambda_controlled=True)
        aa_sterics_custom_nonbonded_force = create_force(openmm.CustomNonbondedForce, sterics_energy_expression,
                                                         'lambda_sterics_{}'.format(RegionName), alchemical_region.annihilate_sterics)
        all_sterics_custom_nonbonded_forces = [na_sterics_custom_nonbonded_force, aa_sterics_custom_nonbonded_force]

        # Add parameters and configure CustomNonbondedForces to match reference force
        for force in all_sterics_custom_nonbonded_forces:
            force.addPerParticleParameter("sigma")  # Lennard-Jones sigma
            force.addPerParticleParameter("epsilon")  # Lennard-Jones epsilon
            force.setUseSwitchingFunction(nonbonded_force.getUseSwitchingFunction())
            force.setCutoffDistance(nonbonded_force.getCutoffDistance())
            force.setSwitchingDistance(nonbonded_force.getSwitchingDistance())
            if self.disable_alchemical_dispersion_correction:
                force.setUseLongRangeCorrection(False)
            else:
                force.setUseLongRangeCorrection(nonbonded_force.getUseDispersionCorrection())

            if is_periodic_method:
                force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
            else:
                force.setNonbondedMethod(nonbonded_force.getNonbondedMethod())

        # With exact PME treatment, we create a single CustomNonbondedForce
        # to store the original alchemically-modified charges.
        if use_exact_pme_treatment:
            # We add the lambda_electrostatics global variable even
            # if it doesn't affect the force to make sure we have
            # a very quick way to check the status of the charges.
            original_charges_custom_nonbonded_force = create_force(openmm.CustomNonbondedForce, electrostatics_energy_expression,
                                                                   'lambda_electrostatics_{}'.format(RegionName), is_lambda_controlled=True)
            # Keep an extra global parameter to flag whether changes in
            # electrostatics through updateParametersInContext are allowed.
            original_charges_custom_nonbonded_force.addGlobalParameter(_UPDATE_ALCHEMICAL_CHARGES_PARAMETER, 1)
            all_electrostatics_custom_nonbonded_forces = [original_charges_custom_nonbonded_force]
        else:
            # Create CustomNonbondedForces to handle electrostatics particle interactions between
            # non-alchemical/alchemical atoms (na) and alchemical/alchemical atoms (aa). Fix lambda
            # to 1.0 for decoupled interactions in alchemical/alchemical force.
            na_electrostatics_custom_nonbonded_force = create_force(openmm.CustomNonbondedForce, electrostatics_energy_expression,
                                                                    'lambda_electrostatics_{}'.format(RegionName), is_lambda_controlled=True)
            aa_electrostatics_custom_nonbonded_force = create_force(openmm.CustomNonbondedForce, electrostatics_energy_expression,
                                                                    'lambda_electrostatics_{}'.format(RegionName), alchemical_region.annihilate_electrostatics)
            all_electrostatics_custom_nonbonded_forces = [na_electrostatics_custom_nonbonded_force,
                                                          aa_electrostatics_custom_nonbonded_force]

        # Common parameters and configuration for electrostatics CustomNonbondedForces.
        for force in all_electrostatics_custom_nonbonded_forces:
            force.addPerParticleParameter("charge")  # partial charge
            # The sigma parameter is necessary only if we don't treat PME exactly.
            if not use_exact_pme_treatment:
                force.addPerParticleParameter("sigma")  # Lennard-Jones sigma
            if ((is_ewald_method and self.alchemical_pme_treatment == 'coulomb') or
                    (is_rf_method and self.alchemical_rf_treatment == 'switched')):
                # Use switching function for alchemical electrostatics to ensure force continuity at cutoff.
                force.setUseSwitchingFunction(True)
            else:
                force.setUseSwitchingFunction(False)
            force.setSwitchingDistance(nonbonded_force.getCutoffDistance() - self.switch_width)
            force.setCutoffDistance(nonbonded_force.getCutoffDistance())
            force.setUseLongRangeCorrection(False)  # long-range dispersion correction is meaningless for electrostatics

            if is_periodic_method:
                force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
            else:
                force.setNonbondedMethod(nonbonded_force.getNonbondedMethod())

        # Create CustomBondForces to handle sterics 1,4 exceptions interactions between
        # non-alchemical/alchemical atoms (na) and alchemical/alchemical atoms (aa). Fix lambda
        # to 1.0 for decoupled interactions in alchemical/alchemical force.
        na_sterics_custom_bond_force = create_force(openmm.CustomBondForce, exceptions_sterics_energy_expression,
                                                    'lambda_sterics_{}'.format(RegionName), is_lambda_controlled=True)
        aa_sterics_custom_bond_force = create_force(openmm.CustomBondForce, exceptions_sterics_energy_expression,
                                                    'lambda_sterics_{}'.format(RegionName), alchemical_region.annihilate_sterics)
        all_sterics_custom_bond_forces = [na_sterics_custom_bond_force, aa_sterics_custom_bond_force]

        for force in all_sterics_custom_bond_forces:
            force.addPerBondParameter("sigma")  # Lennard-Jones effective sigma
            force.addPerBondParameter("epsilon")  # Lennard-Jones effective epsilon

        # Create CustomBondForces to handle electrostatics 1,4 exceptions interactions between
        # non-alchemical/alchemical atoms (na) and alchemical/alchemical atoms (aa). Fix lambda
        # to 1.0 for decoupled interactions in alchemical/alchemical force.
        na_electrostatics_custom_bond_force = create_force(openmm.CustomBondForce, exceptions_electrostatics_energy_expression,
                                                           'lambda_electrostatics_{}'.format(RegionName), is_lambda_controlled=True)
        aa_electrostatics_custom_bond_force = create_force(openmm.CustomBondForce, exceptions_electrostatics_energy_expression,
                                                           'lambda_electrostatics_{}'.format(RegionName), alchemical_region.annihilate_electrostatics)
        all_electrostatics_custom_bond_forces = [na_electrostatics_custom_bond_force, aa_electrostatics_custom_bond_force]

        # Create CustomBondForce to handle exceptions for electrostatics
        for force in all_electrostatics_custom_bond_forces:
            force.addPerBondParameter("chargeprod")  # charge product
            force.addPerBondParameter("sigma")  # Lennard-Jones effective sigma

        # -------------------------------------------------------------------------------
        # Distribute particle interactions contributions in appropriate nonbonded forces
        # -------------------------------------------------------------------------------

        # Create atom groups.
        alchemical_atomset = alchemical_region.alchemical_atoms
        all_atomset = set(range(reference_force.getNumParticles()))  # all atoms, including alchemical region
        all_alchemical_atoms = [item for sublist in all_alchemical_atoms for item in sublist]
        nonalchemical_atomset = all_atomset.difference(all_alchemical_atoms)

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
            # Set sterics parameters to custom forces.
            for force in all_sterics_custom_nonbonded_forces:
                force.addParticle([sigma, epsilon])
            # Set electrostatics parameters to custom forces.
            if use_exact_pme_treatment:
                if particle_index in alchemical_atomset:
                    electrostatics_parameters = [charge]
                else:
                    electrostatics_parameters = [abs(0.0*charge)]
            else:
                electrostatics_parameters = [charge, sigma]
            for force in all_electrostatics_custom_nonbonded_forces:
                force.addParticle(electrostatics_parameters)

        # Turn off interactions contribution from alchemically-modified particles in unmodified
        # NonbondedForce that will be handled by all other forces
        for particle_index in range(nonbonded_force.getNumParticles()):
            # Retrieve parameters.
            [charge, sigma, epsilon] = nonbonded_force.getParticleParameters(particle_index)
            # Wit exact treatment of the PME electrostatics, the NonbondedForce handles the electrostatics.
            if not use_exact_pme_treatment:
                charge = abs(0.0*charge)
            if particle_index in alchemical_atomset:
                nonbonded_force.setParticleParameters(particle_index, charge, sigma, abs(0*epsilon))

        # Restrict interaction evaluation of CustomNonbondedForces to their respective atom groups.
        na_sterics_custom_nonbonded_force.addInteractionGroup(nonalchemical_atomset, alchemical_atomset)
        aa_sterics_custom_nonbonded_force.addInteractionGroup(alchemical_atomset, alchemical_atomset)
        if use_exact_pme_treatment:
            # The custom force only holds the original charges. It doesn't contribute to the energy.
            original_charges_custom_nonbonded_force.addInteractionGroup({}, alchemical_atomset)
        else:
            na_electrostatics_custom_nonbonded_force.addInteractionGroup(nonalchemical_atomset, alchemical_atomset)
            aa_electrostatics_custom_nonbonded_force.addInteractionGroup(alchemical_atomset, alchemical_atomset)

        # ---------------------------------------------------------------
        # Distribute exceptions contributions in appropriate bond forces
        # ---------------------------------------------------------------

        all_custom_nonbonded_forces = all_sterics_custom_nonbonded_forces + all_electrostatics_custom_nonbonded_forces

        # Move all NonbondedForce exception terms for alchemically-modified particles to CustomBondForces.
        for exception_index in range(nonbonded_force.getNumExceptions()):
            # Retrieve parameters.
            [iatom, jatom, chargeprod, sigma, epsilon] = nonbonded_force.getExceptionParameters(exception_index)

            # Exclude this atom pair in CustomNonbondedForces. All nonbonded forces
            # must have the same number of exceptions/exclusions on CUDA platform.
            for force in all_custom_nonbonded_forces:
                force.addExclusion(iatom, jatom)

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

        all_custom_forces = (all_custom_nonbonded_forces +
                             all_sterics_custom_bond_forces +
                             all_electrostatics_custom_bond_forces)
        for force in all_custom_forces:
            add_global_parameters(force)

        # With exact treatment of PME electrostatics, the NonbondedForce
        # is affected by lambda electrostatics as well.
        forces_by_lambda = {
            'lambda_electrostatics_{}'.format(RegionName): all_electrostatics_custom_nonbonded_forces + all_electrostatics_custom_bond_forces,
            'lambda_sterics_{}'.format(RegionName): all_sterics_custom_nonbonded_forces + all_sterics_custom_bond_forces,
        }
        if use_exact_pme_treatment:
            forces_by_lambda['lambda_electrostatics_{}'.format(RegionName)].append(nonbonded_force)
        else:
            forces_by_lambda[''] = [nonbonded_force]
        return forces_by_lambda

    def _alchemically_modify_AmoebaMultipoleForce(self, reference_force, alchemical_region, _, __):
        raise Exception("Not implemented; needs CustomMultipleForce")

    def _alchemically_modify_AmoebaVdwForce(self, reference_force, alchemical_region, _, __):
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
    def _alchemically_modify_GBSAOBCForce(reference_force, alchemical_region, _, __, sasa_model='ACE'):
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
        custom_force.addGlobalParameter("lambda_electrostatics", 1.0)
        custom_force.addPerParticleParameter("charge")
        custom_force.addPerParticleParameter("radius")
        custom_force.addPerParticleParameter("scale")
        custom_force.addPerParticleParameter("alchemical")

        # Set nonbonded method.
        custom_force.setNonbondedMethod(reference_force.getNonbondedMethod())
        custom_force.setCutoffDistance(reference_force.getCutoffDistance())

        # Add global parameters.
        custom_force.addGlobalParameter("solventDielectric", reference_force.getSolventDielectric())
        custom_force.addGlobalParameter("soluteDielectric", reference_force.getSoluteDielectric())
        custom_force.addGlobalParameter("offset", 0.009)

        custom_force.addComputedValue("I",  "(lambda_electrostatics*alchemical2 + (1-alchemical2))*step(r+sr2-or1)*0.5*(1/L-1/U+0.25*(r-sr2^2/r)*(1/(U^2)-1/(L^2))+0.5*log(L/U)/r+C);"
                                "U=r+sr2;"
                                "C=2*(1/or1-1/L)*step(sr2-r-or1);"
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

        return {'lambda_electrostatics': [custom_force]}

    def _alchemically_modify_CustomGBForce(self, reference_force, alchemical_region, _, __):
        """Create alchemically-modified version of CustomGBForce.

        The GB functions are meta-programmed using the following rules:
        - 'lambda_electrostatics' is added as a global parameter.
        - 'alchemical' is added as a per-particle parameter. All atoms in
          the alchemical group have this parameter set to 1; otherwise 0.
        - Any single-particle energy term (`CustomGBForce.SingleParticle`)
          is scaled by `(lambda_electrostatics*alchemical+(1-alchemical))`
        - Any two-particle energy term (`CustomGBForce.ParticlePairNoExclusions`)
          has charge 1 (`charge1`) replaced by `(lambda_electrostatics*alchemical1+(1-alchemical1))*charge1`
          and charge 2 (`charge2`) replaced by `(lambda_electrostatics*alchemical2+(1-alchemical2))*charge2`.
        - Any single-particle computed value (`CustomGBForce.SingleParticle`)
          remains unmodified
        - Any two-particle computed value (`CustomGBForce.ParticlePairNoExclusions`)
          is scaled by `(lambda_electrostatics*alchemical2 + (1-alchemical2))`

        Scaling of a term should always prepend and capture the value with
        an intermediate variable. For example, prepending `scaling * unscaled; unscaled =`
        will capture the value of the expression as `unscaled` and multiple by `scaled`.
        This avoids the need to identify the head expression and add parentheses.

        .. warning::
            This may not work correctly for all GB models.

        Parameters
        ----------
        reference_force : simtk.openmm.GBSAOBCForce
            The reference GBSAOBCForce to be alchemically modify.
        alchemical_region : AlchemicalRegion
            The alchemical region containing the indices of the atoms to
            alchemically modify.

        Returns
        -------
        custom_force : simtk.openmm.CustomGBForce
            The alchemical version of the reference force.

        """
        custom_force = openmm.CustomGBForce()

        # Add global parameters
        for index in range(reference_force.getNumGlobalParameters()):
            name = reference_force.getGlobalParameterName(index)
            default_value = reference_force.getGlobalParameterDefaultValue(index)
            custom_force.addGlobalParameter(name, default_value)
        custom_force.addGlobalParameter("lambda_electrostatics", 1.0)

        # Add per-particle parameters.
        for index in range(reference_force.getNumPerParticleParameters()):
            name = reference_force.getPerParticleParameterName(index)
            custom_force.addPerParticleParameter(name)
        custom_force.addPerParticleParameter("alchemical")

        # Set nonbonded methods.
        custom_force.setNonbondedMethod(reference_force.getNonbondedMethod())
        custom_force.setCutoffDistance(reference_force.getCutoffDistance())

        # Add computations.
        for index in range(reference_force.getNumComputedValues()):
            name, expression, computation_type = reference_force.getComputedValueParameters(index)

            # Alter expression for particle pair terms only.
            if computation_type is not openmm.CustomGBForce.SingleParticle:
                prepend = ('alchemical_scaling*unscaled; '
                           'alchemical_scaling = (lambda_electrostatics*alchemical2 + (1-alchemical2)); '
                           'unscaled = ')
                expression = prepend + expression

            custom_force.addComputedValue(name, expression, computation_type)

        # Add energy terms.
        for index in range(reference_force.getNumEnergyTerms()):
            expression, computation_type = reference_force.getEnergyTermParameters(index)

            # Alter expressions
            if computation_type is openmm.CustomGBForce.SingleParticle:
                prepend = ('alchemical_scaling*unscaled; '
                           'alchemical_scaling = (lambda_electrostatics*alchemical + (1-alchemical)); '
                           'unscaled = ')
                expression = prepend + expression
            else:
                expression = expression.replace('charge1', 'alchemically_scaled_charge1')
                expression = expression.replace('charge2', 'alchemically_scaled_charge2')
                expression += ' ; alchemically_scaled_charge1 = (lambda_electrostatics*alchemical1+(1-alchemical1)) * charge1;'
                expression += ' ; alchemically_scaled_charge2 = (lambda_electrostatics*alchemical2+(1-alchemical2)) * charge2;'

            custom_force.addEnergyTerm(expression, computation_type)

        # Add particle parameters
        for particle_index in range(reference_force.getNumParticles()):
            parameters = reference_force.getParticleParameters(particle_index)
            # Append alchemical parameter
            parameters = list(parameters)
            if particle_index in alchemical_region.alchemical_atoms:
                parameters.append(1.0)
            else:
                parameters.append(0.0)
            custom_force.addParticle(parameters)

        # Add tabulated functions
        for function_index in range(reference_force.getNumTabulatedFunctions()):
            name = reference_force.getTabulatedFunctionName(function_index)
            function = reference_force.getTabulatedFunction(function_index)
            function_copy = copy.deepcopy(function)
            custom_force.addTabulatedFunction(name, function_copy)

        # Add exclusions
        for exclusion_index in range(reference_force.getNumExclusions()):
            [particle1, particle2] = reference_force.getExclusionParticles(exclusion_index)
            custom_force.addExclusion(particle1, particle2)

        return {'lambda_electrostatics': [custom_force]}

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

        def check_energy_expression(custom_force, parameter):
            try:
                found = parameter in custom_force.getEnergyFunction()
            except AttributeError:  # CustomGBForce
                found = False
                for index in range(custom_force.getNumEnergyTerms()):
                    expression, _ = custom_force.getEnergyTermParameters(index)
                    if parameter in expression:
                        found = True
                        break
            return found

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
                if check_energy_expression(force, 'unscaled'):
                    add_label('alchemically modified CustomGBForce', force_index)
                else:
                    add_label('alchemically modified GBSAOBCForce', force_index)
            elif isinstance(force, openmm.CustomBondForce) and check_energy_expression(force, 'lambda'):
                if check_energy_expression(force, 'lambda_sterics'):
                    sterics_bond_forces.append([force_index, force])
                else:
                    electro_bond_forces.append([force_index, force])
            elif (isinstance(force, openmm.CustomNonbondedForce) and force.getEnergyFunction() == '0.0;' and
                          force.getGlobalParameterName(0) == 'lambda_electrostatics'):
                add_label('CustomNonbondedForce holding alchemical atoms unmodified charges', force_index)
            elif isinstance(force, openmm.CustomNonbondedForce) and check_energy_expression(force, 'lambda'):
                if check_energy_expression(force, 'lambda_sterics'):
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
            elif force1.getNumBonds() == 0 and force2.getNumBonds() == 0:
                add_label(label.format(''), force_index1)
                add_label(label.format('non-'), force_index2)

            # We check that the bond atoms are both alchemical or not.
            else:
                atom_i, atom_j, _ = force2.getBondParameters(0)
                both_alchemical = atom_i in alchemical_atoms and atom_j in alchemical_atoms
                if both_alchemical:
                    add_label(label.format(''), force_index2)
                    add_label(label.format('non-'), force_index1)
                else:
                    add_label(label.format('non-'), force_index2)
                    add_label(label.format(''), force_index1)

        return force_labels


if __name__ == '__main__':
    import doctest
    doctest.testmod()
    # doctest.run_docstring_examples(AlchemicalFunction, globals())

