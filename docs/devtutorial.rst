.. _devtutorial:

Advanced features and developer's tutorial
******************************************

This tutorial describes more advanced features that can be useful for people developing their software using or extending
OpenMMTools.

.. contents::

.. testsetup::

    import copy
    import numpy

Using and implementing integrator and force objects
===================================================

Copy and serialization utilities
--------------------------------

The :ref:`integrators <integrators>` and :ref:`forces <forces>` in OpenMMTools are usually implemented by extending
custom classes in OpenMM. For example, the declaration of our ``LangevinIntegrator`` and the ``HarmonicRestraintForce``
goes boils down to

.. code-block:: python

    class LangevinIntegrator(mmtools.utils.RestorableOpenMMObject, openmm.CustomIntegrator):
        pass

    class HarmonicRestraintForce(mmtools.utils.RestorableOpenMMObject, openmm.CustomCentroidBondForce):
        pass

The purpose of inheriting from :ref:`openmmtools.utils.RestorableOpenMMObject <utils>` class has to do with copies and
serialization. Without ``RestorableOpenMMObject``, these objects can still be copied and go through the standard
serialization and deserialization in OpenMM without errors

.. testcode::

    from simtk import openmm, unit

    class VelocityVerlet(openmm.CustomIntegrator):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.addComputePerDof("x", "x+dt*v")
            self.addComputePerDof("v", "v+0.5*dt*f/m")

        def my_method(self):
            return 0.0

    integrator = VelocityVerlet(1*unit.femtosecond)
    copied_integrator = copy.deepcopy(integrator)
    integrator_serialization = openmm.XmlSerializer.serialize(integrator)
    deserialized_integrator = openmm.XmlSerializer.deserialize(integrator_serialization)

However, copies and serializations in OpenMM are performed at the C++ level, and thus they don't keep track of the Python
class and methods.

.. doctest::

    >>> print(type(copied_integrator))
    <class 'simtk.openmm.openmm.CustomIntegrator'>

    >>> deserialized_integrator.my_method()
    Traceback (most recent call last):
    ...
    AttributeError: type object 'object' has no attribute '__getattr__'

Inheriting from :ref:`openmmtools.utils.RestorableOpenMMObject <utils>`, allows you to easily recover the original interface
after copying or deserializing. This happens automatically for copies, but you'll have to use ``RestorableOpenMMObject.restore_interface()``
after deserialization.

.. testcode::

    from openmmtools import utils

    class VelocityVerlet(utils.RestorableOpenMMObject, openmm.CustomIntegrator):

        def __init(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.addComputePerDof("x", "x+dt*v")
            self.addComputePerDof("v", "v+0.5*dt*f/m")

        def my_method(self):
            return 0.0

    integrator = VelocityVerlet(1*unit.femtosecond)

.. doctest::

    >>> copied_integrator = copy.deepcopy(integrator)
    >>> isinstance(copied_integrator, VelocityVerlet)
    True

.. doctest::

    >>> integrator_serialization = openmm.XmlSerializer.serialize(integrator)
    >>> deserialized_integrator = openmm.XmlSerializer.deserialize(integrator_serialization)
    >>> utils.RestorableOpenMMObject.restore_interface(deserialized_integrator)
    True
    >>> deserialized_integrator.my_method()
    0.0

For forces, the function :ref:`openmmtools.forces.find_forces(system) <forces>` automatically calls
``RestorableOpenMMObject.restore_interface()`` on all ``system`` forces so there's usually no need to perform that
call after deserialization.

Integrators coupled to a heat bath
----------------------------------

If you implement an integrator coupled to a heat bath, you have to expose `getTemperature` and `setTemperature` methods
or ``ThermodynamicState`` won't have any way to recognize it, and it will add an ``AndersenThermostat`` force when
initializing the OpenMM ``Context`` object.

The base class :ref:`openmmtools.integrators.ThermostatedIntegrator <integrators>` is a convenience class implemented for
this purpose. Inheriting from ``ThermostatedIntegrator`` will implicitly add the ``RestorableOpenMMObject`` functionalities
as well.

.. doctest::

    >>> from openmmtools import integrators

    >>> class MyIntegrator(integrators.ThermostatedIntegrator):
    ...     def __init__(self, temperature=298.0*unit.kelvin, timestep=1.0*unit.femtoseconds):
    ...         super().__init__(temperature, timestep)
    ...
    >>> integrator = MyIntegrator(temperature=350*unit.kelvin)
    >>> integrator.getTemperature()
    Quantity(value=350.0, unit=kelvin)
    >>> integrator.setTemperature(380.0*unit.kelvin)

Using standard Python attribute in custom integrators and forces
----------------------------------------------------------------

You should avoid having pure Python attributes when inheriting from custom OpenMM integrators and forces and instead
favor using properties that read that attribute from the underlying OpenMM object as, for example, a global variable.

For example, an integrator exposing the temperature should **not** hold a simple ``temperature`` Python attribute
internally such as

.. testcode::

    class INCORRECTIntegrator(openmm.CustomIntegrator):

        def __init__(self, *args, temperature=298.15*unit.kelvin, **kwargs):
            super().__init__(*args, **kwargs)
            self.temperature = temperature

but it expose it as a getter or a property similarly to the follow.

.. testcode::

    class CorrectIntegrator(openmm.CustomIntegrator):

        def __init__(self, *args, temperature=298.15*unit.kelvin, **kwargs):
            super().__init__(*args, **kwargs)
            self.addGlobalVariable('temperature', temperature)

        @property
        def temperature(self):
            return self.getGlobalVariableByName('temperature') * unit.kelvin

This is because:

    1. If the parameter doesn't affect serialization ``ContextCache`` won't be able to distinguish between two integrators
       that differ by that parameter, and it may return an incorrect integrator.
    2. Python attribute cannot be restored by ``RestorableOpenMMObject`` since there's no information about them in the XML
       string, and thus they will be lost with serialization.

|

Handling (compound) thermodynamic states
========================================

In the examples that follow, we'll use a simple ``ThermodynamicState``, but everything applies to ``CompoundThermodynamicState``
as well as ``CompoundThermodynamicState`` is a subclass of ``ThermodynamicState``.

Modifying a System object buried in a ThermodynamicState
--------------------------------------------------------

Setting a thermodynamic parameter in ``ThermodynamicState`` is practically instantaneous, but modifying anything else
involves the copy of the internal ``System`` object so it can be very slow.

.. testcode::

    from openmmtools import states
    from openmmtools import testsystems

    system = testsystems.TolueneVacuum().system
    thermo_state = states.ThermodynamicState(system, temperature=300*unit.kelvin)

    # This is very fast.
    thermo_state.temperature = 400.0*unit.kelvin

    system = thermo_state.system  # This is a copy! Changes to this System won't affect thermo_state.
    # Make your changes to system.
    thermo_state.system = system  # This involves another System copy.

The copies are there to ensure the consistency of ``ThermodynamicState`` internal state. If you need to consistently
modifying part of the systems during the simulation consider implementing a composable state that handle those degrees
of freedom (see section `Implementing a new ComposableState`_).

Another thing to keep in mind is that by default the property ``ThermodynamicState.system`` will return a ``System``
containing an ``AndersenThermostat`` force. If you only use ``ThermodynaicState.create_context()`` or the ``ContextCache``
class to create OpenMM ``Context`` objects, this shouldn't cause issues, but if for any reason you don't want that
thermostat you can use the getter instead of the property.

.. testcode::

    system = thermo_state.get_system(remove_thermostat=True)

Implementation details of compatibility checks
----------------------------------------------

Internally, ``ThermodynamicState`` associates a unique hash to a ``System`` in a particular ensemble, and it compares
this hash to check for compatibility. The function that performs this task looks like this:

.. code-block:: python

    @classmethod
    def _standardize_and_hash(cls, system):
        """Standardize the system and return its hash."""
        cls._standardize_system(system)
        system_serialization = openmm.XmlSerializer.serialize(system)
        return system_serialization.__hash__()

The ``_standardize_system()`` functions sets the thermodynamic parameters controlled by the ``ThermodynamicState`` to a
standard value so that ``System`` that differ by only those parameters will have identical XML serialized strings, and
thus identical hashes.

The section `Implementing a new ComposableState`_ has information on how the composable states expand the concept of
compatibility to thermodynamic parameters other than temperature and pressure.

.. note:: As a consequence of how the compatibility hash is computed, two ThermodynamicStates to be compatible must have Systems with the same particles and forces in the same order, or the XML serialization will be different.

Copying and initialization of multiple thermodynamic states
-----------------------------------------------------------

Because of some memory optimizations, copying a ``ThermodynamicState`` or a ``CompoundThermodynamicState`` does not copy
the internal ``System`` so it is practically instantaneous. On the other hand, initializing a new ``ThermodynamicState``
or a ``CompoundThermodynamicState`` object does involve a ``System`` copy.

.. testcode::

    thermo_state1 = states.ThermodynamicState(system, temperature=300*unit.kelvin)

    # Very fast.
    thermo_state2 = copy.deepcopy(thermo_state)
    thermo_state2.temperature = 350*unit.kelvin

    # Slow.
    thermo_state2 = states.ThermodynamicState(system, temperature=350*unit.kelvin)

The function :ref:`openmmtools.states.create_thermodynamic_state_protocol <states>` takes advantage of this to make it easy
to instantiate a list of ``ThermodynamicState`` or ``CompoundThermodynamicState`` objects that differ only by the controlled
parameters.

|

Implementing a new ComposableState
==================================

The IComposableInterface
------------------------

Composable states allow to control thermodynamic parameters of the simulation while masking their implementation details.
There are no restrictions on the implementation details, but the class must implement the :ref:`openmmtools.states.IComposableState <states>`
interface. You can see the API docs for contract details, but here is a list of the methods.

.. code-block:: python

    class IComposableState:

        def apply_to_system(self, system):
            """Modify an OpenMM System to be in this thermodynamic state."""

        def check_system_consistency(self, system):
            """Raise AlchemicalStateError if system has different parameters."""

        def apply_to_context(self, context):
            """Modify an OpenMM Context to be in this thermodynamic state."""

        def _standardize_system(cls, system):
            """Modify the System to be in the standard thermodynamic state."""

        def _on_setattr(self, standard_system, attribute_name, old_attribute_value):
            """Callback that checks if standard system needs to be updated after a state attribute is set."""

        def _find_force_groups_to_update(self, context, current_context_state, memo)
            """Find the force groups whose energy must be recomputed after apply_to_context."""
            # Optional. This is used only for optimizations.

The ``_standardize_system`` method effectively determines which other states will be compatible (see also section
`Implementation details of compatibility checks`_). The purpose of ``_standardize_system`` is to set the parameters of
the ``System`` that can be manipulated in the ``Context`` to the same value so that their XML serialization string and
their hash will be identical. Systems that after standardization are identical are assigned to the same ``Context`` by
``ContextCache.get_context()``.

Relatedly, the callback ``_on_setattr()`` is called by ``CompoundThermodynamicState`` after a thermodynamic parameter
has been set. The method must return ``True`` if the change in the thermodynamic parameter has caused the standard system
to have a different hash. For example, in the basic ``ThermodynamicState`` class this happens when the ``pressure``
parameter goes from ``None`` to any valid value because states in NVT and NPT are not compatible.

The method ``_find_force_groups_to_update`` is optional and related to the optimization described in
`Computing the reduced potential of one configuration at multiple thermodynamic states`_.

ComposableStates that control a Force global parameter
------------------------------------------------------

Often, a thermodynamic parameter can be implemented with OpenMM as a global parameter added to a custom force. For
example, to alchemically soften torsions, ``alchemy.AbsoluteAlchemicalFactory`` substitute some of the torsion potential
terms using a ``openmm.CustomTorsionForce`` whose energy is multiplied by a global parameter called ``lambda_torsions``.

.. code-block:: python

    energy_function = "lambda_torsions * k*(1+cos(periodicity*theta-phase))"
    custom_force = openmm.CustomTorsionForce(energy_function)
    custom_force.addGlobalParameter('lambda_torsions', 1.0)
    # Other force configurations.
    system.addForce(custom_force)

When this is the case, the base class ``openmmtools.states.GlobalParameterState`` can be used to create a composable state
very quickly.

.. testcode::

    from openmmtools.states import GlobalParameterState

    class MyComposableState(GlobalParameterState):

        lambda_torsions = GlobalParameterState.GlobalParameter('lambda_torsions', standard_value=1.0)

It is possible to perform checks on the assigned value by adding a validator.

.. testcode::

    class MyComposableState(GlobalParameterState):

        lambda_torsions = GlobalParameterState.GlobalParameter('lambda_torsions', standard_value=1.0)

        @lambda_torsions.validator
            def lambda_torsions(self, instance, new_value):
                if new_value is not None and not (0.0 <= new_value <= 1.0):
                    raise ValueError('lambda_torsions must be between 0.0 and 1.0')
                return new_value

The example above allows only values between 0.0 and 1.0 for ``lambda_torsions``.

Computing the reduced potential of one configuration at multiple thermodynamic states
-------------------------------------------------------------------------------------

When computing the potential energy of a single configuration at multiple thermodynamic states, it is often unnecessary
to compute the whole Hamiltonian multiple times but just the terms of the Hamiltonian that change from one state to
another. OpenMM makes this possible to compute only the energy of a subset of forces through the force groups mechanism.

.. code-block:: python

    force = openmm.CustomBondForce('(K/2)*(r-r0)^2;')
    force.setForceGroup(5)

The utility function ``openmmtools.states.reduced_potential_at_states()`` takes advantage of forces separated in different
groups to efficiently compute the reduced potentials at the thermodynamic states.

.. testcode::

    from openmmtools import alchemy
    from openmmtools import cache

    alanine = testsystems.AlchemicalAlanineDipeptide()
    protocol = {'lambda_sterics': [1.0, 0.5, 0.0],
                'lambda_electrostatics': [1.0, 0.5, 0.0]}
    constants = {'temperature': 300*unit.kelvin}
    composable_states = [alchemy.AlchemicalState.from_system(alanine.system)]
    compound_states = states.create_thermodynamic_state_protocol(alanine.system, protocol,
                                                                 constants, composable_states)

    sampler_state = states.SamplerState(positions=alanine.positions)
    reduced_potentials = states.reduced_potential_at_states(sampler_state, compound_states,
                                                            cache.global_context_cache)

In order for the optimization to take effect, the composable states must implement the method
``_find_force_groups_to_update(self, context, current_context_state, memo)``. This method inspects the ``System``
associated to the ``context`` and return the force groups that will have an updated energy after the state will be changed
from ``current_context_state`` to ``self``. The ``memo`` dictionary can be use to store the force groups to inspect in
subsequent calls of the method within a ``reduced_potential_at_states`` execution so that the ``System`` must be parsed
only the first time.

|

Implementing a new MCMCMove
===========================

The MCMCMove interface
----------------------

An ``MCMCMove`` requires exclusively the implementation of an ``apply`` method with the following signature (see the
:ref:`API documentation <mcmc>` for more details.

.. code-block:: python

    class MCMCMove(SubhookedABCMeta):

        def apply(self, thermodynamic_state, sampler_state):
            pass

Anything can happen inside ``apply`` as long as ``thermodynamic_state`` and ``sampler_state`` are updated correctly.
It is usually a good idea to include in the constructor a ``context_cache`` argument to let the user specify how the
``Context`` should be created and on which platform.

OpenMM integrators that modify the thermodnamic state
-----------------------------------------------------

Custom OpenMM integrators can modify global variables that effectively change the thermodynamic state of the ``Context``.

.. important:: Remember to update the ``thermodynamic_state`` object correctly at the end of ``apply`` if the integrator changes the thermodynamic state of the simulation.

When this is the case, it's not possible to cast your integrator into an ``MCMCMove`` with ``IntegratorMove``.
Nevertheless, it's still possible to take advantage of the extra features already offered by ``IntegratorMove`` by
subclassing the `openmmtools.mcmc.BaseIntegratorMove <mcmc>` class. ``IntegratorMove`` inherits from this base class. An
implementation would look more or less like this (see the API documentation for the details).

.. code-block:: python

    class MyMove(BaseIntegratorMove):
        def __init__(self, timestep, n_steps, **kwargs):
            super(MyMove, self).__init__(n_steps, **kwargs)
            self.timestep = timestep

        def _get_integrator(self, thermodynamic_state):
            return MyIntegrator(self.timestep, thermodynamic_state.temperature)

        def _before_integration(self, context, thermodynamic_state):
            # Optional: Any operation performed after the context
            # was created but before integration.

        def _after_integration(self, context, thermodynamic_state):
            # Update thermodynamic_state from context parameters.
            # Optional: Read statistics from context.getIntegrator() parameters.

Metropolized MCMCMoves
----------------------

The `mcmc` module contains a base class for Metropolized moves as well. The following class implement an example that
simply adds the unit vector to the initial coordinates.

.. testcode::

    from openmmtools import mcmc

    class AddOneVector(mcmc.MetropolizedMove):
        def _propose_positions(self, initial_positions):
            print('Propose new positions')
            displacement = numpy.array([1.0, 1.0, 1.0]) * unit.angstrom
            return initial_positions + displacement

The parent class will take care of implementing the Metropolis acceptance criteria, collecting acceptance statistics,
and updating the ``SamplerState`` correctly. The constructor accepts an optional ``atom_subset`` to limit the move to
certain atoms. In this case, the ``initial_positions`` will be the positions of the atom subset only.

.. doctest::

    >>> alanine = testsystems.AlanineDipeptideVacuum()
    >>> sampler_state = states.SamplerState(alanine.positions)
    >>> thermodynamic_state = states.ThermodynamicState(alanine.system, 300*unit.kelvin)
    >>> move = AddOneVector(atom_subset=list(range(sampler_state.n_particles)))
    >>> move.apply(thermodynamic_state, sampler_state)
    Propose new positions
    >>> move.n_accepted
    1
    >>> move.n_proposed
    1
