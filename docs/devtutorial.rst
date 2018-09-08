.. _devtutorial:

Advanced features and developer's tutorial (WIP)
************************************************

This tutorial describes more advanced features that can be useful for people developing their software using or extending
OpenMMTools.

.. contents::

Using and implementing integrator and force objects
===================================================

Serialization/deserialization utilities
---------------------------------------

The :ref:`integrators <integrators>` and :ref:`forces <forces>` in OpenMMTools are usually implemented by extending
custom classes in OpenMM. For example, the declaration of our ``LangevinIntegrator`` and the ``HarmonicRestraintForce``
goes boils down to

.. code-block:: python

    class LangevinIntegrator(openmm.CustomIntegrator, mmtools.utils.RestorableOpenMMObject):
        pass

    class HarmonicRestraintForce(openmm.CustomCentroidBondForce, mmtools.utils.RestorableOpenMMObject):
        pass

But what is the purpose of the :ref:`openmmtools.utils.RestorableOpenMMObject <utils>` class? These objects are compatible
with the standard serialization and deserialization in OpenMM

.. code-block:: python

    import openmmtools as mmtools

    integrator = mmtools.integrators.LangevinIntegrator()
    integrator_serialization = openmm.XmlSerializer.serialize(integrator)
    deserialized_integrator = openmm.XmlSerializer.deserialize(integrator_serialization)

    t4_system = mmtools.testsystems.LysozymeImplicit().system
    harmonic_restraint = mmtools.forces.HarmonicRestraintForce(spring_constant=0.2*unit.kilocalories_per_mole/unit.angstrom**2,
                                                               restrained_atom_indices1=[0, 1, 2],
                                                               restrained_atom_indices2=[2604, 2605, 206])
    t4_system.addForce(harmonic_restraint)
    t4_system_serialization = openmm.XmlSerializer.serialize(t4_system)
    deserialized_t4_system = openmm.XmlSerializer.deserialize(t4_system_serialization)

However, the serialization in OpenMM is performed at the C++ level, and thus it doesn't keep track of the Python class
and methods.

.. code-block:: python

    >>> deserialized_integrator.getTemperature()
    Traceback (most recent call last):
    ...
    AttributeError: type object 'object' has no attribute '__getattr__'

Inheriting from :ref:`openmmtools.utils.RestorableOpenMMObject <utils>` allows you to easily recover the original interface
after deserializing.

    >>> mmtools.RestorableOpenMMObject.restore_interface(deserialized_integrator)
    True
    >>> deserialized_integrator.getTemperature()
    Quantity(value=380.0, unit=kelvin)
    >>> isinstance(deserialized_integrator, mmtools.integrators.LangevinIntegrator)
    True

For forces, the function :ref:`openmmtools.forces.find_forces(system) <forces>` automatically calls
``RestorableOpenMMObject.restore_interface()`` on all ``system`` forces.

Integrators coupled to a heat bath
----------------------------------

If you implement an integrator coupled to a heat bath, you have to expose a getter and setter for the temperature or
``ThermodynamicState`` won't have any way to recognize it, and it will add an ``AndersenThermostat`` force when
initializing the OpenMM ``Context`` object.

The base class :ref:`openmmtools.integrators.ThermostatedIntegrator <integrators>` is a convenience class implemented for
this purpose. Inheriting from ``ThermostatedIntegrator`` will implicitly add the ``RestorableOpenMMObject`` functionalities
as well.

.. code-block:: python

    >>> class TestIntegrator(mmtools.integrators.ThermostatedIntegrator):
    ...     def __init__(self, temperature=298.0*unit.kelvin, timestep=1.0*unit.femtoseconds):
    ...         super(TestIntegrator, self).__init__(temperature, timestep)
    ...

    >>> integrator = TestIntegrator(temperature=350*unit.kelvin)
    >>> integrator.getTemperature()
    Quantity(value=350.0, unit=kelvin)
    >>> integrator.setTemperature(380.0*unit.kelvin)

Using standard Python attribute in custom integrators and forces
----------------------------------------------------------------

**TODO: Any Python parameter not affecting the serialization string is a problem.**

|

Handling thermodynamic states and the ContextCache
==================================================

Modifying a System object buried in a ThermodynamicState
--------------------------------------------------------

Setting a thermodynamic parameter in ``ThermodynamicState`` is practically instantaneous, but modifying anything else
involves the copy of the internal ``System`` object so it can be very slow.

.. code-block:: python

    thermo_state = ThermodynamicState(system, temperature=300*unit.kelvin)
    thermo_state.pressure = 1.0*unit.atmosphere  # This is super fast.
    system = thermo_state.system  # This is a copy! Changes to this System won't affect thermo_state.
    # Make your changes to system.
    thermo_state.system = system  # This involves another System copy.

The copies are there to ensure the consistency of ``ThermodynamicState`` internal state. If you need to consistently
modifying part of the systems during the simulation consider implementing a composable state that handle those degrees
of freedom (see next section).

Another thing to keep in mind is that by default the property ``ThermodynamicState.system`` will return a ``System``
containing an ``AndersenThermostat`` force. If you only use ``ThermodynaicState.create_context()`` or the ``ContextCache``
class to create OpenMM ``Context`` objects, this shouldn't cause issues, but if for any reason you don't want that
thermostat you can use the getter instead of the property.

.. code-block:: python

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
standard value so that ``System`` XML serialized strings that differ by only those parameters will be identical.

See next section for info on how the composable states expand the concept of compatibility to thermodynamic parameters
other than temperature and pressure.

.. note:: As a consequence of how the compatibility hash is computed, two ThermodynamicStates to be compatible must have Systems with the same particles and forces in the same order, or the XML serialization will be different.

Copying and initialization of multiple thermodynamic states
-----------------------------------------------------------

Because of some memory optimizations, copying a ``ThermodynamicState`` or a ``CompoundThermodynamicState`` does not copy
the internal ``System`` so it is practically instantaneous. On the other hand, initializing a new ``ThermodynamicState``
or a ``CompoundThermodynamicState`` object does involve a ``System`` copy.

.. code-block:: python

    thermo_state1 = ThermodynamicState(system, temperature=300*unit.kelvin)

    # Very fast.
    new_thermo_state = copy.deepcopy(thermo_state)
    new_thermo_state.temperature = 350*unit.kelvin

    # Slow.
    new_thermo_state = ThermodynamicState(system, temperature=350*unit.kelvin)

**TODO: Example optimized function to initialize a bunch of (Compound)ThermodynamicStates.**

Optimized computation of the energies of a single configuration at multiple thermodynamic states
------------------------------------------------------------------------------------------------

**TODO: Example optimized function to compute only a the terms of the Hamiltonian that have changed.**

|

Implementing a new ComposableState
==================================

Just need to implement the :ref:`IComposableState <states>` interface.

.. code-block:: python

    class AlchemicalState(object):

        def __init__(self):
            self.lambda_electrostatics = 1.0
            self.lambda_sterics = 1.0

        def apply_to_system(self, system):
            """Set lambda parameters in the system."""

        def check_system_consistency(self, system):
            """Raise AlchemicalStateError if system has different lambda parameters."""

        def apply_to_context(self, context):
            """Set lambda parameters in the context."""

        def _standardize_system(cls, system):
            """Set all lambda parameters of the system to 1.0"""

        def _on_setattr(self, standard_system, attribute_name):
            """Check if standard system needs to be updated after a state attribute is set."""
            return False  # No change in alchemical state can alter the standard system.

        def _find_force_groups_to_update(self, context, current_context_state, memo)
            """Find the force groups whose energy must be recomputed after apply_to_context."""
            pass  # Optional, optimization.

If your composable state control a global variable of your ``System`` then creating a composable state class is very
easy. This is literally the most recent implementation of the composable state used to control the restraints in YANK.

.. code-block:: python

    class RestraintState(mmtools.states.GlobalParameterState):

        lambda_restraints = mmtools.states.GlobalParameterState.GlobalParameter('lambda_restraints', standard_value=1.0)

        @lambda_restraints.validator
        def lambda_restraints(self, instance, new_value):
            if new_value is not None and not (0.0 <= new_value <= 1.0):
                raise ValueError('lambda_restraints must be between 0.0 and 1.0')
            return new_value

        # A constructor just to give parameters_name_suffix a more meaningful name.
        def __init__(self, restraint_name=None, **kwargs):
            super().__init__(parameters_name_suffix=restraint_name, **kwargs)

|

ContextCache and Platform with MCMCMoves
========================================

- Context cache and platform in MCMCMove
   - global context cache
   - local context cache and platforms
   - deactivate caching
   - in SequenceMove they can be all different

.. code-block:: python

    local_cache = ContextCache(platform=openmm.Platform.getPlatformByName('CPU'))
    dummy_cache = DummyContextCache()  # Create a new Context everytime. Basically disables caching.
    move = SequenceMove(move_list=[
        MCDisplacementMove(atom_subset=ligand_atoms, context_cache=local_cache),
        MCRotationMove(atom_subset=ligand_atoms, context_cache=dummy_cache),
        LangevinSplittingDynamicsMove()  # Uses global_context_cache.
    ])

|

Implementing a new MCMCMove
===========================
- Implement apply. Anything can happen in there as long as you update the states. Use context cache.
- Casting integrators in MCMCMoves
   - Doesn't work if your integrator changes the thermodynamic state but you can subclass BaseIntegratorMove.
- Extending Metropolized moves
