.. _gettingstarted:

Getting started tutorial
************************

This tutorial will give you an overview of what you can find in OpenMMTools and how you can use the library.

.. contents::

.. testsetup::

    import copy
    import numpy
    n_steps = 0  # Step 0 so that doctest don't take forever.

    from simtk import openmm
    platform = openmm.Platform.getPlatformByName('CPU')

Test systems, integrators, and forces
=====================================

In its basic usage, OpenMMTools extends OpenMM by providing pre-packaged systems, integrators of force objects that are
not natively implemented in OpenMM.

Test systems
------------

The :ref:`testsystems <testsystems>` module comes with many simulation-ready molecular systems (from analytically-solvable
systems to a kinase in explicit solvent) that can be useful for prototyping, validation, testing, and benchmarking. The
code below creates a TIP3P water cubic box of 2nm side using PME.

.. testcode::

    from simtk import openmm, unit
    from openmmtools import testsystems

    water_box = testsystems.WaterBox(box_edge=2.0*unit.nanometer)
    system = water_box.system  # An OpenMM System object.
    positions = water_box.positions  # Initial coordinates for the system with associated units.

You can use select a subset of the system atoms using the `atom selection domain-specific language (DSL) <http://mdtraj.org/latest/atom_selection.html>`_
implemented in MDTraj. For example, the following snippet create a T4-Lysozyme system in implicit OBC GBSA solvent bound
to a p-xylene molecule, and finds the atom indices corresponding to the heavy atoms of p-xylene and few residues surrounding
the binding site of T4-Lysozyme.

.. testcode::

    lysozyme_pxylene = testsystems.LysozymeImplicit()
    t4_system = lysozyme_pxylene.system
    pxylene_dsl = '(resname TMP) and (mass > 1.5)'  # Select heavy atoms of p-xylene.
    binding_site_dsl = ('(resi 77 or resi 86 or resi 101 or resi 110 or '
                        ' resi 117 or resi 120) and (mass > 1.5)')
    pxylene_atom_indices = lysozyme_pxylene.mdtraj_topology.select(pxylene_dsl).tolist()
    binding_site_atom_indices = lysozyme_pxylene.mdtraj_topology.select(binding_site_dsl).tolist()

Integrators
-----------

The systems created by ``testsystems`` can then be propagated in the usual way with OpenMM. The :ref:`integrators <integrators>`
module provide several high-quality integrators for equilibrium and non-equilibrium simulations in OpenMM.

.. testcode::

    from openmmtools import integrators

    integrator = integrators.LangevinIntegrator(temperature=298.0*unit.kelvin,
                                                collision_rate=1.0/unit.picoseconds,
                                                timestep=1.0*unit.femtoseconds)
    context = openmm.Context(t4_system, integrator)
    context.setPositions(lysozyme_pxylene.positions)
    integrator.step(n_steps)

Our ``LangevinIntegrator`` allows you to specify the splitting used to carry out the numerical integration.
By default, OpenMMTools will construct a BAOAB integrator (i.e. with `V R O R V` splitting), which was shown empirically
to add a very small integration error in configurational space, but other solutions are possible.

.. testcode::

    integrator = integrators.LangevinIntegrator(splitting="V0 V1 R R O R R V1 R R O R R V1 V0",
                                                measure_shadow_work=True, measure_heat=True)
    context = openmm.Context(t4_system, integrator)
    context.setPositions(lysozyme_pxylene.positions)
    integrator.step(n_steps)

    # Obtain the dissipated heat accumulated during Langevin dynamics in molar energy units.
    heat = integrator.get_heat()

The integrator above, for example, implements the geodesic-BAOAB Langevin integrator with solute-solvent splitting, and
it collects statistics on the dissipated heat and the shadow work during the propagation (at the cost of a computational
overhead).

Forces
------

The :ref:`forces <forces>` module is still under construction, but it already provides a few convenient utility
functions and force objects. Let's create a T4-Lysozyme system in implicit OBC GBSA solvent bound to a p-xylene and add
a harmonic restraint between the two molecules.

.. testcode::

    from openmmtools import forces

    harmonic_restraint = forces.HarmonicRestraintForce(spring_constant=0.2*unit.kilocalories_per_mole/unit.angstrom**2,
                                                       restrained_atom_indices1=binding_site_atom_indices,
                                                       restrained_atom_indices2=pxylene_atom_indices)
    t4_system.addForce(harmonic_restraint)

The restraint force above will place a single harmonic potential between the centers of mass of the heavy atoms of the
p-xylene molecule and the binding site of T4-Lysozyme.

The function ``forces.find_forces()`` provides a convenient way to search for particular force objects in the OpenMM
``System``.

.. testcode::

    # Retrieve our harmonic restraint force.
    forces.find_forces(t4_system, force_type=forces.HarmonicRestraintForce)

    # Find all forces that inherit from an OpenMM CustomBondForce object.
    forces.find_forces(t4_system, force_type=openmm.CustomBondForce, include_subclasses=True)

    # Search for force names using regular expressions.
    # Return all openmm.HarmonicBondForce, openmm.HarmonicAngleForce,
    # and forces.HarmonicRestraintForce force objects.
    forces.find_forces(t4_system, '.*Harmonic.*')

|

Alchemical transformations
==========================

The :ref:`alchemy <alchemy>` module provides helper classes to perform alchemical transformations with OpenMM.

AbsoluteAlchemicalFactory and AlchemicalState
---------------------------------------------

The ``AbsoluteAlchemicalFactory`` class prepare OpenMM ``System`` objects for alchemical manipulation. Let's create an
alchemical system that we can use to alchemically decouple p-xylene from T4-lysozyme's binding pocket.

.. doctest::

    >>> from openmmtools import alchemy

    >>> # Create the reference OpenMM System that will be alchemically modified.
    >>> lysozyme_pxylene = testsystems.LysozymeImplicit()
    >>> t4_system = lysozyme_pxylene.system

    >>> # Define the region of the System to be alchemically modified.
    >>> pxylene_atoms = lysozyme_pxylene.mdtraj_topology.select('resname TMP')
    >>> alchemical_region = alchemy.AlchemicalRegion(alchemical_atoms=pxylene_atoms)

    >>> factory = alchemy.AbsoluteAlchemicalFactory()
    >>> alchemical_system = factory.create_alchemical_system(t4_system, alchemical_region)

At this point, the p-xylene in alchemical ``System`` is in its interacting state and it can be then simulated normally

.. doctest::

    >>> integrator = integrators.LangevinIntegrator()
    >>> context = openmm.Context(alchemical_system, integrator)
    >>> context.setPositions(lysozyme_pxylene.positions)
    >>> integrator.step(n_steps)

The alchemical degrees of freedom of the Hamiltonian can be controlled during the simulation through the ``AlchemicalState``
class.

.. doctest::

    >>> alchemical_state = alchemy.AlchemicalState.from_system(alchemical_system)
    >>> alchemical_state.lambda_electrostatics = 0.0
    >>> alchemical_state.lambda_sterics = 0.5
    >>> alchemical_state.apply_to_context(context)

The snippet above modifies the simulated ``System`` to completely turn off the electrostatics interaction and halve the
Lennard-Jones potential between p-xylene and its environment.

.. note:: In OpenMMTools, the convention is to have the interacting state at lambda=1 and the non-interacting state at lambda=0. Some packages adopt the opposite convention.

.. note:: The ``AbsoluteAlchemicalFactory`` class is currently specialized for absolute calculations in the sense that it cannot prepare an OpenMM ``System`` to have an atom changing its element or turn on part of a molecule while decoupling another set of atoms. We're planning to provide these capabilities in the near future.

Decoupling vs annihilating and softcore nonbonded interactions
--------------------------------------------------------------

By default, the alchemical ``System`` is prepared to annihilate electrostatics (i.e. turn off the alchemical atoms' charges)
and decouple the sterics (i.e. preserve the intra-molecular Lennard-Jones interactions), but you can maintain the
intra-molecular charges, for example, by configuring the alchemical region.

.. testcode::

    alchemical_region = alchemy.AlchemicalRegion(alchemical_atoms=pxylene_atoms,
                                                 annihilate_electrostatics=True)
    alchemical_system = factory.create_alchemical_system(t4_system, alchemical_region)

Similarly, you can set specific softcore parameters for the sterics and electrostatics interactions (see the API documentation
for a detailed explanation of the parameters).

.. testcode::

    alchemical_region = alchemy.AlchemicalRegion(alchemical_atoms=pxylene_atoms,
                                                 softcore_alpha=0.5, softcore_c=6)

Softening torsions, angles, and bonds
-------------------------------------

Beside nonbonded interactions, it is possible to modify other terms of the potentials. The following alchemical region
is configured to modify the OpenMM ``System`` to enable torsion softening of all the p-xylene dihedrals. The Hamiltonian
parameter controlling the torsion, angles, and bond potential terms can be controlled with ``AlchemicalState`` in the
same way as with nonbonded interactions.

.. testcode::

    alchemical_region = alchemy.AlchemicalRegion(alchemical_atoms=pxylene_atoms,
                                                 alchemical_torsions=True)
    alchemical_system = factory.create_alchemical_system(t4_system, alchemical_region)
    context = openmm.Context(alchemical_system, integrators.LangevinIntegrator())

    alchemical_state = alchemy.AlchemicalState.from_system(alchemical_system)
    alchemical_state.lambda_torsions = 0.8
    alchemical_state.apply_to_context(context)

Alchemical functions
--------------------

Finally you can enslave the degrees of freedom of the Hamiltonian to a variable through a custom function. The code
below configure the ``AlchemicalState`` to turn off first electrostatic and the steric interactions one after the other
as a generic variable called ``lambda`` goes from ``1.0`` to ``0.0``.

.. testcode::

    # Enslave lambda_sterics and lambda_electrostatics to a generic lambda variable.
    alchemical_state.set_function_variable('lambda', 1.0)

    # The functions here turn off first electrostatic and the steric interactions
    # in sequence as lambda goes from 1.0 to 0.0.
    f_electrostatics = '2*(lambda-0.5)*step(lambda-0.5)'
    f_sterics = '2*lambda*step_hm(0.5-lambda) + step_hm(lambda-0.5)'
    alchemical_state.lambda_electrostatics = alchemy.AlchemicalFunction(f_electrostatics)
    alchemical_state.lambda_sterics = alchemy.AlchemicalFunction(f_sterics)

    alchemical_state.set_function_variable('lambda', 0.75)
    assert alchemical_state.lambda_electrostatics == 0.5
    assert alchemical_state.lambda_sterics == 1.0

    alchemical_state.set_function_variable('lambda', 0.25)
    assert alchemical_state.lambda_electrostatics == 0.0
    assert alchemical_state.lambda_sterics == 0.5

    # Set the alchemical state of the simulated system.
    alchemical_state.apply_to_context(context)

In the example above, ``step_hm`` is the Heaviside step function with half-maximum convention (i.e. ``step_hm(0.0) == 0.5``),
while ``step(0.0) == 0.0``. All the functions in the Python standard module ``math`` can be specified in the string.

|

Manipulating the thermodynamic state of your simulation
=======================================================

The classes in the :ref:`states <states>` module provide a framework to decouple the degrees of freedom (or parameters)
of the simulated thermodynamic state from their implementation details in OpenMM.

Defining temperature and pressure
---------------------------------

The fundamental class in the ``states`` module is ``ThermodynamicState``. This class hold a ``System`` object and controls
the ensemble parameters of temperature and pressure. For example, the code below creates a water box in NVT ensemble at
298 K.

.. doctest::

    >>> from openmmtools import states

    >>> waterbox = testsystems.WaterBox(box_edge=2*unit.nanometers)
    >>> thermo_state = states.ThermodynamicState(system=waterbox.system,
    ...                                          temperature=298.0*unit.kelvin)
    >>> thermo_state.volume.format('%.1f')
    '8.0 nm**3'
    >>> assert thermo_state.pressure is None

The volume is computed from the box vectors associated to the ``System`` object. To convert the system to an NPT state
at 298 K and 1 atm pressure, you can set the ``pressure`` attribute.

.. testcode::

    thermo_state.pressure = 1.0*unit.atmosphere
    assert thermo_state.volume is None

Note that the operation of specifying a constant pressure result in a null volume, as the volume will fluctuate during
the simulation. You can then create an OpenMM ``Context`` object that is guaranteed to be in the specified thermodynamic
state.

.. doctest::

    >>> integrator = integrators.LangevinIntegrator(temperature=298.0*unit.kelvin)
    >>> context = thermo_state.create_context(integrator)
    >>> context.setPositions(waterbox.positions)
    >>> integrator.step(n_steps)

    >>> # ThermodynamicState takes care of adding and configuring a MonteCarloBarostatForce
    >>> # to keep the pressure at 1atm.
    >>> force_index, barostat = forces.find_forces(context.getSystem(),
    ...                                            openmm.MonteCarloBarostat,
    ...                                            only_one=True)
    >>> barostat.getDefaultTemperature().format('%.1f')
    '298.0 K'
    >>> print(barostat.getDefaultPressure())
    1.01325 bar

Consistency checks for free
---------------------------

Using the ``ThermodynamicState`` class means to take advantage of several consistency checks that can avoid bugs in your
application that can be very hard to detect in the first place and then to track down (we speak from personal experience).

For example, trying to create a ``Context`` using Langevin integrator set to the incorrect temperature or trying to add
a barostat to a system in vacuum raises an error.

.. doctest::

    >>> thermo_state.temperature = 298.0*unit.kelvin
    >>> integrator = integrators.LangevinIntegrator(temperature=310.0*unit.kelvin)
    >>> thermo_state.create_context(integrator)
    Traceback (most recent call last):
    ...
    ThermodynamicsError: Integrator is coupled to a heat bath at a different temperature.

.. doctest::

    >>> vacuum_system = testsystems.TolueneVacuum().system
    >>> thermo_state = states.ThermodynamicState(system=vacuum_system,
    ...                                          temperature=298.15*unit.kelvin,
    ...                                          pressure=1.0*unit.atmosphere)
    Traceback (most recent call last):
    ...
    ThermodynamicsError: Non-periodic systems cannot have a barostat.

While, if you create a ``Context`` with an integrator that is not coupled to a heat bath, ``ThermodynamicState`` will
take care of adding an ``AndersenThermostat``.

.. doctest::

    >>> # Use a non-thermostated integrator.
    >>> thermo_state_nvt = states.ThermodynamicState(system=vacuum_system,
    ...                                              temperature=298.15*unit.kelvin)
    >>> integrator = openmm.VerletIntegrator(2.0*unit.femtoseconds)
    >>> context_nvt = thermo_state_nvt.create_context(integrator)
    >>> len(forces.find_forces(context_nvt.getSystem(), openmm.AndersenThermostat))
    1

Manipulating the thermodynamic state: Compatible thermodynamic states
---------------------------------------------------------------------

Once a ``Context`` has been created, is is possible to change the simulation thermodynamic state through the method
``ThermodynamicState.apply_to_context()``. The method will mask the implementation details and take care of modifying
all the OpenMM forces and integrators that depend on the temperature and pressure parameters. In this sense, the
``ThermodynamicState`` class decouples the representation of the thermodynamic parameters from their implementation
details.

.. doctest::

    >>> # Modify temperature and pressure of a system employing a Langevin
    >>> # thermostat and a Monte Carlo barostat.
    >>> thermo_state.temperature = 400.0*unit.kelvin
    >>> thermo_state.pressure = 1.2*unit.bar
    >>> thermo_state.apply_to_context(context)
    >>> context.getIntegrator().getTemperature().format('%.1f')
    '400.0 K'
    >>> context.getParameter(openmm.MonteCarloBarostat.Pressure())
    1.2
    >>> # The MonteCarloBarostat requires also a temperature parameter for the acceptance probability.
    >>> context.getParameter(openmm.MonteCarloBarostat.Temperature())
    400.0

.. doctest::

    >>> # Modify the temperature of a system using an Andersen thermostat.
    >>> thermo_state_nvt.temperature = 400.0*unit.kelvin
    >>> thermo_state_nvt.apply_to_context(context_nvt)
    >>> context_nvt.getParameter(openmm.AndersenThermostat.Temperature())
    400.0

A ``ThermodynamicState`` can be applied to any ``Context`` that was created from a **compatible thermodynamic state**.

.. important:: Two ``ThermodynamicState`` objects ``x, y`` are compatible if a ``context`` created by ``x`` can be modified to be in the ``y`` thermodynamic state through ``y.apply_to_context(context)`` and viceversa.

This is not always possible in OpenMM because of some implementation details related to optimizations. In short,
two ``ThermodynamicState``s are compatible if they have the same ``System`` and they are in the same ensemble (i.e. NVT
and NPT thermodynamic states are incompatible).

.. doctest::

    >>> alanine = testsystems.AlanineDipeptideExplicit()
    >>> state1 = states.ThermodynamicState(alanine.system, 273*unit.kelvin)
    >>> state2 = states.ThermodynamicState(alanine.system, 310*unit.kelvin)
    >>> state1.is_state_compatible(state2)
    True

    # Switch state1 from NVT to NPT ensemble.
    >>> state1.pressure = 1.0*unit.atmosphere
    >>> state1.is_state_compatible(state2)
    False

Luckily, the class :ref:`openmmtools.cache.ContextCache <cache>` takes care of checking for compatibility and decide
whether it's possible to modifying a previously created ``Context`` object or if it is necessary to create a separate
one.

Using the ContextCache
----------------------
.. important:: Using ``ContextCache`` is the recommended way of creating ``Context`` objects within the OpenMMTools framework.

The ``openmmtools.cache.ContextCache`` class has the role of maintaining the *minimum number of compatible Contexts allocated on the GPU*,
allowing virtually an infinite number of thermodynamic states to be simulated on finite-memory hardware, and minimizing
the number of expensive ``Context`` creation/destruction.

To obtain a ``Context`` simply use the ``ContextCache.get_context()`` method.

.. testcode::

    from openmmtools import cache

    alanine = testsystems.AlanineDipeptideExplicit()
    thermo_state = states.ThermodynamicState(alanine.system, 310*unit.kelvin)
    integrator = integrators.LangevinIntegrator(temperature=310*unit.kelvin)

    context_cache = cache.ContextCache()
    context, context_integrator = context_cache.get_context(thermo_state,
                                                            integrator)
    context.setPositions(alanine.positions)
    context_integrator.step(n_steps)

Note that ``get_context()`` returns also an ``Integrator`` that may be a different instance of the ``integrator`` passed
as a parameter. This is because an OpenMM ``Context`` can be associated with a single integrator instance, thus reusing
a previously instantiated ``Context`` requires using the previously instantiated integrator as well. Nevertheless,
``context_integrator`` is guaranteed to be identical to ``integrator``.

Requesting a context in a compatible ``ThermodynamicState`` returns the same ``Context`` object correctly configured to
simulate the requested thermodynamic state.

.. doctest::

    >>> compatible_state = states.ThermodynamicState(alanine.system, 400*unit.kelvin)
    >>> compatible_integrator = integrators.LangevinIntegrator(temperature=400*unit.kelvin)
    >>> compatible_context, compatible_integrator = context_cache.get_context(compatible_state,
    ...                                                                       compatible_integrator)
    >>> id(context) == id(compatible_context)
    True
    >>> len(context_cache)  # The number of Contexts maintained in memory.
    1
    >>> compatible_integrator.getTemperature().format('%.1f')
    '400.0 K'

Requesting a context in a different ensemble causes the creation of another ``Context``.

.. doctest::

    >>> thermo_state_npt = copy.deepcopy(thermo_state)
    >>> thermo_state_npt.pressure = 1.0*unit.atmosphere
    >>> integrator = integrators.LangevinIntegrator(temperature=thermo_state_npt.temperature)
    >>> context_npt, integrator_npt = context_cache.get_context(thermo_state_npt, integrator)
    >>> id(context) == id(context_npt)
    False
    >>> len(context_cache)
    2

You can set a capacity and a time to live for contexts. The time to live is currently measured in number of accesses to
the ``ContextCache``.

.. doctest::

    >>> context_cache = cache.ContextCache(capacity=1, time_to_live=5)
    >>> integrator = openmm.VerletIntegrator(1.0*unit.femtosecond)
    >>> context1, integrator1 = context_cache.get_context(thermo_state,
    ...                                                   copy.deepcopy(integrator))
    >>> context2, integrator2 = context_cache.get_context(thermo_state_npt,
    ...                                                   copy.deepcopy(integrator))
    >>> len(context_cache)
    1

In the example above, the maximum capacity of the cache is 1, so the first context is deallocated to make space for the
second ``Context`` created with the incompatible thermodynamic state.

Finally, you can force the ``ContextCache`` to create contexts on a specific platform.

.. testcode::

    platform = openmm.Platform.getPlatformByName('Reference')
    context_cache = cache.ContextCache(platform=platform)

The global ContextCache
-----------------------

The :ref:`openmmtools.cache <cache>` module exposes a global variable that provides a shared ``ContextCache`` for all the
classes in the framework.

.. testcode::

    cache.global_context_cache.platform = openmm.Platform.getPlatformByName('CPU')
    cache.global_context_cache.capacity = 2
    cache.global_context_cache.time_to_live = 10
    verlet_integrator = openmm.VerletIntegrator(1.0*unit.femtosecond)
    context, integrator = cache.global_context_cache.get_context(thermo_state,
                                                                 verlet_integrator)

Usually, you'll want to create a ``Context`` using the ``global_context_cache`` to minimize the number of created contexts
overall. This is, for example, the context cache used by default by all the ``MCMCMove`` objects internally, which we'll
touch shortly.

Extending ThermodynamicState to control arbitrary parameters
------------------------------------------------------------

It is possible to extend the ``ThermodynamicState`` to manipulate other thermodynamic parameters of the ``System``
through the ``states.CompoundThermodynamicState`` class and one or more *composable states*. An example may clarify
this. Remember the ``alchemy.AlchemicalState`` class we discussed above? ``AlchemicalState`` is a composable state.

.. testcode::

    # Prepare T4-Lysozyme + p-xylene system for alchemical perturbation.
    factory = alchemy.AbsoluteAlchemicalFactory()
    alchemical_region = alchemy.AlchemicalRegion(alchemical_atoms=pxylene_atoms)
    alchemical_system = factory.create_alchemical_system(t4_system, alchemical_region)

    # Define the basic thermodynamic state of the system.
    thermo_state = states.ThermodynamicState(alchemical_system, temperature=298*unit.kelvin)

    # Extend the definition of thermodynamic state to consider alchemical parameters as well.
    alchemical_state = alchemy.AlchemicalState.from_system(alchemical_system)
    compound_state = states.CompoundThermodynamicState(thermodynamic_state=thermo_state,
                                                       composable_states=[alchemical_state])

At this point, ``compound_state`` is *both* a ``ThermodynamicState`` and an ``AlchemicalState`` in the sense that it
exposes the interface to modify the thermodynamic parameters controlled by both objects.

.. testcode::

    context = compound_state.create_context(integrators.LangevinIntegrator())
    compound_state.temperature = 350*unit.kelvin  # Increase temperature of simulation.
    compound_state.lambda_sterics = 0.2  # Soften torsions.
    compound_state.apply_to_context(context)

Obviously, ``CompoundThermodynamicState`` is not compatible exclusively with ``AlchemicalState`` but with any object
implementing the ``states.IComposableState`` interface. A quick way to define your own composable state is described
in the :ref:`developer's tutorial <devtutorial>`.

The power of this abstraction will become evident when we'll implement a simple replica-exchange algorithm at the end of
the tutorial.

|

MCMC framework
==============

The Markov chain Monte Carlo (MCMC) framework implemented in the :ref:`mcmc <mcmc>` module take advantage of the thermodynamic
state objects described above to provide an easy way to experiment with different propagation schemes mixing Monte
Carlo moves and dynamics.

Basic usage
-----------

The basic object in the module is the ``mcmc.MCMCMove`` abstract class that provides a common interface for both
integrators and Monte Carlo to propagate the state of the system.

.. testcode::

    from openmmtools import mcmc

    # Define the thermodynamic state of the T4-Lysozyme + p-xylene system
    thermo_state = states.ThermodynamicState(t4_system, temperature=300*unit.kelvin)

    # Create a SamplerState system holding the coordinates of the system.
    sampler_state = states.SamplerState(positions=lysozyme_pxylene.positions)

    # Propagate the system with a GHMC integrator.
    ghmc_move = mcmc.GHMCMove(timestep=1.0*unit.femtosecond, n_steps=n_steps)
    ghmc_move.apply(thermo_state, sampler_state)

The ``SamplerState`` object in the snippet above holds the configurational degrees of freedom of the ``System`` (e.g.,
positions, velocities, and eventually box vectors). The sampler state is updated by ``MCMCMove.apply`` to hold the
coordinates and velocities after 1000 steps of GHMC integration. Note however that, in princple, the framework allows
an ``MCMCMove`` to change also the thermodynamic degrees of freedom in ``thermo_state``.

OpenMM integrators as MCMCMoves
-------------------------------

The :ref:`mcmc <mcmc>` module provides a few integrators in the form of an ``MCMCMove``, including ``openmmtools.integrators.LangevinIntegrator``.
Casting integrators in the form of an ``MCMCMove`` object makes it easy to combine them with Monte Carlo techniques.
Moreover, integrator ``MCMCMove``s provide a few extra features such as automatic recovery after a NaN.

.. testcode::

    langevin_move = mcmc.LangevinSplittingDynamicsMove(splitting='V R O R V',
                                                       n_steps=n_steps,
                                                       n_restart_attempts=5)
    langevin_move.apply(thermo_state, sampler_state)

Propagating your system through Langevin dynamics has always a non-zero probability of incurring into a NaN error. When
this happens, instead of crashing, the Langevin move above will restore the state of the ``System`` before integrating
and try again, relying on the stochastic component of the propagation to obtain a different solution. This is repeated
to a maximum of 5 times before giving up and throwing an error. The raised exception exposes a method to serialize the
simulation objects automatically for further debugging.

.. testcode::

    try:
        langevin_move.apply(thermo_state, sampler_state)
    except mcmc.IntegratorMoveError as e:
        # This saves to disk the OpenMM System, Integrator, and State objects.
        e.serialize_error(path_files_prefix='debug/langevin')

When a NaN occurr, the code above serializes the OpenMM ``System``, ``Integrator``, and ``State`` objects on disk at
``debug/langevin-system.xml``, ``debug/langevin-integrator.xml``, and ``debug/langevin-state.xml`` respectively.

This feature can easily be extended to other integrators that are not explicitly provided in the :ref:`mcmc <mcmc>` module.

.. testcode::

    integrator = integrators.HMCIntegrator(timestep=1.0*unit.femtosecond)
    HMC_move = mcmc.IntegratorMove(integrator, n_steps=n_steps, n_restart_attempts=4)

Combining Monte Carlo and dynamics
----------------------------------

Combining and mixing multiple ``MCMCMove`` is usually performed through the ``mcmc.SequenceMove`` object

.. testcode::

    sequence_move = mcmc.SequenceMove(move_list=[
        mcmc.MCDisplacementMove(atom_subset=pxylene_atoms),
        mcmc.MCRotationMove(atom_subset=pxylene_atoms),
        mcmc.LangevinSplittingDynamicsMove(timestep=2.0*unit.femtoseconds, n_steps=n_steps,
                                           reassign_velocities=True, n_restart_attempts=6)
    ])

    sequence_move.apply(thermo_state, sampler_state)

The ``MCMCMove`` above performs in sequence a Metropolized Monte Carlo rigid translation and rotation of the p-xylene
molecule followed by Langevin dynamics after randomizing the velocities according to the Boltzmann distribution
at the temperature of ``thermo_state``.

ContextCache and Platform with MCMCMoves
----------------------------------------

All ``MCMCMove`` objects implemented in OpenMMTools accept a ``context_cache`` in the constructor. This parameter
defaults to ``mmtools.cache.global_context_cache``, but you can pass a local cache to trigger other behaviors.

.. testcode::

    local_cache = cache.ContextCache(platform=openmm.Platform.getPlatformByName('CPU'))
    dummy_cache = cache.DummyContextCache()  # Disable caching.
    move = mcmc.SequenceMove(move_list=[
        mcmc.MCDisplacementMove(atom_subset=pxylene_atoms, context_cache=local_cache),
        mcmc.MCRotationMove(atom_subset=pxylene_atoms, context_cache=dummy_cache),
        mcmc.LangevinSplittingDynamicsMove(n_steps=n_steps)  # Uses global_context_cache.
    ])

In the example above, applying the ``move`` will perform an MC translation of the ligands atom using a local ``ContextCache``
that runs on the CPU, then an MC rotation using the ``DummyContextCache``, which recreates context every time effectively
deactivating caching, and finally propagates the system with Langevin dynamics using the global cache on the fastest
platform available.

|

Example: A minimal implementation of a general replica-exchange simulation class
================================================================================

Our most recent enhanced-sampling facilities are currently hosted in `YANK <http://getyank.org/latest/api/multistate_api/index.html>`_,
and they are still waiting to be moved to OpenMMTools. However, the following minimal implementation of a replica exchange
simulation class should give you an idea of what is possible to do when taking advantage of the full framework.

.. testcode::

    import math
    from random import random, randint

    class ReplicaExchange:

        def __init__(self, thermodynamic_states, sampler_states, mcmc_move):
            self._thermodynamic_states = thermodynamic_states
            self._replicas_sampler_states = sampler_states
            self._mcmc_move = mcmc_move

        def run(self, n_iterations=1):
            for iteration in range(n_iterations):
                self._mix_replicas()
                self._propagate_replicas()

        def _propagate_replicas(self):
            # _thermodynamic_state[i] is associated to the replica configuration in _replicas_sampler_states[i].
            for thermo_state, sampler_state in zip(self._thermodynamic_states, self._replicas_sampler_states):
                self._mcmc_move.apply(thermo_state, sampler_state)

        def _mix_replicas(self, n_attempts=1):
            # Attempt to switch two replicas at random. Obviously, this scheme can be improved.
            for attempt in range(n_attempts):
                # Select two replicas at random.
                i = randint(0, len(self._thermodynamic_states)-1)
                j = randint(0, len(self._thermodynamic_states)-1)
                sampler_state_i, sampler_state_j = (self._replicas_sampler_states[k] for k in [i, j])
                thermo_state_i, thermo_state_j = (self._thermodynamic_states[k] for k in [i, j])

                # Compute the energies.
                energy_ii = self._compute_reduced_potential(sampler_state_i, thermo_state_i)
                energy_jj = self._compute_reduced_potential(sampler_state_j, thermo_state_j)
                energy_ij = self._compute_reduced_potential(sampler_state_i, thermo_state_j)
                energy_ji = self._compute_reduced_potential(sampler_state_j, thermo_state_i)

                # Accept or reject the swap.
                log_p_accept = - (energy_ij + energy_ji) + energy_ii + energy_jj
                if log_p_accept >= 0.0 or random() < math.exp(log_p_accept):
                    # Swap states in replica slots i and j.
                    self._thermodynamic_states[i] = thermo_state_j
                    self._thermodynamic_states[j] = thermo_state_i

        def _compute_reduced_potential(self, sampler_state, thermo_state):
            # Obtain a Context to compute the energy with OpenMM. Any integrator will do.
            context, integrator = cache.global_context_cache.get_context(thermo_state)
            # Compute the reduced potential of the sampler_state configuration
            # in the given thermodynamic state.
            sampler_state.apply_to_context(context)
            return thermo_state.reduced_potential(context)

The first observation is that the bulk of the code complexity lies in the replica swapping code, while most of the other
details are handled by the specialized classes of the framework. From a software engineering perspective, this is a good
sign as it is compatible with the single responsibility principle.

Secondly, the class can be used to implement a variety of algorithm. A few examples follow.

Parallel tempering
------------------

To run a parallel tempering simulation, we just have initialize the ``ReplicaExchange`` object with a list of thermodynamic
states that vary in temperature. You can make use of the utility function ``states.create_thermodynamic_state_protocol``
to initialize efficiently a list of ``ThermodynamicState`` or ``CompoundThermodynamicState``.

.. doctest::

    >>> # Initialize thermodynamic states at different temperatures.
    >>> host_guest = testsystems.HostGuestVacuum()
    >>> protocol = {'temperature': [300, 310, 330, 370, 450] * unit.kelvin}
    >>> thermo_states = states.create_thermodynamic_state_protocol(host_guest.system, protocol)

    >>> # Initialize replica initial configurations.
    >>> sampler_states = [states.SamplerState(positions=host_guest.positions)
    ...                   for _ in thermo_states]

    >>> # Propagate the replicas with Langevin dynamics.
    >>> langevin_move = mcmc.LangevinSplittingDynamicsMove(timestep=2.0*unit.femtosecond,
    ...                                                    n_steps=n_steps)

    >>> # Run the parallel tempering simulation.
    >>> parallel_tempering = ReplicaExchange(thermo_states, sampler_states, langevin_move)
    >>> parallel_tempering.run()

This example creates 5 replicas starting from the same configurations but at the temperatures of 300, 310, ..., 450 K,
and propagates the system with Langevin dynamics.

Hamiltonian replica exchange + parallel tempering
-------------------------------------------------

Let's say we want to implement an enhanced sampling scheme that increases the temperature while alchemically softening
part of a system.

.. doctest::

    >>> # Prepare the T4 Lysozyme + p-xylene system for alchemical modification.
    >>> guest_atoms = host_guest.mdtraj_topology.select('resname B2')
    >>> alchemical_region = alchemy.AlchemicalRegion(alchemical_atoms=guest_atoms)
    >>> factory = alchemy.AbsoluteAlchemicalFactory()
    >>> alchemical_system = factory.create_alchemical_system(host_guest.system, alchemical_region)

    >>> # Initialize compound thermodynamic states at different temperatures and alchemical states.
    >>> protocol = {'temperature': [300, 310, 330, 370, 450] * unit.kelvin,
    ...             'lambda_electrostatics': [1.0, 0.5, 0.0, 0.0, 0.0],
    ...             'lambda_sterics': [1.0, 1.0, 1.0, 0.5, 0.0]}
    >>> alchemical_state = alchemy.AlchemicalState.from_system(alchemical_system)
    >>> compound_states = states.create_thermodynamic_state_protocol(
    ...     alchemical_system, protocol=protocol, composable_states=[alchemical_state])

    >>> # Run the combined Hamiltonian replica exchange + parallel tempering simulation.
    >>> hrex_tempering = ReplicaExchange(compound_states, sampler_states, langevin_move)

Hamiltonian replica exchange + parallel tempering mixing Monte Carlo and dynamics
---------------------------------------------------------------------------------

Finally, let's mix Monte Carlo and dynamics for propagation.

.. doctest::

    >>> sequence_move = mcmc.SequenceMove(move_list=[
    ...     mcmc.MCDisplacementMove(atom_subset=pxylene_atoms),
    ...     mcmc.MCRotationMove(atom_subset=pxylene_atoms),
    ...     mcmc.LangevinSplittingDynamicsMove(timestep=2.0*unit.femtoseconds, n_steps=n_steps,
    ...                                        reassign_velocities=True, n_restart_attempts=6)
    ... ])

    >>> # Run the combined Hamiltonian replica exchange + parallel tempering simulation
    >>> # using a combination of Monte Carlo moves and Langevin dynamics.
    >>> hrex_tempering = ReplicaExchange(compound_states, sampler_states, sequence_move)
