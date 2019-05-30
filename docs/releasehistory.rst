Release History
***************

0.18.2 - Bugfix release
=======================

Bugfixes
--------
- A bug in the multistate samplers where``logsumexp`` was imported from ``scipy.misc`` (now in ``scipy.special``) was fixed  (`#423 <https://github.com/choderalab/openmmtools/pull/423>`_).
- Improve the robustness of opening the netcdf file on resuming of the multi-state samplers by setting the environment variable HDF5_USE_FILE_LOCKING to FALSE after 4 failed attempts (`#426 <https://github.com/choderalab/openmmtools/pull/426>`_).
- Fixed a crash during exception handling (`#426 <https://github.com/choderalab/openmmtools/pull/426>`_).

Other
-----
- Update build infrastructure to match `MolSSI cookiecutter <https://github.com/MolSSI/cookiecutter-cms>`_  (`#424 <https://github.com/choderalab/openmmtools/pull/424>`_, `#426 <https://github.com/choderalab/openmmtools/pull/426>`_).

0.18.1 - Bugfix release
=======================

This is a minor bugfix release.

New features
------------
- Improvements for ``HostGuest*`` classes
  - add ``oemols``, ``host_oemol``, and ``guest_oemol`` properties to retrieve OpenEye Toolkit ``OEMol`` objects (requires toolkit license and installation)
  - these classes can now accept overriding ``kwargs``

Bugfixes
--------
- ``openmmtools.multistate`` experimental API warning is only issued when ``openmmtools.multistate`` is imported
- ``AlchemicalNonequilibriumLangevinIntegrator.reset()`` now correctly resets the nonequilibrium work

0.18.0 - Added multistate samplers
==================================

New features
------------
- Add a number of classes that can use MCMC to sample from multiple thermodynamic states:
  - ``MultiStateSampler``: sample independently from multiple thermodynamic states
  - ``ReplicaExchangeSampler``: replica exchange among thermodynamic states
  - ``SAMSSampler``: self-adjusted mixture sampling (SAMS) sampling
- All samplers can use MPI via the `mpiplus <https://github.com/choderalab/mpiplus>`_ package

0.17.0 - Removed Py2 support, faster exact PME treatment
========================================================

New features
------------
- Add ``GlobalParameterFunction`` that allows to enslave a ``GlobalParameter`` to an arbitrary function of controlling variables (`#380 <https://github.com/choderalab/openmmtools/pull/380>`_).
- Allow to ignore velocities when building the dict representation of a ``SamplerState``. This can be useful for example to save bandwidth when sending a ``SamplerState`` over the network and velocities are not required (`#386 <https://github.com/choderalab/openmmtools/pull/386>`_).
- Add ``DoubleWellDimer_WCAFluid`` and ``DoubleWellChain_WCAFluid`` test systems (`#389 <https://github.com/choderalab/openmmtools/pull/389>`_).

Enhancements
------------
- New implementation of the exact PME handling that uses the parameter offset feature in OpenMM 7.3. This comes with a
considerable speed improvement over the previous implementation (`#380 <https://github.com/choderalab/openmmtools/pull/380>`_).
- Exact PME is now the default for the ``alchemical_pme_treatment`` parameter in the constructor of
``AbsoluteAchemicalFactory`` (`#386 <https://github.com/choderalab/openmmtools/pull/386>`_).
- It is now possible to have multiple composable states exposing the same attributes/getter/setter in a
``CompoundThermodynamicState`` (`#380 <https://github.com/choderalab/openmmtools/pull/380>`_).

Bug fixes
---------
- Fixed a bug involving the ``NoseHooverChainVelocityVerletIntegrator`` with ``System`` with constraints. The constraints were not taken into account when calculating the number of degrees of freedom resulting in the temperature not converging to the target value. (`#384 <https://github.com/choderalab/openmmtools/pull/384>`_)
- Fixed a bug affecting ``reduced_potential_at_states`` when computing the reduced potential of systems in different ``AlchemicalState``s when the same alchemical parameter appeared in force objects split in different force groups. (`#385 <https://github.com/choderalab/openmmtools/pull/385>`_)

Deprecated and API breaks
-------------------------
- Python 2 and 3.5 is not supported anymore.
- The ``update_alchemical_charges`` attribute of ``AlchemicalState`, which was deprecated in 0.16.0, has now been removed since it doesn't make sense with the new parameter offset implementation.
- The methods ``AlchemicalState.get_alchemical_variable`` and ``AlchemicalState.set_alchemical_variable`` have been deprecated. Use ``AlchemicalState.get_alchemical_function`` and ``AlchemicalState.set_alchemical_function`` instead.


0.16.0 - Py2 deprecated, GlobalParameterState class, SamplerState reads CVs
===========================================================================

New features
------------
- Add ability for ``SamplerState`` to access new `OpenMM Custom CV Force Variables
  <http://docs.openmm.org/development/api-python/generated/simtk.openmm.openmm.CustomCVForce.html#simtk.openmm.openmm.CustomCVForce>`_
  (`#362 <https://github.com/choderalab/openmmtools/pull/362>`_).
- ``SamplerState.update_from_context`` now has keywords to support finer grain updating from the Context. This is only
  recommended for advanced users (`#362 <https://github.com/choderalab/openmmtools/pull/362>`_).
- Added the new class ``states.GlobalParameterState`` designed to simplify the implementation of composable states that
  control global variables (`#363 <https://github.com/choderalab/openmmtools/pull/363>`_).
- Allow restraint force classes to be controlled by a parameter other than ``lambda_restraints``. This will enable
  multi-restraints simulations (`#363 <https://github.com/choderalab/openmmtools/pull/363>`_).

Enhancements
------------
- Global variables of integrators are now automatically copied over the integrator returned by ``ContextCache.get_context``.
  It is possible to specify exception through ``ContextCache.INCOMPATIBLE_INTEGRATOR_ATTRIBUTES`` (`#364 <https://github.com/choderalab/openmmtools/pull/364>`_).

Others
------
- Integrator ``MCMCMove``s now attempt to recover from NaN automatically by default (with ``n_restart_attempts`` set to
  4) (`#364 <https://github.com/choderalab/openmmtools/pull/364>`_).

Deprecated
----------
- Python2 is officially deprecated. Support will be dropped in future versions.
- Deprecated the signature of ``IComposableState._on_setattr`` to fix a bug where the objects were temporarily left in
  an inconsistent state when an exception was raised and caught.
- Deprecated ``update_alchemical_charges`` in ``AlchemicalState`` in anticipation of the new implementation of the
  exact PME that will be based on the ``NonbondedForce`` offsets rather than ``updateParametersInContext()``.


0.15.0 - Restraint forces
=========================
- Add radially-symmetric restraint custom forces (`#336 <https://github.com/choderalab/openmmtools/pull/336>`_).
- Copy Python attributes of integrators on ``deepcopy()`` (`#336 <https://github.com/choderalab/openmmtools/pull/336>`_).
- Optimization of ``states.CompoundThermodynamicState`` deserialization (`#338 <https://github.com/choderalab/openmmtools/pull/338>`_).
- Bugfixes (`#332 <https://github.com/choderalab/openmmtools/pull/332>`_, `#343 <https://github.com/choderalab/openmmtools/pull/343>`_).


0.14.0 - Exact treatment of alchemical PME electrostatics, water cluster test system, optimizations
===================================================================================================

New features
------------
- Add a ``WaterCluster`` testsystem (`#322 <https://github.com/choderalab/openmmtools/pull/322>`_)
- Add exact treatment of PME electrostatics in `alchemy.AbsoluteAlchemicalFactory`. (`#320 <https://github.com/choderalab/openmmtools/pull/320>`_)
- Add method in ``ThermodynamicState`` for the efficient computation of the reduced potential at a list of states. (`#320 <https://github.com/choderalab/openmmtools/pull/320>`_)

Enhancements
------------
- When a ``SamplerState`` is applied to many ``Context``s, the units are stripped only once for optimization. (`#320 <https://github.com/choderalab/openmmtools/pull/320>`_)

Bug fixes
---------
- Copy thermodynamic state on compound state initialization. (`#320 <https://github.com/choderalab/openmmtools/pull/320>`_)


0.13.4 - Barostat/External Force Bugfix, Restart Robustness
===========================================================

Bug fixes
---------
- Fixed implementation bug where ``CustomExternalForce`` restraining atoms to absolute coordinates caused an issue
  when a Barostat was used (`#310 <https://github.com/choderalab/openmmtools/issues/310>`_)

Enhancements
------------
- MCMC Integrators now attempt to re-initialize the ``Context`` object on the last restart attempt when NaN's are
  encountered. This has internally been shown to correct some instances where normally resetting positions does
  not work around the NaN's. This is a slow step relative to just resetting positions, but better than simulation
  crashing.


0.13.3 - Critical Bugfix to SamplerState Context Manipulation
=============================================================

Critical Fixes
--------------

- ``SamplerState.apply_to_context()`` applies box vectors before positions are set to prevent a bug on non-Reference
  OpenMM Platforms which can re-order system atoms. (`#305 <https://github.com/choderalab/openmmtools/issues/305>`_)

Additional Fixes
----------------

- LibYAML is now optional (`#304 <https://github.com/choderalab/openmmtools/issues/304>`_)
- Fix AppVeyor testing against Python 3.4 (now Python 3.5/3.6 and NumPy 1.12)
  (`#307 <https://github.com/choderalab/openmmtools/issues/307>`_)
- Release History now included in online Docs


0.13.2 - SamplerState Slicing and BitWise And/Or Ops
====================================================

Added support for SamplerState slicing (`#298 <https://github.com/choderalab/openmmtools/issues/298>`_)
Added bit operators ``and`` and ``or`` to ``math_eval`` (`#301 <https://github.com/choderalab/openmmtools/issues/301>`_)



0.13.1 - Bugfix release
=======================

- Fix pickling of ``CompoundThermodynamicState`` (`#284 <https://github.com/choderalab/openmmtools/issues/284>`_).
- Add missing term to OBC2 GB alchemical Force (`#288 <https://github.com/choderalab/openmmtools/issues/288>`_).
- Generalize ``forcefactories.restrain_atoms()`` to non-protein receptors
  (`#290 <https://github.com/choderalab/openmmtools/issues/290>`_).
- Standardize integrator global variables in ContextCache
  (`#291 <https://github.com/choderalab/openmmtools/issues/291>`_).



0.13.0 - Alternative reaction field models, Langevin splitting MCMCMove
=======================================================================

New Features
------------

- Storage Interface module with automatic disk IO handling
- Option for shifted or switched Reaction Field
- ``LangevinSplittingDynamic`` MCMC move with specifiable sub step ordering
- Nose-Hoover Chain Thermostat

Bug Fixes
---------

- Many doc string cleanups
- Tests are based on released versions of OpenMM
- Tests also compare against development OpenMM, but do not fail because of it
- Fixed bug in Harmonic Oscillator tests' error calculation
- Default collision rate in Langevin Integrators now matches docs



0.12.1 - Add virtual sites support in alchemy
=============================================

- Fixed AbsoluteAlchemicalFactory treatment of virtual sites that were previously ignored
  (`#259 <https://github.com/choderalab/openmmtools/issues/259>`_).
- Add possibility to add ions to the WaterBox test system
  (`#259 <https://github.com/choderalab/openmmtools/issues/259>`_).



0.12.0 - GB support in alchemy and new forces module
====================================================

New features
------------

- Add AbsoluteAlchemicalFactory support for all GB models
  (`#250 <https://github.com/choderalab/openmmtools/issues/250>`_)
- Added ``forces`` and ``forcefactories`` modules implementing ``UnishiftedReactionFieldForce`` and
  ``replace_reaction_field`` respectively. The latter has been moved from ``AbsoluteAlchemicalFactory``
  (`#253 <https://github.com/choderalab/openmmtools/issues/253>`_)
- Add ``restrain_atoms`` to restrain molecule conformation through an harmonic restrain
  (`#255 <https://github.com/choderalab/openmmtools/issues/255>`_)

Bug fixes
---------

- Bugfix for ``testsystems`` that use implicit solvent (`#250 <https://github.com/choderalab/openmmtools/issues/250>`_)
- Bugfix for ``ContextCache``: two consecutive calls retrieve the same ``Context`` with same thermodynamic state and no
  integrator (`#252 <https://github.com/choderalab/openmmtools/issues/252>`_)


0.11.2 - Bugfix release
=======================

- Hotfix in fringe Python2/3 compatibility issue when using old style serialization systems in Python 2



0.11.1 - Optimizations
======================

- Adds Drew-Dickerson DNA dodecamer test system (`#223 <https://github.com/choderalab/openmmtools/issues/223>`_)
- Bugfix and optimization to ``ContextCache`` (`#235 <https://github.com/choderalab/openmmtools/issues/235>`_)
- Compress serialized ``ThermodynamicState`` strings for speed and size
  (`#232 <https://github.com/choderalab/openmmtools/issues/232>`_)
- Backwards compatible with uncompressed serialized ``ThermodynamicStates``


0.11.0 - Conda forge installation
=================================

New Features
------------

- ``LangevinIntegrator`` now sets ``measure_heat=False`` by default for increased performance
  (`#211 <https://github.com/choderalab/openmmtools/issues/211>`_)
- ``AbsoluteAlchemicalFactory`` now supports ``disable_alchemical_dispersion_correction`` to prevent 600x slowdowns with
  nonequilibrium integration (`#218 <https://github.com/choderalab/openmmtools/issues/218>`_)
- We now require conda-forge as a dependency for testing and deployment
  (`#216 <https://github.com/choderalab/openmmtools/issues/216>`_)
- Conda-forge added as channel to conda packages



0.10.0 - Optimizations of ThermodynamicState, renamed AlchemicalFactory
=======================================================================

- BREAKS API: Renamed AlchemicalFactory to AbsoluteAlchemicalFactory
  (`#206 <https://github.com/choderalab/openmmtools/issues/206>`_)
- Major optimizations of ThermodynamicState (`#200 <https://github.com/choderalab/openmmtools/issues/177>`_,
  `#205 <https://github.com/choderalab/openmmtools/issues/205>`_)

    * Keep in memory only a single System object per compatible state
    * Fast copy/deepcopy
    * Enable custom optimized serialization for multiple states

- Added readthedocs documentation (`#191 <https://github.com/choderalab/openmmtools/issues/191>`_)
- Bugfix for serialization of context when NaN encountered
  (`#199 <https://github.com/choderalab/openmmtools/issues/199>`_)
- Added tests for Python 3.6 (`#184 <https://github.com/choderalab/openmmtools/issues/184>`_)
- Added tests for integrators (`#186 <https://github.com/choderalab/openmmtools/issues/186>`_,
  `#187 <https://github.com/choderalab/openmmtools/issues/187>`_)


0.9.4 - Nonequilibrium integrators overhaul
===========================================

Major changes
-------------

- Overhaul of ``LangevinIntegrator`` and subclasses to better support nonequilibrium integrators
- Add true reaction-field support to ``AlchemicalFactory``
- Add some alchemical test systems

Updates to ``openmmtools.integrators.LangevinIntegrator`` and friends
---------------------------------------------------------------------

API-breaking changes
^^^^^^^^^^^^^^^^^^^^

- The nonequilibrium integrators are now called ``AlchemicalNonequilibriumLangevinIntegrator`` and
  ``ExternalPerturbationLangevinIntegrator``, and both are subclasses of a common ``NonequilibriumLangevinIntegrator``
  that provides a consistent interface to setting and getting ``protocol_work``
- ``AlchemicalNonequilibriumLangevinIntegrator`` now has a default ``alchemical_functions`` to eliminate need for every
  test to treat it as a special case (`#180 <https://github.com/choderalab/openmmtools/issues/180>`_)
- The ``get_protocol_work()`` method allows you to retrieve the protocol work from any
  ``NonequilibriumLangevinIntegrator`` subclass and returns a unit-bearing work. The optional ``dimensionless=True``
  argument returns a dimensionless float in units of kT.
- Integrator global variables now store all energies in natural OpenMM units (kJ/mol) but the new accessor methods
  (see below) should b used instead of getting integrator global variables for work and heat.
  (`#181 <https://github.com/choderalab/openmmtools/issues/181>`_)
- Any private methods for adding steps to the integrator have been prepended with ``_`` to hide them from the public
  API.

New features
^^^^^^^^^^^^

- Order of arguments for all ``LangevinIntegrator`` derivatives matches ``openmm.LangevinIntegrator`` so it can act as a drop-in
  replacement. (`#176 <https://github.com/choderalab/openmmtools/issues/176>`_)
- The ``get_shadow_work()`` and ``get_heat()`` methods are now available for any ``LangevinIntegrator`` subclass, as
  well as the corresponding properties ``shadow_work`` and heat. The functions also support ``dimensionless=True.``
  (`#163 <https://github.com/choderalab/openmmtools/issues/163>`_)
- The ``shadow_work`` and ``heat`` properties were added to all LangevinIntegrator subclasses, returning the values of
  these properties (if the integrator was constructed with the appropriate ``measure_shadow_work=True`` or
  ``measure_heat=True`` flags) as unit-bearing quantities
- The ``get_protocol_work()`` and ``get_total_work()`` methods are now available for any
  ``NonequilibriumLangevinIntegrator``, returning unit-bearing quantities unless ``dimensionless=True`` is provided in
  which case they return the work in implicit units of kT. ``get_total_work()`` requires the integrator to have been
  constructed with ``measure_shadow_work=True``.
- The ``protocol_work`` and ``total_work`` properties were added to all ``NonequilibriumLangevinIntegrator`` subclasses,
  and return the unit-bearing work quantities. ``total_work`` requires the integrator to have been constructed with
  ``measure_shadow_work=True``.
- The subclasses have been reworked to support any kwargs that the base classes support, and defaults have all been made
  consistent.
- Various reset() methods have been added to reset statistics for all ``LangevinIntegrator`` subclasses.
- All custom integrators support ``.pretty_format()`` and ``.pretty_print()`` with optional highlighting of specific
  step types.

Bugfixes
^^^^^^^^

- Zero-step perturbations now work correctly (`#177 <https://github.com/choderalab/openmmtools/issues/177>`_)
- ``AlchemicalNonequilibriumLangevinIntegrator`` now correctly supports multiple ``H`` steps.

Internal changes
^^^^^^^^^^^^^^^^

- Adding new LangevinIntegrator step methods now uses a ``self._register_step_method(step_string, callback_function, supports_force_groups=False)`` call to simplify this process.
- Code duplication has been reduced through the use of calling base class methods whenever possible.
- ``run_nonequilibrium_switching()`` test now uses BAR to test dragging a harmonic oscillator and tests a variety of
  integrator splittings ``(["O { V R H R V } O", "O V R H R V O", "R V O H O V R", "H R V O V R H"])``.
- Integrator tests use deterministic PME and mixed precision when able.

Updates to openmmtools.alchemy.AlchemicalFactory
------------------------------------------------

- Reaction field electrostatics now removes the shift, setting ``c_rf = 0``.

- A convenience method AlchemicalFactory.replace_reaction_field() has been added to allow fully-interacting systems to
  be modified to force ``c_rf = 0`` by recoding reaction-field electrostatics as a ``CustomNonbondedForce``

New ``openmmtools.testsystems`` classes
---------------------------------------

- AlchemicalWaterBox was added, which has the first water molecule in the system alchemically modified
