Release History
===============

Development snapshot
====================
New features
------------
- Add a ``WaterCluster`` testsystem (`#322 <https://github.com/choderalab/openmmtools/pull/322>`_)
- Add exact treatment of PME electrostatics in `alchemy.AbsoluteAlchemicalFactory`. (`#320 <https://github.com/choderalab/openmmtools/pull/320>`_)
- Add method in ``ThermodynamicState`` for the efficient computation of the reduced potential at a list of states. (`#320 <https://github.com/choderalab/openmmtools/pull/320>`_)
- When a ``SamplerState`` is applied to many ``Context``s, the units are stripped only once for optimization. (`#320 <https://github.com/choderalab/openmmtools/pull/320>`_)


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



OpenMMTools 0.13.0
==================

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


Hotfix 0.11.2
=============

Hotfix in fringe Python2/3 compatibility issue when using old style serialization systems in Python 2



Release 0.11.1: Optimizations
=============================

- Adds Drew-Dickerson DNA dodecamer test system (`#223 <https://github.com/choderalab/openmmtools/issues/223>`_)
- Bugfix and optimization to ``ContextCache`` (`#235 <https://github.com/choderalab/openmmtools/issues/235>`_)
- Compress serialized ``ThermodynamicState`` strings for speed and size
  (`#232 <https://github.com/choderalab/openmmtools/issues/232>`_)
- Backwards compatible with uncompressed serialized ``ThermodynamicStates``


0.11.0
======

New Features:

- ``LangevinIntegrator`` now sets ``measure_heat=False`` by default for increased performance
  (`#211 <https://github.com/choderalab/openmmtools/issues/211>`_)
- ``AbsoluteAlchemicalFactory`` now supports ``disable_alchemical_dispersion_correction`` to prevent 600x slowdowns with
  nonequilibrium integration (`#218 <https://github.com/choderalab/openmmtools/issues/218>`_)
- We now require conda-forge as a dependency for testing and deployment
  (`#216 <https://github.com/choderalab/openmmtools/issues/216>`_)
- Conda-forge added as channel to conda packages



Release 0.10.0 - Optimizations of ThermodynamicState, renamed AlchemicalFactory
===============================================================================

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


Release 0.9.4 - Nonequilibrium integrators overhaul
===================================================

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
