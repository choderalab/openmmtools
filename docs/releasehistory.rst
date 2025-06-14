Release History
***************

0.25.0
======

This release removes the requirement that the ``online_analysis_interval`` is a multiple of ``checkpoint_interval``. See `#799 <https://github.com/choderalab/openmmtools/pull/779>`_ for more details.

Enhancements
------------

- Removes the requirement that the ``online_analysis_interval`` is a multiple of ``checkpoint_interval`` 
  - Issue a logger warning rather than raise a ``ValueError``
  - Note that the real time analysis output file may contain redundant information after restoring from checkpoints that would result in the repeated calculation of a specific iteration index
  - See `#799 <https://github.com/choderalab/openmmtools/pull/779>`_ for more details

0.24.2 - Numpy 2 support and FIRE minimization improvements
===========================================================

This release enables numpy 2 support and makes the FIRE minimization more stable by disabling the barostat during the minimization.

Enhancements
------------

- Add AWS Tags (`#766 <https://github.com/choderalab/openmmtools/pull/766>`_) by @mikemhenry
- chore: migrate to new OMSF start/stop runners (`#775 <https://github.com/choderalab/openmmtools/pull/775>`_) by @ethanholz
- Disable the barostat during FIRE minimization (`#773 <https://github.com/choderalab/openmmtools/pull/773>`_) by @hannahbaumann

Bug Fixes
---------

- Fixes for numpy 2.0 (ruff NPY201) (`#777 <https://github.com/choderalab/openmmtools/pull/777>`_) by @IAlibay

0.24.1 - Differential storage of positions and velocities
=========================================================

Enhancements
------------

- ``MultiStateReporter`` now accepts variable position and velocity checkpointing intervals. Note that when resuming simulations users have to specify again the keyword arguments for the reporter (`Pull Request #767 <https://github.com/choderalab/openmmtools/pull/767>`_).


0.24.0 - pyMBAR Behavior Changes + HIP Platform Added 
=====================================================

Bug Fixes
---------

- Update docstring default for ``alchemical_pme_treatment`` (`Pull Request #644`_).


Behavior Changes
----------------
- Use ``robust`` solver for  ``pyMBAR`` by default. 
  ``pyMBAR`` 3 & 4 used two different solvers by default.
  We now use the ``robust`` solver by default regardless of the ``pyMBAR`` version.
  We still respect whichever solver is specified in ``analysis_kwargs`` (i.e ``analysis_kwargs["solver_protocol"] = "robust"`` when creating the analyzer, but now set the solver to ``"robust"`` if no solver is specified.
  This should improve convergence performance (`Pull Request #735`_).

Enhancements
------------

- Added OpenMM's "HIP" platform as a selectable platform.
  With the release of OpenMM 8.2, the "HIP" platform is now available to use on compatible AMD GPUs. This update will allow ``openmmtools`` to automatically select the HIP platform if it is available (`Pull Request #753`_).
- Added ``effective_length`` to ``MultiStateSamplerAnalyzer`` (`Pull Request #589`_).
- Create ``alchemy`` subpackage (`Pull Request #721`_).



Testing
-------

- Testing framework overhauled to use pytest and flaky tests now automatically re-run if they fail (`Pull Request #714`_, `Pull Request #746`_, `Pull Request #749`_,  `Pull Request #751`_)
- Use OMSF's `gha-runner`_ to test on GPUs.

.. _gha-runner: https://github.com/omsf-eco-infra/gha-runner
.. _Pull Request #589: https://github.com/choderalab/openmmtools/pull/589
.. _Pull Request #714: https://github.com/choderalab/openmmtools/pull/714
.. _Pull Request #721: https://github.com/choderalab/openmmtools/pull/721
.. _Pull Request #644: https://github.com/choderalab/openmmtools/pull/644
.. _Pull Request #744: https://github.com/choderalab/openmmtools/pull/744
.. _Pull Request #746: https://github.com/choderalab/openmmtools/pull/746
.. _Pull Request #749: https://github.com/choderalab/openmmtools/pull/749
.. _Pull Request #735: https://github.com/choderalab/openmmtools/pull/735
.. _Pull Request #751: https://github.com/choderalab/openmmtools/pull/751
.. _Pull Request #753: https://github.com/choderalab/openmmtools/pull/753


0.23.1 - Bugfix release
=======================

Bugfixes
--------

- Fix issue where if ``None`` was used for ``online_analysis_interval`` an error would be thrown (issue `#708 <https://github.com/choderalab/openmmtools/issues/708>`_ PR `#710 <https://github.com/choderalab/openmmtools/pull/710`_)

0.23.0 - latest numba support and real time stats enhancements
==============================================================

Please note that there is an API breaking change. To ensure consistency of the data when appending real time stats make sure that you make the ``online_analysis_interval`` of your ``MultiStateSampler`` object match the ``checkpoint_interval`` of your ``MultiStateReporter``. It will error if this is not the case.

Enhancements
------------
- Running with NVIDIA GPUs in Exclusive Process mode now raises a warning (issue `#697 <https://github.com/choderalab/openmmtools/issues/697>`_, PR `#699 <https://github.com/choderalab/openmmtools/pull/699>`_)

Bugfixes
--------
- Fix metadata for netcdf files, specifying openmmtools for the ``program`` metadata (issue `#694 <https://github.com/choderalab/openmmtools/issues/694>`_, PR `#704 <https://github.com/choderalab/openmmtools/pull/704>`_).
- Real time statistics YAML file gets appended instead of overwritten when extending or resumimng simulations (issue `#691 <https://github.com/choderalab/openmmtools/issues/691>`_, PR `#692 <https://github.com/choderalab/openmmtools/pull/692>`_).
- Error when resuming simulations with numba 0.57 fixed by avoiding using ``numpy.MaskedArray`` when deserializing ``.nc`` files (issue `#700 <https://github.com/choderalab/openmmtools/issues/700>`_, PR `#701 <https://github.com/choderalab/openmmtools/pull/701>`_)


0.22.1 - Bugfix release
=======================

Bugfixes
--------

- Fixed issue where the error message thrown from openMM changed, so we need a case insensitive check. This was already fixed in most of the code base but one spot was missed. (PR `#684 <https://github.com/choderalab/openmmtools/pull/684>`_)

0.22.0 - pymbar 4 support and gentle equilibration
==================================================

Enhancements
------------
- Openmmtools now supports both Pymbar 3 and 4 versions. (PR `#659 <https://github.com/choderalab/openmmtools/pull/659>`_)
- Gentle equilibration protocol utility function available in ``openmmtools.utils.gentle_equilibration`` (PR `#669 <https://github.com/choderalab/openmmtools/pull/669>`_).
- Timing information for multiple state sampler is now reported by default (PRs `#679 <https://github.com/choderalab/openmmtools/pull/679>`_ and `#671 <https://github.com/choderalab/openmmtools/issues/671>`_).

Bugfixes
--------
- Users were not able to distinguish the exceptions caught during dynamics. Warnings are now raised when an exception is being caught (Issue `#643 <https://github.com/choderalab/openmmtools/issues/643>`_ PR `#658 <https://github.com/choderalab/openmmtools/pull/658>`_).
- Deserializing MCMC moves objects from versions <=0.21.4 resulted in error finding the key. Fixed by catching the exception and raising a warning when key is not found (Issue `#618 <https://github.com/choderalab/openmmtools/issues/618>`_ PR `#675 <https://github.com/choderalab/openmmtools/pull/675>`_).
- Different improvements in documentation strings and readthedocs documentation generation (Issues `#620 <https://github.com/choderalab/openmmtools/issues/620>`_ `#641 <https://github.com/choderalab/openmmtools/issues/641>`_ `#548 <https://github.com/choderalab/openmmtools/issues/548>`_. PR `#676 <https://github.com/choderalab/openmmtools/pull/676>`_)
- Support for newer NetCDF versions (1.6 branch) by not using zlib compression for varying length variables. (PR `#654 <https://github.com/choderalab/openmmtools/pull/654>`_).

0.21.5 - Bugfix release
=======================

Changed behaviors
-----------------
- ``LangevinDynamicsMove`` now uses ``openmm.LangevinMiddleIntegrator`` (a BAOAB integrator) instead of ``openmm.LangevinIntegrator`` (an OBABO integrator). Issue `#599 <https://github.com/choderalab/openmmtools/issues/579>`_ (PR `#600 <https://github.com/choderalab/openmmtools/pull/5600>`_).

Bugfixes
--------
- Velocities were being incorrectly updated as zeros when resuming simulations or broadcasting from different mpi processes. Fixed by specifying ``ignore_velocities=False`` in ``_propagate_replica``. Issue `#531 <https://github.com/choderalab/openmmtools/issues/531>`_ (PR `#602 <https://github.com/choderalab/openmmtools/pull/602>`_).
- Bug in equilibration detection #1: The user was allowed to specify ``statistical_inefficiency`` without specifying ``n_equilibration_iterations``, which doesn't make sense, as ``n_equilibration_iterations`` and ``n_effective_max`` cannot be computed from ``statistical_inefficiency`` alone. Fixed by preventing user from specifying ``statistical_inefficiency`` without ``n_equilibration_iterations``. Issue `#609 <https://github.com/choderalab/openmmtools/issues/609>`_ (PR `#610 <https://github.com/choderalab/openmmtools/pull/610>`_). 
- Bug in equilibration detection #2: If the user specified ``n_equilibration_iterations`` but not ``statistical_inefficiency``, the returned ``n_equilibration_iterations`` did not include number of equilibration iterations as computed from ``_get_equilibration_data_per_sample()``. Fixed by always including the ``_get_equilibration_data_per_sample()`` result in  in the returned ``n_equilibration_iterations``. Issue `#609 <https://github.com/choderalab/openmmtools/issues/609>`_ (PR `#610 <https://github.com/choderalab/openmmtools/pull/610>`_).
- Bug in equilibration detection #3: ``get_equilibration_data_per_sample`` returns 0 for ``n_equilibration_iterations``. Fixed by always discarding the first time origin returned by ``get_equilibration_data_per_sample``. To control the amount of data discarded by the first time origin, the user can now specify ``max_subset`` when initializing ``MultiStateSamplerAnalyzer``. Issue `#609 <https://github.com/choderalab/openmmtools/issues/609>`_ (PR `#610 <https://github.com/choderalab/openmmtools/pull/610>`_).
- Deserializing simulations from ``openmmtools<0.21.3`` versions resulted in error. Fixed by catching the missing key, ``KeyError`` exception, when deserializing. Issue `#612 <https://github.com/choderalab/openmmtools/issues/612>`_, PR `#613 <https://github.com/choderalab/openmmtools/pull/613>`_.
- Not specifying a subdirectory for the reporter file resulted in ``PermissionError`` when writing the real time analysis file. Fixed by using ``os.path.join`` for creating the output paths. Issue `#615 <https://github.com/choderalab/openmmtools/issues/615>`_, PR `#616 <https://github.com/choderalab/openmmtools/pull/616>`_.

Enhancements
------------
- ``LangevinDynamicsMove`` now allows ``constraint_tolerance`` parameter and public attribute, for specifying the fraction of the constrained distance within which constraints are maintained for the integrator (Refer to `Openmm's documentation <http://docs.openmm.org/latest/api-python/generated/openmm.openmm.LangevinMiddleIntegrator.html#openmm.openmm.LangevinMiddleIntegrator.setConstraintTolerance>`_ for more information). Issue `#608 <https://github.com/choderalab/openmmtools/issues/608>`_, PR `#611 <https://github.com/choderalab/openmmtools/pull/611>`_.
- Platform is now reported in the logs in DEBUG mode. Issue `#583 <https://github.com/choderalab/openmmtools/issues/583>`_, PR `#605 <https://github.com/choderalab/openmmtools/pull/605>`_.

0.21.4 - Bugfix release
=======================

Bugfixes
--------
- Bug in statistical inefficiency computation -- where self.max_n_iterations wasn't being used -- was fixed (`#577 <https://github.com/choderalab/openmmtools/pull/577>`_).
- Bug in estimated performance in realtime yaml file -- fixed by iterating through all MCMC moves (`#578 <https://github.com/choderalab/openmmtools/pull/578>`_)
- Potential bug fixed by explicitly updating and broadcasting thermodynamic states in replicas, when used in an MPI (distributed) context. Issue `#579 <https://github.com/choderalab/openmmtools/issues/579>`_ (`#587 <https://github.com/choderalab/openmmtools/pull/587>`_).
- Bug in handling unsampled states in realtime/offline analysis -- fixed by using ``MultiStateSampler._unsampled_states`` to build the mbar estimate array. Issue `#592 <https://github.com/choderalab/openmmtools/issues/592>`_ (`#593 <https://github.com/choderalab/openmmtools/pull/593>`_)

Enhancements
------------
- DHFR test system does not require ``parmed`` as dependency, since OpenMM can now handle prmtop/inpcrd files. Issue `#539 <https://github.com/choderalab/openmmtools/issues/539>`_ (`#588 <https://github.com/choderalab/openmmtools/pull/588>`_).
- ``MultiStateSamplerAnalyzer`` now allows to manually specify ``n_equilibrium_iterations`` and ``statistical_inefficiency`` parameters. (`#586 <https://github.com/choderalab/openmmtools/pull/586>`_).


0.21.3 - Bugfix release
=======================

Bugfixes
--------
- Bug in replica mixing in MPI multi-GPU runs--where some replicas were simulated in incorrect states--was fixed (`#449 <https://github.com/choderalab/openmmtools/pull/449>`_) & (`#562  <https://github.com/choderalab/openmmtools/pull/562>`_).
- Velocities are now stored in the checkpoint file to eliminate issue with "cold restart". Fixes issue `#531 <https://github.com/choderalab/openmmtools/issues/531>`_ (`#555 <https://github.com/choderalab/openmmtools/pull/555>`_).
- Documentation now correctly builds via CI. Fixes issue `#548 <https://github.com/choderalab/openmmtools/issues/548>`_ (`#554 <https://github.com/choderalab/openmmtools/pull/554>`_).
- Failing windows CI (issue `#567 <https://github.com/choderalab/openmmtools/issues/567>`_) is fixed. (`#573 <https://github.com/choderalab/openmmtools/pull/573>`_)

Enhancements
------------
- Real time MBAR analysis and timing information is now produced in yaml format at user-specified intervals (`#565 <https://github.com/choderalab/openmmtools/pull/565>`_), (`#561 <https://github.com/choderalab/openmmtools/pull/561>`_) & (`#572 <https://github.com/choderalab/openmmtools/pull/572>`_).
- Information of what CUDA devices are available is now provided in log output (`#570 <https://github.com/choderalab/openmmtools/pull/570>`_).
- Replica exchanges are now attempted during equilibration phase to enhance mixing (`#556 <https://github.com/choderalab/openmmtools/pull/556>`_).
- An example of resuming a MultiStateSampler simulation using API is now provided (`#569 <https://github.com/choderalab/openmmtools/pull/569>`_)


0.21.2 - Bugfix release
=======================

Bugfixes
--------
- Fixed UnboundLocalError when using a string to specify platform in ``platform_supports_precision`` (`#551 <https://github.com/choderalab/openmmtools/pull/551>`_). 


0.21.1 - Bugfix release
=======================

Bugfixes
--------
- More streamlined context cache usage using instance attributes (`#547 <https://github.com/choderalab/openmmtools/pull/547>`_).
- Improved docstring and examples for ``MultiStateSampler`` object.

0.21.0 - Bugfix release
=======================


Bugfixes
--------
- Fixes TestAbsoluteAlchemicalFactory.test_overlap NaNs (`#534 <https://github.com/choderalab/openmmtools/pull/534>`_)
- Try closing reporter in test for windows fix (`#535 <https://github.com/choderalab/openmmtools/pull/535>`_) 
- Follow NEP 29 and test newer python versions and drop old ones (`#542 <https://github.com/choderalab/openmmtools/pull/542>`_)
- Update to handle the new OpenMM 7.6 package namespace (`#528 <https://github.com/choderalab/openmmtools/pull/528>`_)
- Context cache usage cleanup (`#538 <https://github.com/choderalab/openmmtools/pull/538>_`). Avoiding memory issues and more streamlined API usage of `ContextCache` objects.


Known issues
------------
- Correctly raises an error when a ``CustomNonbondedForce`` made by OpenMM's ``LennardJonesGenerator`` is detected (`#511 <https://github.com/choderalab/openmmtools/pull/511>`_)

Enhancement
-----------
- Use of CODATA 2018 constants information from OpenMM 7.6.0. (`#522 <https://github.com/choderalab/openmmtools/pull/522>`_) & (`#525 <https://github.com/choderalab/openmmtools/pull/525>_`)
- Use new way of importing OpenMM >= 7.6. (`#528 <https://github.com/choderalab/openmmtools/pull/528>`_)
- Remove logic for missing file when retrying to open a dataset (`#515 <https://github.com/choderalab/openmmtools/pull/515>`_) 


`Full Changelog <https://github.com/choderalab/openmmtools/compare/0.20.3...0.20.4>`_

0.20.3 - Bugfix release
=======================

Bugfixes
--------
- Fixes [#505](https://github.com/choderalab/openmmtools/issues/505): GPU contexts would silently fail to enable 'mixed' precision; corrects reporting of available precisions

0.20.2 - Bugfix release
=======================

Remove leftover support for python 2.7

Cleanup
-------
- Remove leftover `six` imports and `xrange` (`#504 <https://github.com/choderalab/openmmtools/pull/504>`_)

0.20.1 - Bugfix release
=======================

Enhancements
------------
- ``openmmtools.utils.get_available_platforms()`` and ``.get_fastest_platform()`` now filter OpenMM Platforms based on specified minimum precision support, which defaults to ``mixed``

Bugfixes
--------
- Replace the `cython <https://cython.org/>`_ accelerated ``all-swap`` replica mixing scheme with a `numba <https://numba.pydata.org>`_ implementation for better stability, and portability, and speed
- Fixes incorrect temperature spacing in ``ParallelTemperingSampler`` constructor
- Do unit conversion first to improve precision PR #501 (fixes issue #500)

Misc
----
- Resolve ``numpy 1.20`` ``DeprecationWarning`` about ``np.float``

0.20.0 - Periodic alchemical integrators
========================================

Enhancements
------------
- Add `PeriodicNonequilibriumIntegrator`, a simple extension of `AlchemicalNonequilibriumLangevinIntegrator` that supports periodic alchemical protocols

0.19.1 - Bugfix release
=======================

Bugfixes
--------
- Fixed a crash during the restraint unbiasing for systems with an unexpected order of atoms of receptor and ligands (`#462 <https://github.com/choderalab/openmmtools/pull/462>`_).


0.19.0 - Multiple alchemical regions
====================================

New features
------------
- Added support in ``AbsoluteAlchemicalFactory`` for handling multiple independent alchemical regions (`#438 <https://github.com/choderalab/openmmtools/pull/438>`_).
- Added support for anisotropic and membrane barostats in `ThermodynamicState` (`#437 <https://github.com/choderalab/openmmtools/pull/437>`_).
- Added support for platform properties in ContextCache (e.g. for mixed and double precision CUDA in multistate sampler) (`#437 <https://github.com/choderalab/openmmtools/pull/437>`_).

Bugfixes
--------
- The multistate samplers now issue experimental API warnings via ``logger.warn()`` rather than ``warnings.warn()`` (`#446 <https://github.com/choderalab/openmmtools/pull/446>`_).
- Fix return value in ``states.reduced_potential_at_states`` (`#444 <https://github.com/choderalab/openmmtools/pull/444>`_).

Known issues
------------
- Using parallel MPI processes causes poor mixing of the odd thermodynamic states while the mixing of the even states is
  normal. We're still investigating whether the issue is caused by a change in the MPI library or an internal bug. For
  now, we recommend running calculations using only 1 GPU (see also `#449 <https://github.com/choderalab/openmmtools/issues/449>`_
  and `yank#1130 <https://github.com/choderalab/yank/issues/1130>`_).

0.18.3 - Storage enhancements and bugfixes
==========================================

Bugfixes
--------
- Fixed a bug in ``multistateanalyzer.py`` where a function was imported from ``openmmtools.utils`` instead of ``openmmtools.multistate.utils`` (`#430 <https://github.com/choderalab/openmmtools/pull/430>`_).
- Fixed a few imprecisions in the documentation (`#432 <https://github.com/choderalab/openmmtools/pull/432>`_).

Enhancements
------------
- Writing on disk is much faster when the `checkpoint_interval` of multi-state samplers is large. This was due
  to the dimension of the netcdf chunk size increasing with the checkpoint interval and surpassing the dimension
  of the netcdf chunk cache. The chunk size of the iteration dimension is now always set to 1 (`#432 <https://github.com/choderalab/openmmtools/pull/432>`_).

0.18.2 - Bugfix release
=======================

Bugfixes
--------
- A bug in the multistate samplers where``logsumexp`` was imported from ``scipy.misc`` (now in ``scipy.special``) was fixed (`#423 <https://github.com/choderalab/openmmtools/pull/423>`_).
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
