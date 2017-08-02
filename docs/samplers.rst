.. _samplers:

.. warning::

   These classes are experimental and their API is subject to change.

Multistate sampling
===================

The :mod:`openmmtools.samplers` module includes a number of classes that can be layered on top of :class:`openmmtools.mcmc.MCMCSampler` to create samplers (or stacks of samplers) that efficiently sample multiple thermodynamic states either serially or in parallel.

Expanded ensembles and self-adjusted mixture sampling (SAMS)
------------------------------------------------------------

Expanded ensembles
""""""""""""""""""

The method of Lyubartsev [1] with updates to the thermodynamic made through several possible Gibbs sampling approaches [2] is available via the :class:`openmmtools.samplers.sams.ExpandedEnsemble` class.

::

    [1] Lyubartsev AP, Martsinovski AA, Shevkunov SV, and Vorontsov-Velyaminov PN. New approach to Monte Carlo calculation of the free energy: Method of expanded ensembles. JCP 96:1776, 1992
    http://dx.doi.org/10.1063/1.462133
    [2] Chodera JD and Shirts MR. Replica exchange and expanded ensemble simulations as Gibbs sampling: Simple improvements for enhanced mixing. JCP 135:194110, 2011.
    http://dx.doi.org/10.1063/1.3660669

The thermodynamic states are first defined using :class:`openmmtools.states.ThermodynamicState`:

::

    # Create an alanine dipeptide test system
    from testsystem import AlanineDipeptideVacuum
    testsystem = AlanineDipeptideVacuum()
    # Specify a list of temperatures
    from numpy import geomspace
    from simtk.unit import kelvin
    nstates = 10
    temperatures = geomspace(300, 400, nstates) * kelvin
    # Construct a set of thermodynamic states differing in temperature
    from openmmtools.states import ThermodynamicState
    thermodynamic_states = [ ThermodynamicState(testsystem.system, temperature=temperature) for temperature in temperatures ]

The initial state for the sampler uses :class:`openmmtools.states.SamplerState`:

::

    # Define the initial sampler state in terms of positions only
    from openmmtools.states import SamplerState
    sampler_state = SamplerState(testsystem.positions)

We first need to specify an :class:`MCMCSampler` that will be used to update the positions:

::

    # Create an MCMCSampler to propagate the SamplerState
    from openmmtools.mcmc import MCMCSampler, GCMCMove
    ghmc_move = GHMCMove(timestep=1.0*unit.femtosecond, n_steps=500)
    mcmc_sampler = MCMCSampler(thermodynamic_state, sampler_state, move=ghmc_move)

Next, an expanded ensemble sampler could be specified:

::

    # Create an expanded ensembles sampler with initial zero log weights guess
    from openmmtools.samplers import ExpandedEnsemble
    exen_sampler = ExpandedEnsemble(mcmc_sampler, thermodynamic_states)

Now, you can run the simulation:

::

    # Run the simulation
    exen_sampler.run(niterations=10)

Analysis to estimate the free energy is straightforward:

::

    # Estimate relative free energies between the thermodynamic states
    [Delta_f_ij, dDelta_f_ij] = exen_sampler.compute_free_energies()

.. todo:: Describe storage and binding to storage files.

Self-adjusted mixture sampling (SAMS)
"""""""""""""""""""""""""""""""""""""

A self-adjusted mixture sampling (SAMS) sampler [3] is available through the :class:`SAMS` class.

::

    [3] Tan, Z. Optimally adjusted mixture sampling and locally weighted histogram analysis, Journal of Computational and Graphical Statistics 26:54, 2017.
    http://dx.doi.org/10.1080/10618600.2015.1113975

To use it, first configure an :class:`ExpandedEnsemble` whose weights will be automatically adjusted by SAMS to achieve the target probabilities.
If equal sampling of all states is desired, ``log_target_probabilities`` can be left unspecified.

::

    # Create a SAMS sampler
    from openmmtools.samplers import SAMS
    sams_sampler = SAMS(exen_sampler)

Now, you can run the simulation:

::

    # Run the simulation
    sams_sampler.run(niterations=10)

Analysis uses a similar interface to :class:`ExpandedEnsemble`:

::

    # Estimate relative free energies between the thermodynamic states
    [Delta_f_ij, dDelta_f_ij] = sams_sampler.compute_free_energies()

Alternatively, you can obtain the online (SAMS) estimates of free energies,
though these are currently provided without an uncertainty estimate:

::

    # Estimate relative free energies between the thermodynamic states
    Delta_f_ij = sams_sampler.online_free_energies()

.. todo: Describe storage and binding to storage files.

.. currentmodule:: openmmtools.samplers.sams
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    ExpandedEnsemble
    SAMS

Analysis
""""""""

.. todo: This will be folded into the :class:`ExpandedEnsemble` class.

.. currentmodule:: openmmtools.samplers.sams.analysis
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    analyze
    write_trajectory
    write_trajectory_dcd


SAMS test systems
"""""""""""""""""

A number of SAMS test systems are provided to make testing of SAMS schemes more convenient.
Objects for these classes are initialized with a SAMS sampler stack.

::

    from openmmtools.samplers.sams.testsystems import AlanineDipeptideExplicitSimulatedTempering
    testsystem = AlanineDipeptideExplicitSimulatedTempering()
    testsystem.mcmc_sampler.run(2)
    testsystem.exen_sampler.run(2)
    testsystem.sams_sampler.run(2)

.. currentmodule:: openmmtools.samplers.sams.testsystems
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    SAMSTestSystem
    HarmonicOscillatorSimulatedTempering
    AlanineDipeptideVacuumSimulatedTempering
    AlanineDipeptideExplicitSimulatedTempering
    AlchemicalSAMSTestSystem
    AlanineDipeptideVacuumAlchemical
    AlanineDipeptideExplicitAlchemical
    WaterBoxAlchemical
    HostGuestAlchemical
    AblImatinibVacuumAlchemical
    AblImatinibExplicitAlchemical
