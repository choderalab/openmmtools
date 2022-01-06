import math


def dipeptide_toy_simulation(n_replicas=2, n_steps=10, n_iterations=10, checkpoint_interval=5, do_run=False,
                             local_context_cache=True):
    """
    Performs a small replica exchange toy simulation for Alanine dipeptide system.

    Meant to be used for testing purposes.

    .. note:: Designed to be run in a temporary directory.

    Parameters
    ----------
        n_replicas: int, optional, default=2
            Number of replicas to be used in simulation
        n_steps: int, Optional, default=10
            Number of integrator steps to be run per iteration
        n_iterations: int, optional, default=10
            Total number of replica exchange iterations to be run.
        checkpoint_interval: int, optional, default=5
            Interval in steps to be used for storing checkpoints.
        do_run: bool, optional, default=False
            Tells if simulation will be actually run, instead of just getting created.
        local_context_cache: bool, optional, default=True
            Specifies whether to create a local context_cache. If False, uses global context cache.

    Returns
    -------
        simulation: object
            Openmmtools simulation object.
        storage_path: str
            String with the relative path to the reporter storage file.
    """
    # required imports
    from openmm import unit
    from openmmtools import cache
    from openmmtools import multistate
    from openmmtools import testsystems
    from openmmtools import mcmc
    from openmmtools.multistate import ReplicaExchangeSampler
    from openmmtools.states import SamplerState
    from openmmtools.states import ThermodynamicState
    from openmmtools.utils import get_fastest_platform

    # test system of alanine dipeptide
    testsystem = testsystems.AlanineDipeptideExplicit()

    # Specifying context cache
    if local_context_cache:
        platform = get_fastest_platform()
        context_cache = cache.ContextCache(capacity=None, time_to_live=None, platform=platform)
    else:
        context_cache = None

    # Simulation parameters
    n_replicas = n_replicas  # Number of temperature replicas.
    T_min = 300.0 * unit.kelvin  # Minimum temperature.
    T_max = 600.0 * unit.kelvin  # Maximum temperature.
    temperatures = [
        T_min
        + (T_max - T_min)
        * (math.exp(float(i) / float(n_replicas - 1)) - 1.0)
        / (math.e - 1.0)
        for i in range(n_replicas)
    ]

    thermodynamic_states = [
        ThermodynamicState(system=testsystem.system, temperature=T) for T in temperatures
    ]

    # MCMC Move
    move = mcmc.LangevinSplittingDynamicsMove(
        timestep=4.0 * unit.femtoseconds,
        n_steps=n_steps,
        collision_rate=5.0 / unit.picosecond,
        reassign_velocities=False,
        n_restart_attempts=20,
        constraint_tolerance=1e-06,
        context_cache=context_cache,
    )

    simulation = ReplicaExchangeSampler(
        mcmc_moves=move,
        number_of_iterations=n_iterations
    )

    # Warning. Stores reporter file in local directory
    storage_path = f"alanine_dipeptide_test.nc"
    reporter = multistate.MultiStateReporter(storage_path, checkpoint_interval=checkpoint_interval)

    # Generating simulation
    simulation.create(
        thermodynamic_states=thermodynamic_states,
        sampler_states=SamplerState(
            testsystem.positions,
            box_vectors=testsystem.system.getDefaultPeriodicBoxVectors(),
        ),
        storage=reporter,
    )

    # Running simulation if specified
    if do_run:
        simulation.run()

    # Return simulation object and storage path
    return simulation, storage_path
