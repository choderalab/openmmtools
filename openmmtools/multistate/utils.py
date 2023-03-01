#!/usr/local/bin/env python

# ==============================================================================
# MODULE DOCSTRING
# ==============================================================================

"""
Multistate Utilities
====================

Sampling Utilities for the YANK Multistate Package. A collection of functions and small classes
which are common to help the samplers and analyzers and other public hooks.

COPYRIGHT

Current version by Andrea Rizzi <andrea.rizzi@choderalab.org>, Levi N. Naden <levi.naden@choderalab.org> and
John D. Chodera <john.chodera@choderalab.org> while at Memorial Sloan Kettering Cancer Center.

Original version by John D. Chodera <jchodera@gmail.com> while at the University of
California Berkeley.

LICENSE

This code is licensed under the latest available version of the MIT License.

"""
import logging
import warnings
import numpy as np

from pymbar import timeseries  # for statistical inefficiency analysis

logger = logging.getLogger(__name__)

__all__ = [
    'generate_phase_name',
    'get_decorrelation_time',
    'get_equilibration_data',
    'get_equilibration_data_per_sample',
    'remove_unequilibrated_data',
    'subsample_data_along_axis',
    'SimulationNaNError'
]


# =============================================================================================
# Sampling Exceptions
# =============================================================================================

class SimulationNaNError(Exception):
    """Error when a simulation goes to NaN"""
    pass


# =============================================================================================
# MODULE FUNCTIONS
# =============================================================================================

def generate_phase_name(current_name, name_list):
    """
    Provide a regular way to generate unique human-readable names from base names.

    Given a base name and a list of existing names, a number will be appended to the base name until a unique string
    is generated.

    Parameters
    ----------
    current_name : string
        The base name you wish to ensure is unique. Numbers will be appended to this string until a unique string
        not in the name_list is provided
    name_list : iterable of strings
        The current_name, and its modifiers, are compared against this list until a unique string is found

    Returns
    -------
    name : string
        Unique string derived from the current_name that is not in name_list.
        If the parameter current_name is not already in the name_list, then current_name is returned unmodified.
    """
    base_name = 'phase{}'
    counter = 0
    if current_name is None:
        name = base_name.format(counter)
        while name in name_list:
            counter += 1
            name = base_name.format(counter)
    elif current_name in name_list:
        name = current_name + str(counter)
        while name in name_list:
            counter += 1
            name = current_name + str(counter)
    else:
        name = current_name
    return name


def get_decorrelation_time(timeseries_to_analyze):
    """
    Compute the decorrelation times given a timeseries.

    See the ``pymbar.timeseries.statisticalInefficiency`` for full documentation
    """
    return timeseries.statisticalInefficiency(timeseries_to_analyze)


def get_equilibration_data_per_sample(timeseries_to_analyze, fast=True, max_subset=100):
    """
    Compute the correlation time and n_effective per sample with tuning to how you want your data formatted

    This is a modified pass-through to ``pymbar.timeseries.detectEquilibration`` does, returning the per sample data.

    It has been modified to specify the maximum number of time points to consider, evenly spaced over the timeseries.
    This is different than saying "I want analysis done every X for total points Y = len(timeseries)/X",
    this is "I want Y total analysis points"

    Note that the returned arrays will be of size max_subset - 1, because we always discard data from the first time 
    origin due to equilibration.

    See the ``pymbar.timeseries.detectEquilibration`` function for full algorithm documentation

    Parameters
    ----------
    timeseries_to_analyze : np.ndarray
        1-D timeseries to analyze for equilibration
    max_subset : int >= 1 or None, optional, default: 100
        Maximum number of points in the ``timeseries_to_analyze`` on which to analyze the equilibration on.
        These are distributed uniformly over the timeseries so the final output (after discarding the first point 
        due to equilibration) will be size max_subset - 1 where indices are placed  approximately every 
        ``(len(timeseries_to_analyze) - 1) / max_subset``.
        The full timeseries is used if the timeseries is smaller than ``max_subset`` or if ``max_subset`` is None
    fast : bool, optional. Default: True
        If True, will use faster (but less accurate) method to estimate correlation time
        passed on to timeseries module.

    Returns
    -------
    i_t : np.ndarray of int
        Indices of the timeseries which were sampled from
    g_i : np.ndarray of float
        Estimated statistical inefficiency at t in units of index count.
        Equal to 1 + 2 tau, where tau is the correlation time
        Will always be >= 1

        e.g. If g_i[x] = 4.3, then choosing x as your equilibration point means the every ``ceil(4.3)`` in
        ``timeseries_to_analyze`` will be decorrelated, so the fully equilibrated decorrelated timeseries would be
        indexed by [x, x+5, x+10, ..., X) where X is the final point in the ``timeseries_to_analyze``.

        The "index count" in this case is the by count of the ``timeseries_to_analyze`` indices, NOT the ``i_t``

    n_effective_i : np.ndarray of float
        Number of effective samples by subsampling every ``g_i`` from index t, does include fractional value, so true
        number of points will be the floor of this output.

        The "index count" in this case is the by count of the ``timeseries_to_analyze`` indices, NOT the ``i_t``

    """
    # Cast to array if not already
    series = np.array(timeseries_to_analyze)
    # Special trap for constant series
    time_size = series.size
    set_size = time_size - 1  # Cannot analyze the last entry
    # Set maximum
    if max_subset is None or set_size < max_subset:
        max_subset = set_size
    # Special trap for series of size 1
    if max_subset == 0:
        max_subset = 1
    # Special trap for constant or size 1 series
    if series.std() == 0.0 or max_subset == 1:
        return (np.arange(max_subset, dtype=int),  # i_t
                np.array([1]*max_subset),  # g_i
                np.arange(time_size, time_size-max_subset, -1)  # n_effective_i
                )
    g_i = np.ones([max_subset], np.float32)
    n_effective_i = np.ones([max_subset], np.float32)
    counter = np.arange(max_subset)
    i_t = np.floor(counter * time_size / max_subset).astype(int)
    for i, t in enumerate(i_t):
        try:
            g_i[i] = timeseries.statisticalInefficiency(series[t:], fast=fast)
        except:
            g_i[i] = (time_size - t + 1)
        n_effective_i[i] = (time_size - t + 1) / g_i[i]

    # We should never choose data from the first time origin as the equilibrated data because 
    # it contains snapshots warming up from minimization, which causes problems with correlation time detection
    # By default (max_subset=100), the first 1% of the data is discarded. If 1% is not ideal, user can specify
    # max_subset to change the percentage (e.g. if 0.5% is desired, specify max_subset=200).
    return i_t[1:], g_i[1:], n_effective_i[1:]


def get_equilibration_data(timeseries_to_analyze, fast=True, max_subset=1000):
    """
    Compute equilibration method given a timeseries

    See the ``pymbar.timeseries.detectEquilibration`` function for full documentation

    Parameters
    ----------
    timeseries_to_analyze : np.ndarray
        1-D timeseries to analyze for equilibration
    max_subset : int or None, optional, default: 1000
        Maximum number of points in the ``timeseries_to_analyze`` on which to analyze the equilibration on.
        These are distributed uniformly over the timeseries so the final output will be size max_subset where indices
        are placed  approximately every ``(len(timeseries_to_analyze) - 1) / max_subset``.
        The full timeseries is used if the timeseries is smaller than ``max_subset`` or if ``max_subset`` is None
    fast : bool, optional. Default: True
        If True, will use faster (but less accurate) method to estimate correlation time
        passed on to timeseries module.

    Returns
    -------
    n_equilibration : int
        Iteration at which system becomes equilibrated
        Computed by point which maximizes the number of samples preserved
    g_t : float
        Number of indices between each decorelated sample
    n_effective_max : float
        How many indices are preserved at most.

    See Also
    --------
    get_equilibration_data_per_sample
    """
    warnings.warn("This function will be removed in future versions of YANK due to redundancy, "
                  "Please use the more general `get_equilibration_data_per_sample` function instead.")
    i_t, g_i, n_effective_i = get_equilibration_data_per_sample(timeseries_to_analyze, fast=fast, max_subset=max_subset)
    n_effective_max = n_effective_i.max()
    i_max = n_effective_i.argmax()
    n_equilibration = i_t[i_max]
    g_t = g_i[i_max]
    return n_equilibration, g_t, n_effective_max


def remove_unequilibrated_data(data, number_equilibrated, axis):
    """
    Remove the number_equilibrated samples from a dataset

    Discards number_equilibrated number of indices from given axis

    Parameters
    ----------
    data : np.array-like of any dimension length
        This is the data which will be paired down
    number_equilibrated : int
        Number of indices that will be removed from the given axis, i.e. axis will be shorter by number_equilibrated
    axis : int
        Axis index along which to remove samples from. This supports negative indexing as well

    Returns
    -------
    equilibrated_data : ndarray
        Data with the number_equilibrated number of indices removed from the beginning along axis

    """
    cast_data = np.asarray(data)
    # Define the slice along an arbitrary dimension
    slc = [slice(None)] * len(cast_data.shape)
    # Set the dimension we are truncating
    slc[axis] = slice(number_equilibrated, None)
    # Slice
    equilibrated_data = cast_data[tuple(slc)]
    return equilibrated_data


def subsample_data_along_axis(data, subsample_rate, axis):
    """
    Generate a decorrelated version of a given input data and subsample_rate along a single axis.

    Parameters
    ----------
    data : np.array-like of any dimension length
    subsample_rate : float or int
        Rate at which to draw samples. A sample is considered decorrelated after every ceil(subsample_rate) of
        indices along data and the specified axis
    axis : int
        axis along which to apply the subsampling

    Returns
    -------
    subsampled_data : ndarray of same number of dimensions as data
        Data will be subsampled along the given axis

    """
    # TODO: find a name for the function that clarifies that decorrelation
    # TODO:             is determined exclusively by subsample_rate?
    cast_data = np.asarray(data)
    data_shape = cast_data.shape
    # Since we already have g, we can just pass any appropriate shape to the subsample function
    indices = timeseries.subsampleCorrelatedData(np.zeros(data_shape[axis]), g=subsample_rate)
    subsampled_data = np.take(cast_data, indices, axis=axis)
    return subsampled_data

# =============================================================================================
# SPECIAL MIXINS
# =============================================================================================

class NNPCompatibilityMixin(object):
    """
    Mixin for subclasses of `MultistateSampler` that supports `openmm-ml` exchanges of `lambda_interpolate`
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def setup(self, n_states, mixed_system, 
              init_positions, temperature, storage_kwargs, 
              n_replicas=None, lambda_schedule=None, 
              lambda_protocol=None, setup_equilibration_intervals=None,
              steps_per_setup_equilibration_interval=None,
              **unused_kwargs):
        """try to gently equilibrate the setup of the different thermodynamic states;
        make the number of `setup_equilibration_intervals` some multiple of `n_states`.
        The number of initial equilibration steps will be equal to
        `setup_equilibration_intervals * steps_per_setup_equilibration_interval`
        """
        import openmm
        from openmm import unit
        from openmmtools.states import ThermodynamicState, SamplerState, CompoundThermodynamicState
        from openmmtools.alchemy import NNPAlchemicalState
        from copy import deepcopy
        from openmmtools.multistate import MultiStateReporter
        from openmmtools.utils import get_fastest_platform
        from openmmtools import cache
        platform = get_fastest_platform(minimum_precision='mixed')
        context_cache = cache.ContextCache(capacity=None, time_to_live=None, platform=platform)

        lambda_zero_alchemical_state = NNPAlchemicalState.from_system(mixed_system)
        thermostate = ThermodynamicState(mixed_system,
            temperature=temperature)
        compound_thermostate = CompoundThermodynamicState(thermostate,
            composable_states=[lambda_zero_alchemical_state])
        thermostate_list, sampler_state_list, unsampled_thermostate_list = [], [], []
        if n_replicas is None:
            n_replicas = n_states
        else:
            raise NotImplementedError(f"""the number of states was given as {n_states} 
                                        but the number of replicas was given as {n_replicas}. 
                                        We currently only support equal states and replicas""")
        if lambda_schedule is None:
            lambda_schedule = np.linspace(0., 1., n_states)
        else:
            assert len(lambda_schedule) == n_states
            assert np.isclose(lambda_schedule[0], 0.)
            assert np.isclose(lambda_schedule[-1], 1.)

        if setup_equilibration_intervals is not None:
            # attempt to gently equilibrate
            assert setup_equilibration_intervals % n_states == 0, f"""
              the number of `n_states` must be divisible into `setup_equilibration_intervals`"""
            interval_stepper = setup_equilibration_intervals // n_states
        else:
            raise Exception(f"At present, we require setup equilibration interval work.")
        
        if lambda_protocol is None:
            from openmmtools.alchemy import NNPProtocol
            lambda_protocol = NNPProtocol()
        else:
            raise NotImplementedError(f"""`lambda_protocol` is currently placeholding; only default `None` 
                                      is allowed until the `lambda_protocol` class is appropriately generalized""")
        
        init_sampler_state = SamplerState(init_positions, box_vectors = mixed_system.getDefaultPeriodicBoxVectors())

        # first, a context, integrator to equilibrate and minimize state 0
        eq_context, eq_integrator = context_cache.get_context(deepcopy(compound_thermostate),
                                                              openmm.LangevinMiddleIntegrator(temperature, 1., 0.001))
        forces = eq_context.getSystem().getForces()
        for force in forces: # this is to make sure i am not fucking force groups up
            print(f"{force.__class__.__name__}: {force.getForceGroup()}")
        init_sampler_state.apply_to_context(eq_context) # don't forget to set particle positions, bvs
        openmm.LocalEnergyMinimizer.minimize(eq_context) # don't forget to minimize
        init_sampler_state.update_from_context(eq_context) # update from context for good measure
        eq_context.setVelocitiesToTemperature(temperature) # set velocities at appropriate temperature

        logger.info(f"making lambda states...")
        lambda_subinterval_schedule = np.linspace(0., 1., setup_equilibration_intervals)

        # add unsampled state at lambda = 0 (also sample this...)
        compound_thermostate_copy = deepcopy(compound_thermostate) # copy thermostate
        compound_thermostate_copy.set_alchemical_parameters(0., lambda_protocol) # update thermostate
        unsampled_thermostate_list.append(compound_thermostate_copy)

        

        print(f"running thermolist population...")
        for lambda_subinterval in lambda_subinterval_schedule:
            print(f"running lambda subinterval {lambda_subinterval}.")
            compound_thermostate_copy = deepcopy(compound_thermostate) # copy thermostate
            compound_thermostate_copy.set_alchemical_parameters(lambda_subinterval, lambda_protocol) # update thermostate
            compound_thermostate_copy.apply_to_context(eq_context) # apply new alch val to context
            eq_integrator.step(steps_per_setup_equilibration_interval) # step the integrator
            init_sampler_state.update_from_context(eq_context) # update sampler_state

            # pull the energy of the force groups
            #int_state = eq_context.getState(getEnergy=True, groups = {1})
            #int_g1_energy = int_state.getPotentialEnergy()
            #print(f"\tinternal g1 energy: {int_g1_energy}")

            matchers = [np.isclose(lambda_subinterval, i) for i in lambda_schedule]
            ml_endstate_matcher = np.isclose(lambda_subinterval, 1.) # this is the last state, and we want to make it unsampled
            if ml_endstate_matcher:
                unsampled_thermostate_list.append(compound_thermostate_copy)
                #sampler_state_list.append(deepcopy(init_sampler_state))
            elif any(matchers): # if the lambda subinterval is in the lambda protocol, add thermostate and sampler state
                print(f"this subinterval ({lambda_subinterval}) matched; adding to state...")
                thermostate_list.append(compound_thermostate_copy)
                sampler_state_list.append(deepcopy(init_sampler_state))

        # put context, integrator into garbage collector
        del eq_context
        del eq_integrator
        reporter = MultiStateReporter(**storage_kwargs)
        print(f"thermostate len: {len(thermostate_list)}; samplerstate len: {len(sampler_state_list)}; unsampled: {len(unsampled_thermostate_list)}")
        self.create(thermodynamic_states = thermostate_list, sampler_states = sampler_state_list, storage=reporter,
            unsampled_thermodynamic_states = unsampled_thermostate_list)
