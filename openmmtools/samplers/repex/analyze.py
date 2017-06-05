#!/usr/local/bin/env python

# =============================================================================================
# Analyze datafiles produced by YANK.
# =============================================================================================

# =============================================================================================
# REQUIREMENTS
#
# The netcdf4-python module is now used to provide netCDF v4 support:
# http://code.google.com/p/netcdf4-python/
#
# This requires NetCDF with version 4 and multithreading support, as well as HDF5.
# =============================================================================================

import os
import os.path

import yaml
import abc
import copy
import numpy as np

import openmmtools as mmtools
from .repex import Reporter
import netCDF4 as netcdf  # netcdf4-python

from pymbar import MBAR  # multi-state Bennett acceptance ratio
from pymbar import timeseries  # for statistical inefficiency analysis

import mdtraj
import simtk.unit as units

from openmmtools import utils
from openmmtools.constants import kB

import logging
logger = logging.getLogger(__name__)

ABC = abc.ABCMeta('ABC', (object,), {})  # compatible with Python 2 *and* 3

# =============================================================================================
# PARAMETERS
# =============================================================================================

def generate_phase_name(current_name, name_list):
    """Provide a regular way to generate unique names"""
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


def get_analyzer(file_base_path):
    """
    Utility function to convert storage file to a Reporter and Analyzer by reading the data on file

    For now this is mostly placeholder functions, but creates the API for the user to work with.
    """
    reporter = Reporter(file_base_path)  # Eventually extend this to get more reporters, but for now simple placeholder\
    """
    storage = infer_storage_format_from_extension('complex.nc')  # This is always going to be nc for now.
    metadata = storage.metadata
    sampler_class = metadata['sampler_full_name']
    module_name, cls_name = sampler_full_name.rsplit('.', 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, cls_name)
    reporter = cls.create_reporter('complex.nc')
    """
    # Eventually change this to auto-detect simulation from reporter:
    if True:
        analyzer = RepexPhase(reporter)
    else:
        raise RuntimeError("Cannot automatically determine analyzer for Reporter: {}".format(reporter))
    return analyzer


class _ObservablesRegistry(object):
    """
    Registry of computable observables.

    This is a hidden class accessed by the YankPhaseAnalyzer and MultiPhaseAnalyzer objects to check which observables
    can be computed, and then provide a regular categorization of them. This is a static registry.

    To define your own methods
    """

    ########################
    # Define the observables
    ########################
    @staticmethod
    def observables():
        """
        Set of observables which are derived from the subsets below
        """
        observables = set()
        for subset in (_ObservablesRegistry.observables_defined_by_two_states(),
                       _ObservablesRegistry.observables_defined_by_single_state(),
                       _ObservablesRegistry.observables_defined_by_phase()):
            observables = observables.union(set(subset))
        return tuple(observables)

    # ------------------------------------------------
    # Exclusive Observable categories
    # The intersection of these should be the null set
    # ------------------------------------------------
    @staticmethod
    def observables_defined_by_two_states():
        """
        Observables that require an i and a j state to define the observable accurately between phases
        """
        return 'entropy', 'enthalpy', 'free_energy'

    @staticmethod
    def observables_defined_by_single_state():
        """
        Defined observables which are fully defined by a single state, and not by multiple states such as differences
        """
        return tuple()

    @staticmethod
    def observables_defined_by_phase():
        """
        Observables which are defined by the phase as a whole, and not defined by any 1 or more states
        e.g. Standard State Correction
        """
        return ('standard_state_correction',)

    ##########################################
    # Define the observables which carry error
    # This should be a subset of observables()
    ##########################################
    @staticmethod
    def observables_with_error():
        observables = set()
        for subset in (_ObservablesRegistry.observables_with_error_adding_quadrature(),
                       _ObservablesRegistry.observables_with_error_adding_linear()):
            observables = observables.union(set(subset))
        return tuple(observables)

    # ------------------------------------------------
    # Exclusive Error categories
    # The intersection of these should be the null set
    # ------------------------------------------------
    @staticmethod
    def observables_with_error_adding_quadrature():
        return 'entropy', 'enthalpy', 'free_energy'

    @staticmethod
    def observables_with_error_adding_linear():
        return tuple()


class YankPhaseAnalyzer(ABC):
    """
    Analyzer for a single phase of a YANK simulation. Uses the reporter from the simulation to determine the location
    of all variables.

    To compute a specific observable, add it to the ObservableRegistry and then implement a "compute_X" where X is the
    name of the observable you want to compute.

    Analyzer works in units of kT unless specifically stated otherwise. To convert back to a unit set, just multiply by
    the .kT property.
    """
    def __init__(self, reporter, name=None, reference_states=(0, -1)):
        """
        The reporter provides the hook into how to read the data, all other options control where differences are
        measured from and how each phase interfaces with other phases.

        Parameters
        ----------
        reporter : Reporter instance
            Reporter from Repex which ties to the simulation data on disk.
        name : str, Optional
            Unique name you want to assign this phase, this is the name that will appear in MultiPhaseAnalyzer's. If not
            set, it will be given the arbitrary name "phase#" where # is an integer, chosen in order that it is
            assigned to the MultiPhaseAnalyzer.
        reference_states: tuple of ints, length 2, Optional, Default: (0,-1)
            Integers i and j of the state that is used for reference in observables, "O". These values are only used
            when reporting single numbers or combining observables through MultiPhaseAnalyzer (since the number of states
            between phases can be different). Calls to functions such as `get_free_energy` in a single Phase results in
            the O being returned for all states.
            For O completely defined by the state itself (i.e. no differences between states, e.g. Temperature)"
                O[i] is returned
                O[j] is not used
            For O where differences between states are required (e.g. Free Energy):
                O[i,j] = O[j] - O[i]
        """
        if not reporter.is_open():
            reporter.open(mode='r')
        self._reporter = reporter
        observables = []
        # Auto-determine the computable observables by inspection of non-flagged methods
        # We determine valid observables by negation instead of just having each child implement the method to enforce
        # uniform function naming conventions.
        self._computed_observables = {}  # Cache of observables so the phase can be retrieved once computed
        for observable in _ObservablesRegistry.observables():
            if hasattr(self, "get_" + observable):
                observables.append(observable)
                self._computed_observables[observable] = None
        # Cast observables to an immutable
        self._observables = tuple(observables)
        # Internal properties
        self._name = name
        # Start as default sign +, handle all sign conversion at peration time
        self._sign = '+'
        self._equilibration_data = None  # Internal tracker so the functions can get this data without recalculating it
        # External properties
        self._reference_states = None  # initialize the cache object
        self.reference_states = reference_states
        self._mbar = None
        self._kT = None

    @property
    def name(self):
        """User-readable name of the phase"""
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def observables(self):
        """
        Access the list of observables that the instanced analyzer can compute/fetch.

        This list is automatically compiled upon __init__ based on the functions implemented in the subclass
        """
        return self._observables

    @property
    def mbar(self):
        """Access the MBAR object tied to this phase"""
        if self._mbar is None:
            self._create_mbar_from_scratch()
        return self._mbar

    @property
    def reference_states(self):
        """Provide a way to access the i,j states"""
        return self._reference_states

    @reference_states.setter
    def reference_states(self, value):
        """Provide a way to re-assign the i,j states in a protected way"""
        i, j = value[0], value[1]
        if type(i) is not int or type(j) is not int:
            raise ValueError("reference_states must be a length 2 iterable of ints")
        self._reference_states = (i, j)

    @property
    def kT(self):
        """Fetch the kT of the phase"""
        if self._kT is None:
            thermodynamic_states, _ = self._reporter.read_thermodynamic_states()
            temperature = thermodynamic_states[0].temperature
            self._kT = kB * temperature
        return self._kT

    @property
    def reporter(self):
        """Return the reporter tied to this object..."""
        return self._reporter

    @reporter.setter
    def reporter(self, value):
        """... and then make sure users cannot overwrite it."""
        raise ValueError("You cannot re-assign the reporter for this analyzer!")

    # Abstract methods
    @abc.abstractmethod
    def analyze_phase(self, *args, **kwargs):
        """
        Function which broadly handles "auto-analysis" for those that do not wish to call all the methods on their own.
        This should be have like the old "analyze" function from versions of YANK pre-1.0.

        Returns a dictionary of analysis objects
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _create_mbar_from_scratch(self):
        """
        This method should automatically do everything needed to make the MBAR object from file. It should make all
        the assumptions needed to make the MBAR object.  Typically alot of these functions will be needed for the
        analyze_phase function,

        Returns nothing, but the self.mbar object should be set after this
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def extract_energies(self):
        """
        Extract the deconvoluted energies from a phase. Energies from this are NOT decorrelated.

        Returns
        -------
        sampled_energy_matrix : Deconvoluted energy of sampled states evaluated at other sampled states.
            Has shape (K,K,N) = (number of sampled states, number of sampled states, number of iterations)
            Indexed by [k,l,n]
            where an energy drawn from sampled state [k] is evaluated in sampled state [l] at iteration [n]
        unsampled_energy_matrix
            Has shape (K, L, N) = (number of sampled states, number of UN-sampled states, number of iterations)
            Indexed by [k,l,n]
            where an energy drawn from sampled state [k] is evaluated in un-sampled state [l] at iteration [n]
        """
        raise NotImplementedError()

    # This SHOULD be an abstract static method since its related to the analyzer, but could handle any input data
    # Until we drop Python 2.7, we have to keep this method
    # @abc.abstractmethod
    @staticmethod
    def get_timeseries(passed_timeseries):
        """
        Generate the timeseries that is generated for this function

        Returns
        -------
        generated_timeseries : 1-D iterable
            timeseries which can be fed into get_decorrelation_time to get the decorrelation
        """

        raise NotImplementedError("This class has not implemented this function")

    # Static methods
    @staticmethod
    def get_decorrelation_time(timeseries_to_analyze):
        return timeseries.statisticalInefficiency(timeseries_to_analyze)

    @staticmethod
    def get_equilibration_data(timeseries_to_analyze):
        [n_equilibration, g_t, n_effective_max] = timeseries.detectEquilibration(timeseries_to_analyze)
        return n_equilibration, g_t, n_effective_max

    @staticmethod
    def remove_unequilibrated_data(data, number_equilibrated, axis):
        """
        Remove the number_equilibrated samples from a dataset by discarding number_equilibrated number of indices from
        given axis

        Parameters
        ----------
        data: np.array-like of any dimension length
        number_equilibrated: int
            Number of indices that will be removed from the given axis, i.e. axis will be shorter by number_equilibrated
        axis: int
            axis index along wich to remove samples from

        Returns
        -------
        equilibrated_data: ndarray
            Data with the number_equilibrated number of indices removed from the begining along axis

        """
        cast_data = np.asarray(data)
        # Define the slice along an arbitrary dimension
        slc = [slice(None)] * len(cast_data.shape)
        # Set the dimension we are truncating
        slc[axis] = slice(number_equilibrated, None)
        # Slice
        equilibrated_data = cast_data[slc]
        return equilibrated_data

    @staticmethod
    def decorrelate_data(data, subsample_rate, axis):
        """
        Generate a decorrelated version of a given input data and subsample_rate along a single axis.

        Parameters
        ----------
        data: np.array-like of any dimension length
        subsample_rate : float or int
            Rate at which to draw samples. A sample is considered decorrelated after every ceil(subsample_rate) of
            indicies along data and the specified axis
        axis: int
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

    # Private Class Methods
    @abc.abstractmethod
    def _prepare_mbar_input_data(self, *args, **kwargs):
        """
        Prepare a set of data for MBAR, because each analyzer may need to do something else to prepare for MBAR, it
        should have its own function to do that with.

        This is not a public function

        Parameters
        ----------
        args, kwargs: whatever is needed to generate the appropriate outputs

        Returns
        -------
        energy_matrix : energy matrix of shape (K,L,N), indexed by k,l,n
            K is the total number of sampled states
            L is the total states we want MBAR to analyze
            N is the total number of samples
            The kth sample was drawn from state k at iteration n,
                the nth configuration of kth state is evaluated in thermodynamic state l
        samples_per_state: 1-D iterable of shape L
            The total number of samples drawn from each lth state
        """
        raise NotImplementedError()

    # Shared methods
    def _create_mbar(self, energy_matrix, samples_per_state):
        """
        Initialize MBAR for Free Energy and Enthalpy estimates, this may take a while.
        This function is helpful for those who want to create a slightly different mbar object with different
        parameters.

        This function is hidden from the user unless they really, really need to create their own mbar object

        energy_matrix : array of numpy.float64, optional, default=None
           Reduced potential energies of the replicas; if None, will be extracted from the ncfile
        samples_per_state : array of ints, optional, default=None
           Number of samples drawn from each kth state; if None, will be extracted from the ncfile

        """

        # Delete observables cache since we are now resetting the estimator
        for observable in self.observables:
            self._computed_observables[observable] = None

        # Initialize MBAR (computing free energy estimates, which may take a while)
        logger.info("Computing free energy differences...")
        mbar = MBAR(energy_matrix, samples_per_state)

        self._mbar = mbar

    def _combine_phases(self, other, operator='+'):
        phases = [self]
        names = []
        signs = [self._sign]
        # Reset self._sign
        self._sign = '+'
        if self.name is None:
            names.append(generate_phase_name(self, []))
        else:
            names.append(self.name)
        if isinstance(other, MultiPhaseAnalyzer):
            new_phases = other.phases
            new_signs = other.signs
            new_names = other.names
            final_new_names = []
            for name in new_names:
                other_names = [n for n in new_names if n != name]
                final_new_names.append(generate_phase_name(name, other_names + names))
            names.extend(final_new_names)
            for new_sign in new_signs:
                if operator != '+' and new_sign == '+':
                    signs.append('-')
                else:
                    signs.append('+')
            phases.extend(new_phases)
        elif isinstance(other, YankPhaseAnalyzer):
            names.append(generate_phase_name(other.name, names))
            if operator != '+' and other._sign == '+':
                signs.append('-')
            else:
                signs.append('+')
            # Reset the other's sign if it got set to negative
            other._sign = '+'
            phases.append(other)
        else:
            baseerr = "cannot {} 'YankPhaseAnalyzer' and '{}' objects"
            if operator == '+':
                err = baseerr.format('add', type(other))
            else:
                err = baseerr.format('subtract', type(other))
            raise TypeError(err)
        phase_pass = {'phases': phases, 'signs': signs, 'names': names}
        return MultiPhaseAnalyzer(phase_pass)

    def __add__(self, other):
        return self._combine_phases(other, operator='+')

    def __sub__(self, other):
        return self._combine_phases(other, operator='-')

    def __neg__(self):
        """Internally handle the internal sign"""
        if self._sign == '+':
            self._sign = '-'
        else:
            self._sign = '+'
        return self


class RepexPhase(YankPhaseAnalyzer):

    def generate_mixing_statistics(self, number_equilibrated=0):
        """
        Generate the mixing statistics

        Parameters
        ----------
        number_equilibrated : int, optional, default=0
           If specified, only samples number_equilibrated:end will be used in analysis

        Returns
        -------
        mixing_stats : np.array of shape [nstates, nstates]
            Transition matrix estimate
        mu : np.array
            Eigenvalues of the Transition matrix sorted in descending order
        """

        # Get mixing stats from reporter
        n_accepted_matrix, n_proposed_matrix = self._reporter.read_mixing_statistics()
        # Add along iteration dim
        n_accepted_matrix = n_accepted_matrix[number_equilibrated:].sum(axis=0).astype(float)  # Ensure float division
        n_proposed_matrix = n_proposed_matrix[number_equilibrated:].sum(axis=0)
        # Compute empirical transition count matrix
        t_ij = 1 - n_accepted_matrix/n_proposed_matrix

        # Estimate eigenvalues
        mu = np.linalg.eigvals(t_ij)
        mu = -np.sort(-mu)  # Sort in descending order

        return t_ij, mu

    def show_mixing_statistics(self, cutoff=0.05, number_equilibrated=0):
        """
        Print summary of mixing statistics. Passes information off to generate_mixing_statistics then prints it out to
        the logger

        Parameters
        ----------

        cutoff : float, optional, default=0.05
           Only transition probabilities above 'cutoff' will be printed
        number_equilibrated : int, optional, default=0
           If specified, only samples number_equilibrated:end will be used in analysis

        """

        Tij, mu = self.generate_mixing_statistics(number_equilibrated=number_equilibrated)

        # Print observed transition probabilities.
        nstates = Tij.shape[1]
        logger.info("Cumulative symmetrized state mixing transition matrix:")
        str_row = "{:6s}".format("")
        for jstate in range(nstates):
            str_row += "{:6d}".format(jstate)
        logger.info(str_row)

        for istate in range(nstates):
            str_row = ""
            str_row += "{:-6d}".format(istate)
            for jstate in range(nstates):
                P = Tij[istate, jstate]
                if P >= cutoff:
                    str_row += "{:6.3f}".format(P)
                else:
                    str_row += "{:6s}".format("")
            logger.info(str_row)

        # Estimate second eigenvalue and equilibration time.
        if mu[1] >= 1:
            logger.info("Perron eigenvalue is unity; Markov chain is decomposable.")
        else:
            logger.info("Perron eigenvalue is {0:9.5f}; state equilibration timescale is ~ {1:.1f} iterations".format(
                mu[1], 1.0 / (1.0 - mu[1]))
            )

    def extract_energies(self):
        """
        Extract and decorelate energies from the ncfile to gather energies common data for other functions

        """
        logger.info("Reading energies...")
        energy_thermodynamic_states, energy_unsampled_states = self._reporter.read_energies()
        n_iterations, _, n_states = energy_thermodynamic_states.shape
        _, _, n_unsampled_states = energy_unsampled_states.shape
        energy_matrix_replica = np.zeros([n_states, n_states, n_iterations], np.float64)
        unsampled_energy_matrix_replica = np.zeros([n_states, n_unsampled_states, n_iterations], np.float64)
        for n in range(n_iterations):
            energy_matrix_replica[:, :, n] = energy_thermodynamic_states[n, :, :]
            unsampled_energy_matrix_replica[:, :, n] = energy_unsampled_states[n, :, :]
        logger.info("Done.")

        logger.info("Deconvoluting replicas...")
        energy_matrix = np.zeros([n_states, n_states, n_iterations], np.float64)
        unsampled_energy_matrix = np.zeros([n_states, n_unsampled_states, n_iterations], np.float64)
        for iteration in range(n_iterations):
            state_indices = self._reporter.read_replica_thermodynamic_states(iteration)
            energy_matrix[state_indices, :, iteration] = energy_matrix_replica[:, :, iteration]
            unsampled_energy_matrix[state_indices, :, iteration] = unsampled_energy_matrix_replica[:, :, iteration]
        logger.info("Done.")

        return energy_matrix, unsampled_energy_matrix

    @staticmethod
    def get_timeseries(passed_timeseries):
        """
        Parameters
        ----------
        passed_timeseries: ndarray of shape (K,L,N), indexed by k,l,n
            K is the total number of sampled states
            L is the total states we want MBAR to analyze
            N is the total number of samples
            The kth sample was drawn from state k at iteration n,
                the nth configuration of kth state is evaluated in thermodynamic state l
        """
        niterations = passed_timeseries.shape[-1]
        u_n = np.zeros([niterations], np.float64)
        # Compute total negative log probability over all iterations.
        for iteration in range(niterations):
            u_n[iteration] = np.sum(np.diagonal(passed_timeseries[:, :, iteration]))
        return u_n

    def _prepare_mbar_input_data(self, sampled_energy_matrix, unsampled_energy_matrix):
        nstates, _, niterations = sampled_energy_matrix.shape
        _, nunsampled, _ = unsampled_energy_matrix.shape
        # Subsample data to obtain uncorrelated samples
        N_k = np.zeros(nstates, np.int32)
        N = niterations  # number of uncorrelated samples
        N_k[:] = N
        mbar_ready_energy_matrix = sampled_energy_matrix
        if nunsampled > 0:
            fully_interacting_u_ln = unsampled_energy_matrix[:, 0, :]
            noninteracting_u_ln = unsampled_energy_matrix[:, 1, :]
            # Augment u_kln to accept the new state
            new_energy_matrix = np.zeros([nstates + 2, nstates + 2, N], np.float64)
            N_k_new = np.zeros(nstates + 2, np.int32)
            # Insert energies
            new_energy_matrix[1:-1, 0, :] = fully_interacting_u_ln
            new_energy_matrix[1:-1, -1, :] = noninteracting_u_ln
            # Fill in other energies
            new_energy_matrix[1:-1, 1:-1, :] = sampled_energy_matrix
            N_k_new[1:-1] = N_k
            # Notify users
            logger.info("Found expanded cutoff states in the energies!")
            logger.info("Free energies will be reported relative to them instead!")
            # Reset values, last step in case something went wrong so we dont overwrite u_kln on accident
            mbar_ready_energy_matrix = new_energy_matrix
            N_k = N_k_new
        return mbar_ready_energy_matrix, N_k

    def _compute_free_energy(self):
        """
        Estimate free energies of all alchemical states.
        """

        # Create MBAR object if not provided
        if self._mbar is None:
            self._create_mbar_from_scratch()

        nstates = self.mbar.N_k.size

        # Get matrix of dimensionless free energy differences and uncertainty estimate.
        logger.info("Computing covariance matrix...")

        try:
            # pymbar 2
            (Deltaf_ij, dDeltaf_ij) = self.mbar.getFreeEnergyDifferences()
        except ValueError:
            # pymbar 3
            (Deltaf_ij, dDeltaf_ij, theta_ij) = self.mbar.getFreeEnergyDifferences()

        # Matrix of free energy differences
        logger.info("Deltaf_ij:")
        for i in range(nstates):
            str_row = ""
            for j in range(nstates):
                str_row += "{:8.3f}".format(Deltaf_ij[i, j])
            logger.info(str_row)

        # Matrix of uncertainties in free energy difference (expectations standard
        # deviations of the estimator about the true free energy)
        logger.info("dDeltaf_ij:")
        for i in range(nstates):
            str_row = ""
            for j in range(nstates):
                str_row += "{:8.3f}".format(dDeltaf_ij[i, j])
            logger.info(str_row)

        # Return free energy differences and an estimate of the covariance.
        free_energy_dict = {'value': Deltaf_ij, 'error': dDeltaf_ij}
        self._computed_observables['free_energy'] = free_energy_dict

    def get_free_energy(self):
        """
        Return the free energy and error in free energy from the MBAR object

        Return changes based on if there are expanded cutoff states detected in the sampler

        Returns
        -------
        DeltaF_ij : ndarray of floats, shape (K,K) or (K+2, K+2)
            Difference in free energy from each state relative to each other state
        dDeltaF_ij: ndarray of floats, shape (K,K) or (K+2, K+2)
            Error in the difference in free energy from each state relative to each other state
        """
        if self._computed_observables['free_energy'] is None:
            self._compute_free_energy()
        free_energy_dict = self._computed_observables['free_energy']
        return free_energy_dict['value'], free_energy_dict['error']

    def _compute_enthalpy_and_entropy(self):
        """Function to compute the cached values of enthalpy and entropy"""
        if self._mbar is None:
            self._create_mbar_from_scratch()
        (f_k, df_k, H_k, dH_k, S_k, dS_k) = self.mbar.computeEntropyAndEnthalpy()
        enthalpy = {'value': H_k, 'error': dH_k}
        entropy = {'value': S_k, 'error': dS_k}
        self._computed_observables['enthalpy'] = enthalpy
        self._computed_observables['entropy'] = entropy

    def get_enthalpy(self):
        """
        Compute the difference in enthalpy and error in that estimate from the MBAR object

        Return changes based on if there are expanded cutoff states detected in the sampler

        Returns
        -------
        DeltaH_ij : ndarray of floats, shape (K,K) or (K+2, K+2)
            Difference in enthalpy from each state relative to each other state
        dDeltaH_ij: ndarray of floats, shape (K,K) or (K+2, K+2)
            Error in the difference in enthalpy from each state relative to each other state
        """
        if self._computed_observables['enthalpy'] is None:
            self._compute_enthalpy_and_entropy()
        enthalpy_dict = self._computed_observables['enthalpy']
        return enthalpy_dict['value'], enthalpy_dict['error']

    def get_entropy(self):
        """
        Return the difference in entropy and error in that estimate from the MBAR object
        """
        if self._computed_observables['entropy'] is None:
            self._compute_enthalpy_and_entropy()
        entropy_dict = self._computed_observables['entropy']
        return entropy_dict['value'], entropy_dict['error']

    def get_standard_state_correction(self):
        """
        Compute the standard state correction free energy associated with the Reporter.
        This usually is just a stored variable, but it may need other calculations

        Returns
        -------
        standard_state_correction: float
            Free energy contribution from the standard_state_correction

        """
        if self._computed_observables['standard_state_correction'] is None:
            ssc = self._reporter.read_dict('metadata')['standard_state_correction']
            self._computed_observables['standard_state_correction'] = ssc
        return self._computed_observables['standard_state_correction']

    def _create_mbar_from_scratch(self):
        u_kln, unsampled_u_kln = self.extract_energies()
        u_n = self.get_timeseries(u_kln)

        # Discard equilibration samples.
        # TODO: if we include u_n[0] (the energy right after minimization) in the equilibration detection,
        # TODO:         then number_equilibrated is 0. Find a better way than just discarding first frame.
        number_equilibrated, g_t, Neff_max = self.get_equilibration_data(u_n[1:])
        self._equilibration_data = number_equilibrated, g_t, Neff_max
        u_kln = self.remove_unequilibrated_data(u_kln, number_equilibrated, -1)
        unsampled_u_kln = self.remove_unequilibrated_data(unsampled_u_kln, number_equilibrated, -1)

        # decorrelate_data subsample the energies only based on g_t so both ends up with same indices.
        u_kln = self.decorrelate_data(u_kln, g_t, -1)
        unsampled_u_kln = self.decorrelate_data(unsampled_u_kln, g_t, -1)

        mbar_ukln, mbar_N_k = self._prepare_mbar_input_data(u_kln, unsampled_u_kln)
        self._create_mbar(mbar_ukln, mbar_N_k)

    def analyze_phase(self, cutoff=0.05):
        if self._mbar is None:
            self._create_mbar_from_scratch()
        number_equilibrated, g_t, _ = self._equilibration_data
        self.show_mixing_statistics(cutoff=cutoff, number_equilibrated=number_equilibrated)
        data = {}
        # Accumulate free energy differences
        Deltaf_ij, dDeltaf_ij = self.get_free_energy()
        DeltaH_ij, dDeltaH_ij = self.get_enthalpy()
        data['DeltaF'] = Deltaf_ij[self.reference_states[0], self.reference_states[1]]
        data['dDeltaF'] = dDeltaf_ij[self.reference_states[0], self.reference_states[1]]
        data['DeltaH'] = DeltaH_ij[self.reference_states[0], self.reference_states[1]]
        data['dDeltaH'] = dDeltaH_ij[self.reference_states[0], self.reference_states[1]]
        data['DeltaF_standard_state_correction'] = self.get_standard_state_correction()

        return data


# https://choderalab.slack.com/files/levi.naden/F4G6L9X8S/quick_diagram.png

class MultiPhaseAnalyzer(object):
    """
    Multiple Phase Analyzer creator, not to be directly called itself, but instead called by adding or subtracting
    different YankPhaseAnalyzer or MultiPhaseAnalyzers's
    """
    def __init__(self, phases):
        """
        Create the compound phase which is any combination of phases to generate a new MultiPhaseAnalyzer.

        The observables of this phase are determined through inspection of all the passed in phases and only
        observables which are shared can be computed.

        e.g.
            PhaseA has .get_free_energy and .get_entropy
            PhaseB has .get_free_energy and .get_enthalpy
            Only .get_free_energy will be available to this MultiPhaseAnalyzer

        The user themselves should not attempt to make this class by calling it directly, but instead through doing
        addition and subtraction of YankPhaseAnalyzer objects and/or other MultiPhaseAnalyzer objects.
        This class is public to see its API.

        Parameters
        ----------
        phases: dict
            has keys "phases", "names", and "signs"
        """
        # Determine
        observables = []
        for observable in _ObservablesRegistry.observables():
            shared_observable = True
            for phase in phases['phases']:
                if observable not in phase.observables:
                    shared_observable = False
                    break
            if shared_observable:
                observables.append(observable)
        if len(observables) == 0:
            raise RuntimeError("There are no shared computable observable between the phases, combining them will do "
                               "nothing.")
        self._observables = tuple(observables)
        self._phases = phases['phases']
        self._names = phases['names']
        self._signs = phases['signs']
        # Set the methods shared between both objects
        for observable in self.observables:
            setattr(self, "get_" + observable, lambda: self._compute_observable(observable))

    @property
    def observables(self):
        """List of observables this MultiPhaseAnalyzer can generate"""
        return self._observables

    @property
    def phases(self):
        """List of SinglePhase objects this MultiPhaseAnalyzer is tied to"""
        return self._phases

    @property
    def names(self):
        """
        Unique list of names identifying this phase. If this MultiPhaseAnalyzer is combined with another,
        its possible that new names will be generated unique to that MultiPhaseAnalyzer, but will still reference
        the same phase.

        When in doubt, use .phases to get the actual phase objects.
        """
        return self._names

    @property
    def signs(self):
        """
        List of signs that are used by the MultiPhaseAnalyzer to
        """
        return self._signs

    def _combine_phases(self, other, operator='+'):
        """
        Function to combine the phases regardless of operator to reduce code duplication. Creates a new
        MultiPhaseAnalyzer object based on the combined phases of the other. Accepts either a YankPhaseAnalyzer
        or a MultiPhaseAnalyzer.

        If the names have collision, they are re-named with an extra digit at the end.

        Parameters
        ----------
        other: MultiPhaseAnalyzer or YankPhaseAnalyzer
        operator: sign of the operator connecting the two objects

        Returns
        -------
        output: MultiPhaseAnalyzer
            New MultiPhaseAnalyzer where the phases are the combined list of the individual phases from each component.
            Because the memory pointers to the individual phases are the same, changing any SinglePhase's
            reference_state objects updates all MultiPhaseAnalyzer objects they are tied to

        """
        phases = []
        names = []
        signs = []
        # create copies
        phases.extend(self.phases)
        names.extend(self.names)
        signs.extend(self.signs)
        if isinstance(other, MultiPhaseAnalyzer):
            new_phases = other.phases
            new_signs = other.signs
            new_names = other.names
            final_new_names = []
            for name in new_names:
                other_names = [n for n in new_names if n != name]
                final_new_names.append(generate_phase_name(name, other_names + names))
            names.extend(final_new_names)
            for new_sign in new_signs:
                if (operator == '-' and new_sign == '+') or (operator == '+' and new_sign == '-'):
                    signs.append('-')
                else:
                    signs.append('+')
            signs.extend(new_signs)
            phases.extend(new_phases)
        elif isinstance(other, YankPhaseAnalyzer):
            names.append(generate_phase_name(other.name, names))
            if (operator == '-' and other._sign == '+') or (operator == '+' and other._sign == '-'):
                signs.append('-')
            else:
                signs.append('+')
            other._sign = '+'  # Recast to positive if negated
            phases.append(other)
        else:
            baseerr = "cannot {} 'MultiPhaseAnalyzer' and '{}' objects"
            if operator == '+':
                err = baseerr.format('add', type(other))
            else:
                err = baseerr.format('subtract', type(other))
            raise TypeError(err)
        phase_pass = {'phases': phases, 'signs': signs, 'names': names}
        return MultiPhaseAnalyzer(phase_pass)

    def __add__(self, other):
        return self._combine_phases(other, operator='+')

    def __sub__(self, other):
        return self._combine_phases(other, operator='-')

    def __neg__(self):
        """
        Return a SHALLOW copy of self with negated signs so that the phase objects all still point to the same
        objects
        """
        new_signs = []
        for sign in self._signs:
            if sign == '+':
                new_signs.append('-')
            else:
                new_signs.append('+')
        # return a *shallow* copy of self with the signs reversed
        output = copy.copy(self)
        output._signs = new_signs
        return output

    def __str__(self):
        """Simplified string output"""
        header = "MultiPhaseAnalyzer<{}>"
        output_string = ""
        for phase_name, sign in zip(self.names, self.signs):
            if output_string == "" and sign == '-':
                output_string += '{}{} '.format(sign, phase_name)
            elif output_string == "":
                output_string += '{} '.format(phase_name)
            else:
                output_string += '{} {} '.format(sign, phase_name)
        return header.format(output_string)

    def __repr__(self):
        """Generate a detailed representation of the MultiPhase"""
        header = "MultiPhaseAnalyzer <\n{}>"
        output_string = ""
        for phase, phase_name, sign in zip(self.phases, self.names, self.signs):
            if output_string == "" and sign == '-':
                output_string += '{}{} ({})\n'.format(sign, phase_name, phase)
            elif output_string == "":
                output_string += '{} ({})\n'.format(phase_name, phase)
            else:
                output_string += '    {} {} ({})\n'.format(sign, phase_name, phase)
        return header.format(output_string)

    def _compute_observable(self, observable_name):
        """
        Helper function to compute arbitrary observable in both phases

        Parameters
        ----------
        observable_name: str
            Name of the observable as its defined in the ObservablesRegistry

        Returns
        -------
        observable_value
            The observable as its combined between all the phases

        """
        def prepare_phase_observable(single_phase):
            """Helper function to cast the observable in terms of observable's registry"""
            observable = getattr(single_phase, "get_" + observable_name)()
            if isinstance(single_phase, MultiPhaseAnalyzer):
                if observable_name in _ObservablesRegistry.observables_with_error():
                    observable_payload = {}
                    observable_payload['value'], observable_payload['error'] = observable
                else:
                    observable_payload = observable
            else:
                raise_registry_error = False
                if observable_name in _ObservablesRegistry.observables_with_error():
                    observable_payload = {}
                    if observable_name in _ObservablesRegistry.observables_defined_by_phase():
                        observable_payload['value'], observable_payload['error'] = observable
                    elif observable_name in _ObservablesRegistry.observables_defined_by_single_state():
                        observable_payload['value'] = observable[0][single_phase.reference_states[0]]
                        observable_payload['error'] = observable[1][single_phase.reference_states[0]]
                    elif observable_name in _ObservablesRegistry.observables_defined_by_two_states():
                        observable_payload['value'] = observable[0][single_phase.reference_states[0],
                                                                    single_phase.reference_states[1]]
                        observable_payload['error'] = observable[1][single_phase.reference_states[0],
                                                                    single_phase.reference_states[1]]
                    else:
                        raise_registry_error = True
                else:  # No error
                    if observable_name in _ObservablesRegistry.observables_defined_by_phase():
                        observable_payload = observable
                    elif observable_name in _ObservablesRegistry.observables_defined_by_single_state():
                        observable_payload = observable[single_phase.reference_states[0]]
                    elif observable_name in _ObservablesRegistry.observables_defined_by_two_states():
                        observable_payload = observable[single_phase.reference_states[0],
                                                        single_phase.reference_states[1]]
                    else:
                        raise_registry_error = True
                if raise_registry_error:
                    raise RuntimeError("You have requested an observable that is improperly registered in the "
                                       "ObservablesRegistry!")
            return observable_payload

        def modify_final_output(passed_output, payload, sign):
            if observable_name in _ObservablesRegistry.observables_with_error():
                if sign == '+':
                    passed_output['value'] += payload['value']
                else:
                    passed_output['value'] -= payload['value']
                if observable_name in _ObservablesRegistry.observables_with_error_adding_linear():
                    passed_output['error'] += payload['error']
                elif observable_name in _ObservablesRegistry.observables_with_error_adding_quadrature():
                    passed_output['error'] += (passed_output['error']**2 + payload['error']**2)**0.5
            else:
                if sign == '+':
                    passed_output += payload
                else:
                    passed_output -= payload
            return passed_output

        if observable_name in _ObservablesRegistry.observables_with_error():
            final_output = {'value': 0, 'error': 0}
        else:
            final_output = 0
        for phase, phase_sign in zip(self.phases, self.signs):
            phase_observable = prepare_phase_observable(phase)
            final_output = modify_final_output(final_output, phase_observable, phase_sign)
        if observable_name in _ObservablesRegistry.observables_with_error():
            # Cast output to tuple
            final_output = (final_output['value'], final_output['error'])
        return final_output


def analyze_directory(source_directory):
    """
    Analyze contents of store files to compute free energy differences.

    This function is needed to preserve the old auto-analysis style of YANK. What it exactly does can be refined when
    more analyzers and simulations are made available. For now this function exposes the API.

    Parameters
    ----------
    source_directory : string
       The location of the simulation storage files.

    """
    analysis_script_path = os.path.join(source_directory, 'analysis.yaml')
    if not os.path.isfile(analysis_script_path):
        err_msg = 'Cannot find analysis.yaml script in {}'.format(source_directory)
        logger.error(err_msg)
        raise RuntimeError(err_msg)
    with open(analysis_script_path, 'r') as f:
        analysis = yaml.load(f)
    phase_names = [phase_name for phase_name, sign in analysis]
    data = dict()
    for phase_name, sign in analysis:
        phase_path = os.path.join(source_directory, phase_name + '.nc')
        phase = get_analyzer(phase_path)
        data[phase_name] = phase.analyze_phase()
        kT = phase.kT

    # Compute free energy and enthalpy
    DeltaF = 0.0
    dDeltaF = 0.0
    DeltaH = 0.0
    dDeltaH = 0.0
    for phase_name, sign in analysis:
        DeltaF -= sign * (data[phase_name]['DeltaF'] + data[phase_name]['DeltaF_standard_state_correction'])
        dDeltaF += data[phase_name]['dDeltaF']**2
        DeltaH -= sign * (data[phase_name]['DeltaH'] + data[phase_name]['DeltaF_standard_state_correction'])
        dDeltaH += data[phase_name]['dDeltaH']**2
    dDeltaF = np.sqrt(dDeltaF)
    dDeltaH = np.sqrt(dDeltaH)

    # Attempt to guess type of calculation
    calculation_type = ''
    for phase in phase_names:
        if 'complex' in phase:
            calculation_type = ' of binding'
        elif 'solvent1' in phase:
            calculation_type = ' of solvation'

    # Print energies
    logger.info("")
    logger.info("Free energy{}: {:16.3f} +- {:.3f} kT ({:16.3f} +- {:.3f} kcal/mol)".format(
        calculation_type, DeltaF, dDeltaF, DeltaF * kT / units.kilocalories_per_mole,
        dDeltaF * kT / units.kilocalories_per_mole))
    logger.info("")

    for phase in phase_names:
        logger.info("DeltaG {:<25} : {:16.3f} +- {:.3f} kT".format(phase, data[phase]['DeltaF'],
                                                                   data[phase]['dDeltaF']))
        if data[phase]['DeltaF_standard_state_correction'] != 0.0:
            logger.info("DeltaG {:<25} : {:25.3f} kT".format('restraint',
                                                             data[phase]['DeltaF_standard_state_correction']))
    logger.info("")
    logger.info("Enthalpy{}: {:16.3f} +- {:.3f} kT ({:16.3f} +- {:.3f} kcal/mol)".format(
        calculation_type, DeltaH, dDeltaH, DeltaH * kT / units.kilocalories_per_mole,
        dDeltaH * kT / units.kilocalories_per_mole))


"""
Everything below here is the old code. Currently left in for comparison
"""









# =============================================================================================
# SUBROUTINES
# =============================================================================================


def generate_mixing_statistics(ncfile, number_equilibrated=0):
    """
    Generate the mixing statistics

    Parameters
    ----------
    ncfile : netCDF4.Dataset
       NetCDF file
    number_equilibrated : int, optional, default=0
       If specified, only samples number_equilibrated:end will be used in analysis

    Returns
    -------
    mixing_stats : np.array of shape [nstates, nstates]
        Transition matrix estimate
    mu : np.array
        Eigenvalues of the Transition matrix sorted in descending order
    """

    # Get dimensions.
    niterations = ncfile.variables['states'].shape[0]
    nstates = ncfile.variables['states'].shape[1]

    # Compute empirical transition count matrix.
    Nij = np.zeros([nstates, nstates], np.float64)
    for iteration in range(number_equilibrated, niterations-1):
        for ireplica in range(nstates):
            istate = ncfile.variables['states'][iteration, ireplica]
            jstate = ncfile.variables['states'][iteration+1, ireplica]
            Nij[istate, jstate] += 1

    # Compute transition matrix estimate.
    # TODO: Replace with maximum likelihood reversible count estimator from msmbuilder or pyemma.
    Tij = np.zeros([nstates,nstates], np.float64)
    for istate in range(nstates):
        denom = (Nij[istate,:].sum() + Nij[:,istate].sum())
        if denom > 0:
            for jstate in range(nstates):
                Tij[istate, jstate] = (Nij[istate, jstate] + Nij[jstate, istate]) / denom
        else:
            Tij[istate, istate] = 1.0

    # Estimate eigenvalues
    mu = np.linalg.eigvals(Tij)
    mu = -np.sort(-mu)  # Sort in descending order

    return Tij, mu


def show_mixing_statistics(ncfile, cutoff=0.05, number_equilibrated=0):
    """
    Print summary of mixing statistics. Passes information off to generate_mixing_statistics then prints it out to
    the logger

    Parameters
    ----------

    ncfile : netCDF4.Dataset
       NetCDF file
    cutoff : float, optional, default=0.05
       Only transition probabilities above 'cutoff' will be printed
    number_equilibrated : int, optional, default=0
       If specified, only samples number_equilibrated:end will be used in analysis

    """

    Tij, mu = generate_mixing_statistics(ncfile, number_equilibrated=number_equilibrated)

    # Print observed transition probabilities.
    nstates = ncfile.variables['states'].shape[1]
    logger.info("Cumulative symmetrized state mixing transition matrix:")
    str_row = "%6s" % ""
    for jstate in range(nstates):
        str_row += "%6d" % jstate
    logger.info(str_row)

    for istate in range(nstates):
        str_row = ""
        str_row += "%-6d" % istate
        for jstate in range(nstates):
            P = Tij[istate,jstate]
            if P >= cutoff:
                str_row += "%6.3f" % P
            else:
                str_row += "%6s" % ""
        logger.info(str_row)

    # Estimate second eigenvalue and equilibration time.
    if mu[1] >= 1:
        logger.info("Perron eigenvalue is unity; Markov chain is decomposable.")
    else:
        logger.info("Perron eigenvalue is {0:9.5f}; state equilibration timescale is ~ {1:.1f} iterations".format(
            mu[1], 1.0 / (1.0 - mu[1]))
        )

    return


def extract_ncfile_energies(ncfile, ndiscard=0, nuse=None, g=None):
    """
    Extract and decorelate energies from the ncfile to gather common data for other functions

    Parameters
    ----------
    ncfile : NetCDF
       Input YANK netcdf file
    ndiscard : int, optional, default=0
       Number of iterations to discard to equilibration
    nuse : int, optional, default=None
       Maximum number of iterations to use (after discarding)
    g : int, optional, default=None
       Statistical inefficiency to use if desired; if None, will be computed.

    TODO
    ----
    * Automatically determine 'ndiscard'.

    """
    # Get current dimensions.
    niterations = ncfile.variables['energies'].shape[0]
    nstates = ncfile.variables['energies'].shape[1]
    natoms = ncfile.variables['energies'].shape[2]

    # Extract energies.
    logger.info("Reading energies...")
    energies = ncfile.variables['energies']
    u_kln_replica = np.zeros([nstates, nstates, niterations], np.float64)
    for n in range(niterations):
        u_kln_replica[:,:,n] = energies[n,:,:]
    logger.info("Done.")

    # Deconvolute replicas
    logger.info("Deconvoluting replicas...")
    u_kln = np.zeros([nstates, nstates, niterations], np.float64)
    for iteration in range(niterations):
        state_indices = ncfile.variables['states'][iteration,:]
        u_kln[state_indices,:,iteration] = energies[iteration,:,:]
    logger.info("Done.")

    # Compute total negative log probability over all iterations.
    u_n = np.zeros([niterations], np.float64)
    for iteration in range(niterations):
        u_n[iteration] = np.sum(np.diagonal(u_kln[:,:,iteration]))

    # Discard initial data to equilibration.
    u_kln_replica = u_kln_replica[:,:,ndiscard:]
    u_kln = u_kln[:,:,ndiscard:]
    u_n = u_n[ndiscard:]

    # Truncate to number of specified conforamtions to use
    if (nuse):
        u_kln_replica = u_kln_replica[:,:,0:nuse]
        u_kln = u_kln[:,:,0:nuse]
        u_n = u_n[0:nuse]

    # Subsample data to obtain uncorrelated samples
    N_k = np.zeros(nstates, np.int32)
    indices = timeseries.subsampleCorrelatedData(u_n, g=g) # indices of uncorrelated samples
    #print(u_n) # DEBUG
    #indices = range(0,u_n.size) # DEBUG - assume samples are uncorrelated
    N = len(indices) # number of uncorrelated samples
    N_k[:] = N
    u_kln = u_kln[:,:,indices]
    logger.info("number of uncorrelated samples:")
    logger.info(N_k)
    logger.info("")

    # Check for the expanded cutoff states, and subsamble as needed
    try:
        u_ln_full_raw = ncfile.variables['fully_interacting_expanded_cutoff_energies'][:].T #Its stored as nl, need in ln
        u_ln_non_raw = ncfile.variables['noninteracting_expanded_cutoff_energies'][:].T
        fully_interacting_u_ln = np.zeros(u_ln_full_raw.shape)
        noninteracting_u_ln = np.zeros(u_ln_non_raw.shape)
        # Deconvolute the fully interacting state
        for iteration in range(niterations):
            state_indices = ncfile.variables['states'][iteration,:]
            fully_interacting_u_ln[state_indices,iteration] = u_ln_full_raw[:,iteration]
            noninteracting_u_ln[state_indices,iteration] = u_ln_non_raw[:,iteration]
        # Discard non-equilibrated samples
        fully_interacting_u_ln = fully_interacting_u_ln[:,ndiscard:]
        fully_interacting_u_ln = fully_interacting_u_ln[:,indices]
        noninteracting_u_ln = noninteracting_u_ln[:,ndiscard:]
        noninteracting_u_ln = noninteracting_u_ln[:,indices]
        # Augment u_kln to accept the new state
        u_kln_new = np.zeros([nstates + 2, nstates + 2, N], np.float64)
        N_k_new = np.zeros(nstates + 2, np.int32)
        # Insert energies
        u_kln_new[1:-1,0,:] = fully_interacting_u_ln
        u_kln_new[1:-1,-1,:] = noninteracting_u_ln
        # Fill in other energies
        u_kln_new[1:-1,1:-1,:] = u_kln
        N_k_new[1:-1] = N_k
        # Notify users
        logger.info("Found expanded cutoff states in the energies!")
        logger.info("Free energies will be reported relative to them instead!")
        # Reset values, last step in case something went wrong so we dont overwrite u_kln on accident
        u_kln = u_kln_new
        N_k = N_k_new
    except:
        pass

    return u_kln, N_k, u_n


def initialize_MBAR(ncfile, u_kln=None, N_k=None):
    """
    Initialize MBAR for Free Energy and Enthalpy estimates, this may take a while.

    ncfile : NetCDF
       Input YANK netcdf file
    u_kln : array of numpy.float64, optional, default=None
       Reduced potential energies of the replicas; if None, will be extracted from the ncfile
    N_k : array of ints, optional, default=None
       Number of samples drawn from each kth replica; if None, will be extracted from the ncfile

    TODO
    ----
    * Ensure that the u_kln and N_k are decorrelated if not provided in this function

    """

    if u_kln is None or N_k is None:
        (u_kln, N_k, u_n) = extract_ncfile_energies(ncfile)

    # Initialize MBAR (computing free energy estimates, which may take a while)
    logger.info("Computing free energy differences...")
    mbar = MBAR(u_kln, N_k)

    return mbar


def estimate_free_energies(ncfile, mbar=None):
    """
    Estimate free energies of all alchemical states.

    Parameters
    ----------
    ncfile : NetCDF
       Input YANK netcdf file
    mbar : pymbar MBAR object, optional, default=None
       Initilized MBAR object from simulations; if None, it will be generated from the ncfile

    """

    # Create MBAR object if not provided
    if mbar is None:
        mbar = initialize_MBAR(ncfile)

    nstates = mbar.N_k.size

    # Get matrix of dimensionless free energy differences and uncertainty estimate.
    logger.info("Computing covariance matrix...")

    try:
        # pymbar 2
        (Deltaf_ij, dDeltaf_ij) = mbar.getFreeEnergyDifferences()
    except ValueError:
        # pymbar 3
        (Deltaf_ij, dDeltaf_ij, theta_ij) = mbar.getFreeEnergyDifferences()

    # Matrix of free energy differences
    logger.info("Deltaf_ij:")
    for i in range(nstates):
        str_row = ""
        for j in range(nstates):
            str_row += "%8.3f" % Deltaf_ij[i, j]
        logger.info(str_row)

    # Matrix of uncertainties in free energy difference (expectations standard
    # deviations of the estimator about the true free energy)
    logger.info("dDeltaf_ij:")
    for i in range(nstates):
        str_row = ""
        for j in range(nstates):
            str_row += "%8.3f" % dDeltaf_ij[i, j]
        logger.info(str_row)

    # Return free energy differences and an estimate of the covariance.
    return Deltaf_ij, dDeltaf_ij


def estimate_enthalpies(ncfile, mbar=None):
    """
    Estimate enthalpies of all alchemical states.

    Parameters
    ----------
    ncfile : NetCDF
       Input YANK netcdf file
    mbar : pymbar MBAR object, optional, default=None
       Initilized MBAR object from simulations; if None, it will be generated from the ncfile

    TODO
    ----
    * Check if there is an output/function name difference between pymbar 2 and 3
    """

    # Create MBAR object if not provided
    if mbar is None:
        mbar = initialize_MBAR(ncfile)

    nstates = mbar.N_k.size

    # Compute average enthalpies
    (f_k, df_k, H_k, dH_k, S_k, dS_k) = mbar.computeEntropyAndEnthalpy()

    return H_k, dH_k


def extract_u_n(ncfile):
    """
    Extract timeseries of u_n = - log q(X_n) from store file

    where q(X_n) = \pi_{k=1}^K u_{s_{nk}}(x_{nk})

    with X_n = [x_{n1}, ..., x_{nK}] is the current collection of replica configurations
    s_{nk} is the current state of replica k at iteration n
    u_k(x) is the kth reduced potential

    Parameters
    ----------
    ncfile : str
       The filename of the repex NetCDF file.

    Returns
    -------
    u_n : numpy array of numpy.float64
       u_n[n] is -log q(X_n)

    TODO
    ----
    Move this to repex.

    """

    # Get current dimensions.
    niterations = ncfile.variables['energies'].shape[0]
    nstates = ncfile.variables['energies'].shape[1]
    natoms = ncfile.variables['energies'].shape[2]

    # Extract energies.
    logger.info("Reading energies...")
    energies = ncfile.variables['energies']
    u_kln_replica = np.zeros([nstates, nstates, niterations], np.float64)
    for n in range(niterations):
        u_kln_replica[:,:,n] = energies[n,:,:]
    logger.info("Done.")

    # Deconvolute replicas
    logger.info("Deconvoluting replicas...")
    u_kln = np.zeros([nstates, nstates, niterations], np.float64)
    for iteration in range(niterations):
        state_indices = ncfile.variables['states'][iteration,:]
        u_kln[state_indices,:,iteration] = energies[iteration,:,:]
    logger.info("Done.")

    # Compute total negative log probability over all iterations.
    u_n = np.zeros([niterations], np.float64)
    for iteration in range(niterations):
        u_n[iteration] = np.sum(np.diagonal(u_kln[:,:,iteration]))

    return u_n

# =============================================================================================
# SHOW STATUS OF STORE FILES
# =============================================================================================


def print_status(store_directory):
    """
    Print a quick summary of simulation progress.

    Parameters
    ----------
    store_directory : string
       The location of the NetCDF simulation output files.

    Returns
    -------
    success : bool
       True is returned on success; False if some files could not be read.

    """
    # Get NetCDF files
    phases = utils.find_phases_in_store_directory(store_directory)

    # Process each netcdf file.
    for phase, fullpath in phases.items():

        # Check that the file exists.
        if not os.path.exists(fullpath):
            # Report failure.
            logger.info("File %s not found." % fullpath)
            logger.info("Check to make sure the right directory was specified, and 'yank setup' has been run.")
            return False

        # Open NetCDF file for reading.
        logger.debug("Opening NetCDF trajectory file '%(fullpath)s' for reading..." % vars())
        ncfile = netcdf.Dataset(fullpath, 'r')

        # Read dimensions.
        niterations = ncfile.variables['positions'].shape[0]
        nstates = ncfile.variables['positions'].shape[1]
        natoms = ncfile.variables['positions'].shape[2]

        # Print summary.
        logger.info("%s" % phase)
        logger.info("  %8d iterations completed" % niterations)
        logger.info("  %8d alchemical states" % nstates)
        logger.info("  %8d atoms" % natoms)

        # TODO: Print average ns/day and estimated completion time.

        # Close file.
        ncfile.close()

    return True

# =============================================================================================
# ANALYZE STORE FILES
# =============================================================================================


def analyze(source_directory):
    """
    Analyze contents of store files to compute free energy differences.

    Parameters
    ----------
    source_directory : string
       The location of the NetCDF simulation storage files.

    """
    analysis_script_path = os.path.join(source_directory, 'analysis.yaml')
    if not os.path.isfile(analysis_script_path):
        err_msg = 'Cannot find analysis.yaml script in {}'.format(source_directory)
        logger.error(err_msg)
        raise RuntimeError(err_msg)
    with open(analysis_script_path, 'r') as f:
        analysis = yaml.load(f)
    phases = [phase_name for phase_name, sign in analysis]

    # Storage for different phases.
    data = dict()

    # Process each netcdf file.
    for phase in phases:
        ncfile_path = os.path.join(source_directory, phase + '.nc')

        # Open NetCDF file for reading.
        logger.info("Opening NetCDF trajectory file %(ncfile_path)s for reading..." % vars())
        try:
            ncfile = netcdf.Dataset(ncfile_path, 'r')

            logger.debug("dimensions:")
            for dimension_name in ncfile.dimensions.keys():
                logger.debug("%16s %8d" % (dimension_name, len(ncfile.dimensions[dimension_name])))

            # Read dimensions.
            niterations = ncfile.variables['positions'].shape[0]
            nstates = ncfile.variables['positions'].shape[1]
            logger.info("Read %(niterations)d iterations, %(nstates)d states" % vars())

            DeltaF_standard_state_correction = 0.0
            if 'metadata' in ncfile.groups:
                # Read phase direction and standard state correction free energy.
                # Yank sets correction to 0 if there are no standard_state_correction
                DeltaF_standard_state_correction = ncfile.groups['metadata'].variables['standard_state_correction'][0]

            # Choose number of samples to discard to equilibration
            MIN_ITERATIONS = 10 # minimum number of iterations to use automatic detection
            if niterations > MIN_ITERATIONS:
                from pymbar import timeseries
                u_n = extract_u_n(ncfile)
                u_n = u_n[1:] # discard initial frame of zero energies TODO: Get rid of initial frame of zero energies
                [number_equilibrated, g_t, Neff_max] = timeseries.detectEquilibration(u_n)
                number_equilibrated += 1 # account for initial frame of zero energies
                logger.info([number_equilibrated, Neff_max])
            else:
                number_equilibrated = 1  # discard first frame
                g_t = 1
                Neff_max = niterations

            # Examine acceptance probabilities.
            show_mixing_statistics(ncfile, cutoff=0.05, number_equilibrated=number_equilibrated)

            # Extract equilibrated, decorrelated energies, check for fully interacting state
            (u_kln, N_k, u_n) = extract_ncfile_energies(ncfile, ndiscard=number_equilibrated, g=g_t)

            # Create MBAR object to use for free energy and entropy states
            mbar = initialize_MBAR(ncfile, u_kln=u_kln, N_k=N_k)

            # Estimate free energies, use fully interacting state if present
            (Deltaf_ij, dDeltaf_ij) = estimate_free_energies(ncfile, mbar=mbar)

            # Estimate average enthalpies
            (DeltaH_i, dDeltaH_i) = estimate_enthalpies(ncfile, mbar=mbar)

            # Accumulate free energy differences
            entry = dict()
            entry['DeltaF'] = Deltaf_ij[0, -1]
            entry['dDeltaF'] = dDeltaf_ij[0, -1]
            entry['DeltaH'] = DeltaH_i[0, -1]
            entry['dDeltaH'] = dDeltaH_i[0, -1]
            entry['DeltaF_standard_state_correction'] = DeltaF_standard_state_correction
            data[phase] = entry

            # Get temperatures.
            ncvar = ncfile.groups['thermodynamic_states'].variables['temperatures']
            temperature = ncvar[0] * units.kelvin
            kT = kB * temperature

        finally:
            ncfile.close()

    # Compute free energy and enthalpy
    DeltaF = 0.0
    dDeltaF = 0.0
    DeltaH = 0.0
    dDeltaH = 0.0
    for phase, sign in analysis:
        DeltaF -= sign * (data[phase]['DeltaF'] + data[phase]['DeltaF_standard_state_correction'])
        dDeltaF += data[phase]['dDeltaF']**2
        DeltaH -= sign * (data[phase]['DeltaH'] + data[phase]['DeltaF_standard_state_correction'])
        dDeltaH += data[phase]['dDeltaH']**2
    dDeltaF = np.sqrt(dDeltaF)
    dDeltaH = np.sqrt(dDeltaH)

    # Attempt to guess type of calculation
    calculation_type = ''
    for phase in phases:
        if 'complex' in phase:
            calculation_type = ' of binding'
        elif 'solvent1' in phase:
            calculation_type = ' of solvation'

    # Print energies
    logger.info("")
    logger.info("Free energy{}: {:16.3f} +- {:.3f} kT ({:16.3f} +- {:.3f} kcal/mol)".format(
        calculation_type, DeltaF, dDeltaF, DeltaF * kT / units.kilocalories_per_mole,
        dDeltaF * kT / units.kilocalories_per_mole))
    logger.info("")

    for phase in phases:
        logger.info("DeltaG {:<25} : {:16.3f} +- {:.3f} kT".format(phase, data[phase]['DeltaF'],
                                                                   data[phase]['dDeltaF']))
        if data[phase]['DeltaF_standard_state_correction'] != 0.0:
            logger.info("DeltaG {:<25} : {:25.3f} kT".format('restraint',
                                                             data[phase]['DeltaF_standard_state_correction']))
    logger.info("")
    logger.info("Enthalpy{}: {:16.3f} +- {:.3f} kT ({:16.3f} +- {:.3f} kcal/mol)".format(
        calculation_type, DeltaH, dDeltaH, DeltaH * kT / units.kilocalories_per_mole,
        dDeltaH * kT / units.kilocalories_per_mole))


# ==============================================================================
# Extract trajectory from NetCDF4 file
# ==============================================================================

def extract_trajectory(output_path, nc_path, nc_checkpoint_file=None, state_index=None, replica_index=None,
                       start_frame=0, end_frame=-1, skip_frame=1, keep_solvent=True,
                       discard_equilibration=False, image_molecules=False):
    """Extract phase trajectory from the NetCDF4 file.

    Parameters
    ----------
    output_path : str
        Path to the trajectory file to be created. The extension of the file
        determines the format.
    nc_path : str
        Path to the primary nc_file storing the analysis options
    nc_checkpoint_file : str or None, Optional
        File name of the checkpoint file housing the main trajectory
        Used if the checkpoint file is differently named from the default one chosen by the nc_path file.
        Default: None
    state_index : int, optional
        The index of the alchemical state for which to extract the trajectory.
        One and only one between state_index and replica_index must be not None
        (default is None).
    replica_index : int, optional
        The index of the replica for which to extract the trajectory. One and
        only one between state_index and replica_index must be not None (default
        is None).
    start_frame : int, optional
        Index of the first frame to include in the trajectory (default is 0).
    end_frame : int, optional
        Index of the last frame to include in the trajectory. If negative, will
        count from the end (default is -1).
    skip_frame : int, optional
        Extract one frame every skip_frame (default is 1).
    keep_solvent : bool, optional
        If False, solvent molecules are ignored (default is True).
    discard_equilibration : bool, optional
        If True, initial equilibration frames are discarded (see the method
        pymbar.timeseries.detectEquilibration() for details, default is False).

    """
    # Check correct input
    if (state_index is None) == (replica_index is None):
        raise ValueError('One and only one between "state_index" and '
                         '"replica_index" must be specified.')
    if not os.path.isfile(nc_path):
        raise ValueError('Cannot find file {}'.format(nc_path))

    # Import simulation data
    try:
        reporter = Reporter(nc_path, open_mode='r', checkpoint_storage_file=nc_checkpoint_file)
        metadata = reporter.read_dict('metadata')
        reference_system = mmtools.utils.deserialize(metadata['reference_state']).system
        topology = mmtools.utils.deserialize(metadata['topography']).topology

        # Determine if system is periodic
        is_periodic = reference_system.usesPeriodicBoundaryConditions()
        logger.info('Detected periodic boundary conditions: {}'.format(is_periodic))

        # Get dimensions
        n_iterations = reporter._storage_checkpoint.variables['positions'].shape[0]
        n_atoms = reporter._storage_checkpoint.variables['positions'].shape[2]
        logger.info('Number of iterations: {}, atoms: {}'.format(n_iterations, n_atoms))

        # Determine frames to extract
        if start_frame <= 0:
            # Discard frame 0 with minimized energy which
            # throws off automatic equilibration detection.
            start_frame = 1
        if end_frame < 0:
            end_frame = n_iterations + end_frame + 1
        frame_indices = range(start_frame, end_frame, skip_frame)
        if len(frame_indices) == 0:
            raise ValueError('No frames selected')
        logger.info('Extracting frames from {} to {} every {}'.format(
            start_frame, end_frame, skip_frame))

        # Discard equilibration samples
        if discard_equilibration:
            u_n = extract_u_n(reporter._storage_analysis)[frame_indices]
            n_equil, g, n_eff = timeseries.detectEquilibration(u_n)
            logger.info(("Discarding initial {} equilibration samples (leaving {} "
                         "effectively uncorrelated samples)...").format(n_equil, n_eff))
            frame_indices = frame_indices[n_equil:-1]

        written_indices = []
        for frame in frame_indices:
            if reporter.get_previous_checkpoint(frame):
                written_indices.append(frame)
        # Extract state positions and box vectors
        positions = np.zeros((len(written_indices), n_atoms, 3))
        if is_periodic:
            box_vectors = np.zeros((len(written_indices), 3, 3))
        if state_index is not None:
            logger.info('Extracting positions of state {}...'.format(state_index))

            # Deconvolute state indices
            state_indices = np.zeros(len(written_indices))
            for i, iteration in enumerate(written_indices):
                replica_indices = reporter._storage_analysis.variables['states'][iteration, :]
                state_indices[i] = np.where(replica_indices == state_index)[0][0]

            # Extract state positions and box vectors
            for i, iteration in enumerate(written_indices):
                replica_index = state_indices[i]
                positions[i, :, :] = reporter._storage_checkpoint.variables['positions'][i, replica_index, :, :].astype(np.float32)
                if is_periodic:
                    box_vectors[i, :, :] = reporter._storage_checkpoint.variables['box_vectors'][i, replica_index, :, :].astype(np.float32)

        else:  # Extract replica positions and box vectors
            logger.info('Extracting positions of replica {}...'.format(replica_index))

            for i, iteration in enumerate(written_indices):
                positions[i, :, :] = reporter._storage_checkpoint.variables['positions'][i, replica_index, :, :].astype(np.float32)
                if is_periodic:
                    box_vectors[i, :, :] = reporter._storage_checkpoint.variables['box_vectors'][i, replica_index, :, :].astype(np.float32)
    finally:
        reporter.close()

    # Create trajectory object
    logger.info('Creating trajectory object...')
    trajectory = mdtraj.Trajectory(positions, topology)
    if is_periodic:
        trajectory.unitcell_vectors = box_vectors

    # Force periodic boundary conditions to molecules positions
    if image_molecules:
        logger.info('Applying periodic boundary conditions to molecules positions...')
        trajectory.image_molecules(inplace=True)

    # Remove solvent
    if not keep_solvent:
        logger.info('Removing solvent molecules...')
        trajectory = trajectory.remove_solvent()

    # Detect format
    extension = os.path.splitext(output_path)[1][1:]  # remove dot
    try:
        save_function = getattr(trajectory, 'save_' + extension)
    except AttributeError:
        raise ValueError('Cannot detect format from extension of file {}'.format(output_path))

    # Create output directory and save trajectory
    logger.info('Creating trajectory file: {}'.format(output_path))
    output_dir = os.path.dirname(output_path)
    if output_dir != '' and not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    save_function(output_path)
