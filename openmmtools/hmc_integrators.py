"""Hamiltonian Monte Carlo Integrators

Notes
-----
The code in this module is considered EXPERIMENTAL until further notice.
"""
import time
import logging
import pandas as pd
import numpy as np

import simtk.unit as u
import simtk.openmm as mm

from .constants import kB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_groups(groups, guess=True):
    """Check that `groups` is list of tuples suitable for force group / RESPA."""
    if groups is None or len(groups) == 0:
        if guess:
            logger.info("No force groups specified, using [(0, 1)]!")
            groups = [(0, 1)]
        else:
            raise ValueError("No force groups specified")

    groups = sorted(groups, key=lambda x: x[1])
    return groups


def guess_force_groups(system, nonbonded=1, fft=1, others=0, multipole=1):
    """Set NB short-range to 1 and long-range to 1, which is usually OK.
    This is useful for RESPA multiple timestep integrators.
    """
    for force in system.getForces():
        if isinstance(force, mm.openmm.NonbondedForce):
            force.setForceGroup(nonbonded)
            force.setReciprocalSpaceForceGroup(fft)
        elif isinstance(force, mm.openmm.CustomGBForce):
            force.setForceGroup(nonbonded)
        elif isinstance(force, mm.openmm.GBSAOBCForce):
            force.setForceGroup(nonbonded)
        elif isinstance(force, mm.AmoebaMultipoleForce):
            force.setForceGroup(multipole)
        elif isinstance(force, mm.AmoebaVdwForce):
            force.setForceGroup(nonbonded)
        else:
            force.setForceGroup(others)


class GHMCBase(mm.CustomIntegrator):
    """Generalized hybrid Monte Carlo integrator base class.

    Notes
    -----
    This loosely follows the definition of GHMC given in the two below
    references.  Specifically, the velocities are corrupted (partially or completely),
    several steps of hamiltonian dynamics are performed, and then
    an accept / reject move is taken.

    This class is the base class for a number of more specialized versions
    of GHMC.

    References
    ----------
    C. M. Campos, J. M. Sanz-Serna, J. Comp. Phys. 281, (2015)
    J. Sohl-Dickstein, M. Mudigonda, M. DeWeese.  ICML (2014)
    """

    def step(self, n_steps):
        """Do `n_steps` of dynamics while accumulating some timing information."""
        if not hasattr(self, "elapsed_time"):
            self.elapsed_time = 0.0

        t0 = time.time()
        mm.CustomIntegrator.step(self, n_steps)
        dt = time.time() - t0

        self.elapsed_time += dt

    def reset_time(self):
        """Resets benchmark timing counters, avoids counting the CustomIntegrator setup time."""
        self.step(1)
        self.elapsed_time = 0.0
        self.setGlobalVariableByName("steps_taken", 0.0)
        self.setGlobalVariableByName("steps_accepted", 0.0)
        # Note: XCHMC has additional counters n_i that are not reset here.  This should not be a problem as they aren't used in benchmarking

    @property
    def time_per_step(self):
        return (self.elapsed_time / self.steps_taken)

    @property
    def days_per_step(self):
        return self.time_per_step / (60. * 60. * 24.)

    @property
    def effective_ns_per_day(self):
        return (self.effective_timestep / self.days_per_step) / u.nanoseconds

    @property
    def ns_per_day(self):
        return (self.timestep / self.days_per_step) / u.nanoseconds

    def vstep(self, n_steps, verbose=False):
        """Do n_steps of dynamics and return a summary dataframe."""

        data = []
        for i in range(n_steps):
            self.step(1)

            d = self.summary()
            data.append(d)

        data = pd.DataFrame(data)

        print(data.to_string(formatters=[lambda x: "%.4g" % x for x in range(data.shape[1])]))
        return data

    def create(self):
        self.initialize_variables()
        self.add_draw_velocities_step()
        self.add_cache_variables_step()
        self.add_hmc_iterations()
        self.add_accept_or_reject_step()

    def add_hmc_iterations(self):
        """Add self.steps_per_hmc iterations of symplectic hamiltonian dynamics."""
        logger.debug("Adding (G)HMCIntegrator steps.")
        for step in range(self.steps_per_hmc):
            self.add_hamiltonian_step()

    def add_hamiltonian_step(self):
        """Add a single step of hamiltonian integration.

        Notes
        -----
        This function will be overwritten in RESPA subclasses!
        """
        logger.debug("Adding step of hamiltonian dynamics.""")
        self.addComputePerDof("v", "v+0.5*dt*f/m")
        self.addComputePerDof("x", "x+dt*v")
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions()
        self.addComputePerDof("v", "v+0.5*dt*f/m+(x-x1)/dt")
        self.addConstrainVelocities()

    @property
    def b(self):
        """The scaling factor for preserving versus randomizing velocities."""
        return np.exp(-self.collision_rate * self.timestep)

    @property
    def kT(self):
        """The thermal energy."""
        return kB * self.temperature

    def summary(self):
        """Return a dictionary of relevant state variables for XHMC, useful for debugging.
        Append self.summary() to a list and print out as a dataframe.
        """
        d = {}
        d["arate"] = self.acceptance_rate
        d["effective_timestep"] = self.effective_timestep / u.femtoseconds
        d["effective_ns_per_day"] = self.effective_ns_per_day
        d["ns_per_day"] = self.ns_per_day
        keys = ["accept", "ke", "Enew", "Eold"]

        for key in keys:
            d[key] = self.getGlobalVariableByName(key)

        d["deltaE"] = d["Enew"] - d["Eold"]

        return d

    @property
    def steps_taken(self):
        """Total number of hamiltonian steps taken."""
        return self.getGlobalVariableByName("steps_taken")

    @property
    def steps_accepted(self):
        """Total number of hamiltonian steps accepted."""
        return self.getGlobalVariableByName("steps_accepted")

    @property
    def accept(self):
        """Return True if the last step taken was accepted."""
        return bool(self.getGlobalVariableByName("accept"))

    @property
    def acceptance_rate(self):
        """The acceptance rate, evaluated via the number of force evaluations.

        Notes
        -----
        This should be sufficiently general to apply to both XCGHMC and GHMC.
        In the latter case, the number of force evaluations taken (and accepted)
        are both proportional to the number of MC moves taken.
        """
        return self.steps_accepted / self.steps_taken

    @property
    def effective_timestep(self):
        """The acceptance rate times the timestep."""
        return self.acceptance_rate * self.timestep

    @property
    def is_GHMC(self):
        return self.collision_rate is not None

class GHMCIntegrator(GHMCBase):
    """Generalized hybrid Monte Carlo (GHMC) integrator.

    Parameters
    ----------
    temperature : numpy.unit.Quantity compatible with kelvin, default: 298*simtk.unit.kelvin
       The temperature.
    steps_per_hmc : int, default: 10
       The number of velocity Verlet steps to take per round of hamiltonian dynamics
    timestep : numpy.unit.Quantity compatible with femtoseconds, default: 1*simtk.unit.femtoseconds
       The integration timestep.  The total time taken per iteration
       will equal timestep * steps_per_hmc
    collision_rate : numpy.unit.Quantity compatible with 1 / femtoseconds, default: None
       The collision rate for the velocity corruption (GHMC).  If None,
       velocities information will be discarded after each round (HMC).

    Notes
    -----
    This loosely follows the definition of GHMC given in the two below
    references.  Specifically, the velocities are corrupted,
    several steps of hamiltonian dynamics are performed, and then
    an accept / reject move is taken.  If collision_rate is set to None, however,
    we will do non-generalized HMC, where the velocities information is discarded
    at each iteration.

    This class is the base class for a number of more specialized versions
    of GHMC.

    References
    ----------
    C. M. Campos, J. M. Sanz-Serna, J. Comp. Phys. 281, (2015)
    J. Sohl-Dickstein, M. Mudigonda, M. DeWeese.  ICML (2014)
    """

    def __init__(self, temperature=298.0 * u.kelvin, steps_per_hmc=10, timestep=1 * u.femtoseconds, collision_rate=None):
        mm.CustomIntegrator.__init__(self, timestep)

        self.steps_per_hmc = steps_per_hmc
        self.temperature = temperature
        self.collision_rate = collision_rate
        self.timestep = timestep
        self.create()

    def initialize_variables(self):

        self.addGlobalVariable("kT", self.kT)  # thermal energy
        self.addPerDofVariable("sigma", 0)
        self.addGlobalVariable("ke", 0)  # kinetic energy
        self.addPerDofVariable("xold", 0)  # old positions
        self.addPerDofVariable("vold", 0)  # old velocities
        self.addGlobalVariable("Eold", 0)  # old energy
        self.addGlobalVariable("Enew", 0)  # new energy
        self.addGlobalVariable("accept", 0)  # accept or reject
        self.addPerDofVariable("x1", 0)  # for constraints

        self.addGlobalVariable("steps_accepted", 0)  # Number of productive hamiltonian steps
        self.addGlobalVariable("steps_taken", 0)  # Number of total hamiltonian steps

        if self.is_GHMC:
            self.addGlobalVariable("b", self.b)  # velocity mixing parameter

        self.addComputePerDof("sigma", "sqrt(kT/m)")
        self.addUpdateContextState()

    def add_draw_velocities_step(self):
        """Draw perturbed velocities."""

        if self.is_GHMC:
            self.addComputePerDof("v", "sqrt(b)*v + sqrt(1-b)*sigma*gaussian")
        else:
            self.addComputePerDof("v", "sigma*gaussian")

        self.addConstrainVelocities()

    def add_cache_variables_step(self):
        """Store old positions and energies."""
        self.addComputeSum("ke", "0.5*m*v*v")
        self.addComputeGlobal("Eold", "ke + energy")
        self.addComputePerDof("xold", "x")
        self.addComputePerDof("vold", "v")

    def add_accept_or_reject_step(self):
        logger.debug("GHMC: add_accept_or_reject_step()")
        self.addComputeSum("ke", "0.5*m*v*v")
        self.addComputeGlobal("Enew", "ke + energy")
        self.addComputeGlobal("accept", "step(exp(-(Enew-Eold)/kT) - uniform)")

        self.addComputePerDof("x", "select(accept, x, xold)")

        if self.is_GHMC:
            self.addComputePerDof("v", "select(accept, v, -1*vold)")

        self.addComputeGlobal("steps_accepted", "steps_accepted + accept * %d" % (self.steps_per_hmc))
        self.addComputeGlobal("steps_taken", "steps_taken + %d" % (self.steps_per_hmc))


class RESPAMixIn(object):
    """Mixin object to provide RESPA timestepping for an HMC integrator."""

    def add_hamiltonian_step(self):
        """Add a single step of RESPA hamiltonian integration."""
        logger.debug("Adding step of RESPA hamiltonian dynamics.""")
        self._create_substeps(1, self.groups)
        self.addConstrainVelocities()

    def _create_substeps(self, parentSubsteps, groups):

        group, substeps = groups[0]

        str_group, str_sub = str(group), str(substeps)

        stepsPerParentStep = substeps / parentSubsteps

        if stepsPerParentStep < 1 or stepsPerParentStep != int(stepsPerParentStep):
            raise ValueError("The number for substeps for each group must be a multiple of the number for the previous group")

        stepsPerParentStep = int(stepsPerParentStep)  # needed for Python 3.x

        if group < 0 or group > 31:
            raise ValueError("Force group must be between 0 and 31")

        for i in range(stepsPerParentStep):
            self.addComputePerDof("v", "v+0.5*(dt/%s)*f%s/m" % (str_sub, str_group))
            if len(groups) == 1:
                self.addComputePerDof("x1", "x")
                self.addComputePerDof("x", "x+(dt/%s)*v" % (str_sub))
                self.addConstrainPositions()
                self.addComputePerDof("v", "(x-x1)/(dt/%s)" % (str_sub))
            else:
                self._create_substeps(substeps, groups[1:])
            self.addComputePerDof("v", "v+0.5*(dt/%s)*f%s/m" % (str_sub, str_group))


class GHMCRESPAIntegrator(RESPAMixIn, GHMCIntegrator):
    """Generalized Hamiltonian Monte Carlo (GHMC) with a rRESPA multiple
    time step integration algorithm.  Combines GHMCIntegrator with
    MTSIntegrator.

    Parameters
    ----------
    temperature : numpy.unit.Quantity compatible with kelvin, default: 298*simtk.unit.kelvin
        The temperature.
    steps_per_hmc : int, default: 10
        The number of velocity Verlet steps to take per round of hamiltonian dynamics
    timestep : numpy.unit.Quantity compatible with femtoseconds, default: 1*simtk.unit.femtoseconds
        The integration timestep.  The total time taken per iteration
        will equal timestep * steps_per_hmc
    collision_rate : numpy.unit.Quantity compatible with 1 / femtoseconds, default: None
       The collision rate for the velocity corruption (GHMC).  If None,
       velocities information will be discarded after each round (HMC).
    groups : list of tuples, optional, default=None
        A list of tuples defining the force groups.  The first element
        of each tuple is the force group index, and the second element
        is the number of times that force group should be evaluated in
        one time step.  If None, a default choice of [(0, 1)] will be used!!!

    Notes
    -----

    This integrator allows different forces to be evaluated at different frequencies,
    for example to evaluate the expensive, slowly changing forces less frequently than
    the inexpensive, quickly changing forces.

    To use it, you must first divide your forces into two or more groups (by calling
    setForceGroup() on them) that should be evaluated at different frequencies.  When
    you create the integrator, you provide a tuple for each group specifying the index
    of the force group and the frequency (as a fraction of the outermost time step) at
    which to evaluate it.  For example:

    >>> integrator = GHMCRESPAIntegrator(timestep=4*simtk.unit.femtoseconds, groups=[(0,1), (1,2), (2,8)])

    This specifies that the outermost time step is 4 fs, so each step of the integrator
    will advance time by that much.  It also says that force group 0 should be evaluated
    once per time step, force group 1 should be evaluated twice per time step (every 2 fs),
    and force group 2 should be evaluated eight times per time step (every 0.5 fs).

    The RESPA multiple timestep splitting should closely follow the code
    in MTSIntegrator.

    References
    ----------
    C. M. Campos, J. M. Sanz-Serna, J. Comp. Phys. 281, (2015)
    J. Sohl-Dickstein, M. Mudigonda, M. DeWeese.  ICML (2014)
    Tuckerman et al., J. Chem. Phys. 97(3) pp. 1990-2001 (1992)
    """

    def __init__(self, temperature=298.0 * u.kelvin, steps_per_hmc=10, timestep=1 * u.femtoseconds, collision_rate=1.0 / u.picoseconds, groups=None):
        mm.CustomIntegrator.__init__(self, timestep)

        self.groups = check_groups(groups)
        self.steps_per_hmc = steps_per_hmc

        self.collision_rate = collision_rate
        self.timestep = timestep
        self.temperature = temperature

        self.create()


class XCGHMCIntegrator(GHMCIntegrator):
    """Extra Chance generalized hybrid Monte Carlo (XCGHMC) integrator.

        Parameters
        ----------
        temperature : numpy.unit.Quantity compatible with kelvin, default: 298*simtk.unit.kelvin
            The temperature.
        steps_per_hmc : int, default: 10
            The number of velocity Verlet steps to take per round of hamiltonian dynamics
        timestep : numpy.unit.Quantity compatible with femtoseconds, default: 1*simtk.unit.femtoseconds
            The integration timestep.  The total time taken per iteration
            will equal timestep * steps_per_hmc
        extra_chances : int, optional, default=2
            The number of extra chances.  If the initial move is rejected, up to
            `extra_chances` rounds of additional moves will be attempted to find
            an accepted proposal.  `extra_chances=0` correponds to vanilla (G)HMC.
        steps_per_extra_hmc : int, optional, default=1
            During each extra chance, do this many steps of hamiltonian dynamics.
        collision_rate : numpy.unit.Quantity compatible with 1 / femtoseconds, default: None
           The collision rate for the velocity corruption (GHMC).  If None,
           velocities information will be discarded after each round (HMC).

    Notes
    -----
    This integrator attempts to circumvent rejections by propagating up to
    `extra_chances` steps of additional dynamics.

    References
    ----------
    C. M. Campos, J. M. Sanz-Serna, J. Comp. Phys. 281, (2015)
    J. Sohl-Dickstein, M. Mudigonda, M. DeWeese.  ICML (2014)
    """

    def __init__(self, temperature=298.0 * u.kelvin, steps_per_hmc=10, timestep=1 * u.femtoseconds, extra_chances=2, steps_per_extra_hmc=1, collision_rate=None):
        mm.CustomIntegrator.__init__(self, timestep)

        self.temperature = temperature
        self.steps_per_hmc = steps_per_hmc
        self.steps_per_extra_hmc = steps_per_extra_hmc
        self.timestep = timestep
        self.extra_chances = extra_chances
        self.collision_rate = collision_rate

        self.create()

    @property
    def all_counts(self):
        """Return a pandas series of moves accepted after step 0, 1, ... extra_chances - 1, and flip"""
        d = {}
        for i in range(1 + self.extra_chances):
            d[i] = self.getGlobalVariableByName("n%d" % i)

        d["flip"] = self.n_flip

        return pd.Series(d)

    @property
    def all_probs(self):
        """Return a pandas series of probabilities of moves accepted after step 0, 1, ... extra_chances - 1, and flip"""
        d = self.all_counts
        return d / d.sum()

    @property
    def n_flip(self):
        """The total number of momentum flips."""
        return self.getGlobalVariableByName("nflip")

    def create(self):
        self.initialize_variables()
        self.add_draw_velocities_step()
        self.add_cache_variables_step()
        for i in range(1 + self.extra_chances):
            self.add_hmc_iterations(i)
            self.add_accept_or_reject_step(i)

        self.addComputeGlobal("flip", "(1 - done)")
        self.addComputePerDof("x", "select(flip, xold, xfinal)")

        if self.is_GHMC:
            self.addComputePerDof("v", "select(flip, -vold, vfinal)")  # Requires OMM Build on May 6, 2015

        self.addComputeGlobal("nflip", "nflip + flip")

    def initialize_variables(self):

        self.addGlobalVariable("accept", 1.0)  # accept or reject
        self.addGlobalVariable("r", 0.0)  # Metropolis ratio: ratio probabilities

        self.addGlobalVariable("extra_chances", self.extra_chances)  # Maximum number of rounds of dynamics
        self.addGlobalVariable("flip", 0.0)  # Indicator variable whether this iteration was a flip

        self.addGlobalVariable("mu", 0.0)  #
        self.addGlobalVariable("mu1", 0.0)  # XCGHMC Fig. 3 O1

        for i in range(1 + self.extra_chances):
            self.addGlobalVariable("n%d" % i, 0.0)  # Number of times accepted when k = i

        self.addGlobalVariable("nflip", 0)  # number of momentum flips (e.g. complete rejections)
        self.addGlobalVariable("nrounds", 0)  # number of "rounds" of XHMC, e.g. the number of times k = 0

        # Below this point is possible base class material

        self.addGlobalVariable("kT", self.kT)  # thermal energy
        self.addPerDofVariable("sigma", 0)
        self.addGlobalVariable("ke", 0)  # kinetic energy
        self.addPerDofVariable("xold", 0)  # old positions
        self.addGlobalVariable("Eold", 0)  # old energy
        self.addGlobalVariable("Enew", 0)  # new energy

        self.addGlobalVariable("done", 0)  # Becomes true once we find a good value of (x, p)

        self.addPerDofVariable("x1", 0)  # for constraints

        self.addPerDofVariable("xfinal", 0)
        self.addPerDofVariable("vfinal", 0)

        self.addGlobalVariable("steps_accepted", 0)  # Number of productive hamiltonian steps
        self.addGlobalVariable("steps_taken", 0)  # Number of total hamiltonian steps

        self.addComputePerDof("sigma", "sqrt(kT/m)")
        if self.is_GHMC:
            self.addGlobalVariable("b", self.b)  # velocity mixing parameter
        self.addUpdateContextState()

    def add_draw_velocities_step(self):
        """Draw perturbed velocities."""

        self.addUpdateContextState()
        self.addConstrainPositions()

        self.addComputeGlobal("done", "0.0")
        if self.is_GHMC:
            self.addComputePerDof("v", "sqrt(b)*v + sqrt(1-b)*sigma*gaussian")
        else:
            self.addComputePerDof("v", "sigma*gaussian")

        self.addConstrainVelocities()

    def add_cache_variables_step(self):
        """Store old positions and energies."""

        self.addComputeSum("ke", "0.5*m*v*v")
        self.addComputeGlobal("Eold", "ke + energy")
        self.addComputePerDof("xold", "x")

        self.addComputeGlobal("mu1", "0.0")  # XCGHMC Fig. 3 O1

    def add_hmc_iterations(self, i):
        """Add self.steps_per_hmc or self.steps_per_extra_hmc iterations of symplectic hamiltonian dynamics."""
        logger.debug("Adding XCGHMCIntegrator steps.")

        steps = self.steps_per_hmc

        if i > 0:
            steps = self.steps_per_extra_hmc

        for step in range(steps):
            self.add_hamiltonian_step()

    def add_accept_or_reject_step(self, i):
        logger.debug("XCGHMC: add_accept_or_reject_step()")
        self.addComputeSum("ke", "0.5*m*v*v")
        self.addComputeGlobal("Enew", "ke + energy")

        self.addComputeGlobal("r", "exp(-(Enew - Eold) / kT)")
        self.addComputeGlobal("mu", "min(1, r)")  # XCGHMC paper version
        self.addComputeGlobal("mu1", "max(mu1, mu)")
        self.addComputeGlobal("accept", "step(mu1 - uniform) * (1 - done)")

        if i == 0:
            steps = self.steps_per_hmc
        else:
            steps = self.steps_per_extra_hmc

        cumulative_steps = self.steps_per_hmc + i * self.steps_per_extra_hmc

        self.addComputeGlobal("n%d" % i, "n%d + accept" % (i))

        self.addComputeGlobal("steps_accepted", "steps_accepted + accept * %d" % (cumulative_steps))
        self.addComputeGlobal("steps_taken", "steps_taken + %d" % (steps))

        self.addComputePerDof("xfinal", "select(accept, x, xfinal)")

        if self.is_GHMC:
            self.addComputePerDof("vfinal", "select(accept, v, vfinal)")

        self.addComputeGlobal("done", "max(done, accept)")
        # self.addConditionalTermination("done")  #  Conditional termination here would avoid additional force+energy evaluations.


class XCGHMCRESPAIntegrator(RESPAMixIn, XCGHMCIntegrator):
    """Extra Chance Generalized Hybrid Monte Carlo RESPA Integrator.

        Parameters
        ----------
        temperature : numpy.unit.Quantity compatible with kelvin, default: 298*simtk.unit.kelvin
            The temperature.
        steps_per_hmc : int, default: 10
            The number of velocity Verlet steps to take per round of hamiltonian dynamics
        timestep : numpy.unit.Quantity compatible with femtoseconds, default: 1*simtk.unit.femtoseconds
            The integration timestep.  The total time taken per iteration
            will equal timestep * steps_per_hmc
        extra_chances : int, optional, default=2
            The number of extra chances.  If the initial move is rejected, up to
            `extra_chances` rounds of additional moves will be attempted to find
            an accepted proposal.  `extra_chances=0` correponds to vanilla (G)HMC.
        steps_per_extra_hmc : int, optional, default=1
            During each extra chance, do this many steps of hamiltonian dynamics.
        collision_rate : numpy.unit.Quantity compatible with 1 / femtoseconds, default: None
           The collision rate for the velocity corruption (GHMC).  If None,
           velocities information will be discarded after each round (HMC).
        groups : list of tuples, optional, default=None
            A list of tuples defining the force groups.  The first element
            of each tuple is the force group index, and the second element
            is the number of times that force group should be evaluated in
            one time step.  If None, a default choice of [(0, 1)] will be used!!!

    Notes
    -----
    This integrator attempts to circumvent rejections by propagating up to
    `extra_chances` steps of additional dynamics.  During each extra chance,
    `steps_per_extra_hmc` steps of hamiltonian dynamics are taken.

    References
    ----------
    C. M. Campos, J. M. Sanz-Serna, J. Comp. Phys. 281, (2015)
    J. Sohl-Dickstein, M. Mudigonda, M. DeWeese.  ICML (2014)
    """

    def __init__(self, temperature=298.0 * u.kelvin, steps_per_hmc=10, timestep=1 * u.femtoseconds, extra_chances=2, steps_per_extra_hmc=1, collision_rate=None, groups=None):
        mm.CustomIntegrator.__init__(self, timestep)

        self.groups = check_groups(groups)
        self.temperature = temperature
        self.steps_per_hmc = steps_per_hmc
        self.steps_per_extra_hmc = steps_per_extra_hmc
        self.timestep = timestep
        self.extra_chances = extra_chances
        self.collision_rate = collision_rate

        self.create()
