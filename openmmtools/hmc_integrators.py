import time
import pandas as pd
import numpy as np

import simtk.unit as u

import simtk.openmm as mm
from .constants import kB

def check_groups(groups, guess=True):
    """Check that `groups` is list of tuples suitable for force group / RESPA."""
    if groups is None or len(groups) == 0:
        if guess:
            print("No force groups specified, using [(0, 1)]!")
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


class HMCBase(mm.CustomIntegrator):
    """Generalized or non-generalized hybrid Monte Carlo integrator base class.

    Notes
    -----
    This loosely follows the definition of (G)HMC given in the two below
    references.  Specifically, the velocities are corrupted,
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

        if not hasattr(self, "elapsed_steps"):
            self.elapsed_steps = 0.0
        if not hasattr(self, "elapsed_time"):
            self.elapsed_time = 0.0

        t0 = time.time()
        mm.CustomIntegrator.step(self, n_steps)
        dt = time.time() - t0

        self.elapsed_time += dt
        self.elapsed_steps += self.steps_per_hmc * n_steps

    def reset_time(self):
        """Do this before using any benchmark timing info."""
        self.step(1)
        self.elapsed_steps = 0.0
        self.elapsed_time = 0.0

    @property
    def time_per_step(self):
        return (self.elapsed_time / self.elapsed_steps)

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
        self.add_accumulate_statistics_step()

    def add_hmc_iterations(self):
        """Add self.steps_per_hmc iterations of symplectic hamiltonian dynamics."""
        print("Adding (G)HMCIntegrator steps.")
        for step in range(self.steps_per_hmc):
            self.add_hamiltonian_step()

    def add_accumulate_statistics_step(self):
        self.addComputeGlobal("naccept", "naccept + accept")
        self.addComputeGlobal("ntrials", "ntrials + 1")

    def add_hamiltonian_step(self):
        """Add a single step of hamiltonian integration.

        Notes
        -----
        This function will be overwritten in RESPA subclasses!
        """
        print("Adding step of hamiltonian dynamics.""")
        self.addComputePerDof("v", "v+0.5*dt*f/m")
        self.addComputePerDof("x", "x+dt*v")
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions()
        self.addComputePerDof("v", "v+0.5*dt*f/m+(x-x1)/dt")
        self.addConstrainVelocities()

    @property
    def kT(self):
        """The thermal energy."""
        return kB * self.temperature

    @property
    def n_accept(self):
        """The number of accepted HMC moves."""
        return self.getGlobalVariableByName("naccept")

    @property
    def n_trials(self):
        """The total number of attempted HMC moves."""
        return self.getGlobalVariableByName("ntrials")

    @property
    def acceptance_rate(self):
        """The acceptance rate: n_accept  / n_trials."""
        return self.n_accept / float(self.n_trials)

    @property
    def effective_timestep(self):
        """The acceptance rate times the timestep."""
        return self.acceptance_rate * self.timestep

    def summary(self):
        """Return a dictionary of relevant state variables for XHMC, useful for debugging.
        Append self.summary() to a list and print out as a dataframe.
        """
        d = {}
        d["arate"] = self.acceptance_rate
        d["effective_timestep"] = self.effective_timestep / u.femtoseconds
        d["effective_ns_per_day"] = self.effective_ns_per_day
        d["ns_per_day"] = self.ns_per_day
        d["r"] = self.accept_factor
        keys = ["accept", "ke", "Enew", "naccept", "ntrials", "Eold"]

        for key in keys:
            d[key] = self.getGlobalVariableByName(key)

        d["deltaE"] = d["Enew"] - d["Eold"]

        return d

    @property
    def accept_factor(self):
        return np.exp(-(self.getGlobalVariableByName("Enew") - self.getGlobalVariableByName("Eold")) / self.getGlobalVariableByName("kT"))

    @property
    def accept(self):
        """Return True if the last step taken was accepted."""
        return bool(self.getGlobalVariableByName("accept"))

class HMCIntegrator(HMCBase):
    """Hybrid Monte Carlo (HMC) integrator.

    Notes
    -----
    This loosely follows the definition of GHMC given in the two below
    references.  Specifically, the velocities are corrupted,
    several steps of hamiltonian dynamics are performed, and then
    an accept / reject move is taken.

    This class is the base class for a number of more specialized versions
    of HMC.

    References
    ----------
    C. M. Campos, J. M. Sanz-Serna, J. Comp. Phys. 281, (2015)
    J. Sohl-Dickstein, M. Mudigonda, M. DeWeese.  ICML (2014)

    """

    def __init__(self, temperature=298.0*u.kelvin, steps_per_hmc=10, timestep=1*u.femtoseconds):
        """
        Parameters
        ----------
        temperature : numpy.unit.Quantity compatible with kelvin, default: 298*simtk.unit.kelvin
           The temperature.
        steps_per_hmc : int, default: 10
           The number of velocity Verlet steps to take per round of hamiltonian dynamics
           This must be an even number!
        timestep : numpy.unit.Quantity compatible with femtoseconds, default: 1*simtk.unit.femtoseconds
           The integration timestep.  The total time taken per iteration
           will equal timestep * steps_per_hmc
        """

        mm.CustomIntegrator.__init__(self, timestep)

        self.steps_per_hmc = steps_per_hmc
        self.temperature = temperature
        self.timestep = timestep
        self.create()

    def initialize_variables(self):
        self.addGlobalVariable("naccept", 0) # number accepted
        self.addGlobalVariable("ntrials", 0) # number of Metropolization trials

        self.addGlobalVariable("kT", self.kT) # thermal energy
        self.addPerDofVariable("sigma", 0)
        self.addGlobalVariable("ke", 0) # kinetic energy
        self.addPerDofVariable("xold", 0) # old positions
        self.addGlobalVariable("Eold", 0) # old energy
        self.addGlobalVariable("Enew", 0) # new energy
        self.addGlobalVariable("accept", 0) # accept or reject
        self.addPerDofVariable("x1", 0) # for constraints

        self.addComputePerDof("sigma", "sqrt(kT/m)")
        self.addUpdateContextState()

    def add_draw_velocities_step(self):
        """Draw perturbed velocities."""
        self.addComputePerDof("v", "sigma*gaussian")
        self.addConstrainVelocities()

    def add_cache_variables_step(self):
        """Store old positions and energies."""
        self.addComputeSum("ke", "0.5*m*v*v")
        self.addComputeGlobal("Eold", "ke + energy")
        self.addComputePerDof("xold", "x")

    def add_accept_or_reject_step(self):
        print("HMC: add_accept_or_reject_step()")
        self.addComputeSum("ke", "0.5*m*v*v")
        self.addComputeGlobal("Enew", "ke + energy")
        self.addComputeGlobal("accept", "step(exp(-(Enew-Eold)/kT) - uniform)")

        self.addComputePerDof("x", "select(accept, x, xold)")  # Requires OMM Build on May 6, 2015


class GHMCIntegrator(HMCBase):
    """Generalized hybrid Monte Carlo (GHMC) integrator.

    Notes
    -----
    This loosely follows the definition of GHMC given in the two below
    references.  Specifically, the velocities are corrupted,
    several steps of hamiltonian dynamics are performed, and then
    an accept / reject move is taken.

    This class is the base class for a number of more specialized versions
    of GHMC.

    References
    ----------
    C. M. Campos, J. M. Sanz-Serna, J. Comp. Phys. 281, (2015)
    J. Sohl-Dickstein, M. Mudigonda, M. DeWeese.  ICML (2014)

    """

    def __init__(self, temperature=298.0*u.kelvin, steps_per_hmc=10, timestep=1*u.femtoseconds, collision_rate=1.0 / u.picoseconds):
        """
        Parameters
        ----------
        temperature : numpy.unit.Quantity compatible with kelvin, default: 298*simtk.unit.kelvin
           The temperature.
        steps_per_hmc : int, default: 10
           The number of velocity Verlet steps to take per round of hamiltonian dynamics
           This must be an even number!
        timestep : numpy.unit.Quantity compatible with femtoseconds, default: 1*simtk.unit.femtoseconds
           The integration timestep.  The total time taken per iteration
           will equal timestep * steps_per_hmc
        collision_rate : numpy.unit.Quantity compatible with 1 / femtoseconds, default: 1 / picoseconds
           The collision rate for the langevin velocity corruption.
        """

        mm.CustomIntegrator.__init__(self, timestep)

        self.steps_per_hmc = steps_per_hmc
        self.temperature = temperature
        self.collision_rate = collision_rate
        self.timestep = timestep
        self.create()

    def initialize_variables(self):
        self.addGlobalVariable("naccept", 0) # number accepted
        self.addGlobalVariable("ntrials", 0) # number of Metropolization trials

        self.addGlobalVariable("kT", self.kT) # thermal energy
        self.addPerDofVariable("sigma", 0)
        self.addGlobalVariable("ke", 0) # kinetic energy
        self.addPerDofVariable("xold", 0) # old positions
        self.addPerDofVariable("vold", 0) # old velocities
        self.addGlobalVariable("Eold", 0) # old energy
        self.addGlobalVariable("Enew", 0) # new energy
        self.addGlobalVariable("accept", 0) # accept or reject
        self.addPerDofVariable("x1", 0) # for constraints
        self.addGlobalVariable("b", self.b) # velocity mixing parameter

        self.addComputePerDof("sigma", "sqrt(kT/m)")
        self.addUpdateContextState()

    def add_draw_velocities_step(self):
        """Draw perturbed velocities."""
        self.addComputePerDof("v", "sqrt(b)*v + sqrt(1-b)*sigma*gaussian")
        self.addConstrainVelocities()

    def add_cache_variables_step(self):
        """Store old positions and energies."""
        self.addComputeSum("ke", "0.5*m*v*v")
        self.addComputeGlobal("Eold", "ke + energy")
        self.addComputePerDof("xold", "x")
        self.addComputePerDof("vold", "v")

    def add_accept_or_reject_step(self):
        print("GHMC: add_accept_or_reject_step()")
        self.addComputeSum("ke", "0.5*m*v*v")
        self.addComputeGlobal("Enew", "ke + energy")
        self.addComputeGlobal("accept", "step(exp(-(Enew-Eold)/kT) - uniform)")

        self.addComputePerDof("x", "select(accept, x, xold)")  # Requires OMM Build on May 6, 2015
        self.addComputePerDof("v", "select(accept, v, -1*vold)")  # Requires OMM Build on May 6, 2015

    @property
    def b(self):
        """The scaling factor for preserving versus randomizing velocities."""
        return np.exp(-self.collision_rate * self.timestep)


class XCMixin(object):
    """Extra Chance Generalized hybrid Monte Carlo (XCGHMC) integrator Mixin.

    Notes
    -----
    This integrator attempts to circumvent rejections by propagating
    additional dynamics and performing a second metropolization step.

    References
    ----------
    C. M. Campos, J. M. Sanz-Serna, J. Comp. Phys. 281, (2015)
    J. Sohl-Dickstein, M. Mudigonda, M. DeWeese.  ICML (2014)

    """

    def add_accumulate_xc_accept_statistics(self):
        """Increment acceptance counts for each step of xmc"""
        for i in range(1 + self.extra_chances):
            # Increment 1 to n_i if the current move was accepted and k == i
            self.addComputeGlobal("n%d" % i, "n%d + accept * delta(%d - k)" % (i, i))

    def add_accumulate_statistics_step(self):
        self.addComputeGlobal("nflip", "nflip + flip")
        self.addComputeGlobal("naccept", "naccept + accept")
        self.addComputeGlobal("ntrials", "ntrials + 1")

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

    @property
    def n_rounds(self):
        """The total number of rounds of XCMC."""
        return self.getGlobalVariableByName("nrounds")

    @property
    def n_force_wasted(self):
        """The number of wasted force evaluations."""
        return self.n_flip * (1 + self.extra_chances)

    @property
    def fraction_force_wasted(self):
        """The fraction of wasted force evaluations."""
        return self.n_force_wasted / (1.0 * self.n_trials)

    @property
    def acceptance_rate(self):
        """The acceptance rate, in terms of number of force evaluations.

        Notes
        -----
        Each completed "round" of XHMC dynamics involves extra_chances + 1
        force evaluations.  That is, even with zero extra chances there
        is still a single force evaluation.
        """
        return 1.0 - self.n_flip * (1.0 + self.extra_chances) / float(self.n_trials)

    @property
    def round_acceptance_rate(self):
        """The acceptance rate, in terms of number of rounds ending in failure."""
        return 1.0 - self.n_flip / float(self.n_rounds)

    def summary(self):
        """Return a dictionary of relevant state variables for XHMC, useful for debugging.
        Append self.summary() to a list and print out as a dataframe.
        """
        d = {}
        d["arate"] = self.acceptance_rate
        keys = ["accept", "s", "l", "rho", "ke", "Enew", "Unew", "mu", "mu1", "flip", "kold", "k", "naccept", "nflip", "ntrials", "nrounds", "Eold", "Uold", "uni"]
        for key in keys:
            d[key] = self.getGlobalVariableByName(key)

        d["deltaE"] = d["Enew"] - d["Eold"]

        return d

    @property
    def k(self):
        return self.getGlobalVariableByName("k")


class XCHMCIntegrator(XCMixin, HMCIntegrator):
    """Extra Chance hybrid Monte Carlo (XCHMC) integrator.

    Notes
    -----
    This integrator attempts to circumvent rejections by propagating
    additional dynamics and performing a second metropolization step.

    References
    ----------
    C. M. Campos, J. M. Sanz-Serna, J. Comp. Phys. 281, (2015)
    J. Sohl-Dickstein, M. Mudigonda, M. DeWeese.  ICML (2014)

    """
    def __init__(self, temperature=298.0*u.kelvin, steps_per_hmc=10, timestep=1*u.femtoseconds, extra_chances=2, take_debug_steps=False):
        """CURRENTLY BROKEN!!!!!
        """
        self.take_debug_steps = take_debug_steps

        mm.CustomIntegrator.__init__(self, timestep)

        self.temperature = temperature
        self.steps_per_hmc = steps_per_hmc
        self.timestep = timestep
        self.extra_chances = extra_chances

        self.create()


    def initialize_variables(self):

        self.addGlobalVariable("accept", 1.0) # accept or reject
        self.addGlobalVariable("s", 0.0)
        self.addGlobalVariable("l", 0.0)
        self.addGlobalVariable("r", 0.0) # Metropolis ratio: ratio probabilities

        self.addGlobalVariable("extra_chances", self.extra_chances)  # Maximum number of rounds of dynamics
        self.addGlobalVariable("k", 0)  # Current number of rounds of dynamics
        self.addGlobalVariable("kold", 0)  # Previous value of k stored for debugging purposes
        self.addGlobalVariable("flip", 0.0)  # Indicator variable whether this iteration was a flip

        self.addGlobalVariable("rho", 0.0)  # temporary variables for acceptance criterion
        self.addGlobalVariable("mu", 0.0)  #
        self.addGlobalVariable("mu1", 0.0)  # XCHMC Fig. 3 O1

        for i in range(1 + self.extra_chances):
            self.addGlobalVariable("n%d" % i, 0.0)  # Number of times accepted when k = i

        self.addGlobalVariable("Uold", 0.0)
        self.addGlobalVariable("Unew", 0.0)
        self.addGlobalVariable("uni", 0.0)  # Uniform random variable generated once per round of XHMC

        self.addGlobalVariable("nflip", 0) # number of momentum flips (e.g. complete rejections)
        self.addGlobalVariable("nrounds", 0) # number of "rounds" of XHMC, e.g. the number of times k = 0

        # Below this point is possible base class material

        self.addGlobalVariable("naccept", 0) # number accepted
        self.addGlobalVariable("ntrials", 0) # number of Metropolization trials

        self.addGlobalVariable("kT", self.kT) # thermal energy
        self.addPerDofVariable("sigma", 0)
        self.addGlobalVariable("ke", 0) # kinetic energy
        self.addPerDofVariable("xold", 0) # old positions
        self.addGlobalVariable("Eold", 0) # old energy
        self.addGlobalVariable("Enew", 0) # new energy

        self.addPerDofVariable("x1", 0) # for constraints

        self.addComputePerDof("sigma", "sqrt(kT/m)")
        self.addUpdateContextState()

    def add_draw_velocities_step(self):
        """Draw perturbed velocities."""
        self.addComputeGlobal("s", "step(-k)")  # True only on first step of XHMC round
        self.addComputeGlobal("nrounds", "nrounds + s")
        self.addComputeGlobal("l", "step(k - extra_chances)")  # True only only last step of XHMC round

        self.addUpdateContextState()
        self.addConstrainPositions()

        self.addComputePerDof("v", "s * sigma * gaussian + (1 - s) * v")
        self.addConstrainVelocities()

    def add_cache_variables_step(self):
        """Store old positions and energies."""

        self.addComputeSum("ke", "0.5*m*v*v")
        self.addComputeGlobal("Eold", "s * (ke + energy) + (1 - s) * Eold")
        self.addComputeGlobal("Uold", "energy")  # Not strictly necessary, used for debugging

        self.addComputePerDof("xold", "select(s, x, xold)")  # Requires OMM Build on May 6, 2015


        self.addComputeGlobal("mu1", "mu1 * (1 - s)")  # XCHMC Fig. 3 O1
        self.addComputeGlobal("uni", "(1 - s) * uni + uniform * s")  # XCHMC paper version, only draw uniform once

    def add_accept_or_reject_step(self):
        print("XCHMC: add_accept_or_reject_step()")
        self.addComputeSum("ke", "0.5*m*v*v")
        self.addComputeGlobal("Enew", "ke + energy")

        self.addComputeGlobal("Unew", "energy")
        self.addComputeGlobal("r", "exp(-(Enew - Eold) / kT)")
        self.addComputeGlobal("mu", "min(1, r)")  # XCHMC paper version
        self.addComputeGlobal("mu1", "max(mu1, mu)")


        self.addComputeGlobal("accept", "step(mu1 - uni)")

        self.add_accumulate_xc_accept_statistics()

        self.addComputeGlobal("flip", "(1 - accept) * l")  # Flip is True ONLY on rejection at last cycle

        self.addComputePerDof("x", "select(flip, xold, x)")  # Requires OMM Build on May 6, 2015


        self.addComputeGlobal("kold", "k")  # Store the previous value of k for debugging purposes
        self.addComputeGlobal("k", "(k + 1) * (1 - flip) * (1 - accept)")  # Increment by one ONLY if not flipping momenta or accepting, otherwise set to zero

    def step(self, n_steps):
        if self.take_debug_steps:
            super(XCHMCIntegrator, self).step(n_steps)
        else:
            for i in range(n_steps):
                while True:
                    super(XCHMCIntegrator, self).step(1)
                    if self.k == 0:
                        break


class XCGHMCIntegrator(XCMixin, GHMCIntegrator):
    """Extra Chance Generalized hybrid Monte Carlo (XCGHMC) integrator.

    Notes
    -----
    This integrator attempts to circumvent rejections by propagating
    additional dynamics and performing a second metropolization step.

    References
    ----------
    C. M. Campos, J. M. Sanz-Serna, J. Comp. Phys. 281, (2015)
    J. Sohl-Dickstein, M. Mudigonda, M. DeWeese.  ICML (2014)

    """
    def __init__(self, temperature=298.0*u.kelvin, steps_per_hmc=10, timestep=1*u.femtoseconds, collision_rate=1.0 / u.picoseconds, extra_chances=2, take_debug_steps=False):
        """CURRENTLY BROKEN!!!!!
        """
        self.take_debug_steps = take_debug_steps

        mm.CustomIntegrator.__init__(self, timestep)

        self.temperature = temperature
        self.steps_per_hmc = steps_per_hmc
        self.collision_rate = collision_rate
        self.timestep = timestep
        self.extra_chances = extra_chances

        self.create()


    def initialize_variables(self):

        self.addGlobalVariable("accept", 1.0) # accept or reject
        self.addGlobalVariable("s", 0.0)
        self.addGlobalVariable("l", 0.0)
        self.addGlobalVariable("r", 0.0) # Metropolis ratio: ratio probabilities

        self.addGlobalVariable("extra_chances", self.extra_chances)  # Maximum number of rounds of dynamics
        self.addGlobalVariable("k", 0)  # Current number of rounds of dynamics
        self.addGlobalVariable("kold", 0)  # Previous value of k stored for debugging purposes
        self.addGlobalVariable("flip", 0.0)  # Indicator variable whether this iteration was a flip

        self.addGlobalVariable("rho", 0.0)  # temporary variables for acceptance criterion
        self.addGlobalVariable("mu", 0.0)  #
        self.addGlobalVariable("mu1", 0.0)  # XCHMC Fig. 3 O1

        for i in range(1 + self.extra_chances):
            self.addGlobalVariable("n%d" % i, 0.0)  # Number of times accepted when k = i

        self.addGlobalVariable("Uold", 0.0)
        self.addGlobalVariable("Unew", 0.0)
        self.addGlobalVariable("uni", 0.0)  # Uniform random variable generated once per round of XHMC

        self.addGlobalVariable("nflip", 0) # number of momentum flips (e.g. complete rejections)
        self.addGlobalVariable("nrounds", 0) # number of "rounds" of XHMC, e.g. the number of times k = 0

        # Below this point is possible base class material

        self.addGlobalVariable("naccept", 0) # number accepted
        self.addGlobalVariable("ntrials", 0) # number of Metropolization trials

        self.addGlobalVariable("kT", self.kT) # thermal energy
        self.addPerDofVariable("sigma", 0)
        self.addGlobalVariable("ke", 0) # kinetic energy
        self.addPerDofVariable("xold", 0) # old positions
        self.addPerDofVariable("vold", 0) # old velocities
        self.addGlobalVariable("Eold", 0) # old energy
        self.addGlobalVariable("Enew", 0) # new energy

        self.addPerDofVariable("x1", 0) # for constraints
        self.addGlobalVariable("b", self.b) # velocity mixing parameter

        self.addComputePerDof("sigma", "sqrt(kT/m)")
        self.addUpdateContextState()

    def add_draw_velocities_step(self):
        """Draw perturbed velocities."""
        self.addComputeGlobal("s", "step(-k)")  # True only on first step of XHMC round
        self.addComputeGlobal("nrounds", "nrounds + s")
        self.addComputeGlobal("l", "step(k - extra_chances)")  # True only only last step of XHMC round

        self.addUpdateContextState()
        self.addConstrainPositions()

        self.addComputePerDof("v", "s * (sqrt(b) * v + sqrt(1 - b) * sigma * gaussian) + (1 - s) * v")
        self.addConstrainVelocities()

    def add_cache_variables_step(self):
        """Store old positions and energies."""

        self.addComputeSum("ke", "0.5*m*v*v")
        self.addComputeGlobal("Eold", "s * (ke + energy) + (1 - s) * Eold")
        self.addComputeGlobal("Uold", "energy")  # Not strictly necessary, used for debugging

        self.addComputePerDof("xold", "select(s, x, xold)")  # Requires OMM Build on May 6, 2015
        self.addComputePerDof("vold", "select(s, v, vold)")  # Requires OMM Build on May 6, 2015

        self.addComputeGlobal("mu1", "mu1 * (1 - s)")  # XCHMC Fig. 3 O1
        self.addComputeGlobal("uni", "(1 - s) * uni + uniform * s")  # XCHMC paper version, only draw uniform once

    def add_accept_or_reject_step(self):
        print("XCGHMC: add_accept_or_reject_step()")
        self.addComputeSum("ke", "0.5*m*v*v")
        self.addComputeGlobal("Enew", "ke + energy")

        self.addComputeGlobal("Unew", "energy")
        self.addComputeGlobal("r", "exp(-(Enew - Eold) / kT)")
        self.addComputeGlobal("mu", "min(1, r)")  # XCHMC paper version
        self.addComputeGlobal("mu1", "max(mu1, mu)")


        self.addComputeGlobal("accept", "step(mu1 - uni)")
        self.add_accumulate_xc_accept_statistics()

        self.addComputeGlobal("flip", "(1 - accept) * l")  # Flip is True ONLY on rejection at last cycle

        self.addComputePerDof("x", "select(flip, xold, x)")  # Requires OMM Build on May 6, 2015
        self.addComputePerDof("v", "select(flip, -vold, v)")  # Requires OMM Build on May 6, 2015

        self.addComputeGlobal("kold", "k")  # Store the previous value of k for debugging purposes
        self.addComputeGlobal("k", "(k + 1) * (1 - flip) * (1 - accept)")  # Increment by one ONLY if not flipping momenta or accepting, otherwise set to zero

    def step(self, n_steps):
        if self.take_debug_steps:
            super(XCGHMCIntegrator, self).step(n_steps)
        else:
            for i in range(n_steps):
                while True:
                    super(XCGHMCIntegrator, self).step(1)
                    if self.k == 0:
                        break

class RESPAMixIn(object):
    def add_hamiltonian_step(self):
        """Add a single step of hamiltonian integration.

        Notes
        -----
        This function will be overwritten in RESPA subclasses!
        """
        print("Adding step of RESPA hamiltonian dynamics.""")
        self._create_substeps(1, self.groups)
        self.addConstrainVelocities()

    def _create_substeps(self, parentSubsteps, groups):

        group, substeps = groups[0]

        str_group, str_sub = str(group), str(substeps)

        stepsPerParentStep = substeps / parentSubsteps

        if stepsPerParentStep < 1 or stepsPerParentStep != int(stepsPerParentStep):
            raise ValueError("The number for substeps for each group must be a multiple of the number for the previous group")

        stepsPerParentStep = int(stepsPerParentStep) # needed for Python 3.x

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


class HMCRESPAIntegrator(RESPAMixIn, HMCIntegrator):
    """Hamiltonian Monte Carlo (HMC) with a rRESPA multiple
    time step integration algorithm.  Combines HMCIntegrator with
    MTSIntegrator.

    This integrator allows different forces to be evaluated at different frequencies,
    for example to evaluate the expensive, slowly changing forces less frequently than
    the inexpensive, quickly changing forces.

    To use it, you must first divide your forces into two or more groups (by calling
    setForceGroup() on them) that should be evaluated at different frequencies.  When
    you create the integrator, you provide a tuple for each group specifying the index
    of the force group and the frequency (as a fraction of the outermost time step) at
    which to evaluate it.  For example:

    >>> integrator = MTSIntegrator(4*simtk.unit.femtoseconds, [(0,1), (1,2), (2,8)])

    This specifies that the outermost time step is 4 fs, so each step of the integrator
    will advance time by that much.  It also says that force group 0 should be evaluated
    once per time step, force group 1 should be evaluated twice per time step (every 2 fs),
    and force group 2 should be evaluated eight times per time step (every 0.5 fs).

    Notes
    -----
    This loosely follows the definition of GHMC given in the two below
    references.  Specifically, the velocities are corrupted,
    several steps of hamiltonian dynamics are performed, and then
    an accept / reject move is taken.

    The RESPA multiple timestep splitting should closely follow the code
    in MTSIntegrator.

    References
    ----------
    C. M. Campos, J. M. Sanz-Serna, J. Comp. Phys. 281, (2015)
    J. Sohl-Dickstein, M. Mudigonda, M. DeWeese.  ICML (2014)
    Tuckerman et al., J. Chem. Phys. 97(3) pp. 1990-2001 (1992)
    """

    def __init__(self, temperature=298.0*u.kelvin, steps_per_hmc=10, timestep=1*u.femtoseconds, groups=None):
        """Create a generalized hamiltonian Monte Carlo (HMC) integrator with linearly ramped non-uniform timesteps.

        Parameters
        ----------
        temperature : numpy.unit.Quantity compatible with kelvin, default: 298*simtk.unit.kelvin
            The temperature.
        steps_per_hmc : int, default: 10
            The number of velocity Verlet steps to take per round of hamiltonian dynamics
            This must be an even number!
        timestep : numpy.unit.Quantity compatible with femtoseconds, default: 1*simtk.unit.femtoseconds
            The integration timestep.  The total time taken per iteration
            will equal timestep * steps_per_hmc
        groups : list of tuples, optional, default=(0,1)
            A list of tuples defining the force groups.  The first element
            of each tuple is the force group index, and the second element
            is the number of times that force group should be evaluated in
            one time step.
        """

        mm.CustomIntegrator.__init__(self, timestep)

        self.groups = check_groups(groups)
        self.steps_per_hmc = steps_per_hmc

        self.timestep = timestep
        self.temperature = temperature

        self.create()

class GHMCRESPAIntegrator(RESPAMixIn, GHMCIntegrator):
    """Generalized Hamiltonian Monte Carlo (GHMC) with a rRESPA multiple
    time step integration algorithm.  Combines GHMCIntegrator with
    MTSIntegrator.

    This integrator allows different forces to be evaluated at different frequencies,
    for example to evaluate the expensive, slowly changing forces less frequently than
    the inexpensive, quickly changing forces.

    To use it, you must first divide your forces into two or more groups (by calling
    setForceGroup() on them) that should be evaluated at different frequencies.  When
    you create the integrator, you provide a tuple for each group specifying the index
    of the force group and the frequency (as a fraction of the outermost time step) at
    which to evaluate it.  For example:

    >>> integrator = MTSIntegrator(4*simtk.unit.femtoseconds, [(0,1), (1,2), (2,8)])

    This specifies that the outermost time step is 4 fs, so each step of the integrator
    will advance time by that much.  It also says that force group 0 should be evaluated
    once per time step, force group 1 should be evaluated twice per time step (every 2 fs),
    and force group 2 should be evaluated eight times per time step (every 0.5 fs).

    Notes
    -----
    This loosely follows the definition of GHMC given in the two below
    references.  Specifically, the velocities are corrupted,
    several steps of hamiltonian dynamics are performed, and then
    an accept / reject move is taken.

    The RESPA multiple timestep splitting should closely follow the code
    in MTSIntegrator.

    References
    ----------
    C. M. Campos, J. M. Sanz-Serna, J. Comp. Phys. 281, (2015)
    J. Sohl-Dickstein, M. Mudigonda, M. DeWeese.  ICML (2014)
    Tuckerman et al., J. Chem. Phys. 97(3) pp. 1990-2001 (1992)
    """

    def __init__(self, temperature=298.0*u.kelvin, steps_per_hmc=10, timestep=1*u.femtoseconds, collision_rate=1.0/u.picoseconds, groups=None):
        """Create a generalized hamiltonian Monte Carlo (HMC) integrator with linearly ramped non-uniform timesteps.

        Parameters
        ----------
        temperature : numpy.unit.Quantity compatible with kelvin, default: 298*simtk.unit.kelvin
            The temperature.
        steps_per_hmc : int, default: 10
            The number of velocity Verlet steps to take per round of hamiltonian dynamics
            This must be an even number!
        timestep : numpy.unit.Quantity compatible with femtoseconds, default: 1*simtk.unit.femtoseconds
            The integration timestep.  The total time taken per iteration
            will equal timestep * steps_per_hmc
        collision_rate : numpy.unit.Quantity compatible with 1 / femtoseconds, default: 1 / picoseconds
            The collision rate for the langevin velocity corruption.
        groups : list of tuples, optional, default=(0,1)
            A list of tuples defining the force groups.  The first element
            of each tuple is the force group index, and the second element
            is the number of times that force group should be evaluated in
            one time step.
        """

        mm.CustomIntegrator.__init__(self, timestep)

        self.groups = check_groups(groups)
        self.steps_per_hmc = steps_per_hmc

        self.collision_rate = collision_rate
        self.timestep = timestep
        self.temperature = temperature

        self.create()


class XCGHMCRESPAIntegrator(RESPAMixIn, XCGHMCIntegrator):
    """Extra Chance Generalized hybrid Monte Carlo RESPA integrator.
    """
    def __init__(self, temperature=298.0*u.kelvin, steps_per_hmc=10, timestep=1*u.femtoseconds, collision_rate=1.0 / u.picoseconds, extra_chances=2, groups=None, take_debug_steps=False):
        """
        """
        mm.CustomIntegrator.__init__(self, timestep)

        self.groups = check_groups(groups)

        self.take_debug_steps = take_debug_steps
        self.temperature = temperature
        self.steps_per_hmc = steps_per_hmc
        self.collision_rate = collision_rate
        self.timestep = timestep
        self.extra_chances = extra_chances

        self.create()


class XCHMCRESPAIntegrator(RESPAMixIn, XCHMCIntegrator):
    """Extra Chance Generalized hybrid Monte Carlo RESPA integrator.
    """
    def __init__(self, temperature=298.0*u.kelvin, steps_per_hmc=10, timestep=1*u.femtoseconds, extra_chances=2, groups=None, take_debug_steps=False):
        """
        """
        mm.CustomIntegrator.__init__(self, timestep)

        self.groups = check_groups(groups)

        self.take_debug_steps = take_debug_steps
        self.temperature = temperature
        self.steps_per_hmc = steps_per_hmc
        self.timestep = timestep
        self.extra_chances = extra_chances

        self.create()

class UnrolledXCMixin(object):
    """Extra Chance Generalized hybrid Monte Carlo (XCGHMC) integrator Mixin.

    Notes
    -----
    This integrator attempts to circumvent rejections by propagating
    additional dynamics and performing a second metropolization step.

    References
    ----------
    C. M. Campos, J. M. Sanz-Serna, J. Comp. Phys. 281, (2015)
    J. Sohl-Dickstein, M. Mudigonda, M. DeWeese.  ICML (2014)

    """


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

    @property
    def steps_taken(self):
        return self.getGlobalVariableByName("steps_taken")

    @property
    def steps_accepted(self):
        return self.getGlobalVariableByName("steps_accepted")

    @property
    def acceptance_rate(self):
        """The acceptance rate, in terms of number of force evaluations.
        """
        return self.steps_accepted / self.steps_taken


    def create(self):
        self.initialize_variables()
        self.add_draw_velocities_step()
        self.add_cache_variables_step()
        for i in range(1 + self.extra_chances):
            self.add_hmc_iterations(i)
            self.add_accept_or_reject_step(i)

        self.addComputeGlobal("flip", "(1 - done)")

        self.addComputePerDof("x", "select(flip, xold, xfinal)")  # Requires OMM Build on May 6, 2015
        
        self.addComputeGlobal("nflip", "nflip + flip")
        self.addComputeGlobal("naccept", "naccept + accept")
        self.addComputeGlobal("ntrials", "ntrials + 1")


class UnrolledXCHMCIntegrator(UnrolledXCMixin, HMCIntegrator):
    """Extra Chance hybrid Monte Carlo (XCHMC) integrator.

    Notes
    -----
    This integrator attempts to circumvent rejections by propagating
    additional dynamics and performing a second metropolization step.

    References
    ----------
    C. M. Campos, J. M. Sanz-Serna, J. Comp. Phys. 281, (2015)
    J. Sohl-Dickstein, M. Mudigonda, M. DeWeese.  ICML (2014)

    """
    def __init__(self, temperature=298.0*u.kelvin, steps_per_hmc=10, timestep=1*u.femtoseconds, extra_chances=2, steps_per_extra_hmc=1, take_debug_steps=False):
        """CURRENTLY BROKEN!!!!!
        """
        self.take_debug_steps = take_debug_steps

        mm.CustomIntegrator.__init__(self, timestep)

        self.temperature = temperature
        self.steps_per_hmc = steps_per_hmc
        self.steps_per_extra_hmc = steps_per_extra_hmc
        self.timestep = timestep
        self.extra_chances = extra_chances

        self.create()


    def initialize_variables(self):

        self.addGlobalVariable("accept", 1.0) # accept or reject
        self.addGlobalVariable("s", 0.0)
        self.addGlobalVariable("l", 0.0)
        self.addGlobalVariable("r", 0.0) # Metropolis ratio: ratio probabilities

        self.addGlobalVariable("extra_chances", self.extra_chances)  # Maximum number of rounds of dynamics
        self.addGlobalVariable("k", 0)  # Current number of rounds of dynamics
        self.addGlobalVariable("kold", 0)  # Previous value of k stored for debugging purposes
        self.addGlobalVariable("flip", 0.0)  # Indicator variable whether this iteration was a flip

        self.addGlobalVariable("rho", 0.0)  # temporary variables for acceptance criterion
        self.addGlobalVariable("mu", 0.0)  #
        self.addGlobalVariable("mu1", 0.0)  # XCHMC Fig. 3 O1

        for i in range(1 + self.extra_chances):
            self.addGlobalVariable("n%d" % i, 0.0)  # Number of times accepted when k = i

        self.addGlobalVariable("nflip", 0) # number of momentum flips (e.g. complete rejections)
        self.addGlobalVariable("nrounds", 0) # number of "rounds" of XHMC, e.g. the number of times k = 0

        # Below this point is possible base class material

        self.addGlobalVariable("naccept", 0) # number accepted
        self.addGlobalVariable("ntrials", 0) # number of Metropolization trials

        self.addGlobalVariable("kT", self.kT) # thermal energy
        self.addPerDofVariable("sigma", 0)
        self.addGlobalVariable("ke", 0) # kinetic energy
        self.addPerDofVariable("xold", 0) # old positions
        self.addGlobalVariable("Eold", 0) # old energy
        self.addGlobalVariable("Enew", 0) # new energy
        
        self.addGlobalVariable("done", 0) # Becomes true once we find a good value of (x, p)

        self.addPerDofVariable("x1", 0) # for constraints
        
        self.addPerDofVariable("xfinal", 0)
        #self.addPerDofVariable("vfinal", 0)
        
        self.addGlobalVariable("steps_accepted", 0)  # Number of productive hamiltonian steps
        self.addGlobalVariable("steps_taken", 0)  # Number of total hamiltonian steps
        

        self.addComputePerDof("sigma", "sqrt(kT/m)")
        self.addUpdateContextState()

    def add_draw_velocities_step(self):
        """Draw perturbed velocities."""

        self.addUpdateContextState()
        self.addConstrainPositions()

        self.addComputeGlobal("done", "0.0")
        self.addComputePerDof("v", "sigma*gaussian")
        self.addConstrainVelocities()

    def add_cache_variables_step(self):
        """Store old positions and energies."""

        self.addComputeSum("ke", "0.5*m*v*v")
        self.addComputeGlobal("Eold", "ke + energy")
        self.addComputePerDof("xold", "x")

        self.addComputeGlobal("mu1", "0.0")  # XCHMC Fig. 3 O1

    def add_hmc_iterations(self, i):
        """Add self.steps_per_hmc or self.steps_per_extra_hmc iterations of symplectic hamiltonian dynamics."""
        print("Adding XCHMCIntegrator steps.")
        
        steps = self.steps_per_hmc
        
        if i > 0:
            steps = self.steps_per_extra_hmc
        
        for step in range(steps):
            self.add_hamiltonian_step()
                

    def add_accept_or_reject_step(self, i):
        print("XCHMC: add_accept_or_reject_step()")
        self.addComputeSum("ke", "0.5*m*v*v")
        self.addComputeGlobal("Enew", "ke + energy")

        self.addComputeGlobal("r", "exp(-(Enew - Eold) / kT)")
        self.addComputeGlobal("mu", "min(1, r)")  # XCHMC paper version
        self.addComputeGlobal("mu1", "max(mu1, mu)")
        self.addComputeGlobal("accept", "step(mu1 - uniform) * (1 - done)")
        
        if i == 0:
            steps = self.steps_per_hmc
        else:
            steps = self.steps_per_extra_hmc
        
        self.addComputeGlobal("n%d" % i, "n%d + accept" % (i))
        
        self.addComputeGlobal("steps_accepted", "steps_accepted + accept * %d" % (steps))
        self.addComputeGlobal("steps_taken", "steps_taken + %d" % (steps))

        self.addComputePerDof("xfinal", "select(accept, x, xfinal)")
        #self.addComputePerDof("vfinal", "select(accept, v, vfinal)")

        self.addComputeGlobal("done", "max(done, accept)")


class UnrolledXCHMCRESPAIntegrator(RESPAMixIn, UnrolledXCHMCIntegrator):
    """Extra Chance Generalized hybrid Monte Carlo RESPA integrator.
    """
    def __init__(self, temperature=298.0*u.kelvin, steps_per_hmc=10, timestep=1*u.femtoseconds, extra_chances=2, steps_per_extra_hmc=1, groups=None, take_debug_steps=False):
        """
        """
        mm.CustomIntegrator.__init__(self, timestep)

        self.groups = check_groups(groups)

        self.take_debug_steps = take_debug_steps
        self.temperature = temperature
        self.steps_per_hmc = steps_per_hmc
        self.steps_per_extra_hmc = steps_per_extra_hmc
        self.timestep = timestep
        self.extra_chances = extra_chances

        self.create()
