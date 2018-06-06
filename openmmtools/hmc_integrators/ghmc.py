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

from ..integrators import ThermostatedIntegrator
from .utils import warn_experimental

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GHMCBase(ThermostatedIntegrator):
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

    def nan_to_inf(self, outkey, inkey):
        """Add a compute step that converts nan to positive infinity in `inkey` and ouputs in `outkey`."""
        self.addComputeGlobal(outkey, "select(step(exp(%s)), %s, 1/0)" % (inkey, inkey))

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

    def add_compute_steps(self):
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

    def summary(self):
        """Return a dictionary of relevant state variables for XHMC, useful for debugging.
        Append self.summary() to a list and print out as a dataframe.
        """
        d = {}
        d["arate"] = self.acceptance_rate
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
    def is_GHMC(self):
        return self.collision_rate is not None


class GHMCIntegrator(GHMCBase):
    """Generalized hybrid Monte Carlo (GHMC) integrator.

    Parameters
    ----------
    temperature : simtk.unit.Quantity compatible with kelvin, default: 298*simtk.unit.kelvin
       The temperature.
    steps_per_hmc : int, default: 10
       The number of velocity Verlet steps to take per round of hamiltonian dynamics
    timestep : simtk.unit.Quantity compatible with femtoseconds, default: 1*simtk.unit.femtoseconds
       The integration timestep.  The total time taken per iteration
       will equal timestep * steps_per_hmc
    collision_rate : simtk.unit.Quantity compatible with 1 / femtoseconds, default: None
       The collision rate for the velocity corruption (GHMC).  If None,
       velocities information will be discarded after each round (HMC).

    Notes
    -----
    This loosely follows the definition of GHMC given in the two below
    references.  Specifically, the velocities are corrupted,
    several steps of Hamiltonian dynamics are performed, and then
    an accept / reject move is taken.  If collision_rate is set to None, however,
    we will do non-generalized HMC, where the velocity information is discarded
    at each iteration.

    This class is the base class for a number of more specialized versions
    of GHMC.

    References
    ----------
    C. M. Campos, J. M. Sanz-Serna, J. Comp. Phys. 281, (2015)
    J. Sohl-Dickstein, M. Mudigonda, M. DeWeese.  ICML (2014)
    """

    def __init__(self, temperature=298.0 * u.kelvin, steps_per_hmc=10, timestep=1 * u.femtoseconds, collision_rate=None):
        warn_experimental()
        super(GHMCIntegrator, self).__init__(temperature, timestep)

        self.steps_per_hmc = steps_per_hmc
        self.collision_rate = collision_rate
        self.timestep = timestep
        self.add_compute_steps()

    def initialize_variables(self):

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
            val = np.exp(-1.0 * self.collision_rate * self.timestep)
            self.addGlobalVariable("b", val)

        self.addComputeTemperatureDependentConstants({"sigma": "sqrt(kT/m)"})

        self.addUpdateContextState()

    def add_draw_velocities_step(self):
        """Draw perturbed velocities using either partial or complete momentum thermalization."""

        self.addUpdateContextState()
        self.addConstrainPositions()

        if self.is_GHMC:
            self.addComputePerDof("v", "sqrt(b)*v + sqrt(1-b)*sigma*gaussian")
        else:
            self.addComputePerDof("v", "sigma*gaussian")

        self.addConstrainVelocities()

    def add_cache_variables_step(self):
        """Store old positions and energies."""
        self.addComputeSum("ke", "0.5*m*v*v")
        self.nan_to_inf("Eold", "ke + energy")
        self.addComputePerDof("xold", "x")
        self.addComputePerDof("vold", "v")

    def add_accept_or_reject_step(self):
        logger.debug("GHMC: add_accept_or_reject_step()")
        self.addComputeSum("ke", "0.5*m*v*v")
        self.nan_to_inf("Enew", "energy + ke")
        self.addComputeGlobal("accept", "step(exp(-(Enew-Eold)/kT) - uniform)")

        self.addComputePerDof("x", "select(accept, x, xold)")

        if self.is_GHMC:
            self.addComputePerDof("v", "select(accept, v, -1*vold)")

        self.addComputeGlobal("steps_accepted", "steps_accepted + accept * %d" % (self.steps_per_hmc))
        self.addComputeGlobal("steps_taken", "steps_taken + %d" % (self.steps_per_hmc))
