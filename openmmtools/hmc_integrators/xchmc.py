"""Hamiltonian Monte Carlo Integrators

Notes
-----
The code in this module is considered EXPERIMENTAL until further notice.
"""
import logging
import pandas as pd

import simtk.unit as u

from .utils import warn_experimental
from .ghmc import GHMCIntegrator
from .ghmc_respa import RESPAMixIn, check_groups

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        warn_experimental()
        self.extra_chances = extra_chances
        self.steps_per_extra_hmc = steps_per_extra_hmc
        super(XCGHMCIntegrator, self).__init__(temperature=temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, collision_rate=collision_rate)

    @property
    def all_counts(self):
        """Return a pandas series of moves accepted after step 0, 1, . extra_chances - 1, and flip"""
        d = {}
        for i in range(1 + self.extra_chances):
            d[i] = self.getGlobalVariableByName("n%d" % i)

        d["flip"] = self.n_flip

        return pd.Series(d)

    @property
    def all_probs(self):
        """Return a pandas series of probabilities of moves accepted after step 0, 1,  extra_chances - 1, and flip"""
        d = self.all_counts
        return d / d.sum()

    @property
    def n_flip(self):
        """The total number of momentum flips."""
        return self.getGlobalVariableByName("nflip")

    def add_compute_steps(self):
        """The key flow control logic for XCHMC."""
        self.initialize_variables()
        self.add_draw_velocities_step()
        self.add_cache_variables_step()
        for i in range(1 + self.extra_chances):
            self.beginIfBlock("uni > mu1")
            self.add_hmc_iterations()
            self.addComputeSum("ke", "0.5*m*v*v")
            self.nan_to_inf("Enew", "ke + energy")
            self.addComputeGlobal("r", "exp(-(Enew - Eold) / kT)")
            self.addComputeGlobal("mu", "min(1, r)")  # XCGHMC paper version
            self.addComputeGlobal("mu1", "max(mu1, mu)")
            self.addComputeGlobal("terminal_chance", "%d" % i)
            self.addComputeGlobal("n%d" % i, "n%d + step(mu1 - uni)" % (i))
            self.addComputeGlobal("steps_taken", "steps_taken + 1")
            self.endBlock()

        self.beginIfBlock("uni > mu1")
        self.addComputePerDof("x", "xold")
        if self.is_GHMC:
            self.addComputePerDof("v", "-1 * vold")
        else:
            self.addComputePerDof("v", "vold")
        self.addComputeGlobal("nflip", "nflip + 1")
        self.endBlock()

        self.beginIfBlock("uni <= mu1")
        self.addComputeGlobal("steps_accepted", "steps_accepted + terminal_chance + 1")
        self.endBlock()

    def initialize_variables(self):

        self.addGlobalVariable("accept", 1.0)  # accept or reject
        self.addGlobalVariable("r", 0.0)  # Metropolis ratio: ratio probabilities

        self.addGlobalVariable("extra_chances", self.extra_chances)  # Maximum number of rounds of dynamics

        self.addGlobalVariable("mu", 0.0)  #
        self.addGlobalVariable("mu1", 0.0)  # XCGHMC Fig. 3 O1

        for i in range(1 + self.extra_chances):
            self.addGlobalVariable("n%d" % i, 0.0)  # Number of times accepted when k = i

        self.addGlobalVariable("nflip", 0)  # number of momentum flips (e.g. complete rejections)
        self.addGlobalVariable("nrounds", 0)  # number of "rounds" of XHMC, e.g. the number of times k = 0

        # Below this point is possible base class material

        self.addPerDofVariable("sigma", 0)
        self.addGlobalVariable("ke", 0)  # kinetic energy
        self.addPerDofVariable("xold", 0)  # old positions
        self.addPerDofVariable("vold", 0)  # old velocities
        self.addGlobalVariable("Eold", 0)  # old energy
        self.addGlobalVariable("Enew", 0)  # new energy
        self.addGlobalVariable("terminal_chance", 0)

        self.addPerDofVariable("x1", 0)  # for constraints

        self.addGlobalVariable("steps_accepted", 0)  # Number of productive hamiltonian steps
        self.addGlobalVariable("steps_taken", 0)  # Number of total hamiltonian steps

        self.addGlobalVariable("uni", 0)  # Uniform random number draw in XCHMC
        self.addComputeTemperatureDependentConstants({"sigma": "sqrt(kT/m)"})

        if self.is_GHMC:
            self.addGlobalVariable("b", self.b)  # velocity mixing parameter

        self.addUpdateContextState()

    def add_cache_variables_step(self):
        """Store old positions and energies."""

        self.addComputeSum("ke", "0.5*m*v*v")
        self.addComputeGlobal("Eold", "ke + energy")
        self.addComputePerDof("xold", "x")
        self.addComputePerDof("vold", "v")

        self.addComputeGlobal("mu1", "0.0")  # XCGHMC Fig. 3 O1
        self.addComputeGlobal("uni", "uniform")  # XCGHMC Fig. 3 O1


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
        warn_experimental()
        self.groups = check_groups(groups)
        self.steps_per_extra_hmc = steps_per_extra_hmc
        self.extra_chances = extra_chances

        super(XCGHMCRESPAIntegrator, self).__init__(temperature=temperature, steps_per_hmc=steps_per_hmc, timestep=timestep,
                                                    extra_chances=extra_chances, steps_per_extra_hmc=steps_per_extra_hmc, collision_rate=collision_rate)
