import simtk.unit as u
import simtk.openmm as mm

from ..constants import kB
from .ghmc import GHMCBase


class MJHMCIntegrator(GHMCBase):

    """
    Markov Jump Hybrid Monte Carlo (HMC) integrator.

    """

    def __init__(self, temperature=298.0 * u.kelvin, steps_per_hmc=10, timestep=1 * u.femtoseconds, beta_mixing=0.01):
        """
        Create a hybrid Monte Carlo (HMC) integrator.

        Parameters
        ----------
        temperature : numpy.unit.Quantity compatible with kelvin, default: 298*simtk.unit.kelvin
           The temperature.
        nsteps : int, default: 10
           The number of velocity Verlet steps to take per HMC trial.
        timestep : numpy.unit.Quantity compatible with femtoseconds, default: 1*simtk.unit.femtoseconds
           The integration timestep.

        Warning
        -------
        Because 'nsteps' sets the number of steps taken, a call to integrator.step(1) actually takes 'nsteps' steps.

        Notes
        -----
        The velocity is drawn from a Maxwell-Boltzmann distribution, then 'nsteps' steps are taken,
        and the new configuration is either accepted or rejected.

        Additional global variables 'ntrials' and  'naccept' keep track of how many trials have been attempted and
        accepted, respectively.

        TODO
        ----
        Currently, the simulation timestep is only advanced by 'timestep' each step, rather than timestep*nsteps.  Fix this.

        Examples
        --------

        Create an HMC integrator.

        >>> timestep = 1.0 * simtk.unit.femtoseconds # fictitious timestep
        >>> temperature = 298.0 * simtk.unit.kelvin
        >>> nsteps = 10 # number of steps per call
        >>> integrator = HMCIntegrator(temperature, nsteps, timestep)

        """
        mm.CustomIntegrator.__init__(self, timestep)

        self.temperature = temperature
        self.steps_per_hmc = steps_per_hmc
        self.timestep = timestep

        self.beta_mixing = beta_mixing

        self.add_compute_steps()

    def initialize_variables(self):
        # Compute the thermal energy.
        kT = kB * self.temperature

        # Integrator initialization.

        self.addGlobalVariable("ntrials", 0)  # number of Metropolization trials

        self.addGlobalVariable("kT", kT)  # thermal energy
        self.addGlobalVariable("beta_mixing", self.beta_mixing)  # thermalization rate
        self.addPerDofVariable("sigma", 0)
        self.addGlobalVariable("ke", 0)  # kinetic energy
        self.addPerDofVariable("xold", 0)  # old positions
        self.addPerDofVariable("vold", 0)  # old velocities
        self.addGlobalVariable("Eold", 0)  # old energy
        self.addGlobalVariable("Enew", 0)  # new energy
        self.addPerDofVariable("x1", 0)  # for constraints

        # Add some MJHMC specific variables
        self.addGlobalVariable("last_move", -2)  # Use integer coding to store the type of the last move.
        self.addPerDofVariable("xLm", 0)
        self.addPerDofVariable("vLm", 0)
        self.addGlobalVariable("K", 0)  # Kinetic energy of current state
        self.addGlobalVariable("E", 0)  # Potential energy of current state
        self.addGlobalVariable("K0", 0)  # Kinetic energy of starting state
        self.addGlobalVariable("E0", 0)  # Potential energy of starting state
        self.addGlobalVariable("H0", 0)  # Total Hamiltonial of starting state
        self.addGlobalVariable("H", 0)  # Total Hamiltonial of current state
        self.addGlobalVariable("HLm", 0)  # Total Hamiltonial of starting state
        self.addGlobalVariable("ELm", 0)
        self.addGlobalVariable("KLm", 0)
        self.addGlobalVariable("gammaL", 0)
        self.addGlobalVariable("gammaLm", 0)
        self.addGlobalVariable("gammaF", 0)
        self.addGlobalVariable("gammaR", 0)
        self.addGlobalVariable("wL", 0)
        self.addGlobalVariable("wF", 0)
        self.addGlobalVariable("wR", 0)
        self.addGlobalVariable("calculated_xLm", 0)

        # Keep track of the number of times each move type is accepted.
        self.addGlobalVariable("n1", 0)
        self.addGlobalVariable("n2", 0)
        self.addGlobalVariable("n3", 0)
        self.addGlobalVariable("holding", 0)

    def reset_time(self):
        pass  # Necessary to over-ride GHMC-specific version

    def add_compute_steps(self):
        self.initialize_variables()

        # Allow Context updating here, outside of inner loop only.  (KAB: CHECK THIS for NPT LATER!)
        self.addUpdateContextState()

        ########################################################################
        # If we're starting the first iteration, pre-compute some variables and re-thermalize
        ########################################################################
        self.beginIfBlock("last_move < 0")
        self.addComputePerDof("sigma", "sqrt(kT/m)")
        self.addComputePerDof("v", "sigma*gaussian")
        self.addConstrainVelocities()
        self.endBlock()
        ########################################################################

        ########################################################################
        # Store old position and energy.
        ########################################################################
        self.addComputeSum("K0", "0.5*m*v*v")
        self.addComputeGlobal("H0", "K0 + energy")
        self.addComputePerDof("xold", "x")
        self.addComputePerDof("vold", "v")

        ########################################################################
        # Calculate Backwards step
        ########################################################################
        # If we previously accepted flip move or thermalization move
        # Go backwards for 1 round of leapfrog to determine xLm and vLm
        self.beginIfBlock("last_move != 1")  # KAB NOTE: MAY NEED TO always force-recalculation if contextUpdate has occurred
        # Reverse the timestep
        self.addComputeGlobal("dt", "dt * -1")
        self.add_hmc_iterations()
        self.addComputePerDof("xLm", "x")
        self.addComputePerDof("vLm", "v")

        # Store the potential and kinetic energies from the lower ladder position
        self.addComputeSum("KLm", "0.5*m*v*v")
        self.addComputeGlobal("ELm", "energy")
        self.nan_to_inf("HLm", "energy + KLm")

        # Revert the timestep to forward direction and restore the cached coordinates
        self.addComputeGlobal("dt", "dt * -1")
        self.addComputePerDof("x", "xold")
        self.addComputePerDof("v", "vold")
        self.addComputeGlobal("calculated_xLm", "1")  # For debug purposes

        self.endBlock()
        ########################################################################


        ########################################################################
        # Generate Lx via leapfrog steps
        ########################################################################
        self.add_hmc_iterations()
        ########################################################################


        self.addComputeSum("K", "0.5*m*v*v")
        # Check if energy is NAN.  If so, replace it with infinity.  Otherwise keep it as is.
        self.addComputeGlobal("E", "energy")
        self.nan_to_inf("H", "energy + K")


        # Eqn (9)
        self.addComputeGlobal("gammaL", "exp(-0.5 * (H - H0) / kT)")
        self.addComputeGlobal("gammaLm", "exp(-0.5 * (HLm - H0) / kT)")
        self.addComputeGlobal("gammaF", "max(0, gammaLm - gammaL)")
        self.addComputeGlobal("gammaR", "beta_mixing")

        # From wikipedia + Algorithm 1
        self.addComputeGlobal("wL", "-1 * log(uniform) / gammaL")
        self.addComputeGlobal("wF", "-1 * log(uniform) / gammaF")
        self.addComputeGlobal("wR", "-1 * log(uniform) / gammaR")
        ########################################################################


        self.addComputeGlobal("holding", "min(wF, min(wR, wL))")

        ########################################################################
        # Leapfrog move: coded as 1
        self.beginIfBlock("wL < min(wF, wR)")
        self.addComputePerDof("xLm", "xold")
        self.addComputePerDof("vLm", "vold")
        self.addComputeGlobal("last_move", "1")
        self.addComputeGlobal("n1", "n1 + 1")
        self.endBlock()
        ########################################################################

        ########################################################################
        # Flip Momenta move: coded as 2
        self.beginIfBlock("wF < min(wL, wR)")
        # x is already at at position L (x_{-1})
        self.addComputePerDof("x", "xold")
        self.addComputePerDof("v", "vold * -1")
        self.addComputeGlobal("last_move", "2")
        self.addComputeGlobal("n2", "n2 + 1")
        self.endBlock()
        ########################################################################

        ########################################################################
        # Thermalization Move: coded as 3
        self.beginIfBlock("wR < min(wL, wF)")
        # x is already at at position L (x_{-1})
        self.addComputePerDof("x", "xold")
        self.addComputePerDof("v", "sigma*gaussian")
        self.addConstrainVelocities()
        self.addComputeGlobal("last_move", "3")
        self.addComputeGlobal("n3", "n3 + 1")
        self.endBlock()
        ########################################################################
