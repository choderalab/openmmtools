#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Custom integrators for molecular simulation.

DESCRIPTION

This module provides various custom integrators for OpenMM.

EXAMPLES

COPYRIGHT

@author John D. Chodera <john.chodera@choderalab.org>

All code in this repository is released under the GNU General Public License.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <http://www.gnu.org/licenses/>.

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import numpy

import simtk.unit

import simtk.unit as units
import simtk.openmm as mm

#=============================================================================================
# CONSTANTS
#=============================================================================================

kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA

#=============================================================================================
# INTEGRATORS
#=============================================================================================

def DummyIntegrator():
    """
    Construct a dummy integrator that does nothing except update call the force updates.

    Returns
    -------
    integrator : simtk.openmm.CustomIntegrator
        A dummy integrator.

    Examples
    --------

    Create a dummy integrator.

    >>> integrator = DummyIntegrator()

    """

    timestep = 0.0 * units.femtoseconds
    integrator = mm.CustomIntegrator(timestep)
    integrator.addUpdateContextState()
    integrator.addConstrainPositions()
    integrator.addConstrainVelocities()

    return integrator

def GradientDescentMinimizationIntegrator(initial_step_size=0.01*units.angstroms):
    """
    Construct a simple gradient descent minimization integrator.

    Parameters
    ----------
    initial_step_size : numpy.unit.Quantity compatible with nanometers, default: 0.01*simtk.unit.angstroms
        The norm of the initial step size guess.

    Returns
    -------
    integrator : simtk.openmm.CustomIntegrator
        A velocity Verlet integrator.

    Notes
    -----
    An adaptive step size is used.

    References
    ----------

    Examples
    --------

    Create a gradient descent minimization integrator.

    >>> integrator = GradientDescentMinimizationIntegrator()

    """

    timestep = 1.0 * units.femtoseconds
    integrator = mm.CustomIntegrator(timestep)

    integrator.addGlobalVariable("step_size", initial_step_size/units.nanometers)
    integrator.addGlobalVariable("energy_old", 0)
    integrator.addGlobalVariable("energy_new", 0)
    integrator.addGlobalVariable("delta_energy", 0)
    integrator.addGlobalVariable("accept", 0)
    integrator.addGlobalVariable("fnorm2", 0)
    integrator.addPerDofVariable("x_old", 0)

    # Update context state.
    integrator.addUpdateContextState()

    # Constrain positions.
    integrator.addConstrainPositions()

    # Store old energy and positions.
    integrator.addComputeGlobal("energy_old", "energy")
    integrator.addComputePerDof("x_old", "x")

    # Compute sum of squared norm.
    integrator.addComputeSum("fnorm2", "f^2")

    # Take step.
    integrator.addComputePerDof("x", "x+step_size*f/sqrt(fnorm2 + delta(fnorm2))")
    integrator.addConstrainPositions()

    # Ensure we only keep steps that go downhill in energy.
    integrator.addComputeGlobal("energy_new", "energy")
    integrator.addComputeGlobal("delta_energy", "energy_new-energy_old")
    # Accept also checks for NaN
    integrator.addComputeGlobal("accept", "step(-delta_energy) * delta(energy - energy_new)")

    integrator.addComputePerDof("x", "accept*x + (1-accept)*x_old")

    # Update step size.
    integrator.addComputeGlobal("step_size", "step_size * (2.0*accept + 0.5*(1-accept))")

    return integrator

def VelocityVerletIntegrator(timestep=1.0*simtk.unit.femtoseconds):
    """
    Construct a velocity Verlet integrator.

    Parameters
    ----------
    timestep : numpy.unit.Quantity compatible with femtoseconds, default: 1*simtk.unit.femtoseconds
        The integration timestep.

    Returns
    -------
    integrator : simtk.openmm.CustomIntegrator
        A velocity Verlet integrator.

    Notes
    -----
    This integrator is taken verbatim from Peter Eastman's example appearing in the CustomIntegrator header file documentation.

    References
    ----------
    W. C. Swope, H. C. Andersen, P. H. Berens, and K. R. Wilson, J. Chem. Phys. 76, 637 (1982)

    Examples
    --------

    Create a velocity Verlet integrator.

    >>> timestep = 1.0 * simtk.unit.femtoseconds
    >>> integrator = VelocityVerletIntegrator(timestep)

    """

    integrator = mm.CustomIntegrator(timestep)

    integrator.addPerDofVariable("x1", 0)

    integrator.addUpdateContextState()
    integrator.addComputePerDof("v", "v+0.5*dt*f/m")
    integrator.addComputePerDof("x", "x+dt*v")
    integrator.addComputePerDof("x1", "x")
    integrator.addConstrainPositions()
    integrator.addComputePerDof("v", "v+0.5*dt*f/m+(x-x1)/dt")
    integrator.addConstrainVelocities()

    return integrator

def AndersenVelocityVerletIntegrator(temperature=298*simtk.unit.kelvin, collision_rate=91.0/simtk.unit.picoseconds, timestep=1.0*simtk.unit.femtoseconds):
    """
    Construct a velocity Verlet integrator with Andersen thermostat, implemented as per-particle collisions (rather than massive collisions).

    Parameters
    ----------
    temperature : numpy.unit.Quantity compatible with kelvin, default: 298*simtk.unit.kelvin
        The temperature of the fictitious bath.
    collision_rate : numpy.unit.Quantity compatible with 1/picoseconds, default: 91/simtk.unit.picoseconds
        The collision rate with fictitious bath particles.
    timestep : numpy.unit.Quantity compatible with femtoseconds, default: 1*simtk.unit.femtoseconds
        The integration timestep.

    Returns
    -------
    integrator : simtk.openmm.CustomIntegrator
        A velocity Verlet integrator with periodic Andersen thermostat.

    References
    ----------
    Hans C. Andersen "Molecular dynamics simulations at constant pressure and/or temperature", Journal of Chemical Physics 72, 2384-2393 (1980)
    http://dx.doi.org/10.1063/1.439486

    Examples
    --------

    Create a velocity Verlet integrator with Andersen thermostat.

    >>> timestep = 1.0 * simtk.unit.femtoseconds
    >>> collision_rate = 91.0 / simtk.unit.picoseconds
    >>> temperature = 298.0 * simtk.unit.kelvin
    >>> integrator = AndersenVelocityVerletIntegrator(temperature, collision_rate, timestep)

    Notes
    ------
    The velocity Verlet integrator is taken verbatim from Peter Eastman's example in the CustomIntegrator header file documentation.
    The efficiency could be improved by avoiding recomputation of sigma_v every timestep.

    """

    integrator = mm.CustomIntegrator(timestep)

    #
    # Integrator initialization.
    #
    kT = kB * temperature
    integrator.addGlobalVariable("kT", kT) # thermal energy
    integrator.addGlobalVariable("p_collision", timestep * collision_rate) # per-particle collision probability per timestep
    integrator.addPerDofVariable("sigma_v", 0) # velocity distribution stddev for Maxwell-Boltzmann (computed later)
    integrator.addPerDofVariable("collision", 0) # 1 if collision has occured this timestep, 0 otherwise
    integrator.addPerDofVariable("x1", 0) # for constraints

    #
    # Update velocities from Maxwell-Boltzmann distribution for particles that collide.
    #
    integrator.addComputePerDof("sigma_v", "sqrt(kT/m)")
    integrator.addComputePerDof("collision", "step(p_collision-uniform)") # if collision has occured this timestep, 0 otherwise
    integrator.addComputePerDof("v", "(1-collision)*v + collision*sigma_v*gaussian") # randomize velocities of particles that have collided

    #
    # Velocity Verlet step
    #
    integrator.addUpdateContextState()
    integrator.addComputePerDof("v", "v+0.5*dt*f/m")
    integrator.addComputePerDof("x", "x+dt*v")
    integrator.addComputePerDof("x1", "x")
    integrator.addConstrainPositions()
    integrator.addComputePerDof("v", "v+0.5*dt*f/m+(x-x1)/dt")
    integrator.addConstrainVelocities()

    return integrator

def MetropolisMonteCarloIntegrator(temperature=298.0*simtk.unit.kelvin, sigma=0.1*simtk.unit.angstroms, timestep=1*simtk.unit.femtoseconds):
    """
    Create a simple Metropolis Monte Carlo integrator that uses Gaussian displacement trials.

    Parameters
    ----------
    temperature : numpy.unit.Quantity compatible with kelvin, default: 298*simtk.unit.kelvin
        The temperature.
    sigma : numpy.unit.Quantity compatible with nanometers, default: 0.1*simtk.unit.angstroms
        The displacement standard deviation for each degree of freedom.
    timestep : numpy.unit.Quantity compatible with femtoseconds, default: 1.0*simtk.unit.femtoseconds
        The integration timestep, which is purely fictitious---it is just used to advance the simulation clock.

    Returns
    -------
    integrator : simtk.openmm.CustomIntegrator
        A Metropolis Monte Carlo integrator.

    Warning
    -------
    This integrator does not respect constraints.

    Notes
    -----
    The timestep is purely fictitious, and just used to advance the simulation clock.
    Velocities are drawn from a Maxwell-Boltzmann distribution each timestep to generate correct (x,v) statistics.
    Additional global variables 'ntrials' and  'naccept' keep track of how many trials have been attempted and accepted, respectively.

    Examples
    --------

    Create a Metropolis Monte Carlo integrator with specified random displacement standard deviation.

    >>> timestep = 1.0 * simtk.unit.femtoseconds # fictitious timestep
    >>> temperature = 298.0 * simtk.unit.kelvin
    >>> sigma = 1.0 * simtk.unit.angstroms
    >>> integrator = MetropolisMonteCarloIntegrator(temperature, sigma, timestep)

    """

    # Create a new Custom integrator.
    integrator = mm.CustomIntegrator(timestep)

    # Compute the thermal energy.
    kT = kB * temperature

    #
    # Integrator initialization.
    #
    integrator.addGlobalVariable("naccept", 0) # number accepted
    integrator.addGlobalVariable("ntrials", 0) # number of Metropolization trials

    integrator.addGlobalVariable("kT", kT) # thermal energy
    integrator.addPerDofVariable("sigma_x", sigma) # perturbation size
    integrator.addPerDofVariable("sigma_v", 0) # velocity distribution stddev for Maxwell-Boltzmann (set later)
    integrator.addPerDofVariable("xold", 0) # old positions
    integrator.addGlobalVariable("Eold", 0) # old energy
    integrator.addGlobalVariable("Enew", 0) # new energy
    integrator.addGlobalVariable("accept", 0) # accept or reject

    #
    # Context state update.
    #
    integrator.addUpdateContextState();

    #
    # Update velocities from Maxwell-Boltzmann distribution.
    #
    integrator.addComputePerDof("sigma_v", "sqrt(kT/m)")
    integrator.addComputePerDof("v", "sigma_v*gaussian")
    integrator.addConstrainVelocities();

    #
    # propagation steps
    #
    # Store old positions and energy.
    integrator.addComputePerDof("xold", "x")
    integrator.addComputeGlobal("Eold", "energy")
    # Gaussian particle displacements.
    integrator.addComputePerDof("x", "x + sigma_x*gaussian")
    # Accept or reject with Metropolis criteria.
    integrator.addComputeGlobal("accept", "step(exp(-(energy-Eold)/kT) - uniform)")
    integrator.addComputePerDof("x", "(1-accept)*xold + x*accept")
    # Accumulate acceptance statistics.
    integrator.addComputeGlobal("naccept", "naccept + accept")
    integrator.addComputeGlobal("ntrials", "ntrials + 1")

    return integrator

class HMCIntegrator(mm.CustomIntegrator):
    def __init__(self, temperature=298.0*simtk.unit.kelvin, nsteps=10, timestep=1*simtk.unit.femtoseconds):
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

        Returns
        -------
        integrator : simtk.openmm.CustomIntegrator
            A hybrid Monte Carlo integrator.

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
        mm.CustomIntegrator.__init__(self, dt)

        # Compute the thermal energy.
        kT = kB * temperature

        #
        # Integrator initialization.
        #
        self.addGlobalVariable("naccept", 0) # number accepted
        self.addGlobalVariable("ntrials", 0) # number of Metropolization trials

        self.addGlobalVariable("kT", kT) # thermal energy
        self.addPerDofVariable("sigma", 0)
        self.addGlobalVariable("ke", 0) # kinetic energy
        self.addPerDofVariable("xold", 0) # old positions
        self.addGlobalVariable("Eold", 0) # old energy
        self.addGlobalVariable("Enew", 0) # new energy
        self.addGlobalVariable("accept", 0) # accept or reject
        self.addPerDofVariable("x1", 0) # for constraints

        #
        # Pre-computation.
        # This only needs to be done once, but it needs to be done for each degree of freedom.
        # Could move this to initialization?
        #
        self.addComputePerDof("sigma", "sqrt(kT/m)")

        #
        # Allow Context updating here, outside of inner loop only.
        #
        self.addUpdateContextState();

        #
        # Draw new velocity.
        #
        self.addComputePerDof("v", "sigma*gaussian")
        self.addConstrainVelocities();

        #
        # Store old position and energy.
        #
        self.addComputeSum("ke", "0.5*m*v*v")
        self.addComputeGlobal("Eold", "ke + energy")
        self.addComputePerDof("xold", "x")

        #
        # Inner symplectic steps using velocity Verlet.
        #
        for step in range(nsteps):
            self.addComputePerDof("v", "v+0.5*dt*f/m")
            self.addComputePerDof("x", "x+dt*v")
            self.addComputePerDof("x1", "x")
            self.addConstrainPositions()
            self.addComputePerDof("v", "v+0.5*dt*f/m+(x-x1)/dt")
            self.addConstrainVelocities()

        #
        # Accept/reject step.
        #
        self.addComputeSum("ke", "0.5*m*v*v")
        self.addComputeGlobal("Enew", "ke + energy")
        self.addComputeGlobal("accept", "step(exp(-(Enew-Eold)/kT) - uniform)")
        self.addComputePerDof("x", "x*accept + xold*(1-accept)")

        #
        # Accumulate statistics.
        #
        self.addComputeGlobal("naccept", "naccept + accept")
        self.addComputeGlobal("ntrials", "ntrials + 1")

    @property
    def n_accept(self):
        return integrator.getGlobalVariableByName("naccept")

    @property
    def n_trials(self):
        return integrator.getGlobalVariableByName("ntrials")

    @property
    def acceptance_rate(self):
        return self.n_accept / float(self.n_trials)


def GHMCIntegrator(temperature=298.0*simtk.unit.kelvin, collision_rate=91.0/simtk.unit.picoseconds, timestep=1.0*simtk.unit.femtoseconds):
    """
    Create a generalized hybrid Monte Carlo (GHMC) integrator.

    Parameters
    ----------
    temperature : numpy.unit.Quantity compatible with kelvin, default: 298*simtk.unit.kelvin
        The temperature.
    collision_rate : numpy.unit.Quantity compatible with 1/picoseconds, default: 91.0/simtk.unit.picoseconds
        The collision rate.
    timestep : numpy.unit.Quantity compatible with femtoseconds, default: 1.0*simtk.unit.femtoseconds
        The integration timestep.

    Returns
    -------
    integrator : simtk.openmm.CustomIntegrator
        A GHMC integrator.

    Notes
    -----
    This integrator is equivalent to a Langevin integrator in the velocity Verlet discretization with a
    Metrpolization step to ensure sampling from the appropriate distribution.

    Additional global variables 'ntrials' and  'naccept' keep track of how many trials have been attempted and
    accepted, respectively.

    TODO
    ----
    Move initialization of 'sigma' to setting the per-particle variables.

    Examples
    --------

    Create a GHMC integrator.

    >>> temperature = 298.0 * simtk.unit.kelvin
    >>> collision_rate = 91.0 / simtk.unit.picoseconds
    >>> timestep = 1.0 * simtk.unit.femtoseconds
    >>> integrator = GHMCIntegrator(temperature, collision_rate, timestep)

    References
    ----------
    Lelievre T, Stoltz G, and Rousset M. Free Energy Computations: A Mathematical Perspective
    http://www.amazon.com/Free-Energy-Computations-Mathematical-Perspective/dp/1848162472

    """

    # Initialize constants.
    kT = kB * temperature
    gamma = collision_rate

    # Create a new custom integrator.
    integrator = mm.CustomIntegrator(timestep)

    #
    # Integrator initialization.
    #
    integrator.addGlobalVariable("kT", kT) # thermal energy
    integrator.addGlobalVariable("b", numpy.exp(-gamma*timestep)) # velocity mixing parameter
    integrator.addPerDofVariable("sigma", 0)
    integrator.addGlobalVariable("ke", 0) # kinetic energy
    integrator.addPerDofVariable("vold", 0) # old velocities
    integrator.addPerDofVariable("xold", 0) # old positions
    integrator.addGlobalVariable("Eold", 0) # old energy
    integrator.addGlobalVariable("Enew", 0) # new energy
    integrator.addGlobalVariable("accept", 0) # accept or reject
    integrator.addGlobalVariable("naccept", 0) # number accepted
    integrator.addGlobalVariable("ntrials", 0) # number of Metropolization trials
    integrator.addPerDofVariable("x1", 0) # position before application of constraints

    #
    # Pre-computation.
    # This only needs to be done once, but it needs to be done for each degree of freedom.
    # Could move this to initialization?
    #
    integrator.addComputePerDof("sigma", "sqrt(kT/m)")

    #
    # Allow context updating here.
    #
    integrator.addUpdateContextState();

    #
    # Constrain positions.
    #
    integrator.addConstrainPositions();

    #
    # Velocity perturbation.
    #
    integrator.addComputePerDof("v", "sqrt(b)*v + sqrt(1-b)*sigma*gaussian")
    integrator.addConstrainVelocities();

    #
    # Metropolized symplectic step.
    #
    integrator.addComputeSum("ke", "0.5*m*v*v")
    integrator.addComputeGlobal("Eold", "ke + energy")
    integrator.addComputePerDof("xold", "x")
    integrator.addComputePerDof("vold", "v")
    integrator.addComputePerDof("v", "v + 0.5*dt*f/m")
    integrator.addComputePerDof("x", "x + v*dt")
    integrator.addComputePerDof("x1", "x")
    integrator.addConstrainPositions();
    integrator.addComputePerDof("v", "v + 0.5*dt*f/m + (x-x1)/dt")
    integrator.addConstrainVelocities();
    integrator.addComputeSum("ke", "0.5*m*v*v")
    integrator.addComputeGlobal("Enew", "ke + energy")
    integrator.addComputeGlobal("accept", "step(exp(-(Enew-Eold)/kT) - uniform)")
    integrator.addComputePerDof("x", "x*accept + xold*(1-accept)")
    integrator.addComputePerDof("v", "v*accept - vold*(1-accept)")

    #
    # Velocity randomization
    #
    integrator.addComputePerDof("v", "sqrt(b)*v + sqrt(1-b)*sigma*gaussian")
    integrator.addConstrainVelocities();

    #
    # Accumulate statistics.
    #
    integrator.addComputeGlobal("naccept", "naccept + accept")
    integrator.addComputeGlobal("ntrials", "ntrials + 1")

    return integrator

def VVVRIntegrator(temperature=298.0*simtk.unit.kelvin, collision_rate=91.0/simtk.unit.picoseconds, timestep=1.0*simtk.unit.femtoseconds):
    """
    Create a velocity verlet with velocity randomization (VVVR) integrator.

    Parameters
    ----------
    temperature : numpy.unit.Quantity compatible with kelvin, default: 298.0*simtk.unit.kelvin
        The temperature.
    collision_rate : numpy.unit.Quantity compatible with 1/picoseconds, default: 91.0/simtk.unit.picoseconds
        The collision rate.
    timestep : numpy.unit.Quantity compatible with femtoseconds, default: 1.0*simtk.unit.femtoseconds
        The integration timestep.

    Returns
    -------
    integrator : simtk.openmm.CustomIntegrator
        VVVR integrator.

    Notes
    -----
    This integrator is equivalent to a Langevin integrator in the velocity Verlet discretization with a
    timestep correction to ensure that the field-free diffusion constant is timestep invariant.

    The global 'pseudowork' keeps track of the pseudowork accumulated during integration, and can be
    used to correct the sampled statistics or in a Metropolization scheme.

    TODO
    ----
    Move initialization of 'sigma' to setting the per-particle variables.
    We can ditch pseudowork and instead use total energy difference - heat.

    References
    ----------
    David A. Sivak, John D. Chodera, and Gavin E. Crooks.
    Time step rescaling recovers continuous-time dynamical properties for discrete-time Langevin integration of nonequilibrium systems
    http://arxiv.org/abs/1301.3800

    Examples
    --------

    Create a VVVR integrator.

    >>> temperature = 298.0 * simtk.unit.kelvin
    >>> collision_rate = 91.0 / simtk.unit.picoseconds
    >>> timestep = 1.0 * simtk.unit.femtoseconds
    >>> integrator = VVVRIntegrator(temperature, collision_rate, timestep)

    """

    # Compute constants.
    kT = kB * temperature
    gamma = collision_rate

    # Create a new custom integrator.
    integrator = mm.CustomIntegrator(timestep)

    #
    # Integrator initialization.
    #
    integrator.addGlobalVariable("kT", kT) # thermal energy
    integrator.addGlobalVariable("b", numpy.exp(-gamma*timestep)) # velocity mixing parameter
    integrator.addPerDofVariable("sigma", 0)
    integrator.addPerDofVariable("x1", 0) # position before application of constraints

    #
    # Allow context updating here.
    #
    integrator.addUpdateContextState();

    #
    # Pre-computation.
    # This only needs to be done once, but it needs to be done for each degree of freedom.
    # Could move this to initialization?
    #
    integrator.addComputePerDof("sigma", "sqrt(kT/m)")

    #
    # Velocity perturbation.
    #
    integrator.addComputePerDof("v", "sqrt(b)*v + sqrt(1-b)*sigma*gaussian")
    integrator.addConstrainVelocities();

    #
    # Metropolized symplectic step.
    #
    integrator.addComputePerDof("v", "v + 0.5*dt*f/m")
    integrator.addComputePerDof("x", "x + v*dt")
    integrator.addComputePerDof("x1", "x")
    integrator.addConstrainPositions();
    integrator.addComputePerDof("v", "v + 0.5*dt*f/m + (x-x1)/dt")
    integrator.addConstrainVelocities();

    #
    # Velocity randomization
    #
    integrator.addComputePerDof("v", "sqrt(b)*v + sqrt(1-b)*sigma*gaussian")
    integrator.addConstrainVelocities();

    return integrator

