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

import respa
class MTSIntegrator(respa.MTSIntegrator):
    """
    MTSIntegrator implements the rRESPA multiple time step integration algorithm.

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

    For details, see Tuckerman et al., J. Chem. Phys. 97(3) pp. 1990-2001 (1992).

    """
    def __init__(self, timestep=1.0*simtk.unit.femtoseconds, groups=(0,1)):
        """Create an MTSIntegrator.

        Parameters
        ----------
        timestep : simtk.unit.Quantity with units compatible with femtoseconds, optional default=1*femtoseconds
           The largest (outermost) integration time step to use.
        groups : list of tuples, optional, default=(0,1)
           A list of tuples defining the force groups.  The first element of each tuple is the force group index, and the second element is the number of times that force group should be evaluated in one time step.

        """
        super(MTSIntegrator, self).__init__(timestep, groups)

class DummyIntegrator(simtk.openmm.CustomIntegrator):
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
    def __init__(self):
        timestep = 0.0 * units.femtoseconds
        super(DummyIntegrator, self).__init__(timestep)
        self.addUpdateContextState()
        self.addConstrainPositions()
        self.addConstrainVelocities()

class GradientDescentMinimizationIntegrator(simtk.openmm.CustomIntegrator):
    """Simple gradient descent minimizer implemented as an integrator.

    Examples
    --------

    Create a gradient descent minimization integrator.

    >>> integrator = GradientDescentMinimizationIntegrator()

    """

    def __init__(self, initial_step_size=0.01*units.angstroms):
        """
        Construct a simple gradient descent minimization integrator.

        Parameters
        ----------
        initial_step_size : numpy.unit.Quantity compatible with nanometers, default: 0.01*simtk.unit.angstroms
           The norm of the initial step size guess.

        Notes
        -----
        An adaptive step size is used.

        """

        timestep = 1.0 * units.femtoseconds
        super(GradientDescentMinimizationIntegrator, self).__init__(timestep)

        self.addGlobalVariable("step_size", initial_step_size/units.nanometers)
        self.addGlobalVariable("energy_old", 0)
        self.addGlobalVariable("energy_new", 0)
        self.addGlobalVariable("delta_energy", 0)
        self.addGlobalVariable("accept", 0)
        self.addGlobalVariable("fnorm2", 0)
        self.addPerDofVariable("x_old", 0)

        # Update context state.
        self.addUpdateContextState()

        # Constrain positions.
        self.addConstrainPositions()

        # Store old energy and positions.
        self.addComputeGlobal("energy_old", "energy")
        self.addComputePerDof("x_old", "x")

        # Compute sum of squared norm.
        self.addComputeSum("fnorm2", "f^2")

        # Take step.
        self.addComputePerDof("x", "x+step_size*f/sqrt(fnorm2 + delta(fnorm2))")
        self.addConstrainPositions()

        # Ensure we only keep steps that go downhill in energy.
        self.addComputeGlobal("energy_new", "energy")
        self.addComputeGlobal("delta_energy", "energy_new-energy_old")
        # Accept also checks for NaN
        self.addComputeGlobal("accept", "step(-delta_energy) * delta(energy - energy_new)")

        self.addComputePerDof("x", "accept*x + (1-accept)*x_old")

        # Update step size.
        self.addComputeGlobal("step_size", "step_size * (2.0*accept + 0.5*(1-accept))")

class VelocityVerletIntegrator(simtk.openmm.CustomIntegrator):
    """Verlocity Verlet integrator.

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

    def __init__(self, timestep=1.0*simtk.unit.femtoseconds):
        """Construct a velocity Verlet integrator.

        Parameters
        ----------
        timestep : numpy.unit.Quantity compatible with femtoseconds, default: 1*simtk.unit.femtoseconds
           The integration timestep.

        """

        super(VelocityVerletIntegrator, self).__init__(timestep)

        self.addPerDofVariable("x1", 0)

        self.addUpdateContextState()
        self.addComputePerDof("v", "v+0.5*dt*f/m")
        self.addComputePerDof("x", "x+dt*v")
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions()
        self.addComputePerDof("v", "v+0.5*dt*f/m+(x-x1)/dt")
        self.addConstrainVelocities()

class AndersenVelocityVerletIntegrator(simtk.openmm.CustomIntegrator):
    """Velocity Verlet integrator with Andersen thermostat using per-particle collisions (rather than massive collisions).

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

    def __init__(self, temperature=298*simtk.unit.kelvin, collision_rate=91.0/simtk.unit.picoseconds, timestep=1.0*simtk.unit.femtoseconds):
        """Construct a velocity Verlet integrator with Andersen thermostat, implemented as per-particle collisions (rather than massive collisions).

        Parameters
        ----------
        temperature : numpy.unit.Quantity compatible with kelvin, default=298*simtk.unit.kelvin
           The temperature of the fictitious bath.
        collision_rate : numpy.unit.Quantity compatible with 1/picoseconds, default=91/simtk.unit.picoseconds
           The collision rate with fictitious bath particles.
        timestep : numpy.unit.Quantity compatible with femtoseconds, default=1*simtk.unit.femtoseconds
           The integration timestep.

        """
        super(AndersenVelocityVerletIntegrator, self).__init__(timestep)

        #
        # Integrator initialization.
        #
        kT = kB * temperature
        self.addGlobalVariable("kT", kT) # thermal energy
        self.addGlobalVariable("p_collision", timestep * collision_rate) # per-particle collision probability per timestep
        self.addPerDofVariable("sigma_v", 0) # velocity distribution stddev for Maxwell-Boltzmann (computed later)
        self.addPerDofVariable("collision", 0) # 1 if collision has occured this timestep, 0 otherwise
        self.addPerDofVariable("x1", 0) # for constraints

        #
        # Update velocities from Maxwell-Boltzmann distribution for particles that collide.
        #
        self.addComputePerDof("sigma_v", "sqrt(kT/m)")
        self.addComputePerDof("collision", "step(p_collision-uniform)") # if collision has occured this timestep, 0 otherwise
        self.addComputePerDof("v", "(1-collision)*v + collision*sigma_v*gaussian") # randomize velocities of particles that have collided

        #
        # Velocity Verlet step
        #
        self.addUpdateContextState()
        self.addComputePerDof("v", "v+0.5*dt*f/m")
        self.addComputePerDof("x", "x+dt*v")
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions()
        self.addComputePerDof("v", "v+0.5*dt*f/m+(x-x1)/dt")
        self.addConstrainVelocities()

class MetropolisMonteCarloIntegrator(simtk.openmm.CustomIntegrator):
    """
    Metropolis Monte Carlo with Gaussian displacement trials.

    """

    def __init__(self, temperature=298.0*simtk.unit.kelvin, sigma=0.1*simtk.unit.angstroms, timestep=1*simtk.unit.femtoseconds):
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
        super(MetropolisMonteCarloIntegrator, self).__init__(timestep)

        # Compute the thermal energy.
        kT = kB * temperature

        #
        # Integrator initialization.
        #
        self.addGlobalVariable("naccept", 0) # number accepted
        self.addGlobalVariable("ntrials", 0) # number of Metropolization trials

        self.addGlobalVariable("kT", kT) # thermal energy
        self.addPerDofVariable("sigma_x", sigma) # perturbation size
        self.addPerDofVariable("sigma_v", 0) # velocity distribution stddev for Maxwell-Boltzmann (set later)
        self.addPerDofVariable("xold", 0) # old positions
        self.addGlobalVariable("Eold", 0) # old energy
        self.addGlobalVariable("Enew", 0) # new energy
        self.addGlobalVariable("accept", 0) # accept or reject

        #
        # Context state update.
        #
        self.addUpdateContextState();

        #
        # Update velocities from Maxwell-Boltzmann distribution.
        #
        self.addComputePerDof("sigma_v", "sqrt(kT/m)")
        self.addComputePerDof("v", "sigma_v*gaussian")
        self.addConstrainVelocities();

        #
        # propagation steps
        #
        # Store old positions and energy.
        self.addComputePerDof("xold", "x")
        self.addComputeGlobal("Eold", "energy")
        # Gaussian particle displacements.
        self.addComputePerDof("x", "x + sigma_x*gaussian")
        # Accept or reject with Metropolis criteria.
        self.addComputeGlobal("accept", "step(exp(-(energy-Eold)/kT) - uniform)")
        self.addComputePerDof("x", "(1-accept)*xold + x*accept")
        # Accumulate acceptance statistics.
        self.addComputeGlobal("naccept", "naccept + accept")
        self.addComputeGlobal("ntrials", "ntrials + 1")

class HMCIntegrator(simtk.openmm.CustomIntegrator):
    """
    Hybrid Monte Carlo (HMC) integrator.

    """

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

        super(HMCIntegrator, self).__init__(timestep)

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

class GHMCIntegrator(simtk.openmm.CustomIntegrator):
    """
    Generalized hybrid Monte Carlo (GHMC) integrator.

    """

    def __init__(self, temperature=298.0*simtk.unit.kelvin, collision_rate=91.0/simtk.unit.picoseconds, timestep=1.0*simtk.unit.femtoseconds):
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
        super(GHMCIntegrator, self).__init__(timestep)

        #
        # Integrator initialization.
        #
        self.addGlobalVariable("kT", kT) # thermal energy
        self.addGlobalVariable("b", numpy.exp(-gamma*timestep)) # velocity mixing parameter
        self.addPerDofVariable("sigma", 0)
        self.addGlobalVariable("ke", 0) # kinetic energy
        self.addPerDofVariable("vold", 0) # old velocities
        self.addPerDofVariable("xold", 0) # old positions
        self.addGlobalVariable("Eold", 0) # old energy
        self.addGlobalVariable("Enew", 0) # new energy
        self.addGlobalVariable("accept", 0) # accept or reject
        self.addGlobalVariable("naccept", 0) # number accepted
        self.addGlobalVariable("ntrials", 0) # number of Metropolization trials
        self.addPerDofVariable("x1", 0) # position before application of constraints

        #
        # Pre-computation.
        # This only needs to be done once, but it needs to be done for each degree of freedom.
        # Could move this to initialization?
        #
        self.addComputePerDof("sigma", "sqrt(kT/m)")

        #
        # Allow context updating here.
        #
        self.addUpdateContextState();

        #
        # Constrain positions.
        #
        self.addConstrainPositions();

        #
        # Velocity perturbation.
        #
        self.addComputePerDof("v", "sqrt(b)*v + sqrt(1-b)*sigma*gaussian")
        self.addConstrainVelocities();

        #
        # Metropolized symplectic step.
        #
        self.addComputeSum("ke", "0.5*m*v*v")
        self.addComputeGlobal("Eold", "ke + energy")
        self.addComputePerDof("xold", "x")
        self.addComputePerDof("vold", "v")
        self.addComputePerDof("v", "v + 0.5*dt*f/m")
        self.addComputePerDof("x", "x + v*dt")
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions();
        self.addComputePerDof("v", "v + 0.5*dt*f/m + (x-x1)/dt")
        self.addConstrainVelocities();
        self.addComputeSum("ke", "0.5*m*v*v")
        self.addComputeGlobal("Enew", "ke + energy")
        self.addComputeGlobal("accept", "step(exp(-(Enew-Eold)/kT) - uniform)")
        self.addComputePerDof("x", "x*accept + xold*(1-accept)")
        self.addComputePerDof("v", "v*accept - vold*(1-accept)")

        #
        # Velocity randomization
        #
        self.addComputePerDof("v", "sqrt(b)*v + sqrt(1-b)*sigma*gaussian")
        self.addConstrainVelocities();

        #
        # Accumulate statistics.
        #
        self.addComputeGlobal("naccept", "naccept + accept")
        self.addComputeGlobal("ntrials", "ntrials + 1")

class VVVRIntegrator(simtk.openmm.CustomIntegrator):
    """
    Create a velocity Verlet with velocity randomization (VVVR) integrator.

    """

    def __init__(self, temperature=298.0*simtk.unit.kelvin, collision_rate=91.0/simtk.unit.picoseconds, timestep=1.0*simtk.unit.femtoseconds):
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
        super(VVVRIntegrator, self).__init__(timestep)

        #
        # Integrator initialization.
        #
        self.addGlobalVariable("kT", kT) # thermal energy
        self.addGlobalVariable("b", numpy.exp(-gamma*timestep)) # velocity mixing parameter
        self.addPerDofVariable("sigma", 0)
        self.addPerDofVariable("x1", 0) # position before application of constraints

        #
        # Allow context updating here.
        #
        self.addUpdateContextState();

        #
        # Pre-computation.
        # This only needs to be done once, but it needs to be done for each degree of freedom.
        # Could move this to initialization?
        #
        self.addComputePerDof("sigma", "sqrt(kT/m)")

        #
        # Velocity perturbation.
        #
        self.addComputePerDof("v", "sqrt(b)*v + sqrt(1-b)*sigma*gaussian")
        self.addConstrainVelocities();

        #
        # Metropolized symplectic step.
        #
        self.addComputePerDof("v", "v + 0.5*dt*f/m")
        self.addComputePerDof("x", "x + v*dt")
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions();
        self.addComputePerDof("v", "v + 0.5*dt*f/m + (x-x1)/dt")
        self.addConstrainVelocities();

        #
        # Velocity randomization
        #
        self.addComputePerDof("v", "sqrt(b)*v + sqrt(1-b)*sigma*gaussian")
        self.addConstrainVelocities();


