# ============================================================================================
# MODULE DOCSTRING
# ============================================================================================

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

# ============================================================================================
# GLOBAL IMPORTS
# ============================================================================================

import numpy
import logging

import simtk.unit

import simtk.unit as units
import simtk.openmm as mm

from openmmtools.constants import kB
from openmmtools import respa

logger = logging.getLogger(__name__)


# ============================================================================================
# BASE CLASSES
# ============================================================================================

class RestorableIntegrator(mm.CustomIntegrator):
    """A CustomIntegrator that can be restored after being copied.

    Normally, a CustomIntegrator loses its specific class (and all its
    methods) when it is copied. This happens for example when obtaining
    the integrator from a Context with getIntegrator(). This class
    offers a method restore_interface() that restore the original class.

    Parameters
    ----------
    temperature : simtk.unit.Quantity
        The temperature of the integrator heat bath (temperature units).
    timestep : simtk.unit.Quantity
        The timestep to pass to the CustomIntegrator constructor (time
        units)

    """

    def __init__(self, *args, **kwargs):
        super(RestorableIntegrator, self).__init__(*args, **kwargs)
        self.addGlobalVariable('_restorable__class_hash',
                               self._compute_class_hash(self.__class__))

    @staticmethod
    def is_restorable(integrator):
        """Check if the integrator has a restorable interface.

        Parameters
        ----------
        integrator : simtk.openmm.CustomIntegrator
            The custom integrator to check.

        Returns
        -------
        True if the integrator has a restorable interface, False otherwise.

        """
        try:
            integrator.getGlobalVariableByName('_restorable__class_hash')
        except Exception:
            return False
        return True

    @classmethod
    def restore_interface(cls, integrator):
        """Restore the original interface of a CustomIntegrator.

        The function restore the methods of the original class that
        inherited from RestorableIntegrator. Return False if the interface
        could not be restored.

        Parameters
        ----------
        integrator : simtk.openmm.CustomIntegrator
            The integrator to which add methods.

        Returns
        -------
        True if the original class interface could be restored, False otherwise.

        """
        try:
            integrator_hash = integrator.getGlobalVariableByName('_restorable__class_hash')
        except Exception:
            return False

        # Compute the hash table for all subclasses.
        if cls._cached_hash_subclasses is None:
            # Recursive function to find all subclasses.
            def all_subclasses(c):
                return c.__subclasses__() + [subsubcls for subcls in c.__subclasses__()
                                             for subsubcls in all_subclasses(subcls)]
            cls._cached_hash_subclasses = {cls._compute_class_hash(sc): sc
                                           for sc in all_subclasses(cls)}
        # Retrieve integrator class.
        try:
            integrator_class = cls._cached_hash_subclasses[integrator_hash]
        except KeyError:
            return False

        # Restore class interface.
        integrator.__class__ = integrator_class
        return True

    # -------------------------------------------------------------------------
    # Internal-usage
    # -------------------------------------------------------------------------

    _cached_hash_subclasses = None

    @staticmethod
    def _compute_class_hash(integrator_class):
        """Return a numeric hash for the integrator class."""
        # We need to convert to float because some digits may be lost in the conversion
        return float(hash(integrator_class.__name__))


class ThermostatedIntegrator(RestorableIntegrator):
    """Add temperature functions to a CustomIntegrator.

    This class is intended to be inherited by integrators that maintain the
    stationary distribution at a given temperature. The constructor adds a
    global variable named "kT" defining the thermal energy at the given
    temperature. This global variable is updated through the temperature
    setter and getter.

    It also provide a utility function to handle per-DOF constants that
    must be computed only when the temperature changes.

    Notice that the CustomIntegrator internally stored by a Context object
    will loose setter and getter and any extra function you define. The same
    happens when you copy your integrator. You can restore the methods with
    the static method ThermostatedIntegrator.restore_interface().

    Parameters
    ----------
    temperature : simtk.unit.Quantity
        The temperature of the integrator heat bath (temperature units).
    timestep : simtk.unit.Quantity
        The timestep to pass to the CustomIntegrator constructor (time
        units).

    Examples
    --------
    We can inherit from ThermostatedIntegrator to automatically define
    setters and getters for the temperature and to add a per-DOF constant
    "sigma" that we need to update only when the temperature is changed.

    >>> from simtk import openmm, unit
    >>> class TestIntegrator(ThermostatedIntegrator):
    ...     def __init__(self, temperature=298.0*unit.kelvin, timestep=1.0*unit.femtoseconds):
    ...         super(TestIntegrator, self).__init__(temperature, timestep)
    ...         self.addPerDofVariable("sigma", 0)  # velocity standard deviation
    ...         self.addComputeTemperatureDependentConstants({"sigma": "sqrt(kT/m)"})
    ...

    We instantiate the integrator normally.

    >>> integrator = TestIntegrator(temperature=350*unit.kelvin)
    >>> integrator.getTemperature()
    Quantity(value=350.0, unit=kelvin)
    >>> integrator.setTemperature(380.0*unit.kelvin)
    >>> integrator.getTemperature()
    Quantity(value=380.0, unit=kelvin)
    >>> integrator.getGlobalVariableByName('kT')
    3.1594995390636815

    Notice that a CustomIntegrator bound to a context loses any extra method.

    >>> from openmmtools import testsystems
    >>> test = testsystems.HarmonicOscillator()
    >>> context = openmm.Context(test.system, integrator)
    >>> integrator = context.getIntegrator()
    >>> integrator.getTemperature()
    Traceback (most recent call last):
    ...
    AttributeError: type object 'object' has no attribute '__getattr__'

    We can restore the original interface with a class method

    >>> ThermostatedIntegrator.restore_interface(integrator)
    True
    >>> integrator.getTemperature()
    Quantity(value=380.0, unit=kelvin)
    >>> integrator.setTemperature(400.0*unit.kelvin)
    >>> isinstance(integrator, TestIntegrator)
    True

    """
    def __init__(self, temperature, *args, **kwargs):
        super(ThermostatedIntegrator, self).__init__(*args, **kwargs)
        self.addGlobalVariable('kT', kB * temperature)  # thermal energy

    def getTemperature(self):
        """Return the temperature of the heat bath.

        Returns
        -------
        temperature : simtk.unit.Quantity
            The temperature of the heat bath in kelvins.

        """
        kT = self.getGlobalVariableByName('kT') * units.kilojoule_per_mole
        temperature = kT / kB
        return temperature

    def setTemperature(self, temperature):
        """Set the temperature of the heat bath.

        Parameters
        ----------
        temperature : simtk.unit.Quantity
            The new temperature of the heat bath (temperature units).

        """
        kT = kB * temperature
        self.setGlobalVariableByName('kT', kT)

        # Update the changed flag if it exist.
        try:
            self.setGlobalVariableByName('has_kT_changed', 1)
        except Exception:
            pass

    def addComputeTemperatureDependentConstants(self, compute_per_dof):
        """Wrap the ComputePerDof into an if-block executed only when kT changes.

        Parameters
        ----------
        compute_per_dof : dict of str: str
            A dictionary of variable_name: expression.

        """
        # First check if flag variable already exist.
        try:
            self.getGlobalVariableByName('has_kT_changed')
        except Exception:
            self.addGlobalVariable('has_kT_changed', 1)

        # Create if-block that conditionally update the per-DOF variables.
        self.beginIfBlock('has_kT_changed = 1')
        for variable, expression in compute_per_dof.items():
            self.addComputePerDof(variable, expression)
        self.addComputeGlobal('has_kT_changed', '0')
        self.endBlock()

    @classmethod
    def is_thermostated(cls, integrator):
        """Return true if the integrator is a ThermostatedIntegrator.

        This can be useful when you only have access to the Context
        CustomIntegrator, which loses all extra function during serialization.

        Parameters
        ----------
        integrator : simtk.openmm.Integrator
            The integrator to check.

        Returns
        -------
        True if the original CustomIntegrator class inherited from
        ThermostatedIntegrator, False otherwise.

        """
        try:
            integrator.getGlobalVariableByName('kT')
        except Exception:
            return False
        return super(ThermostatedIntegrator, cls).is_restorable(integrator)

    @classmethod
    def restore_interface(cls, integrator):
        """Restore the original interface of a CustomIntegrator.

        The function restore the methods of the original class that
        inherited from ThermostatedIntegrator. Return False if the interface
        could not be restored.

        Parameters
        ----------
        integrator : simtk.openmm.CustomIntegrator
            The integrator to which add methods.

        Returns
        -------
        True if the original class interface could be restored, False otherwise.

        """
        restored = super(ThermostatedIntegrator, cls).restore_interface(integrator)
        # Warn the user if he is implementing a CustomIntegrator
        # that may keep the stationary distribution at a certain
        # temperature without exposing getters and setters.
        if not restored:
            try:
                integrator.getGlobalVariableByName('kT')
            except Exception:
                pass
            else:
                if not hasattr(integrator, 'getTemperature'):
                    logger.warning("The integrator {} has a global variable 'kT' variable "
                                   "but does not expose getter and setter for the temperature. "
                                   "Consider inheriting from ThermostatedIntegrator.")
        return restored

# ============================================================================================
# INTEGRATORS
# ============================================================================================


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

    def __init__(self, timestep=1.0 * simtk.unit.femtoseconds, groups=[(0, 1)]):
        """Create an MTSIntegrator.

        Parameters
        ----------
        timestep : simtk.unit.Quantity with units compatible with femtoseconds, optional default=1*femtoseconds
           The largest (outermost) integration time step to use.
        groups : list of tuples, optional, default=(0,1)
           A list of tuples defining the force groups.  The first element of each tuple is the force group index, and the second element is the number of times that force group should be evaluated in one time step.

        """
        super(MTSIntegrator, self).__init__(timestep, groups)


class DummyIntegrator(mm.CustomIntegrator):

    """
    Construct a dummy integrator that does nothing except update call the force updates.

    Returns
    -------
    integrator : mm.CustomIntegrator
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


class GradientDescentMinimizationIntegrator(mm.CustomIntegrator):

    """Simple gradient descent minimizer implemented as an integrator.

    Examples
    --------

    Create a gradient descent minimization integrator.

    >>> integrator = GradientDescentMinimizationIntegrator()

    """

    def __init__(self, initial_step_size=0.01 * units.angstroms):
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

        self.addGlobalVariable("step_size", initial_step_size / units.nanometers)
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


class VelocityVerletIntegrator(mm.CustomIntegrator):

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

    def __init__(self, timestep=1.0 * simtk.unit.femtoseconds):
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


class AndersenVelocityVerletIntegrator(ThermostatedIntegrator):

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

    def __init__(self, temperature=298 * simtk.unit.kelvin, collision_rate=91.0 / simtk.unit.picoseconds, timestep=1.0 * simtk.unit.femtoseconds):
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
        super(AndersenVelocityVerletIntegrator, self).__init__(temperature, timestep)

        #
        # Integrator initialization.
        #
        self.addGlobalVariable("p_collision", timestep * collision_rate)  # per-particle collision probability per timestep
        self.addPerDofVariable("sigma_v", 0)  # velocity distribution stddev for Maxwell-Boltzmann (computed later)
        self.addPerDofVariable("collision", 0)  # 1 if collision has occured this timestep, 0 otherwise
        self.addPerDofVariable("x1", 0)  # for constraints

        #
        # Update velocities from Maxwell-Boltzmann distribution for particles that collide.
        #
        self.addComputeTemperatureDependentConstants({"sigma_v": "sqrt(kT/m)"})
        self.addComputePerDof("collision", "step(p_collision-uniform)")  # if collision has occured this timestep, 0 otherwise
        self.addComputePerDof("v", "(1-collision)*v + collision*sigma_v*gaussian")  # randomize velocities of particles that have collided

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


class MetropolisMonteCarloIntegrator(ThermostatedIntegrator):

    """
    Metropolis Monte Carlo with Gaussian displacement trials.

    """

    def __init__(self, temperature=298.0 * simtk.unit.kelvin, sigma=0.1 * simtk.unit.angstroms, timestep=1 * simtk.unit.femtoseconds):
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
        super(MetropolisMonteCarloIntegrator, self).__init__(temperature, timestep)

        #
        # Integrator initialization.
        #
        self.addGlobalVariable("naccept", 0)  # number accepted
        self.addGlobalVariable("ntrials", 0)  # number of Metropolization trials

        self.addPerDofVariable("sigma_x", sigma)  # perturbation size
        self.addPerDofVariable("sigma_v", 0)  # velocity distribution stddev for Maxwell-Boltzmann (set later)
        self.addPerDofVariable("xold", 0)  # old positions
        self.addGlobalVariable("Eold", 0)  # old energy
        self.addGlobalVariable("Enew", 0)  # new energy
        self.addGlobalVariable("accept", 0)  # accept or reject

        #
        # Context state update.
        #
        self.addUpdateContextState()

        #
        # Update velocities from Maxwell-Boltzmann distribution.
        #
        self.addComputeTemperatureDependentConstants({"sigma_v": "sqrt(kT/m)"})
        self.addComputePerDof("v", "sigma_v*gaussian")
        self.addConstrainVelocities()

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


class HMCIntegrator(ThermostatedIntegrator):

    """
    Hybrid Monte Carlo (HMC) integrator.

    """

    def __init__(self, temperature=298.0 * simtk.unit.kelvin, nsteps=10, timestep=1 * simtk.unit.femtoseconds):
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

        super(HMCIntegrator, self).__init__(temperature, timestep)

        #
        # Integrator initialization.
        #
        self.addGlobalVariable("naccept", 0)  # number accepted
        self.addGlobalVariable("ntrials", 0)  # number of Metropolization trials

        self.addPerDofVariable("sigma", 0)
        self.addGlobalVariable("ke", 0)  # kinetic energy
        self.addPerDofVariable("xold", 0)  # old positions
        self.addGlobalVariable("Eold", 0)  # old energy
        self.addGlobalVariable("Enew", 0)  # new energy
        self.addGlobalVariable("accept", 0)  # accept or reject
        self.addPerDofVariable("x1", 0)  # for constraints

        #
        # Pre-computation.
        # This only needs to be done once, but it needs to be done for each degree of freedom.
        # Could move this to initialization?
        #
        self.addComputeTemperatureDependentConstants({"sigma": "sqrt(kT/m)"})

        #
        # Allow Context updating here, outside of inner loop only.
        #
        self.addUpdateContextState()

        #
        # Draw new velocity.
        #
        self.addComputePerDof("v", "sigma*gaussian")
        self.addConstrainVelocities()

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


class GHMCIntegrator(ThermostatedIntegrator):

    """
    Generalized hybrid Monte Carlo (GHMC) integrator.

    """

    def __init__(self, temperature=298.0 * simtk.unit.kelvin, collision_rate=91.0 / simtk.unit.picoseconds, timestep=1.0 * simtk.unit.femtoseconds):
        """
        Create a generalized hybrid Monte Carlo (GHMC) integrator.

        Parameters
        ----------
        temperature : simtk.unit.Quantity compatible with kelvin, default: 298*unit.kelvin
           The temperature.
        collision_rate : simtk.unit.Quantity compatible with 1/picoseconds, default: 91.0/unit.picoseconds
           The collision rate.
        timestep : simtk.unit.Quantity compatible with femtoseconds, default: 1.0*unit.femtoseconds
           The integration timestep.

        Notes
        -----
        This integrator is equivalent to a Langevin integrator in the velocity Verlet discretization with a
        Metrpolization step to ensure sampling from the appropriate distribution.

        Additional global variables 'ntrials' and  'naccept' keep track of how many trials have been attempted and
        accepted, respectively.

        TODO
        ----
        * Move initialization of 'sigma' to setting the per-particle variables.
        * Generalize to use MTS inner integrator.

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
        gamma = collision_rate

        # Create a new custom integrator.
        super(GHMCIntegrator, self).__init__(temperature, timestep)

        #
        # Integrator initialization.
        #
        self.addGlobalVariable("b", numpy.exp(-gamma * timestep))  # velocity mixing parameter
        self.addPerDofVariable("sigma", 0) # velocity standard deviation
        self.addGlobalVariable("ke", 0)  # kinetic energy
        self.addPerDofVariable("vold", 0)  # old velocities
        self.addPerDofVariable("xold", 0)  # old positions
        self.addGlobalVariable("Eold", 0)  # old energy
        self.addGlobalVariable("Enew", 0)  # new energy
        self.addGlobalVariable("potential_old", 0)  # old potential energy
        self.addGlobalVariable("potential_new", 0)  # new potential energy
        self.addGlobalVariable("accept", 0)  # accept or reject
        self.addGlobalVariable("naccept", 0)  # number accepted
        self.addGlobalVariable("ntrials", 0)  # number of Metropolization trials
        self.addPerDofVariable("x1", 0)  # position before application of constraints

        #
        # Initialization.
        #
        self.beginIfBlock("ntrials = 0")
        self.addComputePerDof("sigma", "sqrt(kT/m)")
        self.addConstrainPositions()
        self.addConstrainVelocities()
        self.endBlock()

        #
        # Allow context updating here.
        #
        self.addUpdateContextState()

        #
        # Velocity randomization
        #
        self.addComputePerDof("v", "sqrt(b)*v + sqrt(1-b)*sigma*gaussian")
        self.addConstrainVelocities()

        # Compute initial total energy
        self.addComputeSum("ke", "0.5*m*v*v")
        self.addComputeGlobal("potential_old", "energy")
        self.addComputeGlobal("Eold", "ke + potential_old")
        self.addComputePerDof("xold", "x")
        self.addComputePerDof("vold", "v")
        # Velocity Verlet step
        self.addComputePerDof("v", "v + 0.5*dt*f/m")
        self.addComputePerDof("x", "x + v*dt")
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions()
        self.addComputePerDof("v", "v + 0.5*dt*f/m + (x-x1)/dt")
        self.addConstrainVelocities()
        # Compute final total energy
        self.addComputeSum("ke", "0.5*m*v*v")
        self.addComputeGlobal("potential_new", "energy")
        self.addComputeGlobal("Enew", "ke + potential_new")
        # Accept/reject, ensuring rejection if energy is NaN
        self.addComputeGlobal("accept", "step(exp(-(Enew-Eold)/kT) - uniform)")
        self.beginIfBlock("accept != 1")
        self.addComputePerDof("x", "xold")
        self.addComputePerDof("v", "-vold")
        self.addComputeGlobal("potential_new", "potential_old")
        self.endBlock()

        #
        # Velocity randomization
        #
        self.addComputePerDof("v", "sqrt(b)*v + sqrt(1-b)*sigma*gaussian")
        self.addConstrainVelocities()

        #
        # Accumulate statistics.
        #
        self.addComputeGlobal("naccept", "naccept + accept")
        self.addComputeGlobal("ntrials", "ntrials + 1")
    
    def resetStatistics(self):
        """
        Reset the step counter and statistics

        """
        self.setGlobalVariableByName('ntrials', 0)
        self.setGlobalVariableByName('naccept', 0)

    def setTemperature(self, temperature):
        """Set the temperature of the heat bath.

        This also resets the trial statistics.

        Parameters
        ----------
        temperature : simtk.unit.Quantity
            The new temperature of the heat bath (temperature units).

        """
        super(GHMCIntegrator, self).setTemperature(temperature)
        # Reset statistics to ensure 'sigma' is updated on step 0
        self.resetStatistics()

class LangevinSplittingIntegrator(ThermostatedIntegrator):
    """Integrates Langevin dynamics with a prescribed operator splitting.

    One way to divide the Langevin system is into three parts which can each be solved "exactly:"
        - R: Linear "drift" / Constrained "drift"
            Deterministic update of *positions*, using current velocities
            x <- x + v dt

        - V: Linear "kick" / Constrained "kick"
            Deterministic update of *velocities*, using current forces
            v <- v + (f/m) dt
                where f = force, m = mass

        - O: Ornstein-Uhlenbeck
            Stochastic update of velocities, simulating interaction with a heat bath
            v <- av + b sqrt(kT/m) R
                where
                a = e^(-gamma dt)
                b = sqrt(1 - e^(-2gamma dt))
                R is i.i.d. standard normal

    We can then construct integrators by solving each part for a certain timestep in sequence.
    (We can further split up the V step by force group, evaluating cheap but fast-fluctuating
    forces more frequently than expensive but slow-fluctuating forces. Since forces are only
    evaluated in the V step, we represent this by including in our "alphabet" V0, V1, ...)

    When the system contains holonomic constraints, these steps are confined to the constraint
    manifold.

    Examples
    --------
        - VVVR
            splitting="O V R V O"
        - BAOAB:
            splitting="V R O R V"
        - g-BAOAB, with K_r=3:
            splitting="V R R R O R R R V"
        - g-BAOAB with solvent-solute splitting, K_r=K_p=2:
            splitting="V0 V1 R R O R R V1 R R O R R V1 V0"

    Attributes
    ----------
    _kinetic_energy : str
        This is 0.5*m*v*v by default, and is the expression used for the kinetic energy

    References
    ----------
    [Leimkuhler and Matthews, 2015] Molecular dynamics: with deterministic and stochastic numerical methods, Chapter 7
    """

    _kinetic_energy = "0.5 * m * v * v"

    def __init__(self,
                 splitting="V R O R V",
                 temperature=298.0 * simtk.unit.kelvin,
                 collision_rate=1.0 / simtk.unit.picoseconds,
                 timestep=1.0 * simtk.unit.femtoseconds,
                 constraint_tolerance=1e-8,
                 measure_shadow_work=False,
                 measure_heat=True,
                 ):
        """Create a Langevin integrator with the prescribed operator splitting.

        Parameters
        ----------
        splitting : string, default: "V R O R V"
            Sequence of "R", "V", "O" (and optionally "(", ")", "V0", "V1", ...) substeps to be executed each timestep.

            Forces are only used in V-step. Handle multiple force groups by appending the force group index
            to V-steps, e.g. "V0" will only use forces from force group 0. "V" will perform a step using all forces.
            "(" will cause metropolization, and must be followed later by a ")".


        temperature : numpy.unit.Quantity compatible with kelvin, default: 298.0*simtk.unit.kelvin
           Fictitious "bath" temperature

        collision_rate : numpy.unit.Quantity compatible with 1/picoseconds, default: 91.0/simtk.unit.picoseconds
           Collision rate

        timestep : numpy.unit.Quantity compatible with femtoseconds, default: 1.0*simtk.unit.femtoseconds
           Integration timestep

        constraint_tolerance : float, default: 1.0e-8
            Tolerance for constraint solver

        measure_shadow_work : boolean, default: False
            Accumulate the shadow work performed by the symplectic substeps, in the global `shadow_work`

        measure_heat : boolean, default: True
            Accumulate the heat exchanged with the bath in each step, in the global `heat`
        """

        # Compute constants
        gamma = collision_rate

        # Check if integrator is metropolized by checking for M step:
        if splitting.find("(") > -1:
            self._metropolized_integrator = True
            measure_shadow_work = True
        else:
            self._metropolized_integrator = False

        ORV_counts, mts, force_group_nV = self.parse_splitting_string(splitting)

        # Create a new CustomIntegrator
        super(LangevinSplittingIntegrator, self).__init__(temperature, timestep)

        # Initialize
        self.addPerDofVariable("sigma", 0)

        # Velocity mixing parameter: current velocity component
        h = timestep / max(1, ORV_counts['O'])
        self.addGlobalVariable("a", numpy.exp(-gamma * h))

        # Velocity mixing parameter: random velocity component
        self.addGlobalVariable("b", numpy.sqrt(1 - numpy.exp(- 2 * gamma * h)))

        # Positions before application of position constraints
        self.addPerDofVariable("x1", 0)

        # Set constraint tolerance
        self.setConstraintTolerance(constraint_tolerance)

        # Add bookkeeping variables
        if measure_heat:
            self.addGlobalVariable("heat", 0)

        if measure_shadow_work or measure_heat:
            self.addGlobalVariable("old_ke", 0)
            self.addGlobalVariable("new_ke", 0)

        if measure_shadow_work:
            self.addGlobalVariable("old_pe", 0)
            self.addGlobalVariable("new_pe", 0)
            self.addGlobalVariable("shadow_work", 0)

        # If we metropolize, we have to keep track of the before and after (x, v)
        if self._metropolized_integrator:
            self.addGlobalVariable("ntrials", 0)
            self.addGlobalVariable("nreject", 0)
            self.addGlobalVariable("naccept", 0)
            self.addPerDofVariable("vold", 0)
            self.addPerDofVariable("xold", 0)

        # Integrate
        self.addUpdateContextState()
        self.addComputeTemperatureDependentConstants({"sigma": "sqrt(kT/m)"})

        self.add_integrator_steps(splitting, measure_shadow_work, measure_heat, ORV_counts, force_group_nV, mts)

    def add_integrator_steps(self, splitting, measure_shadow_work, measure_heat, ORV_counts, force_group_nV, mts):
        """Add the steps to the integrator--this can be overridden to place steps around the integration.

        Parameters
        ----------
        splitting : str
            The Langevin splitting string
        measure_shadow_work : bool
            Whether to measure shadow work
        measure_heat : bool
            Whether to measure heat
        ORV_counts : dict
            Dictionary of occurrences of O, R, V
        force_group_nV : dict
            Dictionary of the number of Vs per force group
        mts : bool
            Whether this integrator defines an MTS integrator
        """
        for i, step in enumerate(splitting.split()):
            self.substep_function(step, measure_shadow_work, measure_heat, ORV_counts['R'], force_group_nV, mts)

    def sanity_check(self, splitting, allowed_characters="()RVO0123456789"):
        """Perform a basic sanity check on the splitting string to ensure that it makes sense.

        Parameters
        ----------
        splitting : str
            The string specifying the integrator splitting
        mts : bool
            Whether the integrator is a multiple timestep integrator
        allowed_characters : str, optional
            The characters allowed to be present in the splitting string.
            Default RVO and the digits 0-9.
        """

        # Space is just a delimiter--remove it
        splitting_no_space = splitting.replace(" ", "")

        # sanity check to make sure only allowed combinations are present in string:
        for step in splitting.split():
            if step[0]=="V":
                if len(step) > 1:
                    try:
                        force_group_number = int(step[1:])
                        if force_group_number > 31:
                            raise ValueError("OpenMM only allows up to 32 force groups")
                    except ValueError:
                        raise ValueError("You must use an integer force group")
            elif step == "(":
                    if ")" not in splitting:
                        raise ValueError("Use of { must be followed by }")
                    if not self.verify_metropolization(splitting):
                        raise ValueError("Shadow work generating steps found outside the Metropolization block")
            elif step in allowed_characters:
                continue
            else:
                raise ValueError("Invalid step name used")

        # Make sure we contain at least one of R, V, O steps
        assert ("R" in splitting_no_space)
        assert ("V" in splitting_no_space)
        assert ("O" in splitting_no_space)

    def verify_metropolization(self, splitting):
        """Verify that the shadow-work generating steps are all inside the metropolis block

        Returns False if they are not.

        Parameters
        ----------
        splitting : str
            The langevin splitting string

        Returns
        -------
        valid_metropolis : bool
            Whether all shadow-work generating steps are in the {} block
        """
        # check that there is exactly one metropolized region
        if splitting.count(")") != 1 or splitting.count("(") != 1:
            raise ValueError("There can only be one Metropolized region.")

        # find the metropolization steps:
        M_start_index = splitting.find("(")
        M_end_index = splitting.find(")")

        # accept/reject happens before the beginning of metropolis step
        if M_start_index > M_end_index:
            return False

        non_metropolis_string = splitting[:M_start_index] + splitting[M_end_index:]

        if "R" in non_metropolis_string or "V" in non_metropolis_string:
            return False
        else:
            return True



    def R_step(self, measure_shadow_work, n_R):
        """Add an R step (position update) given the velocities.

        Parameters
        ----------
        measure_shadow_work : bool
            Whether to compute the shadow work
        n_R : int
            Number of R steps in total (this determines the size of the timestep)
        """
        if measure_shadow_work:
            self.addComputeGlobal("old_pe", "energy")
            self.addComputeSum("old_ke", self._kinetic_energy)

        # update positions (and velocities, if there are constraints)
        self.addComputePerDof("x", "x + ((dt / {}) * v)".format(n_R))
        self.addComputePerDof("x1", "x")  # save pre-constraint positions in x1
        self.addConstrainPositions()  # x is now constrained
        self.addComputePerDof("v", "v + ((x - x1) / (dt / {}))".format(n_R))
        self.addConstrainVelocities()

        if measure_shadow_work:
            self.addComputeGlobal("new_pe", "energy")
            self.addComputeSum("new_ke", self._kinetic_energy)
            self.addComputeGlobal("shadow_work", "shadow_work + (new_ke + new_pe) - (old_ke + old_pe)")

    def V_step(self, fg, measure_shadow_work, force_group_nV, mts):
        """Deterministic velocity update, using only forces from force-group fg.

        Parameters
        ----------
        fg : string
            Force group to use in this substep.
            "" means all forces, "0" means force-group 0, etc.
        measure_shadow_work : bool
            Whether to compute shadow work
        force_group_nV : dict
            Number of V steps per integrator step per force group--used to compute per-V timestep.
            In non-MTS setting, this is just {0: nV}
        mts : bool
            Whether this integrator is a multiple timestep integrator
        """
        if measure_shadow_work:
            self.addComputeSum("old_ke", self._kinetic_energy)

        # update velocities
        if mts:
            self.addComputePerDof("v", "v + ((dt / {}) * f{} / m)".format(force_group_nV[fg], fg))
        else:
            self.addComputePerDof("v", "v + (dt / {}) * f / m".format(force_group_nV["0"]))

        self.addConstrainVelocities()

        if measure_shadow_work:
            self.addComputeSum("new_ke", self._kinetic_energy)
            self.addComputeGlobal("shadow_work", "shadow_work + (new_ke - old_ke)")

    def O_step(self, measure_heat):
        """Add an O step (stochastic velocity update)

        Parameters
        ----------
        measure_heat : bool
            Whether to compute the heat
        """

        if measure_heat:
            self.addComputeSum("old_ke", self._kinetic_energy)

        # update velocities
        self.addComputePerDof("v", "(a * v) + (b * sigma * gaussian)")
        self.addConstrainVelocities()

        if measure_heat:
            self.addComputeSum("new_ke", self._kinetic_energy)
            self.addComputeGlobal("heat", "heat + (new_ke - old_ke)")

    def substep_function(self, step_string, measure_shadow_work, measure_heat, n_R, force_group_nV, mts):
        """Take step string, and add the appropriate R, V, O step with appropriate parameters.

        The step string input here is a single character (or character + number, for MTS)

        Parameters
        ----------
        step_string : str
            R, O, V, {, }, or Vn (where n is a nonnegative integer specifying force group)
        measure_shadow_work : bool
            Whether the steps should measure shadow work
        measure_heat : bool
            Whether the O step should measure heat
        n_R : int
            The number of R steps per integrator step
        n_V : int
            The number of V steps per integrator step
        force_group_nV : dict
            The number of V steps per integrator step per force group. {0: nV} if not mts
        mts : bool
            Whether the integrator is a multiple timestep integrator
        """

        if step_string == "O":
            self.O_step(measure_heat)
        elif step_string == "R":
            self.R_step(measure_shadow_work, n_R)
        elif step_string == "(":
            self.addComputePerDof("xold", "x")
            self.addComputePerDof("vold", "v")
        elif step_string == ")":
            self.metropolize()
        elif step_string[0] == "V":
            # get the force group for this update--it's the number after the V
            force_group = step_string[1:]
            self.V_step(force_group, measure_shadow_work, force_group_nV, mts)

    def parse_splitting_string(self, splitting_string):
        """Parse the splitting string to check for simple errors and extract necessary information

        Parameters
        ----------
        splitting_string : str
            The string that specifies how to do the integrator splitting

        Returns
        -------
        ORV_counts : dict
            Number of O, R, and V steps
        mts : bool
            Whether the splitting specifies an MTS integrator
        force_group_n_V : dict
            Specifies the number of V steps per force group. {"0": nV} if not MTS
        """
        # convert the string to all caps
        splitting_string = splitting_string.upper()

        # sanity check the splitting string
        self.sanity_check(splitting_string)

        ORV_counts = dict()

        # count number of R, V, O steps:
        ORV_counts["R"] = splitting_string.count("R")
        ORV_counts["V"] = splitting_string.count("V")
        ORV_counts["O"] = splitting_string.count("O")

        # split by delimiter (space)
        step_list = splitting_string.split(" ")

        # populate a list with all the force groups in the system
        force_group_list = []
        for step in step_list:
            # if the length of the step is greater than one, it has a digit after it
            if step[0] == "V" and len(step) > 1:
                force_group_list.append(step[1:])

        # Make a set to count distinct force groups
        force_group_set = set(force_group_list)

        # check if force group list cast to set is longer than one
        # If it is, then multiple force groups are specified
        if len(force_group_set) > 1:
            mts = True
        else:
            mts = False


        # If the integrator is MTS, count how many times the V steps appear for each
        if mts:
            force_group_n_V = {force_group: 0 for force_group in force_group_set}
            for step in step_list:
                if step[0] == "V":
                    # ensure that there are no V-all steps if it's MTS
                    assert len(step) > 1
                    # extract the index of the force group from the step
                    force_group_idx = step[1:]
                    # increment the number of V calls for that force group
                    force_group_n_V[force_group_idx] += 1
        else:
            force_group_n_V = {"0": ORV_counts["V"]}

        return ORV_counts, mts, force_group_n_V

    def metropolize(self):
        """Add a Metropolization (based on shadow work) step to the integrator.

        When Metropolization occurs, shadow work is reset.
        """
        self.addComputeGlobal("accept", "step(exp(-(shadow_work)/kT) - uniform)")
        self.addComputeGlobal("ntrials", "ntrials + 1")
        self.beginIfBlock("accept != 1")
        self.addComputePerDof("x", "xold")
        self.addComputePerDof("v", "-vold")
        self.addComputeGlobal("nreject", "nreject + 1")
        self.endBlock()
        self.addComputeGlobal("naccept", "ntrials - nreject")
        self.addComputeGlobal("shadow_work", 0)

    def begin_metropolize(self):
        """Save the current x and v for a metropolization step later"""
        self.addComputePerDof("xold", "x")
        self.addComputePerDof("vold", "v")

class AlchemicalLangevinSplittingIntegrator(LangevinSplittingIntegrator):
    """Allows nonequilibrium switching based on force parameters specified in alchemical_functions.
    Propagator is based on Langevin splitting, as described below.

    One way to divide the Langevin system is into three parts which can each be solved "exactly:"
        - R: Linear "drift" / Constrained "drift"
            Deterministic update of *positions*, using current velocities
            x <- x + v dt

        - V: Linear "kick" / Constrained "kick"
            Deterministic update of *velocities*, using current forces
            v <- v + (f/m) dt
                where f = force, m = mass

        - O: Ornstein-Uhlenbeck
            Stochastic update of velocities, simulating interaction with a heat bath
            v <- av + b sqrt(kT/m) R
                where
                a = e^(-gamma dt)
                b = sqrt(1 - e^(-2gamma dt))
                R is i.i.d. standard normal

    We can then construct integrators by solving each part for a certain timestep in sequence.
    (We can further split up the V step by force group, evaluating cheap but fast-fluctuating
    forces more frequently than expensive but slow-fluctuating forces. Since forces are only
    evaluated in the V step, we represent this by including in our "alphabet" V0, V1, ...)

    When the system contains holonomic constraints, these steps are confined to the constraint
    manifold.

    Examples
    --------
        - VVVR
            splitting="O V R V O"
        - BAOAB:
            splitting="V R O R V"
        - g-BAOAB, with K_r=3:
            splitting="V R R R O R R R V"
        - g-BAOAB with solvent-solute splitting, K_r=K_p=2:
            splitting="V0 V1 R R O R R V1 R R O R R V1 V0"

    Attributes
    ----------
    _kinetic_energy : str
        This is 0.5*m*v*v by default, and is the expression used for the kinetic energy

    References
    ----------
    [Nilmeier, et al. 2011] Nonequilibrium candidate Monte Carlo is an efficient tool for equilibrium simulation
    [Leimkuhler and Matthews, 2015] Molecular dynamics: with deterministic and stochastic numerical methods, Chapter 7
    """

    def __init__(self,
                 alchemical_functions,
                 splitting="V R O R V",
                 temperature=298.0 * simtk.unit.kelvin,
                 collision_rate=1.0 / simtk.unit.picoseconds,
                 timestep=1.0 * simtk.unit.femtoseconds,
                 constraint_tolerance=1e-8,
                 measure_shadow_work=False,
                 measure_heat=True,
                 direction="forward",
                 nsteps_neq=100):
        """
        Parameters
        ----------
        alchemical_functions : dict of strings
            key: value pairs such as "global_parameter" : function_of_lambda where function_of_lambda is a Lepton-compatible
            string that depends on the variable "lambda"

        splitting : string, default: "V R O R V"
            Sequence of R, V, O (and optionally V{i}), and { }substeps to be executed each timestep.

            Forces are only used in V-step. Handle multiple force groups by appending the force group index
            to V-steps, e.g. "V0" will only use forces from force group 0. "V" will perform a step using all forces.
            ( will cause metropolization, and must be followed later by a ).

        temperature : numpy.unit.Quantity compatible with kelvin, default: 298.0*simtk.unit.kelvin
           Fictitious "bath" temperature

        collision_rate : numpy.unit.Quantity compatible with 1/picoseconds, default: 91.0/simtk.unit.picoseconds
           Collision rate

        timestep : numpy.unit.Quantity compatible with femtoseconds, default: 1.0*simtk.unit.femtoseconds
           Integration timestep

        constraint_tolerance : float, default: 1.0e-8
            Tolerance for constraint solver

        measure_shadow_work : boolean, default: False
            Accumulate the shadow work performed by the symplectic substeps, in the global `shadow_work`

        measure_heat : boolean, default: True
            Accumulate the heat exchanged with the bath in each step, in the global `heat`

        direction : str, default: "forward"
            Whether to move the global lambda parameter from 0 to 1 (forward) or 1 to 0 (reverse).

        nsteps_neq : int, default: 100
            Number of steps in nonequilibrium protocol. Default 100
        """

        self._alchemical_functions = alchemical_functions
        self._direction = direction
        self._n_steps_neq = nsteps_neq

        # add some global variables relevant to the integrator
        self.addGlobalVariable('lambda', 0.0) # parameter switched from 0 <--> 1 during course of integrating internal 'nsteps' of dynamics
        self.addGlobalVariable('kinetic', 0.0) # kinetic energy
        self.addGlobalVariable('nsteps', self._n_steps_neq) # total number of NCMC steps to perform
        self.addGlobalVariable('step', 0) # current NCMC step number

        # collect the system parameters.
        self._system_parameters = {system_parameter for system_parameter in alchemical_functions.keys()}

        # call the base class constructor
        super(AlchemicalLangevinSplittingIntegrator, self).__init__(splitting=splitting, temperature=temperature,
                                                                    collision_rate=collision_rate, timestep=timestep,
                                                                    constraint_tolerance=constraint_tolerance,
                                                                    measure_shadow_work=measure_shadow_work,
                                                                    measure_heat=measure_heat,
                                                                    )

    def update_alchemical_parameters_step(self):
        """
        Update Context parameters according to provided functions.
        """
        for context_parameter in self._alchemical_functions:
            if context_parameter in self._system_parameters:
                self.addComputeGlobal(context_parameter, self._alchemical_functions[context_parameter])

    def alchemical_perturbation_step(self):
        """
        Add alchemical perturbation step, accumulating protocol work.
        """
        # Store initial potential energy
        self.addComputeGlobal("Eold", "energy")

        # Use fractional state
        if self._direction == 'forward':
            self.addComputeGlobal('lambda', '(step+1)/nsteps')
        elif self._direction == 'reverse':
            self.addComputeGlobal('lambda', '(nsteps - step - 1)/nsteps')

        # Update all slaved alchemical parameters
        self.update_alchemical_parameters_step()

        # Accumulate protocol work
        self.addComputeGlobal("Enew", "energy")
        self.addComputeGlobal("protocol_work", "protocol_work + (Enew-Eold)/kT")

    def sanity_check(self, splitting, allowed_characters="H()RVO0123456789"):
        super(AlchemicalLangevinSplittingIntegrator, self).sanity_check(splitting, allowed_characters=allowed_characters)

    def substep_function(self, step_string, measure_shadow_work, measure_heat, n_R, force_group_nV, mts):
        """Take step string, and add the appropriate R, V, O, M, or H step with appropriate parameters.

        The step string input here is a single character (or character + number, for MTS)

        Parameters
        ----------
        step_string : str
            R, O, V, or Vn (where n is a nonnegative integer specifying force group)
        measure_shadow_work : bool
            Whether the steps should measure shadow work
        measure_heat : bool
            Whether the O step should measure heat
        n_R : int
            The number of R steps per integrator step
        force_group_nV : dict
            The number of V steps per integrator step per force group. {0: nV} if not mts
        mts : bool
            Whether the integrator is a multiple timestep integrator
        """

        if step_string == "O":
            self.O_step(measure_heat)
        elif step_string == "R":
            self.R_step(measure_shadow_work, n_R)
        elif step_string == "(":
            self.addComputePerDof("xold", "x")
            self.addComputePerDof("vold", "v")
        elif step_string == ")":
            self.metropolize()
        elif step_string[0] == "V":
            # get the force group for this update--it's the number after the V
            force_group = step_string[1:]
            self.V_step(force_group, measure_shadow_work, force_group_nV, mts)
        elif step_string == "H":
            self.alchemical_perturbation_step()

    def addGlobalVariables(self, nsteps):
        """Add the appropriate global parameters to the CustomIntegrator. nsteps refers to the number of
        total steps in the protocol.

        Parameters
        ----------
        nsteps : int, greater than 0
            The number of steps in the switching protocol.
        """
        self.addGlobalVariable('lambda', 0.0) # parameter switched from 0 <--> 1 during course of integrating internal 'nsteps' of dynamics
        self.addGlobalVariable('kinetic', 0.0) # kinetic energy
        self.addGlobalVariable('nsteps', nsteps) # total number of NCMC steps to perform
        self.addGlobalVariable('step', 0) # current NCMC step number

class ExternalPerturbationLangevinSplittingIntegrator(LangevinSplittingIntegrator):
    """LangevinSplittingIntegrator that accounts for external perturbations and tracks protocol work."""

    def __init__(self,
                 splitting="V R O R V",
                 temperature=298.0 * simtk.unit.kelvin,
                 collision_rate=1.0 / simtk.unit.picoseconds,
                 timestep=1.0 * simtk.unit.femtoseconds,
                 constraint_tolerance=1e-8,
                 measure_shadow_work=False,
                 measure_heat=True):


        super(ExternalPerturbationLangevinSplittingIntegrator, self).__init__(splitting=splitting,
                                                                              temperature=temperature,
                                                                              collision_rate=collision_rate,
                                                                              timestep=timestep,
                                                                              constraint_tolerance=constraint_tolerance,
                                                                              measure_shadow_work=measure_shadow_work,
                                                                              measure_heat=measure_heat)

        self.addGlobalVariable("protocol_work", 0)
        self.addGlobalVariable("perturbed_pe", 0)
        self.addGlobalVariable("unperturbed_pe", 0)

    def add_integrator_steps(self, splitting, measure_shadow_work, measure_heat, ORV_counts, force_group_nV, mts):
        self.addComputeGlobal("perturbed_pe", "energy")
        self.addComputeGlobal("protocol_work", "protocol_work + (perturbed_pe - unperturbed_pe)")
        super(ExternalPerturbationLangevinSplittingIntegrator, self).add_integrator_steps(splitting, measure_shadow_work, measure_heat, ORV_counts, force_group_nV, mts)
        self.addComputeGlobal("unperturbed_pe", "energy")

class VVVRIntegrator(LangevinSplittingIntegrator):
    """Create a velocity Verlet with velocity randomization (VVVR) integrator."""
    def __init__(self,
                 temperature=298.0 * simtk.unit.kelvin,
                 collision_rate=91.0 / simtk.unit.picoseconds,
                 timestep=1.0 * simtk.unit.femtoseconds,
                 constraint_tolerance=1e-8,
                 measure_shadow_work=False,
                 measure_heat=True,
                 ):
        """Create a velocity verlet with velocity randomization (VVVR) integrator.
        -----
        This integrator is equivalent to a Langevin integrator in the velocity Verlet discretization with a
        timestep correction to ensure that the field-free diffusion constant is timestep invariant.
        The global 'heat' keeps track of the heat accumulated during integration, and can be
        used to correct the sampled statistics or in a Metropolization scheme.

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
        LangevinSplittingIntegrator.__init__(self, splitting="O V R V O",
                                             temperature=temperature,
                                             collision_rate=collision_rate,
                                             timestep=timestep,
                                             constraint_tolerance=constraint_tolerance,
                                             measure_shadow_work=measure_shadow_work,
                                             measure_heat=measure_heat,
                                             )

class BAOABIntegrator(LangevinSplittingIntegrator):
    """Create a velocity Verlet with velocity randomization (VVVR) integrator."""
    def __init__(self,
                 temperature=298.0 * simtk.unit.kelvin,
                 collision_rate=1.0 / simtk.unit.picoseconds,
                 timestep=1.0 * simtk.unit.femtoseconds,
                 constraint_tolerance=1e-8,
                 measure_shadow_work=False,
                 measure_heat=True
                 ):
        """Create an integrator of Langevin dynamics using the BAOAB operator splitting.

        Parameters
        ----------
        temperature : numpy.unit.Quantity compatible with kelvin, default: 298.0*simtk.unit.kelvin
           Fictitious "bath" temperature

        collision_rate : numpy.unit.Quantity compatible with 1/picoseconds, default: 91.0/simtk.unit.picoseconds
           Collision rate

        timestep : numpy.unit.Quantity compatible with femtoseconds, default: 1.0*simtk.unit.femtoseconds
           Integration timestep

        constraint_tolerance : float, default: 1.0e-8
            Tolerance for constraint solver

        measure_shadow_work : boolean, default: False
            Accumulate the shadow work performed by the symplectic substeps, in the global `shadow_work`

        measure_heat : boolean, default: True
            Accumulate the heat exchanged with the bath in each step, in the global `heat`

        References
        ----------
        [Leimkuhler and Matthews, 2013] Rational construction of stochastic numerical methods for molecular sampling
        https://academic.oup.com/amrx/article-abstract/2013/1/34/166771/Rational-Construction-of-Stochastic-Numerical

        Examples
        --------
        Create a BAOAB integrator.
        >>> temperature = 298.0 * simtk.unit.kelvin
        >>> collision_rate = 91.0 / simtk.unit.picoseconds
        >>> timestep = 1.0 * simtk.unit.femtoseconds
        >>> integrator = BAOABIntegrator(temperature, collision_rate, timestep)
        """
        LangevinSplittingIntegrator.__init__(self, splitting="V R O R V",
                                             temperature=temperature,
                                             collision_rate=collision_rate,
                                             timestep=timestep,
                                             constraint_tolerance=constraint_tolerance,
                                             measure_shadow_work=measure_shadow_work,
                                             measure_heat=measure_heat
                                             )

class GeodesicBAOABIntegrator(LangevinSplittingIntegrator):
    """Create a geodesic-BAOAB integrator."""

    def __init__(self, K_r=2,
                 temperature=298.0 * simtk.unit.kelvin,
                 collision_rate=1.0 / simtk.unit.picoseconds,
                 timestep=1.0 * simtk.unit.femtoseconds,
                 constraint_tolerance=1e-8,
                 measure_shadow_work=False,
                 measure_heat=True
                 ):
        """Create a geodesic BAOAB Langevin integrator.

        Parameters
        ----------
        K_r : integer, default: 2
            Number of geodesic drift steps.

        temperature : numpy.unit.Quantity compatible with kelvin, default: 298.0*simtk.unit.kelvin
           Fictitious "bath" temperature

        collision_rate : numpy.unit.Quantity compatible with 1/picoseconds, default: 91.0/simtk.unit.picoseconds
           Collision rate

        timestep : numpy.unit.Quantity compatible with femtoseconds, default: 1.0*simtk.unit.femtoseconds
           Integration timestep

        constraint_tolerance : float, default: 1.0e-8
            Tolerance for constraint solver

        measure_shadow_work : boolean, default: False
            Accumulate the shadow work performed by the symplectic substeps, in the global `shadow_work`

        measure_heat : boolean, default: True
            Accumulate the heat exchanged with the bath in each step, in the global `heat`

        References
        ----------
        [Leimkuhler and Matthews, 2016] Efficient molecular dynamics using geodesic integration and solvent-solute splitting
        http://rspa.royalsocietypublishing.org/content/472/2189/20160138

        Examples
        --------
        Create a geodesic BAOAB integrator.
        >>> temperature = 298.0 * simtk.unit.kelvin
        >>> collision_rate = 91.0 / simtk.unit.picoseconds
        >>> timestep = 1.0 * simtk.unit.femtoseconds
        >>> integrator = GeodesicBAOABIntegrator(K_r=3, temperature=temperature, collision_rate=collision_rate, timestep=timestep)
        """
        splitting = " ".join(["V"] + ["R"] * K_r + ["O"] + ["R"] * K_r + ["V"])
        LangevinSplittingIntegrator.__init__(self, splitting=splitting,
                                             temperature=temperature,
                                             collision_rate=collision_rate,
                                             timestep=timestep,
                                             constraint_tolerance=constraint_tolerance,
                                             measure_shadow_work=measure_shadow_work,
                                             measure_heat=measure_heat
                                             )

class GHMCIntegrator(LangevinSplittingIntegrator):

    def __init__(self, temperature=298.0 * simtk.unit.kelvin, collision_rate=1.0 / simtk.unit.picoseconds,
                 timestep=1.0 * simtk.unit.femtoseconds, constraint_tolerance=1.0e-8, measure_shadow_work=False,
                 measure_heat=True):
        """
        Create a generalized hybrid Monte Carlo (GHMC) integrator.

        Parameters
        ----------
        temperature : simtk.unit.Quantity compatible with kelvin, default: 298*unit.kelvin
           The temperature.
        collision_rate : simtk.unit.Quantity compatible with 1/picoseconds, default: 91.0/unit.picoseconds
           The collision rate.
        timestep : simtk.unit.Quantity compatible with femtoseconds, default: 1.0*unit.femtoseconds
           The integration timestep.

        Notes
        -----
        This integrator is equivalent to a Langevin integrator in the velocity Verlet discretization with a
        Metrpolization step to ensure sampling from the appropriate distribution.

        Additional global variables 'ntrials' and  'naccept' keep track of how many trials have been attempted and
        accepted, respectively.

        TODO
        ----
        * Move initialization of 'sigma' to setting the per-particle variables.
        * Generalize to use MTS inner integrator.

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

        super(GHMCIntegrator, self).__init__(splitting="", temperature=temperature, collision_rate=collision_rate,
                                             timestep=timestep,
                                             constraint_tolerance=constraint_tolerance,
                                             measure_shadow_work=measure_shadow_work, measure_heat=measure_heat)
if __name__ == '__main__':
    import doctest
    doctest.testmod()
