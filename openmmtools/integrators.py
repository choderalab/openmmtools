# -*- coding: UTF-8 -*-
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

All code in this repository is released under the MIT License.

This program is free software: you can redistribute it and/or modify it under
the terms of the MIT License.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the MIT License for more details.

You should have received a copy of the MIT License along with this program.

"""

# ============================================================================================
# GLOBAL IMPORTS
# ============================================================================================

import logging
import re

import numpy as np
import simtk.unit as unit
import simtk.openmm as mm

from openmmtools.constants import kB
from openmmtools import respa, utils

logger = logging.getLogger(__name__)

# Energy unit used by OpenMM unit system
_OPENMM_ENERGY_UNIT = unit.kilojoules_per_mole

# ============================================================================================
# BASE CLASSES
# ============================================================================================

class PrettyPrintableIntegrator(object):
    """A PrettyPrintableIntegrator can format the contents of its step program for printing.

    This is a mix-in.

    TODO: We should check that the object (`self`) is a CustomIntegrator or subclass.

    """
    def pretty_format(self, as_list=False, step_types_to_highlight=None):
        """Generate a human-readable version of each integrator step.

        Parameters
        ----------
        as_list : bool, optional, default=False
           If True, a list of human-readable strings will be returned.
           If False, these will be concatenated into a single human-readable string.
        step_types_to_highlight : list of int, optional, default=None
           If specified, these step types will be highlighted.

        Returns
        -------
        readable_lines : list of str
           A list of human-readable versions of each step of the integrator
        """
        step_type_dict = {
            0 : "{target} <- {expr}",
            1: "{target} <- {expr}",
            2: "{target} <- sum({expr})",
            3: "constrain positions",
            4: "constrain velocities",
            5: "allow forces to update the context state",
            6: "if({expr}):",
            7: "while({expr}):",
            8: "end"
        }

        if not hasattr(self, 'getNumComputations'):
            raise Exception('This integrator is not a CustomIntegrator.')

        readable_lines = []
        indent_level = 0
        for step in range(self.getNumComputations()):
            line = ''
            step_type, target, expr = self.getComputationStep(step)
            highlight = True if (step_types_to_highlight is not None) and (step_type in step_types_to_highlight) else False
            if step_type in [8]:
                indent_level -= 1
            if highlight:
                line += '\x1b[6;30;42m'
            line += 'step {:6d} : '.format(step) + '   ' * indent_level + step_type_dict[step_type].format(target=target, expr=expr)
            if highlight:
                line += '\x1b[0m'
            if step_type in [6, 7]:
                indent_level += 1
            readable_lines.append(line)

        if as_list:
            return readable_lines
        else:
            return '\n'.join(readable_lines)

    def pretty_print(self):
        """Pretty-print the computation steps of this integrator."""
        print(self.pretty_format())


class ThermostatedIntegrator(utils.RestorableOpenMMObject, PrettyPrintableIntegrator,
                             mm.CustomIntegrator):
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
    temperature : unit.Quantity
        The temperature of the integrator heat bath (temperature units).
    timestep : unit.Quantity
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

    Notice that a CustomIntegrator loses any extra method after a serialization cycle.

    >>> integrator_serialization = openmm.XmlSerializer.serialize(integrator)
    >>> deserialized_integrator = openmm.XmlSerializer.deserialize(integrator_serialization)
    >>> deserialized_integrator.getTemperature()
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

    @property
    def global_variable_names(self):
        """The set of global variable names defined for this integrator."""
        return set([ self.getGlobalVariableName(index) for index in range(self.getNumGlobalVariables()) ])

    def getTemperature(self):
        """Return the temperature of the heat bath.

        Returns
        -------
        temperature : unit.Quantity
            The temperature of the heat bath in kelvins.

        """
        kT = self.getGlobalVariableByName('kT') * _OPENMM_ENERGY_UNIT
        temperature = kT / kB
        return temperature

    def setTemperature(self, temperature):
        """Set the temperature of the heat bath.

        Parameters
        ----------
        temperature : unit.Quantity
            The new temperature of the heat bath (temperature units).

        """
        kT = kB * temperature
        self.setGlobalVariableByName('kT', kT)

        # Update the changed flag if it exist.
        if 'has_kT_changed' in self.global_variable_names:
            self.setGlobalVariableByName('has_kT_changed', 1)

    def addComputeTemperatureDependentConstants(self, compute_per_dof):
        """Wrap the ComputePerDof into an if-block executed only when kT changes.

        Parameters
        ----------
        compute_per_dof : dict of str: str
            A dictionary of variable_name: expression.

        """
        # First check if flag variable already exist.
        if not 'has_kT_changed' in self.global_variable_names:
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
        global_variable_names = set([ integrator.getGlobalVariableName(index) for index in range(integrator.getNumGlobalVariables()) ])
        if not 'kT' in global_variable_names:
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
            if hasattr(integrator, 'getGlobalVariableName'):
                global_variable_names = set([ integrator.getGlobalVariableName(index) for index in range(integrator.getNumGlobalVariables()) ])
                if 'kT' in global_variable_names:
                    if not hasattr(integrator, 'getTemperature'):
                        logger.warning("The integrator {} has a global variable 'kT' variable "
                                       "but does not expose getter and setter for the temperature. "
                                       "Consider inheriting from ThermostatedIntegrator.")
        return restored

    @property
    def kT(self):
        """The thermal energy in simtk.openmm.Quantity"""
        return self.getGlobalVariableByName("kT") * _OPENMM_ENERGY_UNIT


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

    >>> integrator = MTSIntegrator(4*unit.femtoseconds, [(0,1), (1,2), (2,8)])

    This specifies that the outermost time step is 4 fs, so each step of the integrator
    will advance time by that much.  It also says that force group 0 should be evaluated
    once per time step, force group 1 should be evaluated twice per time step (every 2 fs),
    and force group 2 should be evaluated eight times per time step (every 0.5 fs).

    For details, see Tuckerman et al., J. Chem. Phys. 97(3) pp. 1990-2001 (1992).

    """

    def __init__(self, timestep=1.0 * unit.femtoseconds, groups=[(0, 1)]):
        """Create an MTSIntegrator.

        Parameters
        ----------
        timestep : unit.Quantity with units compatible with femtoseconds, optional default=1*femtoseconds
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
        timestep = 0.0 * unit.femtoseconds
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

    def __init__(self, initial_step_size=0.01 * unit.angstroms):
        """
        Construct a simple gradient descent minimization integrator.

        Parameters
        ----------
        initial_step_size : np.unit.Quantity compatible with nanometers, default: 0.01*unit.angstroms
           The norm of the initial step size guess.

        Notes
        -----
        An adaptive step size is used.

        """

        timestep = 1.0 * unit.femtoseconds
        super(GradientDescentMinimizationIntegrator, self).__init__(timestep)

        self.addGlobalVariable("step_size", initial_step_size / unit.nanometers)
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

    >>> timestep = 1.0 * unit.femtoseconds
    >>> integrator = VelocityVerletIntegrator(timestep)

    """

    def __init__(self, timestep=1.0 * unit.femtoseconds):
        """Construct a velocity Verlet integrator.

        Parameters
        ----------
        timestep : np.unit.Quantity compatible with femtoseconds, default: 1*unit.femtoseconds
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

    >>> timestep = 1.0 * unit.femtoseconds
    >>> collision_rate = 91.0 / unit.picoseconds
    >>> temperature = 298.0 * unit.kelvin
    >>> integrator = AndersenVelocityVerletIntegrator(temperature, collision_rate, timestep)

    Notes
    ------
    The velocity Verlet integrator is taken verbatim from Peter Eastman's example in the CustomIntegrator header file documentation.
    The efficiency could be improved by avoiding recomputation of sigma_v every timestep.

    """

    def __init__(self, temperature=298 * unit.kelvin, collision_rate=91.0 / unit.picoseconds, timestep=1.0 * unit.femtoseconds):
        """Construct a velocity Verlet integrator with Andersen thermostat, implemented as per-particle collisions (rather than massive collisions).

        Parameters
        ----------
        temperature : np.unit.Quantity compatible with kelvin, default=298*unit.kelvin
           The temperature of the fictitious bath.
        collision_rate : np.unit.Quantity compatible with 1/picoseconds, default=91/unit.picoseconds
           The collision rate with fictitious bath particles.
        timestep : np.unit.Quantity compatible with femtoseconds, default=1*unit.femtoseconds
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


class NoseHooverChainVelocityVerletIntegrator(ThermostatedIntegrator):

    """Nosé-Hoover chain thermostat, using the reversible multi time step velocity Verlet algorithm
       detailed in the papers below.

    References
    ----------

    G. J. Martyna, M. E. Tuckerman, D. J. Tobias, and Michael L. Klein  "Explicit reversible
    integrators for extended systems dynamics", Molecular Physics, 87 1117-1157 (1996)
    http://dx.doi.org/10.1080/00268979600100761

    G. J. Martyna, M. L. Klein, and M. Tuckerman "Nosé–Hoover chains: The canonical
    ensemble via continuous dynamics",
    Journal of Chemical Physics 97, 2635-2643 (1992)
    http://dx.doi.org/10.1063/1.463940

    Examples
    --------

    Create a velocity Verlet integrator with Nosé-Hoover chain thermostat.

    >>> from openmmtools import testsystems
    >>> waterbox = testsystems.WaterBox()
    >>> system = waterbox.system
    >>> timestep = 1.0 * unit.femtoseconds
    >>> temperature = 300 * unit.kelvin
    >>> chain_length = 10
    >>> collision_frequency = 50 / unit.picoseconds
    >>> num_mts = 5
    >>> num_yoshidasuzuki = 5

    >>> integrator = NoseHooverChainVelocityVerletIntegrator(system, temperature, collision_frequency, timestep, chain_length, num_mts, num_yoshidasuzuki)

    Notes
    ------

    The velocity Verlet integrator is taken verbatim from Peter Eastman's example in the CustomIntegrator header file documentation.

    Useful tests of the NHC integrator can be performed by monitoring the instantaneous temperature during the simulation and confirming that conserved energy is constant to about 1 part in 10^5.  The instantanous temperature and particle kinetic and potential energies can already be extracted from a snapshot, for example see the OpenMM StateDataReporter implementation for more details.  This integrator also provides heat bath energies (kJ/mol) through the following mechanism:

    >>> waterbox = testsystems.WaterBox()
    >>> system = waterbox.system
    >>> integrator = NoseHooverChainVelocityVerletIntegrator(system)
    >>> heat_bath_kinetic_energy = integrator.getGlobalVariableByName('bathKE')
    >>> heat_bath_potential_energy = integrator.getGlobalVariableByName('bathPE')

    From here, the conserved energy is the sum of the potential and kinetic energies for both the system and the heat bath.

    """


    YSWeights = {
        1 : [ 1.0000000000000000 ],
        3 : [ 0.8289815435887510, -0.6579630871775020,  0.8289815435887510 ],
        5 : [ 0.2967324292201065,  0.2967324292201065, -0.1869297168804260, 0.2967324292201065, 0.2967324292201065 ]
    }

    def __init__(self, system=None, temperature=298*unit.kelvin, collision_frequency=50/unit.picoseconds,
                 timestep=0.001*unit.picoseconds, chain_length=5, num_mts=5, num_yoshidasuzuki=5):
        """ Construct a velocity Verlet integrator with Nosé-Hoover chain thermostat implemented with massive collisions.

        Parameters:
        -----------

        system: openmm.app.System instance
            Required to extract the system's number of degrees of freedom.
            If the system is not passed to the constructor and has constraints
            or a ``CMMotionRemover`` force, the temperature will converge to the
            wrong value.

        temperature: unit.Quantity compatible with kelvin, default=298*unit.kelvin
            The target temperature for the thermostat.

        collision_freqency: unit.Quantity compatible with picoseconds**-1, default=50/unit.picoseconds
            The frequency of collisions with the heat bath.  A very small value will result
            in a distribution approaching the microcanonical ensemble, while a large value will
            cause rapid fluctuations in the temperature before convergence.

        timestep: unit.Quantity compatible with femtoseconds, default=1*unit.femtoseconds
            The integration timestep for particles.

        chain_length: integer, default=5
            The number of thermostat particles in the Nosé-Hoover chain.  Increasing
            this parameter will affect computational cost, but will make the simulation
            less sensitive to thermostat parameters, particularly in stiff systems; see
            the 1992 paper referenced above for more details.

        num_mts: integer, default=5
            The number of timesteps used in the multi-timestep procedure to update the thermostat
            positions.  A higher value will increase the stability of the dynamics, but will also
            increase the compuational cost of the integration.

        num_yoshidasuzuki: integer, default=5
            The number of Yoshida-Suzuki steps used to subdivide each of the multi-timesteps used
            to update the thermostat positions.  A higher value will increase the stability of the
            dynamics, but will also increase the computational cost of the integration; only certain
            values (currently 1,3, or 5) are supported, because weights must be defined.

        """
        super(NoseHooverChainVelocityVerletIntegrator, self).__init__(temperature, timestep)
        #
        # Integrator initialization.
        #
        self.n_c         = num_mts
        self.n_ys        = num_yoshidasuzuki
        try:
            self.weights     = self.YSWeights[self.n_ys]
        except KeyError:
            raise Exception("Invalid Yoshida-Suzuki value. Allowed values are: %s"%
                             ",".join(map(str,self.YSWeights.keys())))
        if chain_length < 0:
            raise Exception("Nosé-Hoover chain length must be at least 0")
        if chain_length == 0:
            logger.warning('Nosé-Hoover chain length is 0; falling back to regular velocity verlet algorithm.')
        self.M           = chain_length

        # Define the "mass" of the thermostat particles (multiply by ndf for particle 0)
        kT = self.getGlobalVariableByName('kT')
        frequency = collision_frequency.value_in_unit(unit.picoseconds**-1)
        Q = kT/frequency**2

        #
        # Compute the number of degrees of freedom.
        #
        if system is None:
            logger.warning('The system was not passed to the NoseHooverChainVelocityVerletIntegrator. '
                           'For systems with constraints, the simulation will run at the wrong temperature.')
            # Fall back to old scheme, which only works for unconstrained systems
            self.addGlobalVariable("ndf", 0)
            self.addPerDofVariable("ones", 1.0)
            self.addComputeSum("ndf", "ones")
        else:
            # same as in openmm.app.StateDataReporter._initializeConstants
            dof = 0
            for i in range(system.getNumParticles()):
                if system.getParticleMass(i) > 0*unit.dalton:
                    dof += 3
            dof -= system.getNumConstraints()
            if any(type(system.getForce(i)) == mm.CMMotionRemover for i in range(system.getNumForces())):
                dof -= 3

            self.addGlobalVariable("ndf", dof)      # number of degrees of freedom

        #
        # Define global variables
        #
        self.addGlobalVariable("bathKE", 0.0) # Thermostat bath kinetic energy
        self.addGlobalVariable("bathPE", 0.0) # Thermostat bath potential energy
        self.addGlobalVariable("KE2", 0.0)    # Twice the kinetic energy
        self.addGlobalVariable("Q", Q)        # Thermostat particle "mass"
        self.addGlobalVariable("scale", 1.0)
        self.addGlobalVariable("aa", 0.0)
        self.addGlobalVariable("wdt", 0.0)
        for w in range(self.n_ys):
            self.addGlobalVariable("w{}".format(w), self.weights[w])

        #
        # Initialize thermostat parameters
        #
        for i in range(self.M):
            self.addGlobalVariable("xi{}".format(i), 0)            # Thermostat particle
            self.addGlobalVariable("vxi{}".format(i), 0)           # Thermostat particle velocities in ps^-1
            self.addGlobalVariable("G{}".format(i), -frequency**2) # Forces on thermostat particles in ps^-2
            self.addGlobalVariable("Q{}".format(i), 0)             # Thermostat "masses" in ps^2 kJ/mol
        # The masses need the number of degrees of freedom, which is approximated here.  Need a
        # better solution eventually, to properly account for constraints, translations, etc.
        self.addPerDofVariable("x1", 0);
        if self.M:
            self.addComputeGlobal("Q0", "ndf*Q")
            for i in range(1, self.M):
                self.addComputeGlobal("Q{}".format(i), "Q")

        #
        # Take a velocity verlet step, with propagation of thermostat before and after
        #
        if self.M: self.propagateNHC()
        self.addUpdateContextState()
        self.addComputePerDof("v", "v+0.5*dt*f/m")
        self.addComputePerDof("x", "x+dt*v")
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions()
        self.addComputePerDof("v", "v+0.5*dt*f/m+(x-x1)/dt")
        self.addConstrainVelocities()
        if self.M: self.propagateNHC()
        # Compute heat bath energies
        self.computeEnergies()

    def propagateNHC(self):
        """ Propagate the Nosé-Hoover chain """
        self.addComputeGlobal("scale", "1.0")
        self.addComputeSum("KE2", "m*v^2")
        self.addComputeGlobal("G0", "(KE2 - ndf*kT)/Q0")
        for ncval in range(self.n_c):
            for nysval in range(self.n_ys):
                self.addComputeGlobal("wdt", "w{}*dt/{}".format(nysval, self.n_c))
                self.addComputeGlobal("vxi{}".format(self.M-1), "vxi{} + 0.25*wdt*G{}".format(self.M-1, self.M-1))
                for j in range(self.M-2, -1, -1):
                    self.addComputeGlobal("aa", "exp(-0.125*wdt*vxi{})".format(j+1))
                    self.addComputeGlobal("vxi{}".format(j), "aa*(aa*vxi{} + 0.25*wdt*G{})".format(j,j))
                # update particle velocities
                self.addComputeGlobal("aa", "exp(-0.5*wdt*vxi0)")
                self.addComputeGlobal("scale", "scale*aa")
                # update the thermostat positions
                for j in range(self.M):
                    self.addComputeGlobal("xi{}".format(j), "xi{} + 0.5*wdt*vxi{}".format(j,j))
                # update the forces
                self.addComputeGlobal("G0", "(scale*scale*KE2 - ndf*kT)/Q0")
                # update thermostat velocities
                for j in range(self.M-1):
                    self.addComputeGlobal("aa", "exp(-0.125*wdt*vxi{})".format(j+1))
                    self.addComputeGlobal("vxi{}".format(j), "aa*(aa*vxi{} + 0.25*wdt*G{})".format(j,j))
                    self.addComputeGlobal("G{}".format(j+1), "(Q{}*vxi{}*vxi{} - kT)/Q{}".format(j,j,j,j+1))
                self.addComputeGlobal("vxi{}".format(self.M-1), "vxi{} + 0.25*wdt*G{}".format(self.M-1, self.M-1))
        # update particle velocities
        self.addComputePerDof("v", "scale*v")

    def computeEnergies(self):
        """ Computes kinetic and potential energies for the heat bath """
        # Bath kinetic energy
        self.addComputeGlobal("bathKE", "0.0")
        for i in range(self.M):
            self.addComputeGlobal("bathKE", "bathKE + 0.5*Q{}*vxi{}^2".format(i,i))
        # Bath potential energy
        self.addComputeGlobal("bathPE", "ndf*xi0")
        for i in range(1,self.M):
            self.addComputeGlobal("bathPE", "bathPE + xi{}".format(i))
        self.addComputeGlobal("bathPE", "kT*bathPE")


class MetropolisMonteCarloIntegrator(ThermostatedIntegrator):

    """
    Metropolis Monte Carlo with Gaussian displacement trials.

    """

    def __init__(self, temperature=298.0 * unit.kelvin, sigma=0.1 * unit.angstroms, timestep=1 * unit.femtoseconds):
        """
        Create a simple Metropolis Monte Carlo integrator that uses Gaussian displacement trials.

        Parameters
        ----------
        temperature : np.unit.Quantity compatible with kelvin, default: 298*unit.kelvin
           The temperature.
        sigma : np.unit.Quantity compatible with nanometers, default: 0.1*unit.angstroms
           The displacement standard deviation for each degree of freedom.
        timestep : np.unit.Quantity compatible with femtoseconds, default: 1.0*unit.femtoseconds
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

        >>> timestep = 1.0 * unit.femtoseconds # fictitious timestep
        >>> temperature = 298.0 * unit.kelvin
        >>> sigma = 1.0 * unit.angstroms
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

    def __init__(self, temperature=298.0 * unit.kelvin, nsteps=10, timestep=1 * unit.femtoseconds):
        """
        Create a hybrid Monte Carlo (HMC) integrator.

        Parameters
        ----------
        temperature : np.unit.Quantity compatible with kelvin, default: 298*unit.kelvin
           The temperature.
        nsteps : int, default: 10
           The number of velocity Verlet steps to take per HMC trial.
        timestep : np.unit.Quantity compatible with femtoseconds, default: 1*unit.femtoseconds
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

        >>> timestep = 1.0 * unit.femtoseconds # fictitious timestep
        >>> temperature = 298.0 * unit.kelvin
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


class LangevinIntegrator(ThermostatedIntegrator):
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
    shadow_work : unit.Quantity with units of energy
       Shadow work (if integrator was constructed with measure_shadow_work=True)
    heat : unit.Quantity with units of energy
       Heat (if integrator was constructed with measure_heat=True)

    References
    ----------
    [Leimkuhler and Matthews, 2015] Molecular dynamics: with deterministic and stochastic numerical methods, Chapter 7
    """

    _kinetic_energy = "0.5 * m * v * v"

    def __init__(self,
                 temperature=298.0 * unit.kelvin,
                 collision_rate=1.0 / unit.picoseconds,
                 timestep=1.0 * unit.femtoseconds,
                 splitting="V R O R V",
                 constraint_tolerance=1e-8,
                 measure_shadow_work=False,
                 measure_heat=False,
                 ):
        """Create a Langevin integrator with the prescribed operator splitting.

        Parameters
        ----------
        splitting : string, default: "V R O R V"
            Sequence of "R", "V", "O" (and optionally "{", "}", "V0", "V1", ...) substeps to be executed each timestep.

            Forces are only used in V-step. Handle multiple force groups by appending the force group index
            to V-steps, e.g. "V0" will only use forces from force group 0. "V" will perform a step using all forces.
            "{" will cause metropolization, and must be followed later by a "}".


        temperature : np.unit.Quantity compatible with kelvin, default: 298.0*unit.kelvin
           Fictitious "bath" temperature

        collision_rate : np.unit.Quantity compatible with 1/picoseconds, default: 1.0/unit.picoseconds
           Collision rate

        timestep : np.unit.Quantity compatible with femtoseconds, default: 1.0*unit.femtoseconds
           Integration timestep

        constraint_tolerance : float, default: 1.0e-8
            Tolerance for constraint solver

        measure_shadow_work : boolean, default: False
            Accumulate the shadow work performed by the symplectic substeps, in the global `shadow_work`

        measure_heat : boolean, default: False
            Accumulate the heat exchanged with the bath in each step, in the global `heat`
        """

        # Compute constants
        gamma = collision_rate
        self._gamma = gamma

        # Check if integrator is metropolized by checking for M step:
        if splitting.find("{") > -1:
            self._metropolized_integrator = True
            # We need to measure shadow work if Metropolization is used
            measure_shadow_work = True
        else:
            self._metropolized_integrator = False

        # Record whether we are measuring heat and shadow work
        self._measure_heat = measure_heat
        self._measure_shadow_work = measure_shadow_work

        ORV_counts, mts, force_group_nV = self._parse_splitting_string(splitting)

        # Record splitting.
        self._splitting = splitting
        self._ORV_counts = ORV_counts
        self._mts = mts
        self._force_group_nV = force_group_nV

        # Create a new CustomIntegrator
        super(LangevinIntegrator, self).__init__(temperature, timestep)

        # Initialize
        self.addPerDofVariable("sigma", 0)

        # Velocity mixing parameter: current velocity component
        h = timestep / max(1, ORV_counts['O'])
        self.addGlobalVariable("a", np.exp(-gamma * h))

        # Velocity mixing parameter: random velocity component
        self.addGlobalVariable("b", np.sqrt(1 - np.exp(- 2 * gamma * h)))

        # Positions before application of position constraints
        self.addPerDofVariable("x1", 0)

        # Set constraint tolerance
        self.setConstraintTolerance(constraint_tolerance)

        # Add global variables
        self._add_global_variables()

        # Add integrator steps
        self._add_integrator_steps()

    @property
    def _step_dispatch_table(self):
        """dict: The dispatch table step_name -> add_step_function."""
        # TODO use methoddispatch (see yank.utils) when dropping Python 2 support.
        dispatch_table = {
            'O': (self._add_O_step, False),
            'R': (self._add_R_step, False),
            '{': (self._add_metropolize_start, False),
            '}': (self._add_metropolize_finish, False),
            'V': (self._add_V_step, True)
        }
        return dispatch_table

    def _add_global_variables(self):
        """Add global bookkeeping variables."""
        if self._measure_heat:
            self.addGlobalVariable("heat", 0)

        if self._measure_shadow_work or self._measure_heat:
            self.addGlobalVariable("old_ke", 0)
            self.addGlobalVariable("new_ke", 0)

        if self._measure_shadow_work:
            self.addGlobalVariable("old_pe", 0)
            self.addGlobalVariable("new_pe", 0)
            self.addGlobalVariable("shadow_work", 0)

        # If we metropolize, we have to keep track of the before and after (x, v)
        if self._metropolized_integrator:
            self.addGlobalVariable("accept", 0)
            self.addGlobalVariable("ntrials", 0)
            self.addGlobalVariable("nreject", 0)
            self.addGlobalVariable("naccept", 0)
            self.addPerDofVariable("vold", 0)
            self.addPerDofVariable("xold", 0)

    def reset_heat(self):
        """Reset heat."""
        if self._measure_heat:
            self.setGlobalVariableByName('heat', 0.0)

    def reset_shadow_work(self):
        """Reset shadow work."""
        if self._measure_shadow_work:
            self.setGlobalVariableByName('shadow_work', 0.0)

    def reset_ghmc_statistics(self):
        """Reset GHMC acceptance rate statistics."""
        if self._metropolized_integrator:
            self.setGlobalVariableByName('ntrials', 0)
            self.setGlobalVariableByName('naccept', 0)
            self.setGlobalVariableByName('nreject', 0)

    def reset(self):
        """Reset all statistics (heat, shadow work, acceptance rates, step).
        """
        self.reset_heat()
        self.reset_shadow_work()
        self.reset_ghmc_statistics()

    def _get_energy_with_units(self, variable_name, dimensionless=False):
        """Retrive an energy/work quantity and return as unit-bearing or dimensionless quantity.

        Parameters
        ----------
        variable_name : str
           Name of the global context variable to retrieve
        dimensionless : bool, optional, default=False
           If specified, the energy/work is returned in reduced (kT) unit.

        Returns
        -------
        work : unit.Quantity or float
           If dimensionless=True, the work in kT (float).
           Otherwise, the unit-bearing work in units of energy.
        """
        work = self.getGlobalVariableByName(variable_name) * _OPENMM_ENERGY_UNIT
        if dimensionless:
            return work / self.kT
        else:
            return work

    def get_shadow_work(self, dimensionless=False):
        """Get the current accumulated shadow work.

        Parameters
        ----------
        dimensionless : bool, optional, default=False
           If specified, the work is returned in reduced (kT) unit.

        Returns
        -------
        work : unit.Quantity or float
           If dimensionless=True, the protocol work in kT (float).
           Otherwise, the unit-bearing protocol work in units of energy.
        """
        if not self._measure_shadow_work:
            raise Exception("This integrator must be constructed with 'measure_shadow_work=True' to measure shadow work.")
        return self._get_energy_with_units("shadow_work", dimensionless=dimensionless)

    @property
    def shadow_work(self):
        return self.get_shadow_work()

    def get_heat(self, dimensionless=False):
        """Get the current accumulated heat.

        Parameters
        ----------
        dimensionless : bool, optional, default=False
           If specified, the work is returned in reduced (kT) unit.

        Returns
        -------
        work : unit.Quantity or float
           If dimensionless=True, the heat in kT (float).
           Otherwise, the unit-bearing heat in units of energy.
        """
        if not self._measure_heat:
            raise Exception("This integrator must be constructed with 'measure_heat=True' in order to measure heat.")
        return self._get_energy_with_units("heat", dimensionless=dimensionless)

    @property
    def heat(self):
        return self.get_heat()

    def get_acceptance_rate(self):
        """Get acceptance rate for Metropolized integrators.

        Returns
        -------
        acceptance_rate : float
           Acceptance rate.
           An Exception is thrown if the integrator is not Metropolized.
        """
        if not self._metropolized_integrator:
            raise Exception("This integrator must be Metropolized to return an acceptance rate.")
        return self.getGlobalVariableByName("naccept") / self.getGlobalVariableByName("ntrials")

    @property
    def acceptance_rate(self):
        """Get acceptance rate for Metropolized integrators."""
        return self.get_acceptance_rate()

    @property
    def is_metropolized(self):
        """Return True if this integrator is Metropolized, False otherwise."""
        return self._metropolized_integrator

    def _add_integrator_steps(self):
        """Add the steps to the integrator--this can be overridden to place steps around the integration.
        """
        # Integrate
        self.addUpdateContextState()
        self.addComputeTemperatureDependentConstants({"sigma": "sqrt(kT/m)"})

        for i, step in enumerate(self._splitting.split()):
            self._substep_function(step)

    def _sanity_check(self, splitting):
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

        allowed_characters = "0123456789"
        for key in self._step_dispatch_table:
            allowed_characters += key

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
            elif step == "{":
                    if "}" not in splitting:
                        raise ValueError("Use of { must be followed by }")
                    if not self._verify_metropolization(splitting):
                        raise ValueError("Shadow work generating steps found outside the Metropolization block")
            elif step in allowed_characters:
                continue
            else:
                raise ValueError("Invalid step name '%s' used; valid step names are %s" % (step, allowed_characters))

        # Make sure we contain at least one of R, V, O steps
        assert ("R" in splitting_no_space)
        assert ("V" in splitting_no_space)
        assert ("O" in splitting_no_space)

    def _verify_metropolization(self, splitting):
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
        #this pattern matches the { literally, then any number of any character other than }, followed by another {
        #If there's a match, then we have an attempt at a nested metropolization, which is unsupported
        regex_nested_metropolis = "{[^}]*{"
        pattern = re.compile(regex_nested_metropolis)
        if pattern.match(splitting.replace(" ", "")):
            raise ValueError("There can only be one Metropolized region.")

        # find the metropolization steps:
        M_start_index = splitting.find("{")
        M_end_index = splitting.find("}")

        # accept/reject happens before the beginning of metropolis step
        if M_start_index > M_end_index:
            return False

        #pattern to find whether any shadow work producing steps lie outside the metropolization region
        RV_outside_metropolis = "[RV](?![^{]*})"
        outside_metropolis_check = re.compile(RV_outside_metropolis)
        if outside_metropolis_check.match(splitting.replace(" ","")):
            return False
        else:
            return True

    def _add_R_step(self):
        """Add an R step (position update) given the velocities.
        """
        if self._measure_shadow_work:
            self.addComputeGlobal("old_pe", "energy")
            self.addComputeSum("old_ke", self._kinetic_energy)

        n_R = self._ORV_counts['R']

        # update positions (and velocities, if there are constraints)
        self.addComputePerDof("x", "x + ((dt / {}) * v)".format(n_R))
        self.addComputePerDof("x1", "x")  # save pre-constraint positions in x1
        self.addConstrainPositions()  # x is now constrained
        self.addComputePerDof("v", "v + ((x - x1) / (dt / {}))".format(n_R))
        self.addConstrainVelocities()

        if self._measure_shadow_work:
            self.addComputeGlobal("new_pe", "energy")
            self.addComputeSum("new_ke", self._kinetic_energy)
            self.addComputeGlobal("shadow_work", "shadow_work + (new_ke + new_pe) - (old_ke + old_pe)")

    def _add_V_step(self, force_group="0"):
        """Deterministic velocity update, using only forces from force-group fg.

        Parameters
        ----------
        force_group : str, optional, default="0"
           Force group to use for this step
        """
        if self._measure_shadow_work:
            self.addComputeSum("old_ke", self._kinetic_energy)

        # update velocities
        if self._mts:
            self.addComputePerDof("v", "v + ((dt / {}) * f{} / m)".format(self._force_group_nV[force_group], force_group))
        else:
            self.addComputePerDof("v", "v + (dt / {}) * f / m".format(self._force_group_nV["0"]))

        self.addConstrainVelocities()

        if self._measure_shadow_work:
            self.addComputeSum("new_ke", self._kinetic_energy)
            self.addComputeGlobal("shadow_work", "shadow_work + (new_ke - old_ke)")

    def _add_O_step(self):
        """Add an O step (stochastic velocity update)
        """
        if self._measure_heat:
            self.addComputeSum("old_ke", self._kinetic_energy)

        # update velocities
        self.addComputePerDof("v", "(a * v) + (b * sigma * gaussian)")
        self.addConstrainVelocities()

        if self._measure_heat:
            self.addComputeSum("new_ke", self._kinetic_energy)
            self.addComputeGlobal("heat", "heat + (new_ke - old_ke)")

    def _substep_function(self, step_string):
        """Take step string, and add the appropriate R, V, O step with appropriate parameters.

        The step string input here is a single character (or character + number, for MTS)
        """
        function, can_accept_force_groups = self._step_dispatch_table[step_string[0]]
        if can_accept_force_groups:
            force_group = step_string[1:]
            function(force_group)
        else:
            function()

    def _parse_splitting_string(self, splitting_string):
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
        self._sanity_check(splitting_string)

        ORV_counts = dict()

        # count number of R, V, O steps:
        for step_symbol in self._step_dispatch_table:
            ORV_counts[step_symbol] = splitting_string.count(step_symbol)

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

    def _add_metropolize_start(self):
        """Save the current x and v for a metropolization step later"""
        self.addComputePerDof("xold", "x")
        self.addComputePerDof("vold", "v")

    def _add_metropolize_finish(self):
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
        self.addComputeGlobal("shadow_work", "0")

class NonequilibriumLangevinIntegrator(LangevinIntegrator):
    """Nonequilibrium integrator mix-in.

    Properties
    ----------
    protocol_work : unit.Quantity with units of energy
       Protocol work
    total_work : unit.Quantity with units of energy
       Total work = protocol work + shadow work

    Public methods
    --------------
    get_protocol_work
        Get the current protocol work (with units), if available.
    get_total_work
        Get the current total work (with units), if available.

    Private methods
    ---------------
    reset_work_step
        Create an integrator step that resets the protocol and shadow work.
    """

    def __init__(self, *args, **kwargs):
        super(NonequilibriumLangevinIntegrator, self).__init__(*args, **kwargs)

    def _add_global_variables(self):
        super(NonequilibriumLangevinIntegrator, self)._add_global_variables()
        self.addGlobalVariable("protocol_work", 0)

    def _add_reset_protocol_work_step(self):
        """
        Add a step that resets protocol and shadow work statistics.
        """
        self.addComputeGlobal("protocol_work", "0.0")

    def reset_protocol_work(self):
        """
        Reset the protocol work.
        """
        self.setGlobalVariableByName("protocol_work", 0)

    def get_protocol_work(self, dimensionless=False):
        """Get the current accumulated protocol work.

        Parameters
        ----------
        dimensionless : bool, optional, default=False
           If specified, the work is returned in reduced (kT) unit.

        Returns
        -------
        work : unit.Quantity or float
           If dimensionless=True, the protocol work in kT (float).
           Otherwise, the unit-bearing protocol work in units of energy.
        """
        return self._get_energy_with_units("protocol_work", dimensionless=dimensionless)

    @property
    def protocol_work(self):
        """Total protocol work in energy units.
        """
        return self.get_protocol_work()

    def get_total_work(self, dimensionless=False):
        """Get the current accumulated total work.

        Note that the integrator must have been constructed with measure_shadow_work=True.

        Parameters
        ----------
        dimensionless : bool, optional, default=False
           If specified, the work is returned in reduced (kT) unit.

        Returns
        -------
        work : unit.Quantity or float
           If dimensionless=True, the total work in kT (float).
           Otherwise, the unit-bearing total work in units of energy.
        """
        return self.get_protocol_work(dimensionless=dimensionless) + self.get_shadow_work(dimensionless=dimensionless)

    @property
    def total_work(self):
        """Total work (protocol work plus shadow work) in energy units.
        """
        return self.get_total_work()

    def reset(self):
        """
        Manually reset protocol work and other statistics.
        """
        self.reset_protocol_work()
        super(NonequilibriumLangevinIntegrator, self).reset()

class AlchemicalNonequilibriumLangevinIntegrator(NonequilibriumLangevinIntegrator):
    """Allows nonequilibrium switching based on force parameters specified in alchemical_functions.
    A variable named lambda is switched from 0 to 1 linearly throughout the nsteps of the protocol.
    The functions can use this to create more complex protocols for other global parameters.

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

        - H: Hamiltonian update step

    We can then construct integrators by solving each part for a certain timestep in sequence.
    (We can further split up the V step by force group, evaluating cheap but fast-fluctuating
    forces more frequently than expensive but slow-fluctuating forces. Since forces are only
    evaluated in the V step, we represent this by including in our "alphabet" V0, V1, ...)

    When the system contains holonomic constraints, these steps are confined to the constraint
    manifold.

    Examples
    --------

    Create a nonequilibrium integrator to switch the center of a harmonic oscillator

    >>> # Create harmonic oscillator testsystem
    >>> from openmmtools import testsystems
    >>> from simtk import openmm, unit
    >>> testsystem = testsystems.HarmonicOscillator()
    >>> # Create a nonequilibrium alchemical integrator
    >>> alchemical_functions = { 'testsystems_HarmonicOscillator_x0' : 'lambda' }
    >>> nsteps_neq = 100 # number of steps in the switching trajectory where lambda is switched from 0 to 1
    >>> integrator = AlchemicalNonequilibriumLangevinIntegrator(temperature=300*unit.kelvin, collision_rate=1.0/unit.picoseconds, timestep=1.0*unit.femtoseconds,
    ...                                                         alchemical_functions=alchemical_functions, splitting="O { V R H R V } O", nsteps_neq=nsteps_neq,
    ...                                                         measure_shadow_work=True)
    >>> # Create a Context
    >>> context = openmm.Context(testsystem.system, integrator)
    >>> # Run the whole switching trajectory
    >>> context.setPositions(testsystem.positions)
    >>> integrator.step(nsteps_neq)
    >>> protocol_work = integrator.protocol_work # retrieve protocol work (excludes shadow work)
    >>> total_work = integrator.total_work # retrieve total work (includes shadow worl)
    >>> # Reset and run again
    >>> context.setPositions(testsystem.positions)
    >>> integrator.reset()
    >>> integrator.step(nsteps_neq)

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
                 alchemical_functions=None,
                 splitting="O { V R H R V } O",
                 nsteps_neq=100,
                 *args,
                 **kwargs):
        """
        Parameters
        ----------
        alchemical_functions : dict of strings, optional, default=None
            key: value pairs such as "global_parameter" : function_of_lambda where function_of_lambda is a Lepton-compatible
            string that depends on the variable "lambda"
            If not specified, no alchemical functions will be used.

        splitting : string, default: "O { V R H R V } O"
            Sequence of R, V, O (and optionally V{i}), and { }substeps to be executed each timestep. There is also an H option,
            which increments the global parameter `lambda` by 1/nsteps_neq for each step.

            Forces are only used in V-step. Handle multiple force groups by appending the force group index
            to V-steps, e.g. "V0" will only use forces from force group 0. "V" will perform a step using all forces.
            ( will cause metropolization, and must be followed later by a ).

        temperature : np.unit.Quantity compatible with kelvin, default: 298.0*unit.kelvin
           Fictitious "bath" temperature

        collision_rate : np.unit.Quantity compatible with 1/picoseconds, default: 91.0/unit.picoseconds
           Collision rate

        timestep : np.unit.Quantity compatible with femtoseconds, default: 1.0*unit.femtoseconds
           Integration timestep

        constraint_tolerance : float, default: 1.0e-8
            Tolerance for constraint solver

        measure_shadow_work : boolean, default: False
            Accumulate the shadow work performed by the symplectic substeps, in the global `shadow_work`

        measure_heat : boolean, default: False
            Accumulate the heat exchanged with the bath in each step, in the global `heat`

        nsteps_neq : int, default: 100
            Number of steps in nonequilibrium protocol. Default 100
            This number cannot be changed without creating a new integrator.
        """

        if alchemical_functions is None:
            alchemical_functions = dict()

        if (nsteps_neq < 0) or (nsteps_neq != int(nsteps_neq)):
            raise Exception('nsteps_neq must be an integer >= 0')

        self._alchemical_functions = alchemical_functions
        self._n_steps_neq = nsteps_neq # number of integrator steps

        # collect the system parameters.
        self._system_parameters = {system_parameter for system_parameter in alchemical_functions.keys()}

        # call the base class constructor
        kwargs['splitting'] = splitting
        super(AlchemicalNonequilibriumLangevinIntegrator, self).__init__(*args, **kwargs)

    @property
    def _step_dispatch_table(self):
        """dict: The dispatch table step_name -> add_step_function."""
        # TODO use methoddispatch (see yank.utils) when dropping Python 2 support.
        dispatch_table = super(AlchemicalNonequilibriumLangevinIntegrator, self)._step_dispatch_table
        dispatch_table['H'] = (self._add_alchemical_perturbation_step, False)
        return dispatch_table

    def reset(self):
        """Reset all statistics, alchemical parameters, and work.
        """
        # Reset statistics
        super(AlchemicalNonequilibriumLangevinIntegrator, self).reset()
        # Trigger update of all context parameters only by running one integrator cycle with step = -1
        self.setGlobalVariableByName('step', -1)
        self.step(1)

    def _add_global_variables(self):
        """Add the appropriate global parameters to the CustomIntegrator. nsteps refers to the number of
        total steps in the protocol.

        Parameters
        ----------
        nsteps : int, greater than 0
            The number of steps in the switching protocol.
        """
        super(AlchemicalNonequilibriumLangevinIntegrator, self)._add_global_variables()
        self.addGlobalVariable('Eold', 0) #old energy value before perturbation
        self.addGlobalVariable('Enew', 0) #new energy value after perturbation
        self.addGlobalVariable('lambda', 0.0) # parameter switched from 0 <--> 1 during course of integrating internal 'nsteps' of dynamics
        self.addGlobalVariable('nsteps', self._n_steps_neq) # total number of NCMC steps to perform; this SHOULD NOT BE CHANGED during the protocol
        self.addGlobalVariable('step', 0) # step counter for handling initialization and terminating integration

        # Keep track of number of Hamiltonian updates per nonequilibrium switch
        n_H = self._ORV_counts['H'] # number of H updates per integrator step
        self._n_lambda_steps = self._n_steps_neq * n_H # number of Hamiltonian increments per switching step
        if self._n_steps_neq == 0:
            self._n_lambda_steps = 1 # instantaneous switching
        self.addGlobalVariable('n_lambda_steps', self._n_lambda_steps) # total number of NCMC steps to perform; this SHOULD NOT BE CHANGED during the protocol
        self.addGlobalVariable('lambda_step', 0)

    def _add_update_alchemical_parameters_step(self):
        """
        Add step to update Context parameters according to provided functions.
        """
        for context_parameter in self._alchemical_functions:
            if context_parameter in self._system_parameters:
                self.addComputeGlobal(context_parameter, self._alchemical_functions[context_parameter])

    def _add_alchemical_perturbation_step(self):
        """
        Add alchemical perturbation step, accumulating protocol work.

        TODO: Extend this to be able to handle force groups?

        """
        # Store initial potential energy
        self.addComputeGlobal("Eold", "energy")

        # Update lambda and increment that tracks updates.
        self.addComputeGlobal('lambda', '(lambda_step+1)/n_lambda_steps')
        self.addComputeGlobal('lambda_step', 'lambda_step + 1')

        # Update all slaved alchemical parameters
        self._add_update_alchemical_parameters_step()

        # Accumulate protocol work
        self.addComputeGlobal("Enew", "energy")
        self.addComputeGlobal("protocol_work", "protocol_work + (Enew-Eold)")

    def _add_integrator_steps(self):
        """
        Override the base class to insert reset steps around the integrator.
        """
        # First step: Constrain positions and velocities and reset work accumulators and alchemical integrators
        self.beginIfBlock('step = 0')
        self.addConstrainPositions()
        self.addConstrainVelocities()
        self._add_reset_protocol_work_step()
        self._add_alchemical_reset_step()
        self.endBlock()

        # Main body
        if self._n_steps_neq == 0:
            # If nsteps = 0, we need to force execution on the first step only.
            self.beginIfBlock('step = 0')
            super(AlchemicalNonequilibriumLangevinIntegrator, self)._add_integrator_steps()
            self.addComputeGlobal("step", "step + 1")
            self.endBlock()
        else:
            #call the superclass function to insert the appropriate steps, provided the step number is less than n_steps
            self.beginIfBlock("step >= 0")
            self.beginIfBlock("step < nsteps")
            super(AlchemicalNonequilibriumLangevinIntegrator, self)._add_integrator_steps()
            self.addComputeGlobal("step", "step + 1")
            self.endBlock()
            self.endBlock()

        # Reset step
        self.beginIfBlock('step = -1')
        self._add_reset_protocol_work_step()
        self._add_alchemical_reset_step() # sets step to 0
        self.endBlock()

    def _add_alchemical_reset_step(self):
        """
        Reset the alchemical lambda to its starting value
        """
        self.addComputeGlobal("lambda", "0")
        self.addComputeGlobal("protocol_work", "0")
        self.addComputeGlobal("step", "0")
        self.addComputeGlobal("lambda_step", "0")
        # Add all dependent parameters
        self._add_update_alchemical_parameters_step()

class ExternalPerturbationLangevinIntegrator(NonequilibriumLangevinIntegrator):
    """
    Create a LangevinSplittingIntegrator that accounts for external perturbations and tracks protocol work.


    Examples
    --------

    >>> # Create harmonic oscillator testsystem
    >>> from openmmtools import testsystems
    >>> from simtk import openmm, unit
    >>> testsystem = testsystems.HarmonicOscillator()
    >>> # Create an external perturbation integrator
    >>> integrator = ExternalPerturbationLangevinIntegrator(temperature=300*unit.kelvin, collision_rate=1.0/unit.picoseconds, timestep=1.0*unit.femtoseconds)
    >>> context = openmm.Context(testsystem.system, integrator)
    >>> context.setPositions(testsystem.positions)
    >>> # Take a step
    >>> integrator.step(1)
    >>> # Perturb the system
    >>> context.setParameter('testsystems_HarmonicOscillator_x0', 0.1)
    >>> # Take another step, integrating work
    >>> integrator.step(1)
    >>> # Retrieve the work
    >>> protocol_work = integrator.protocol_work

    where force is an instance of openmm's force class. The externally performed protocol work is accumulated in the
    "protocol_work" global variable. This variable can be re-initialized by calling

    >>> integrator.reset()

    """

    def __init__(self, *args, **kwargs):
        super(ExternalPerturbationLangevinIntegrator, self).__init__(*args, **kwargs)

    def reset(self):
        """Reset all statistics.
        """
        super(ExternalPerturbationLangevinIntegrator, self).reset()
        # Setting 'step' to 0 will trigger the integrator to reset all statistics prior to the next step
        self.setGlobalVariableByName('step', 0)

    def _add_global_variables(self):
        super(ExternalPerturbationLangevinIntegrator, self)._add_global_variables()
        self.addGlobalVariable("perturbed_pe", 0)
        self.addGlobalVariable("unperturbed_pe", 0)
        self.addGlobalVariable("step", 0)

    def _add_integrator_steps(self):
        self.addComputeGlobal("perturbed_pe", "energy")
        # Assumes no perturbation is done before doing the initial MD step.
        self.beginIfBlock("step < 1")
        self.addComputeGlobal("step", "1")
        self.addComputeGlobal("unperturbed_pe", "energy")
        self.addComputeGlobal("protocol_work", "0.0")
        self.endBlock()

        # Calculate the protocol work
        self.addComputeGlobal("protocol_work", "protocol_work + (perturbed_pe - unperturbed_pe)")

        # Computing context updates, such as from the barostat, _after_ computing protocol work.
        super(ExternalPerturbationLangevinIntegrator, self)._add_integrator_steps()

        # Store final energy before perturbation
        self.addComputeGlobal("unperturbed_pe", "energy")

class VVVRIntegrator(LangevinIntegrator):
    """Create a velocity Verlet with velocity randomization (VVVR) integrator."""
    def __init__(self, *args, **kwargs):
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
        >>> temperature = 298.0 * unit.kelvin
        >>> collision_rate = 1.0 / unit.picoseconds
        >>> timestep = 1.0 * unit.femtoseconds
        >>> integrator = VVVRIntegrator(temperature, collision_rate, timestep)
        """
        kwargs['splitting'] = "O V R V O"
        super(VVVRIntegrator, self).__init__(*args, **kwargs)

class BAOABIntegrator(LangevinIntegrator):
    """Create a BAOAB integrator."""
    def __init__(self, *args, **kwargs):
        """Create an integrator of Langevin dynamics using the BAOAB operator splitting.

        Parameters
        ----------
        temperature : np.unit.Quantity compatible with kelvin, default: 298.0*unit.kelvin
           Fictitious "bath" temperature

        collision_rate : np.unit.Quantity compatible with 1/picoseconds, default: 91.0/unit.picoseconds
           Collision rate

        timestep : np.unit.Quantity compatible with femtoseconds, default: 1.0*unit.femtoseconds
           Integration timestep

        constraint_tolerance : float, default: 1.0e-8
            Tolerance for constraint solver

        measure_shadow_work : boolean, default: False
            Accumulate the shadow work performed by the symplectic substeps, in the global `shadow_work`

        measure_heat : boolean, default: False
            Accumulate the heat exchanged with the bath in each step, in the global `heat`

        References
        ----------
        [Leimkuhler and Matthews, 2013] Rational construction of stochastic numerical methods for molecular sampling
        https://academic.oup.com/amrx/article-abstract/2013/1/34/166771/Rational-Construction-of-Stochastic-Numerical

        Examples
        --------
        Create a BAOAB integrator.
        >>> temperature = 298.0 * unit.kelvin
        >>> collision_rate = 1.0 / unit.picoseconds
        >>> timestep = 1.0 * unit.femtoseconds
        >>> integrator = BAOABIntegrator(temperature, collision_rate, timestep)
        """
        kwargs['splitting'] = "V R O R V"
        super(BAOABIntegrator, self).__init__(*args, **kwargs)


class GeodesicBAOABIntegrator(LangevinIntegrator):
    """Create a geodesic-BAOAB integrator."""

    def __init__(self, *args, **kwargs):
        """Create a geodesic BAOAB Langevin integrator.

        Parameters
        ----------
        K_r : integer, default: 2
            Number of geodesic drift steps.

        temperature : np.unit.Quantity compatible with kelvin, default: 298.0*unit.kelvin
           Fictitious "bath" temperature

        collision_rate : np.unit.Quantity compatible with 1/picoseconds, default: 1.0/unit.picoseconds
           Collision rate

        timestep : np.unit.Quantity compatible with femtoseconds, default: 1.0*unit.femtoseconds
           Integration timestep

        constraint_tolerance : float, default: 1.0e-8
            Tolerance for constraint solver

        measure_shadow_work : boolean, default: False
            Accumulate the shadow work performed by the symplectic substeps, in the global `shadow_work`

        measure_heat : boolean, default: False
            Accumulate the heat exchanged with the bath in each step, in the global `heat`

        References
        ----------
        [Leimkuhler and Matthews, 2016] Efficient molecular dynamics using geodesic integration and solvent-solute splitting
        http://rspa.royalsocietypublishing.org/content/472/2189/20160138

        Examples
        --------
        Create a geodesic BAOAB integrator.
        >>> temperature = 298.0 * unit.kelvin
        >>> collision_rate = 1.0 / unit.picoseconds
        >>> timestep = 1.0 * unit.femtoseconds
        >>> integrator = GeodesicBAOABIntegrator(K_r=3, temperature=temperature, collision_rate=collision_rate, timestep=timestep)
        """
        # TODO: move this as an explicity keyword argument after dropping Python 2 support.
        K_r = kwargs.pop('K_r', 2)
        kwargs['splitting'] = " ".join(["V"] + ["R"] * K_r + ["O"] + ["R"] * K_r + ["V"])
        super(GeodesicBAOABIntegrator, self).__init__(*args, **kwargs)


class GHMCIntegrator(LangevinIntegrator):
    """Create a generalized hybrid Monte Carlo (GHMC) integrator."""

    def __init__(self, *args, **kwargs):
        """


        Parameters
        ----------
        temperature : unit.Quantity compatible with kelvin, default: 298*unit.kelvin
           The temperature.
        collision_rate : unit.Quantity compatible with 1/picoseconds, default: 91.0/unit.picoseconds
           The collision rate.
        timestep : unit.Quantity compatible with femtoseconds, default: 1.0*unit.femtoseconds
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

        >>> temperature = 298.0 * unit.kelvin
        >>> collision_rate = 1.0 / unit.picoseconds
        >>> timestep = 1.0 * unit.femtoseconds
        >>> integrator = GHMCIntegrator(temperature, collision_rate, timestep)

        References
        ----------
        Lelievre T, Stoltz G, and Rousset M. Free Energy Computations: A Mathematical Perspective
        http://www.amazon.com/Free-Energy-Computations-Mathematical-Perspective/dp/1848162472
        """
        kwargs['splitting'] = "O { V R V } O"
        super(GHMCIntegrator, self).__init__(*args, **kwargs)

class FIREMinimizationIntegrator(mm.CustomIntegrator):
    """Fast Internal Relaxation Engine (FIRE) minimization.
    Notes
    -----
    This integrator is taken verbatim from Peter Eastman's example appearing in the CustomIntegrator header file documentation.
    References
    ----------
    Erik Bitzek, Pekka Koskinen, Franz Gaehler, Michael Moseler, and Peter Gumbsch.
    Structural Relaxation Made Simple. PRL 97:170201, 2006.
    http://dx.doi.org/10.1103/PhysRevLett.97.170201
    Examples
    --------
    >>> from openmmtools import testsystems
    >>> from simtk import openmm
    >>> t = testsystems.AlanineDipeptideVacuum()
    >>> system, positions = t.system, t.positions
    >>> integrator = FIREMinimizationIntegrator()
    >>> context = openmm.Context(system, integrator)
    >>> context.setPositions(positions)
    >>> integrator.step(100)
    """

    def __init__(self, timestep=1.0 * unit.femtoseconds, tolerance=None, alpha=0.1, dt_max=10.0 * unit.femtoseconds, f_inc=1.1, f_dec=0.5, f_alpha=0.99, N_min=5):
        """Construct a Fast Internal Relaxation Engine (FIRE) minimization integrator.
        Parameters
        ----------
        timestep : unit.Quantity compatible with femtoseconds, optional, default = 1*femtoseconds
            The integration timestep.
        tolerance : unit.Quantity compatible with kilojoules_per_mole/nanometer, optional, default = None
            Minimization will be terminated when RMS force reaches this tolerance.
        alpha : float, optional default = 0.1
            Velocity relaxation parameter, alpha \in (0,1).
        dt_max : unit.Quantity compatible with femtoseconds, optional, default = 10*femtoseconds
            Maximum allowed timestep.
        f_inc : float, optional, default = 1.1
            Timestep increment multiplicative factor.
        f_dec : float, optional, default = 0.5
            Timestep decrement multiplicative factor.
        f_alpha : float, optional, default = 0.99
            alpha multiplicative relaxation parameter
        N_min : int, optional, default = 5
            Limit on number of timesteps P is negative before decrementing timestep.
        Notes
        -----
        Velocities should be set to zero before using this integrator.
        """

        # Check input ranges.
        if not ((alpha > 0.0) and (alpha < 1.0)):
            raise Exception("alpha must be in the interval (0,1); specified alpha = %f" % alpha)

        if tolerance is None:
            tolerance = 0 * unit.kilojoules_per_mole / unit.nanometers

        super(FIREMinimizationIntegrator, self).__init__(timestep)

        # Use high-precision constraints
        self.setConstraintTolerance(1.0e-8)

        self.addGlobalVariable("alpha", alpha)  # alpha
        self.addGlobalVariable("P", 0)  # P
        self.addGlobalVariable("N_neg", 0.0)
        self.addGlobalVariable("fmag", 0)  # |f|
        self.addGlobalVariable("fmax", 0)  # max|f_i|
        self.addGlobalVariable("ndof", 0)  # number of degrees of freedom
        self.addGlobalVariable("ftol", tolerance.value_in_unit_system(unit.md_unit_system))  # convergence tolerance
        self.addGlobalVariable("vmag", 0)  # |v|
        self.addGlobalVariable("converged", 0) # 1 if convergence threshold reached, 0 otherwise
        self.addPerDofVariable("x0", 0)
        self.addPerDofVariable("v0", 0)
        self.addPerDofVariable("x1", 0)
        self.addGlobalVariable("E0", 0) # old energy associated with x0
        self.addGlobalVariable("dE", 0)
        self.addGlobalVariable("restart", 0)
        self.addGlobalVariable("delta_t", timestep.value_in_unit_system(unit.md_unit_system))

        # Update context state.
        self.addUpdateContextState()

        # Assess convergence
        # TODO: Can we more closely match the OpenMM criterion here?
        self.beginIfBlock('converged < 1')

        # Compute fmag = |f|
        #self.addComputeGlobal('fmag', '0.0')
        self.addComputeSum('fmag', 'f*f')
        self.addComputeGlobal('fmag', 'sqrt(fmag)')

        # Compute ndof
        self.addComputeSum('ndof', '1')

        self.addComputeSum('converged', 'step(ftol - fmag/ndof)')
        self.endBlock()

        # Enclose everything in a block that checks if we have already converged.
        self.beginIfBlock('converged < 1')

        # Store old positions and energy
        self.addComputePerDof('x0', 'x')
        self.addComputePerDof('v0', 'v')
        self.addComputeGlobal('E0', 'energy')

        # MD: Take a velocity Verlet step.
        self.addComputePerDof("v", "v+0.5*delta_t*f/m")
        self.addComputePerDof("x", "x+delta_t*v")
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions()
        self.addComputePerDof("v", "v+0.5*delta_t*f/m+(x-x1)/delta_t")
        self.addConstrainVelocities()

        self.addComputeGlobal('dE', 'energy - E0')

        # Compute fmag = |f|
        #self.addComputeGlobal('fmag', '0.0')
        self.addComputeSum('fmag', 'f*f')
        self.addComputeGlobal('fmag', 'sqrt(fmag)')
        # Compute vmag = |v|
        #self.addComputeGlobal('vmag', '0.0')
        self.addComputeSum('vmag', 'v*v')
        self.addComputeGlobal('vmag', 'sqrt(vmag)')

        # F1: Compute P = F.v
        self.addComputeSum('P', 'f*v')

        # F2: set v = (1-alpha) v + alpha \hat{F}.|v|
        # Update velocities.
        # TODO: This must be corrected to be atomwise redirection of v magnitude along f
        self.addComputePerDof('v', '(1-alpha)*v + alpha*(f/fmag)*vmag')

        # Back up if the energy went up, protecing against NaNs
        self.addComputeGlobal('restart', '1')
        self.beginIfBlock('dE < 0')
        self.addComputeGlobal('restart', '0')
        self.endBlock()
        self.beginIfBlock('restart > 0')
        self.addComputePerDof('v', 'v0')
        self.addComputePerDof('x', 'x0')
        self.addComputeGlobal('P', '-1')
        self.endBlock()

        # If dt goes to zero, signal we've converged!
        dt_min = 1.0e-5 * timestep
        self.beginIfBlock('delta_t <= %f' % dt_min.value_in_unit_system(unit.md_unit_system))
        self.addComputeGlobal('converged', '1')
        self.endBlock()

        # F3: If P > 0 and the number of steps since P was negative > N_min,
        # Increase timestep dt = min(dt*f_inc, dt_max) and decrease alpha = alpha*f_alpha
        self.beginIfBlock('P > 0')
        # Update count of number of steps since P was negative.
        self.addComputeGlobal('N_neg', 'N_neg + 1')
        # If we have enough steps since P was negative, scale up timestep.
        self.beginIfBlock('N_neg > %d' % N_min)
        self.addComputeGlobal('delta_t', 'min(delta_t*%f, %f)' % (f_inc, dt_max.value_in_unit_system(unit.md_unit_system))) # TODO: Automatically convert dt_max to md units
        self.addComputeGlobal('alpha', 'alpha * %f' % f_alpha)
        self.endBlock()
        self.endBlock()

        # F4: If P < 0, decrease the timestep dt = dt*f_dec, freeze the system v=0,
        # and set alpha = alpha_start
        self.beginIfBlock('P < 0')
        self.addComputeGlobal('N_neg', '0.0')
        self.addComputeGlobal('delta_t', 'delta_t*%f' % f_dec)
        self.addComputePerDof('v', '0.0')
        self.addComputeGlobal('alpha', '%f' % alpha)
        self.endBlock()

        # Close block that checks for convergence.
        self.endBlock()


if __name__ == '__main__':
    import doctest
    doctest.testmod()
