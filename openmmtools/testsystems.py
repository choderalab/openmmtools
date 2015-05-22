"""
Module to generate Systems and positions for simple reference molecular systems for testing.

DESCRIPTION

This module provides functions for building a number of test systems of varying complexity,
useful for testing both OpenMM and various codes based on pyopenmm.

Note that the PYOPENMM_SOURCE_DIR must be set to point to where the PyOpenMM package is unpacked.

EXAMPLES

Create a 3D harmonic oscillator.

>>> from openmmtools import testsystems
>>> ho = testsystems.HarmonicOscillator()
>>> system, positions = ho.system, ho.positions

See list of methods for a complete list of provided test systems.

COPYRIGHT

@author John D. Chodera <john.chodera@choderalab.org>
@author Randall J. Radmer <radmer@stanford.edu>

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

TODO

* Add units checking code to check arguments.
* Change default arguments to Quantity objects, rather than None?

"""

import os
import os.path
import numpy as np
import numpy.random
import itertools

import scipy
import scipy.special
import scipy.integrate

from simtk import openmm
from simtk import unit
from simtk.openmm import app

from .constants import kB

pi = np.pi

#=============================================================================================
# SUBROUTINES
#=============================================================================================


def in_openmm_units(quantity):
    """Strip the units from a simtk.unit.Quantity object after converting to natural OpenMM units

    Parameters
    ----------
    quantity : simtk.unit.Quantity
       The quantity to convert

    Returns
    -------
    unitless_quantity : float
       The quantity in natural OpenMM units, stripped of units.

    """

    unitless_quantity = quantity.in_unit_system(unit.md_unit_system)
    unitless_quantity /= unitless_quantity.unit
    return unitless_quantity


def get_data_filename(relative_path):
    """Get the full path to one of the reference files in testsystems.

    In the source distribution, these files are in ``openmmtools/data/*/``,
    but on installation, they're moved to somewhere in the user's python
    site-packages directory.

    Parameters
    ----------
    name : str
        Name of the file to load (with respect to the repex folder).

    """

    from pkg_resources import resource_filename
    fn = resource_filename('openmmtools', relative_path)

    if not os.path.exists(fn):
        raise ValueError("Sorry! %s does not exist. If you just added it, you'll have to re-install" % fn)

    return fn


def halton_sequence(p, n):
    """
    Halton deterministic sequence on [0,1].

    Parameters
    ----------
    p : int
       Prime number for sequence.
    n : int
       Sequence length to generate.

    Returns
    -------
    u : numpy.array of double
       Sequence on [0,1].

    Notes
    -----
    Code source: http://blue.math.buffalo.edu/sauer2py/
    More info: http://en.wikipedia.org/wiki/Halton_sequence

    Examples
    --------
    Generate some sequences with different prime number bases.
    >>> x = halton_sequence(2,100)
    >>> y = halton_sequence(3,100)
    >>> z = halton_sequence(5,100)

    """
    eps = np.finfo(np.double).eps
    b = np.zeros(np.ceil(np.log(n) / np.log(p)) + 1)   # largest number of digits (adding one for halton_sequence(2,64) corner case)
    u = np.empty(n)
    for j in range(n):
        i = 0
        b[0] += 1                       # add one to current integer
        while b[i] > p - 1 + eps:           # this loop does carrying in base p
            b[i] = 0
            i = i + 1
            b[i] += 1
        u[j] = 0
        for k in range(len(b)):         # add up reversed digits
            u[j] += b[k] * p**-(k + 1)
    return u


def subrandom_particle_positions(nparticles, box_vectors, method='sobol'):
    """Generate a deterministic list of subrandom particle positions.

    Parameters
    ----------
    nparticles : int
        The number of particles.
    box_vectors : simtk.unit.Quantity of (3,3) with units compatible with nanometer
        Periodic box vectors in which particles should lie.
    method : str, optional, default='sobol'
        Method for creating subrandom sequence (one of 'halton' or 'sobol')

    Returns
    -------
    positions : simtk.unit.Quantity of (natoms,3) with units compatible with nanometer
        The particle positions.

    Examples
    --------
    >>> nparticles = 216
    >>> box_vectors = openmm.System().getDefaultPeriodicBoxVectors()
    >>> positions = subrandom_particle_positions(nparticles, box_vectors)

    Use halton sequence:

    >>> nparticles = 216
    >>> box_vectors = openmm.System().getDefaultPeriodicBoxVectors()
    >>> positions = subrandom_particle_positions(nparticles, box_vectors, method='halton')

    """
    # Create positions array.
    positions = unit.Quantity(np.zeros([nparticles, 3], np.float32), unit.nanometers)

    if method == 'halton':
        # Fill in each dimension.
        primes = [2, 3, 5]  # prime bases for Halton sequence
        for dim in range(3):
            x = halton_sequence(primes[dim], nparticles)
            l = box_vectors[dim][dim]
            positions[:, dim] = unit.Quantity(x * l / l.unit, l.unit)

    elif method == 'sobol':
        # Generate Sobol' sequence.
        from openmmtools import sobol
        ivec = sobol.i4_sobol_generate(3, nparticles, 1)
        x = np.array(ivec, np.float32)
        for dim in range(3):
            l = box_vectors[dim][dim]
            positions[:, dim] = unit.Quantity(x[dim, :] * l / l.unit, l.unit)

    else:
        raise Exception("method '%s' must be 'halton' or 'sobol'" % method)

    return positions


def build_lattice_cell():
    """Build a single (4 atom) unit cell of a FCC lattice, assuming a cell length
    of 1.0.

    Returns
    -------
    xyz : np.ndarray, shape=(4, 3), dtype=float
        Coordinates of each particle in cell
    """
    xyz = [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5]]
    xyz = np.array(xyz)

    return xyz


def build_lattice(n_particles):
    """Build a FCC lattice with n_particles, where (n_particles / 4) must be a cubed integer.

    Parameters
    ----------
    n_particles : int
        How many particles.

    Returns
    -------
    xyz : np.ndarray, shape=(n_particles, 3), dtype=float
        Coordinates of each particle in box.  Each subcell is based on a unit-sized
        cell output by build_lattice_cell()
    n : int
        The number of cells along each direction.  Because each cell has unit
        length, `n` is also the total box length of the `n_particles` system.

    Notes
    -----
    Equations eyeballed from http://en.wikipedia.org/wiki/Close-packing_of_equal_spheres
    """
    n = ((n_particles / 4.) ** (1 / 3.))

    if np.abs(n - np.round(n)) > 1E-10:
        raise(ValueError("Must input 4 m^3 particles for some integer m!"))
    else:
        n = int(np.round(n))

    xyz = []
    cell = build_lattice_cell()
    x, y, z = np.eye(3)
    for atom, (i, j, k) in enumerate(itertools.product(np.arange(n), repeat=3)):
        xi = cell + i * x + j * y + k * z
        xyz.append(xi)

    xyz = np.concatenate(xyz)

    return xyz, n


def generate_dummy_trajectory(xyz, box):
    """Convert xyz coordinates and box vectors into an MDTraj Trajectory (with Topology)."""
    try:
        import mdtraj as md
        import pandas as pd
    except ImportError as e:
        print("Error: generate_dummy_trajectory() requires mdtraj and pandas!")
        raise(e)

    n_atoms = len(xyz)
    data = []

    for i in range(n_atoms):
        data.append(dict(serial=i, name="H", element="H", resSeq=i + 1, resName="UNK", chainID=0))

    data = pd.DataFrame(data)
    unitcell_lengths = box * np.ones((1, 3))
    unitcell_angles = 90 * np.ones((1, 3))
    top = md.Topology.from_dataframe(data, np.zeros((0, 2), dtype='int'))
    traj = md.Trajectory(xyz, top, unitcell_lengths=unitcell_lengths, unitcell_angles=unitcell_angles)

    return traj


#=============================================================================================
# Thermodynamic state description
#=============================================================================================

class ThermodynamicState(object):

    """Object describing a thermodynamic state obeying Boltzmann statistics.

    Examples
    --------

    Specify an NVT state for a water box at 298 K.

    >>> from openmmtools import testsystems
    >>> system_container = testsystems.WaterBox()
    >>> (system, positions) = system_container.system, system_container.positions
    >>> state = ThermodynamicState(system=system, temperature=298.0*unit.kelvin)

    Specify an NPT state at 298 K and 1 atm pressure.

    >>> state = ThermodynamicState(system=system, temperature=298.0*unit.kelvin, pressure=1.0*unit.atmospheres)

    Note that the pressure is only relevant for periodic systems.

    A barostat will be added to the system if none is attached.

    Notes
    -----

    This state object cannot describe states obeying non-Boltzamnn statistics, such as Tsallis statistics.

    ToDo
    ----

    * Implement a more fundamental ProbabilityState as a base class?
    * Implement pH.

    """

    def __init__(self, system=None, temperature=None, pressure=None):
        """Construct a thermodynamic state with given system and temperature.

        Parameters
        ----------

        system : simtk.openmm.System, optional, default=None
            System object describing the potential energy function for the system
        temperature : simtk.unit.Quantity compatible with 'kelvin', optional, default=None
            Temperature for a system with constant temperature
        pressure : simtk.unit.Quantity compatible with 'atmospheres', optional, default=None
            If not None, specifies the pressure for constant-pressure systems.


        """

        self.system = system
        self.temperature = temperature
        self.pressure = pressure

        return

#=============================================================================================
# Abstract base class for test systems
#=============================================================================================


class TestSystem(object):

    """Abstract base class for test systems, demonstrating how to implement a test system.

    Parameters
    ----------

    Attributes
    ----------
    system : simtk.openmm.System
        System object for the test system
    positions : list
        positions of test system
    topology : list
        topology of the test system

    Notes
    -----

    Unimplemented methods will default to the base class methods, which raise a NotImplementedException.

    Examples
    --------

    Create a test system.

    >>> testsystem = TestSystem()

    Retrieve a deep copy of the System object.

    >>> system = testsystem.system

    Retrieve a deep copy of the positions.

    >>> positions = testsystem.positions

    Retrieve a deep copy of the topology.

    >>> topology = testsystem.topology

    Serialize system and positions to XML (to aid in debugging).

    >>> (system_xml, positions_xml) = testsystem.serialize()

    """

    def __init__(self, **kwargs):
        """Abstract base class for test system.

        Parameters
        ----------

        """

        # Create an empty system object.
        self._system = openmm.System()

        # Store positions.
        self._positions = unit.Quantity(np.zeros([0, 3], np.float), unit.nanometers)

        # Empty topology.
        self._topology = app.Topology()

        return

    @property
    def system(self):
        """The simtk.openmm.System object corresponding to the test system."""
        return self._system

    @system.setter
    def system(self, value):
        self._system = value

    @system.deleter
    def system(self):
        del self._system

    @property
    def positions(self):
        """The simtk.unit.Quantity object containing the particle positions, with units compatible with simtk.unit.nanometers."""
        return self._positions

    @positions.setter
    def positions(self, value):
        self._positions = value

    @positions.deleter
    def positions(self):
        del self._positions

    @property
    def topology(self):
        """The simtk.openmm.app.Topology object corresponding to the test system."""
        return self._topology

    @topology.setter
    def topology(self, value):
        self._topology = value

    @topology.deleter
    def topology(self):
        del self._topology

    @property
    def analytical_properties(self):
        """A list of available analytical properties, accessible via 'get_propertyname(thermodynamic_state)' calls."""
        return [method[4:] for method in dir(self) if (method[0:4] == 'get_')]

    def reduced_potential_expectation(self, state_sampled_from, state_evaluated_in):
        """Calculate the expected potential energy in state_sampled_from, divided by kB * T in state_evaluated_in.

        Notes
        -----

        This is not called get_reduced_potential_expectation because this function
        requires two, not one, inputs.
        """

        if hasattr(self, "get_potential_expectation"):
            U = self.get_potential_expectation(state_sampled_from)
            U_red = U / (kB * state_evaluated_in.temperature)
            return U_red
        else:
            raise AttributeError("Cannot return reduced potential energy because system lacks get_potential_expectation")

    def serialize(self):
        """Return the System and positions in serialized XML form.

        Returns
        -------

        system_xml : str
            Serialized XML form of System object.

        state_xml : str
            Serialized XML form of State object containing particle positions.

        """

        from simtk.openmm import XmlSerializer

        # Serialize System.
        system_xml = XmlSerializer.serialize(self._system)

        # Serialize positions via State.
        if self._system.getNumParticles() == 0:
            # Cannot serialize the State of a system with no particles.
            state_xml = None
        else:
            platform = openmm.Platform.getPlatformByName('Reference')
            integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
            context = openmm.Context(self._system, integrator, platform)
            context.setPositions(self._positions)
            state = context.getState(getPositions=True)
            del context, integrator
            state_xml = XmlSerializer.serialize(state)

        return (system_xml, state_xml)

    @property
    def name(self):
        """The name of the test system."""
        return self.__class__.__name__


class CustomExternalForcesTestSystem(TestSystem):

    """Create a system with an arbitrary number of CustomExternalForces.

    Parameters
    ----------
    energy_expressions : tuple(string)
        Each string in the tuple will add a CustomExternalForce to the
        OpenMM system.  Each force will be assigned a different force
        group, starting with 0.  By default this will be a 3D harmonic oscillator.
    mass : simtk.unit.Quantity, optional, default=39.948 * unit.amu
        particle mass.  Default corresponds to argon.
    n_particles : int, optional, default=500
        Number of (identical) particles to add.

    Notes
    -----
    This may be useful for testing multiple timestep integrators.
    """

    def __init__(self, energy_expressions=("x^2 + y^2 + z^2",), mass=39.948 * unit.amu, n_particles=500, **kwargs):
        TestSystem.__init__(self, **kwargs)

        system = openmm.System()

        for n in range(n_particles):
            system.addParticle(mass)

        positions = unit.Quantity(np.zeros([n_particles, 3], np.float32), unit.angstroms)

        forces = [openmm.CustomExternalForce(energy_expression) for energy_expression in energy_expressions]

        for i, force in enumerate(forces):
            for n in range(n_particles):
                parameters = ()
                force.addParticle(n, parameters)
            force.setForceGroup(i)
            system.addForce(force)

        # Create topology.
        topology = app.Topology()
        element = app.Element.getBySymbol('Ar')
        chain = topology.addChain()
        for particle in range(n_particles):
            residue = topology.addResidue('Ar', chain)
            topology.addAtom('Ar', element, residue)

        self.topology = topology
        self.system, self.positions = system, positions
        self.n_particles = n_particles
        self.mass = mass
        self.ndof = 3 * n_particles

#=============================================================================================
# 3D harmonic oscillator
#=============================================================================================


class HarmonicOscillator(TestSystem):

    """Create a 3D harmonic oscillator, with a single particle confined in an isotropic harmonic well.

    Parameters
    ----------
    K : simtk.unit.Quantity, optional, default=90.0 * unit.kilocalories_per_mole/unit.angstrom**2
        harmonic restraining potential
    mass : simtk.unit.Quantity, optional, default=39.948 * unit.amu
        particle mass

    Attributes
    ----------
    system : simtk.openmm.System
        Openmm system with the harmonic oscillator
    positions : list
        positions of harmonic oscillator

    Notes
    -----

    The natural period of a harmonic oscillator is T = sqrt(m/K), so you will want to use an
    integration timestep smaller than ~ T/10.

    The standard deviation in position in each dimension is sigma = (kT / K)^(1/2)

    The expectation and standard deviation of the potential energy of a 3D harmonic oscillator is (3/2)kT.

    Examples
    --------

    Create a 3D harmonic oscillator with default parameters:

    >>> ho = HarmonicOscillator()
    >>> (system, positions) = ho.system, ho.positions

    Create a harmonic oscillator with specified mass and spring constant:

    >>> mass = 12.0 * unit.amu
    >>> K = 1.0 * unit.kilocalories_per_mole / unit.angstroms**2
    >>> ho = HarmonicOscillator(K=K, mass=mass)
    >>> (system, positions) = ho.system, ho.positions

    Get a list of the available analytically-computed properties.

    >>> print(ho.analytical_properties)
    ['potential_expectation', 'potential_standard_deviation']

    Compute the potential expectation and standard deviation

    >>> import simtk.unit as u
    >>> thermodynamic_state = ThermodynamicState(temperature=298.0*u.kelvin, system=system)
    >>> potential_mean = ho.get_potential_expectation(thermodynamic_state)
    >>> potential_stddev = ho.get_potential_standard_deviation(thermodynamic_state)

    """

    def __init__(self, K=100.0 * unit.kilocalories_per_mole / unit.angstroms**2, mass=39.948 * unit.amu, **kwargs):

        TestSystem.__init__(self, **kwargs)

        # Create an empty system object.
        system = openmm.System()

        # Add the particle to the system.
        system.addParticle(mass)

        # Set the positions.
        positions = unit.Quantity(np.zeros([1, 3], np.float32), unit.angstroms)

        # Add a restrining potential centered at the origin.
        energy_expression = '(K/2.0) * (x^2 + y^2 + z^2);'
        energy_expression += 'K = testsystems_HarmonicOscillator_K;'
        force = openmm.CustomExternalForce(energy_expression)
        force.addGlobalParameter('testsystems_HarmonicOscillator_K', K)
        force.addParticle(0, [])
        system.addForce(force)

        # Create topology.
        topology = app.Topology()
        element = app.Element.getBySymbol('Ar')
        chain = topology.addChain()
        residue = topology.addResidue('OSC', chain)
        topology.addAtom('Ar', element, residue)
        self.topology = topology

        self.K, self.mass = K, mass
        self.system, self.positions = system, positions

        # Number of degrees of freedom.
        self.ndof = 3

    def get_potential_expectation(self, state):
        """Return the expectation of the potential energy, computed analytically or numerically.

        Arguments
        ---------

        state : ThermodynamicState with temperature defined
            The thermodynamic state at which the property is to be computed.

        Returns
        -------

        potential_mean : simtk.unit.Quantity compatible with simtk.unit.kilojoules_per_mole
            The expectation of the potential energy.

        """

        return (3. / 2.) * kB * state.temperature

    def get_potential_standard_deviation(self, state):
        """Return the standard deviation of the potential energy, computed analytically or numerically.

        Arguments
        ---------

        state : ThermodynamicState with temperature defined
            The thermodynamic state at which the property is to be computed.

        Returns
        -------

        potential_stddev : simtk.unit.Quantity compatible with simtk.unit.kilojoules_per_mole
            potential energy standard deviation if implemented, or else None

        """

        return (3. / 2.) * kB * state.temperature


class PowerOscillator(TestSystem):

    """Create a 3D Power oscillator, with a single particle confined in an isotropic x^b well.

    Parameters
    ----------
    K : simtk.unit.Quantity, optional, default=100.0
        harmonic restraining potential.  The units depend on the power,
        so we accept unitless inputs and add units of the form
        unit.kilocalories_per_mole / unit.angstrom ** b
    mass : simtk.unit.Quantity, optional, default=39.948 * unit.amu
        particle mass

    Attributes
    ----------
    system : simtk.openmm.System
        Openmm system with the harmonic oscillator
    positions : list
        positions of harmonic oscillator

    Notes
    -----

    Here we assume a potential energy of the form U(x) = k * x^b.

    By the generalized equipartition theorem, the expectation of the
    potential energy is 3 kT / b.

    """

    def __init__(self, K=100.0, b=2.0, mass=39.948 * unit.amu, **kwargs):

        TestSystem.__init__(self, **kwargs)

        K = K * unit.kilocalories_per_mole / unit.angstroms ** b

        # Create an empty system object.
        system = openmm.System()

        # Add the particle to the system.
        system.addParticle(mass)

        # Set the positions.
        positions = unit.Quantity(np.zeros([1, 3], np.float32), unit.angstroms)

        # Add a restrining potential centered at the origin.
        energy_expression = 'K * (x^%d + y^%d + z^%d);' % (b, b, b)
        energy_expression += 'K = testsystems_PowerOscillator_K;'
        force = openmm.CustomExternalForce(energy_expression)
        force.addGlobalParameter('testsystems_PowerOscillator_K', K)
        force.addParticle(0, [])
        system.addForce(force)

        # Create topology.
        topology = app.Topology()
        element = app.Element.getBySymbol('Ar')
        chain = topology.addChain()
        residue = topology.addResidue('OSC', chain)
        topology.addAtom('Ar', element, residue)
        self.topology = topology

        self.K, self.mass = K, mass
        self.b = b
        self.system, self.positions = system, positions

        # Number of degrees of freedom.
        self.ndof = 3

    def get_potential_expectation(self, state):
        """Return the expectation of the potential energy, computed analytically or numerically.

        Arguments
        ---------

        state : ThermodynamicState with temperature defined
            The thermodynamic state at which the property is to be computed.

        Returns
        -------

        potential_mean : simtk.unit.Quantity compatible with simtk.unit.kilojoules_per_mole
            The expectation of the potential energy.

        """

        return (3.) * kB * state.temperature / self.b

    def _get_power_expectation(self, state, n):
        """Return the power of x^n.  Not currently used"""
        b = 1.0 * self.b
        beta = (1.0 * kB * state.temperature) ** -1.
        gamma = scipy.special.gamma
        return (self.K * beta) ** (-n / b) * gamma((n + 1.) / b) / gamma(1. / b)

    @classmethod
    def reduced_potential(cls, beta, a, b, a2, b2):
        gamma = scipy.special.gamma
        reduced_u = 3 * a2 * (a * beta) ** (-b2 / b) * gamma((b2 + 1.) / b) / gamma(1. / b) * beta
        return reduced_u

#=============================================================================================
# Diatomic molecule
#=============================================================================================


class Diatom(TestSystem):

    """Create a free diatomic molecule with a single harmonic bond between the two atoms.

    Parameters
    ----------
    K : simtk.unit.Quantity, optional, default=290.1 * unit.kilocalories_per_mole / unit.angstrom**2
        harmonic bond potential.  default is GAFF c-c bond
    r0 : simtk.unit.Quantity, optional, default=1.550 * unit.amu
        bond length.  Default is Amber GAFF c-c bond.
    constraint : bool, default=False
        if True, the bond length will be constrained
    m1 : simtk.unit.Quantity, optional, default=12.01 * unit.amu
        particle1 mass
    m2 : simtk.unit.Quantity, optional, default=12.01 * unit.amu
        particle2 mass
    use_central_potential : bool, optional, default=False
        if True, a soft central potential will also be added to keep the system from drifting away


    Notes
    -----

    The natural period of a harmonic oscillator is T = sqrt(m/K), so you will want to use an
    integration timestep smaller than ~ T/10.

    Examples
    --------

    Create a Diatom:

    >>> diatom = Diatom()
    >>> system, positions = diatom.system, diatom.positions

    Create a Diatom with constraint in a central potential
    >>> diatom = Diatom(constraint=True, use_central_potential=True)
    >>> system, positions = diatom.system, diatom.positions

    """

    def __init__(self,
                 K=290.1 * unit.kilocalories_per_mole / unit.angstrom**2,
                 r0=1.550 * unit.angstroms,
                 m1=39.948 * unit.amu,
                 m2=39.948 * unit.amu,
                 constraint=False,
                 use_central_potential=False, **kwargs):

        TestSystem.__init__(self, **kwargs)

        # Create an empty system object.
        system = openmm.System()

        # Add two particles to the system.
        system.addParticle(m1)
        system.addParticle(m2)

        # Add a harmonic bond.
        force = openmm.HarmonicBondForce()
        force.addBond(0, 1, r0, K)
        system.addForce(force)

        if constraint:
            # Add constraint between particles.
            system.addConstraint(0, 1, r0)

        # Set the positions.
        positions = unit.Quantity(np.zeros([2, 3], np.float32), unit.angstroms)
        positions[1, 0] = r0

        if use_central_potential:
            # Add a central restraining potential.
            Kcentral = 1.0 * unit.kilocalories_per_mole / unit.nanometer**2
            energy_expression = '(K/2.0) * (x^2 + y^2 + z^2);'
            energy_expression += 'K = testsystems_Diatom_Kcentral;'
            force = openmm.CustomExternalForce(energy_expression)
            force.addGlobalParameter('testsystems_Diatom_Kcentral', Kcentral)
            force.addParticle(0, [])
            force.addParticle(1, [])
            system.addForce(force)

        # Create topology.
        topology = app.Topology()
        element = app.Element.getBySymbol('N')
        chain = topology.addChain()
        residue = topology.addResidue('N2', chain)
        topology.addAtom('N', element, residue)
        topology.addAtom('N', element, residue)
        self.topology = topology

        self.system, self.positions = system, positions
        self.K, self.r0, self.m1, self.m2, self.constraint, self.use_central_potential = K, r0, m1, m2, constraint, use_central_potential

        # Store number of degrees of freedom.
        self.ndof = 6 - 1 * constraint

    def get_potential_expectation(self, state):
        """Return the expectation of the potential energy, computed analytically or numerically.

        Parameters
        ----------
        state : ThermodynamicState with temperature defined
            The thermodynamic state at which the property is to be computed.

        Returns
        -------

        potential_mean : simtk.unit.Quantity compatible with simtk.unit.kilojoules_per_mole
            The expectation of the potential energy.

        """

        return (self.ndof / 2.) * kB * state.temperature

#=============================================================================================
# Diatomic fluid
#=============================================================================================


class DiatomicFluid(TestSystem):

    """Create a diatomic fluid.

    Note
    ----
    The default reduced_density is set to 0.05 (gas) so that no minimization is needed to simulate the default system.

    Parameters
    ----------
    nmolecules : int, optional, default=250
        Number of molecules.
    K : simtk.unit.Quantity, optional, default=290.1 * unit.kilocalories_per_mole / unit.angstrom**2
        harmonic bond potential.  default is GAFF c-c bond
    r0 : simtk.unit.Quantity, optional, default=1.550 * unit.amu
        bond length.  Default is Amber GAFF c-c bond.
    constraint : bool, default=False
        if True, the bond length will be constrained
    m1 : simtk.unit.Quantity, optional, default=12.01 * unit.amu
        particle1 mass
    m2 : simtk.unit.Quantity, optional, default=12.01 * unit.amu
        particle2 mass
    epsilon : simtk.unit.Quantity, optional, default=0.1700 * unit.kilocalories_per_mole
        particle Lennard-Jones well depth
    sigma : simtk.unit.Quantity, optional, default=1.8240 * unit.angstroms
        particle Lennard-Jones sigma
    charge : simtk.unit.Quantity, optional, default=0.0 * unit.elementary_charge
        charge to place on atomic centers to create a dipole
    reduced_density : float, optional, default=0.05
        Reduced density (density * sigma**3); default is appropriate for gas
    cutoff : simtk.unit.Quantity, optional, default=None
        if specified, the specified cutoff will be used; otherwise, 3.0 * sigma will be used
    switch_width : simtk.unit.Quantity with units compatible with angstroms, optional, default=0.2*unit.angstroms
        switching function is turned on at cutoff - switch_width
        If None, no switch will be applied (e.g. hard cutoff).
    dispersion_correction : bool, optional, default=True
        if True, will use analytical dispersion correction (if not using switching function)


    Notes
    -----

    The natural period of a harmonic oscillator is T = sqrt(m/K), so you will want to use an
    integration timestep smaller than ~ T/10.

    Examples
    --------

    Create an uncharged Diatomic fluid.

    >>> diatom = DiatomicFluid()
    >>> system, positions = diatom.system, diatom.positions

    Create a dipolar fluid.

    >>> diatom = DiatomicFluid(charge=1.0*unit.elementary_charge)
    >>> system, positions = diatom.system, diatom.positions

    Create a Diatomic fluid with constraints instead of harmonic bonds

    >>> diatom = DiatomicFluid(constraint=True)
    >>> system, positions = diatom.system, diatom.positions

    Specify a different system size.

    >>> diatom = DiatomicFluid(constraint=True, nmolecules=200)
    >>> system, positions = diatom.system, diatom.positions

    """

    def __init__(self,
                 nmolecules=250,
                 K=424.0 * unit.kilocalories_per_mole / unit.angstrom**2,
                 r0=1.383 * unit.angstroms,
                 m1=14.01 * unit.amu,
                 m2=14.01 * unit.amu,
                 epsilon=0.1700 * unit.kilocalories_per_mole,
                 sigma=1.8240 * unit.angstroms,
                 charge=0.0 * unit.elementary_charge,
                 reduced_density=0.05,
                 switch_width=0.5 * unit.angstroms,
                 cutoff=None,
                 constraint=False,
                 dispersion_correction=True,
                 **kwargs):

        TestSystem.__init__(self, **kwargs)

        nparticles = 2 * nmolecules

        # Create an empty system object.
        system = openmm.System()

        # Add particles to the system.
        for molecule_index in range(nmolecules):
            system.addParticle(m1)
            system.addParticle(m2)

        if constraint:
            # Add constraint between particles.
            for molecule_index in range(nmolecules):
                system.addConstraint(2 * molecule_index + 0, 2 * molecule_index + 1, r0)
        else:
            # Add a harmonic bonds.
            force = openmm.HarmonicBondForce()
            for molecule_index in range(nmolecules):
                force.addBond(2 * molecule_index + 0, 2 * molecule_index + 1, r0, K)
            system.addForce(force)

        # Set up nonbonded interactions.
        nb = openmm.NonbondedForce()

        # Create particle pairs.
        for atom_index in range(nmolecules):
            nb.addParticle(+charge, sigma, epsilon)
            nb.addParticle(-charge, sigma, epsilon)

        # Determine Lennard-Jones cutoff.
        if cutoff is None:
            cutoff = 3.0 * sigma

        # Determine volume and periodic box vectors.
        number_density = reduced_density / sigma**3
        volume = nparticles * (number_density ** -1)
        box_edge = volume ** (1. / 3.)
        a = unit.Quantity((box_edge,        0 * unit.angstrom, 0 * unit.angstrom))
        b = unit.Quantity((0 * unit.angstrom, box_edge,        0 * unit.angstrom))
        c = unit.Quantity((0 * unit.angstrom, 0 * unit.angstrom, box_edge))
        system.setDefaultPeriodicBoxVectors(a, b, c)

        # Create initial molecule centers of geometry using subrandom positions.
        molecule_positions = subrandom_particle_positions(nmolecules, system.getDefaultPeriodicBoxVectors())  # for molecule centers
        molecule_directions = subrandom_particle_positions(nmolecules, system.getDefaultPeriodicBoxVectors())  # for molecule orientations

        # Compute particle positions.
        positions = unit.Quantity(np.zeros([nparticles, 3], np.float32), unit.angstroms)
        unit_vector = np.array([1, 0, 0], np.float32)
        for molecule_index in range(0, nmolecules):
            vector = molecule_directions[molecule_index, :] - molecule_directions.mean(0)
            unit_vector = vector / unit.norm(vector)
            positions[2 * molecule_index + 0, :] = molecule_positions[molecule_index, :] + 0.5 * r0 * unit_vector
            positions[2 * molecule_index + 1, :] = molecule_positions[molecule_index, :] - 0.5 * r0 * unit_vector

        # Add exceptions for intramolecular forces.
        for molecule_index in range(nmolecules):
            nb.addException(2 * molecule_index + 0, 2 * molecule_index + 1, 0.0 * charge * charge, sigma, 0.0 * epsilon)

        system.addForce(nb)

        nb.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
        nb.setUseDispersionCorrection(dispersion_correction)
        nb.setCutoffDistance(cutoff)

        nb.setUseSwitchingFunction(False)
        if switch_width is not None:
            nb.setUseSwitchingFunction(True)
            nb.setSwitchingDistance(cutoff - switch_width)

        # Store number of degrees of freedom.
        self.ndof = 3 * nparticles - nmolecules * constraint

        # Create topology.
        topology = app.Topology()
        element = app.Element.getBySymbol('N')
        chain = topology.addChain()
        for molecule_index in range(nmolecules):
            residue = topology.addResidue('N2', chain)
            topology.addAtom('N', element, residue)
            topology.addAtom('N', element, residue)
        self.topology = topology

        # Store system and positions.
        self._system = system
        self._positions = positions

    def get_potential_expectation(self, state):
        """Return the expectation of the potential energy, computed analytically or numerically.

        Parameters
        ---------
        state : ThermodynamicState with temperature defined
            The thermodynamic state at which the property is to be computed.

        Returns
        -------
        potential_mean : simtk.unit.Quantity compatible with simtk.unit.kilojoules_per_mole
            The expectation of the potential energy.

        """

        return (self.ndof / 2.) * kB * state.temperature


class UnconstrainedDiatomicFluid(DiatomicFluid):

    """
    Examples
    --------

    Create an unconstrained diatomic fluid.

    >>> test = UnconstrainedDiatomicFluid()
    >>> system, positions = test.system, test.positions

    """

    def __init__(self, *args, **kwargs):
        super(UnconstrainedDiatomicFluid, self).__init__(constraint=False, *args, **kwargs)


class ConstrainedDiatomicFluid(DiatomicFluid):

    """
    Examples
    --------

    Create an constrained diatomic fluid.

    >>> test = ConstrainedDiatomicFluid()
    >>> system, positions = test.system, test.positions

    """

    def __init__(self, *args, **kwargs):
        super(ConstrainedDiatomicFluid, self).__init__(constraint=True, *args, **kwargs)


class DipolarFluid(DiatomicFluid):

    """
    Examples
    --------

    Create a dipolar fluid.

    >>> test = DipolarFluid()
    >>> system, positions = test.system, test.positions

    """

    def __init__(self, *args, **kwargs):
        super(DipolarFluid, self).__init__(charge=0.25 * unit.elementary_charge, *args, **kwargs)


class UnconstrainedDipolarFluid(DipolarFluid):

    """
    Examples
    --------

    Create a dipolar fluid.

    >>> test = UnconstrainedDipolarFluid()
    >>> system, positions = test.system, test.positions

    """

    def __init__(self, *args, **kwargs):
        super(UnconstrainedDipolarFluid, self).__init__(constraint=False, *args, **kwargs)


class ConstrainedDipolarFluid(DipolarFluid):

    """
    Examples
    --------

    Create a dipolar fluid.

    >>> test = ConstrainedDipolarFluid()
    >>> system, positions = test.system, test.positions

    """

    def __init__(self, *args, **kwargs):
        super(ConstrainedDipolarFluid, self).__init__(constraint=True, *args, **kwargs)

#=============================================================================================
# Constraint-coupled harmonic oscillator
#=============================================================================================


class ConstraintCoupledHarmonicOscillator(TestSystem):

    """Create a pair of particles in 3D harmonic oscillator wells, coupled by a constraint.

    Parameters
    ----------
    K : simtk.unit.Quantity, optional, default=1.0 * unit.kilojoules_per_mole / unit.nanometer**2
        harmonic restraining potential
    d : simtk.unit.Quantity, optional, default=1.0 * unit.nanometer
        distance between harmonic oscillators.  Default is Amber GAFF c-c bond.
    mass : simtk.unit.Quantity, default=39.948 * unit.amu
        particle mass

    Attributes
    ----------
    system : simtk.openmm.System
    positions : list

    Notes
    -----

    The natural period of a harmonic oscillator is T = sqrt(m/K), so you will want to use an
    integration timestep smaller than ~ T/10.

    Examples
    --------

    Create a constraint-coupled harmonic oscillator with specified mass, distance, and spring constant.

    >>> ccho = ConstraintCoupledHarmonicOscillator()
    >>> mass = 12.0 * unit.amu
    >>> d = 5.0 * unit.angstroms
    >>> K = 1.0 * unit.kilocalories_per_mole / unit.angstroms**2
    >>> ccho = ConstraintCoupledHarmonicOscillator(K=K, d=d, mass=mass)
    >>> system, positions = ccho.system, ccho.positions
    """

    def __init__(self,
                 K=1.0 * unit.kilojoules_per_mole / unit.nanometer**2,
                 d=1.0 * unit.nanometer,
                 mass=39.948 * unit.amu, **kwargs):

        TestSystem.__init__(self, **kwargs)

        # Create an empty system object.
        system = openmm.System()

        # Add particles to the system.
        system.addParticle(mass)
        system.addParticle(mass)

        # Set the positions.
        positions = unit.Quantity(np.zeros([2, 3], np.float32), unit.angstroms)
        positions[1, 0] = d

        # Add a restrining potential centered at the origin.
        energy_expression = '(K/2.0) * ((x-d)^2 + y^2 + z^2);'
        energy_expression += 'K = testsystems_ConstraintCoupledHarmonicOscillator_K;'
        force = openmm.CustomExternalForce(energy_expression)
        force.addGlobalParameter('testsystems_ConstraintCoupledHarmonicOscillator_K', K)
        force.addPerParticleParameter('d')
        force.addParticle(0, [0.0])
        force.addParticle(1, [d / unit.nanometers])
        system.addForce(force)

        # Add constraint between particles.
        system.addConstraint(0, 1, d)

        # Add a harmonic bond force as well so minimization will roughly satisfy constraints.
        force = openmm.HarmonicBondForce()
        K = 10.0 * unit.kilocalories_per_mole / unit.angstrom**2  # force constant
        force.addBond(0, 1, d, K)
        system.addForce(force)

        # Create topology.
        topology = app.Topology()
        element = app.Element.getBySymbol('N')
        chain = topology.addChain()
        residue = topology.addResidue('N2', chain)
        topology.addAtom('N', element, residue)
        topology.addAtom('N', element, residue)
        self.topology = topology

        self.system, self.positions = system, positions
        self.K, self.d, self.mass = K, d, mass

#=============================================================================================
# Harmonic oscillator array
#=============================================================================================


class HarmonicOscillatorArray(TestSystem):

    """Create a 1D array of noninteracting particles in 3D harmonic oscillator wells.

    Parameters
    ----------
    K : simtk.unit.Quantity, optional, default=90.0 * unit.kilocalories_per_mole/unit.angstroms**2
        harmonic restraining potential
    d : simtk.unit.Quantity, optional, default=1.0 * unit.nanometer
        distance between harmonic oscillators.  Default is Amber GAFF c-c bond.
    mass : simtk.unit.Quantity, default=39.948 * unit.amu
        particle mass
    N : int, optional, default=5
        Number of harmonic oscillators

    Attributes
    ----------
    system : simtk.openmm.System
    positions : list

    Notes
    -----

    The natural period of a harmonic oscillator is T = sqrt(m/K), so you will want to use an
    integration timestep smaller than ~ T/10.

    Examples
    --------

    Create a constraint-coupled 3D harmonic oscillator with default parameters.

    >>> ho_array = HarmonicOscillatorArray()
    >>> mass = 12.0 * unit.amu
    >>> d = 5.0 * unit.angstroms
    >>> K = 1.0 * unit.kilocalories_per_mole / unit.angstroms**2
    >>> ccho = HarmonicOscillatorArray(K=K, d=d, mass=mass)
    >>> system, positions = ccho.system, ccho.positions
    """

    def __init__(self, K=90.0 * unit.kilocalories_per_mole / unit.angstroms**2,
                 d=1.0 * unit.nanometer,
                 mass=39.948 * unit.amu,
                 N=5, **kwargs):

        TestSystem.__init__(self, **kwargs)

        # Create an empty system object.
        system = openmm.System()

        # Add particles to the system.
        for n in range(N):
            system.addParticle(mass)

        # Set the positions for a 1D array of particles spaced d apart along the x-axis.
        positions = unit.Quantity(np.zeros([N, 3], np.float32), unit.angstroms)
        for n in range(N):
            positions[n, 0] = n * d

        # Add a restrining potential for each oscillator.
        energy_expression = '(K/2.0) * ((x-x0)^2 + y^2 + z^2);'
        energy_expression += 'K = testsystems_HarmonicOscillatorArray_K;'
        force = openmm.CustomExternalForce(energy_expression)
        force.addGlobalParameter('testsystems_HarmonicOscillatorArray_K', K)
        force.addPerParticleParameter('x0')
        for n in range(N):
            parameters = (d * n / unit.nanometers, )
            force.addParticle(n, parameters)
        system.addForce(force)

        # Create topology.
        topology = app.Topology()
        element = app.Element.getBySymbol('Ar')
        chain = topology.addChain()
        for particle in range(N):
            residue = topology.addResidue('Ar', chain)
            topology.addAtom('Ar', element, residue)
        self.topology = topology

        self.system, self.positions = system, positions
        self.K, self.d, self.mass, self.N = K, d, mass, N
        self.ndof = 3 * N

    def get_potential_expectation(self, state):
        """Return the expectation of the potential energy, computed analytically or numerically.

        Parameters
        ----------
        state : ThermodynamicState with temperature defined
            The thermodynamic state at which the property is to be computed.

        Returns
        -------
        potential_mean : simtk.unit.Quantity compatible with simtk.unit.kilojoules_per_mole
            The expectation of the potential energy.

        """

        return (self.ndof / 2.) * kB * state.temperature

    def get_potential_standard_deviation(self, state):
        """Return the standard deviation of the potential energy, computed analytically or numerically.

        Parameters
        ---------
        state : ThermodynamicState with temperature defined
            The thermodynamic state at which the property is to be computed.

        Returns
        -------
        potential_stddev : simtk.unit.Quantity compatible with simtk.unit.kilojoules_per_mole
            potential energy standard deviation if implemented, or else None

        """

        return (self.ndof / 2.) * kB * state.temperature

#=============================================================================================
# Sodium chloride FCC crystal.
#=============================================================================================


class SodiumChlorideCrystal(TestSystem):

    """Create an FCC crystal of sodium chloride.

    Each atom is represented by a charged Lennard-Jones sphere in an Ewald lattice.

    switch_width : simtk.unit.Quantity with units compatible with angstroms, optional, default=0.2*unit.angstroms
        switching function is turned on at cutoff - switch_width
        If None, no switch will be applied (e.g. hard cutoff).
    dispersion_correction : bool, optional, default=True
        if True, will use analytical dispersion correction (if not using switching function)

    Notes
    -----

    TODO

    * Lennard-Jones interactions aren't correctly being included now, due to LJ cutoff.  Fix this by hard-coding LJ interactions?
    * Add nx, ny, nz arguments to allow user to specify replication of crystal unit in x,y,z.
    * Choose more appropriate lattice parameters and lattice spacing.

    Examples
    --------

    Create sodium chloride crystal unit.

    >>> crystal = SodiumChlorideCrystal()
    >>> system, positions = crystal.system, crystal.positions
    """

    def __init__(self, switch_width=0.2 * unit.angstroms, dispersion_correction=True, **kwargs):

        TestSystem.__init__(self, **kwargs)

        # Set default parameters (from Tinker).
        mass_Na = 22.990 * unit.amu
        mass_Cl = 35.453 * unit.amu
        q_Na = 1.0 * unit.elementary_charge
        q_Cl = -1.0 * unit.elementary_charge
        sigma_Na = 3.330445 * unit.angstrom
        sigma_Cl = 4.41724 * unit.angstrom
        epsilon_Na = 0.002772 * unit.kilocalorie_per_mole
        epsilon_Cl = 0.118 * unit.kilocalorie_per_mole

        # Create system
        system = openmm.System()

        # Create topology.
        topology = app.Topology()
        chain = topology.addChain()

        # Set box vectors.
        box_size = 5.628 * unit.angstroms  # box width
        a = unit.Quantity(np.zeros([3]), unit.nanometers)
        a[0] = box_size
        b = unit.Quantity(np.zeros([3]), unit.nanometers)
        b[1] = box_size
        c = unit.Quantity(np.zeros([3]), unit.nanometers)
        c[2] = box_size
        system.setDefaultPeriodicBoxVectors(a, b, c)

        # Create nonbonded force term.
        force = openmm.NonbondedForce()

        # Set interactions to be periodic Ewald.
        force.setNonbondedMethod(openmm.NonbondedForce.Ewald)

        # Set cutoff to be less than one half the box length.
        cutoff = box_size / 2.0 * 0.99
        force.setCutoffDistance(cutoff)

        # Set treatment.
        force.setUseDispersionCorrection(dispersion_correction)
        force.setUseSwitchingFunction(False)
        if switch_width is not None:
            force.setUseSwitchingFunction(True)
            force.setSwitchingDistance(cutoff - switch_width)

        # Allocate storage for positions.
        natoms = 2
        positions = unit.Quantity(np.zeros([natoms, 3], np.float32), unit.angstroms)

        # Add sodium ion.
        system.addParticle(mass_Na)
        force.addParticle(q_Na, sigma_Na, epsilon_Na)
        positions[0, 0] = 0.0 * unit.angstrom
        positions[0, 1] = 0.0 * unit.angstrom
        positions[0, 2] = 0.0 * unit.angstrom

        element = app.Element.getBySymbol('Na')
        residue = topology.addResidue('Na+', chain)
        topology.addAtom('Na+', element, residue)

        # Add chloride atom.
        system.addParticle(mass_Cl)
        force.addParticle(q_Cl, sigma_Cl, epsilon_Cl)
        positions[1, 0] = 2.814 * unit.angstrom
        positions[1, 1] = 2.814 * unit.angstrom
        positions[1, 2] = 2.814 * unit.angstrom

        element = app.Element.getBySymbol('Cl')
        residue = topology.addResidue('Cl-', chain)
        topology.addAtom('Cl-', element, residue)

        # Add nonbonded force term to the system.
        system.addForce(force)

        self.topology = topology
        self.system, self.positions = system, positions

#=============================================================================================
# Lennard-Jones cluster
#=============================================================================================


class LennardJonesCluster(TestSystem):

    """Create a non-periodic rectilinear grid of Lennard-Jones particles in a harmonic restraining potential.

    Parameters
    ----------
    nx : int, optional, default=3
        number of particles in the x direction
    ny : int, optional, default=3
        number of particles in the y direction
    nz : int, optional, default=3
        number of particles in the z direction
    K : simtk.unit.Quantity, optional, default=1.0 * unit.kilojoules_per_mole/unit.nanometer**2
        harmonic restraining potential
    cutoff : simtk.unit.Quantity, optional, default=None
        If None, will use NoCutoff for the NonbondedForce.  Otherwise,
        use CutoffNonPeriodic with the specified cutoff.
    switch_width : simtk.unit.Quantity, optional, default=None
        If None, the cutoff is a hard cutoff.  If switch_width is specified,
        use a switching function with this width.

    Examples
    --------

    Create Lennard-Jones cluster.

    >>> cluster = LennardJonesCluster()
    >>> system, positions = cluster.system, cluster.positions

    Create default 3x3x3 Lennard-Jones cluster in a harmonic restraining potential.

    >>> cluster = LennardJonesCluster(nx=10, ny=10, nz=10)
    >>> system, positions = cluster.system, cluster.positions
    """

    def __init__(self, nx=3, ny=3, nz=3, K=1.0 * unit.kilojoules_per_mole / unit.nanometer**2, cutoff=None, switch_width=None, **kwargs):

        TestSystem.__init__(self, **kwargs)

        # Default parameters
        mass_Ar = 39.9 * unit.amu
        q_Ar = 0.0 * unit.elementary_charge
        sigma_Ar = 3.350 * unit.angstrom
        epsilon_Ar = 0.001603 * unit.kilojoule_per_mole

        scaleStepSizeX = 1.0
        scaleStepSizeY = 1.0
        scaleStepSizeZ = 1.0

        # Determine total number of atoms.
        natoms = nx * ny * nz

        # Create an empty system object.
        system = openmm.System()

        # Create a nonperiodic NonbondedForce object.
        nb = openmm.NonbondedForce()

        if cutoff is None:
            nb.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
        else:
            nb.setNonbondedMethod(openmm.NonbondedForce.CutoffNonPeriodic)
            nb.setCutoffDistance(cutoff)
            nb.setUseDispersionCorrection(False)
            nb.setUseSwitchingFunction(False)
            if switch_width is not None:
                nb.setUseSwitchingFunction(True)
                nb.setSwitchingDistance(cutoff - switch_width)

        positions = unit.Quantity(np.zeros([natoms, 3], np.float32), unit.angstrom)

        atom_index = 0
        for ii in range(nx):
            for jj in range(ny):
                for kk in range(nz):
                    system.addParticle(mass_Ar)
                    nb.addParticle(q_Ar, sigma_Ar, epsilon_Ar)
                    x = sigma_Ar * scaleStepSizeX * (ii - nx / 2.0)
                    y = sigma_Ar * scaleStepSizeY * (jj - ny / 2.0)
                    z = sigma_Ar * scaleStepSizeZ * (kk - nz / 2.0)

                    positions[atom_index, 0] = x
                    positions[atom_index, 1] = y
                    positions[atom_index, 2] = z
                    atom_index += 1

        # Add the nonbonded force.
        system.addForce(nb)

        # Create topology.
        topology = app.Topology()
        element = app.Element.getBySymbol('Ar')
        chain = topology.addChain()
        for particle in range(system.getNumParticles()):
            residue = topology.addResidue('Ar', chain)
            topology.addAtom('Ar', element, residue)
        self.topology = topology

        # Add a restrining potential centered at the origin.
        energy_expression = '(K/2.0) * (x^2 + y^2 + z^2);'
        energy_expression += 'K = %f;' % (K / (unit.kilojoules_per_mole / unit.nanometers**2))  # in OpenMM units
        force = openmm.CustomExternalForce(energy_expression)
        for particle_index in range(natoms):
            force.addParticle(particle_index, [])
        system.addForce(force)

        self.system, self.positions = system, positions

#=============================================================================================
# Lennard-Jones fluid
#=============================================================================================


class LennardJonesFluid(TestSystem):

    """Create a periodic fluid of Lennard-Jones particles.
    Initial positions are assigned using a subrandom grid to minimize steric interactions.

    Note
    ----
    The default reduced_density is set to 0.05 (gas) so that no minimization is needed to simulate the default system.

    Parameters
    ----------
    nparticles : int, optional, default=1000
        Number of Lennard-Jones particles.
    reduced_density : float, optional, default=0.05
        Reduced density (density * sigma**3); default is appropriate for gas
    mass : simtk.unit.Quantity, optional, default=39.9 * unit.amu
        mass of each particle; default is appropriate for argon
    sigma : simtk.unit.Quantity, optional, default=3.4 * unit.angstrom
        Lennard-Jones sigma parameter; default is appropriate for argon
    epsilon : simtk.unit.Quantity, optional, default=0.238 * unit.kilocalories_per_mole
        Lennard-Jones well depth; default is appropriate for argon
    cutoff : simtk.unit.Quantity, optional, default=None
        Cutoff for nonbonded interactions.  If None, defaults to 3.0 * sigma
    switch_width : simtk.unit.Quantity with units compatible with angstroms, optional, default=3.4 * unit.angstrom
        switching function is turned on at cutoff - switch_width
        If None, no switch will be applied (e.g. hard cutoff).
        Ignored if `shift=True`.
    shift : bool, optional, default=False
        If True, will shift Lennard-Jones potential so energy will be continuous at cutoff (switch_width is ignored).
    dispersion_correction : bool, optional, default=True
        if True, will use analytical dispersion correction (if not using switching function)
    lattice : bool, optional, default=False
        If True, use fcc sphere packing to generate initial positions.  The box
        size will be determined by `nparticles` and `reduced_density`.
    charge : simtk.unit, optional, default=None
        If not None, use alternating plus and minus `charge` for the particle charges.
        Also, if not None, use PME for electrostatics.  Obviously this is no
        longer a traditional LJ system, but this option could be useful for
        testing the effect of charges in small systems.
    ewaldErrorTolerance : float, optional, default=5E-4
           The Ewald or PME tolerance.  Used only if charge is not None.

    Examples
    --------

    Create default-size Lennard-Jones fluid.

    >>> fluid = LennardJonesFluid()
    >>> system, positions = fluid.system, fluid.positions

    Create a larger box of Lennard-Jones particles with specified reduced density.

    >>> fluid = LennardJonesFluid(nparticles=1000, reduced_density=0.50)
    >>> system, positions = fluid.system, fluid.positions

    Create Lennard-Jones fluid using switched particle interactions (switched off betwee 7 and 9 A) and more particles.

    >>> fluid = LennardJonesFluid(switch_width=2.0*unit.angstroms, cutoff=9.0*unit.angstroms)
    >>> system, positions = fluid.system, fluid.positions

    Create Lennard-Jones fluid using shifted potential.

    >>> fluid = LennardJonesFluid(cutoff=9.0*unit.angstroms, shift=True)
    >>> system, positions = fluid.system, fluid.positions

    """

    def __init__(self,
                 nparticles=1000,
                 reduced_density=0.05,
                 mass=39.9 * unit.amu,  # argon
                 sigma=3.4 * unit.angstrom,  # argon,
                 epsilon=0.238 * unit.kilocalories_per_mole,  # argon,
                 cutoff=None,
                 switch_width=3.4 * unit.angstrom,  # argon
                 shift=False,
                 dispersion_correction=True,
                 lattice=False,
                 charge=None,
                 ewaldErrorTolerance=None,
                 **kwargs):

        TestSystem.__init__(self, **kwargs)

        # Determine Lennard-Jones cutoff.
        if cutoff is None:
            cutoff = 3.0 * sigma

        if charge is None:  # Charge is zero.
            charge = 0.0 * unit.elementary_charge
            cutoff_type = openmm.NonbondedForce.CutoffPeriodic
        else:
            cutoff_type = openmm.NonbondedForce.PME

        # Create an empty system object.
        system = openmm.System()

        # Determine volume and periodic box vectors.
        number_density = reduced_density / sigma**3
        volume = nparticles * (number_density ** -1)
        box_edge = volume ** (1. / 3.)
        a = unit.Quantity((box_edge,        0 * unit.angstrom, 0 * unit.angstrom))
        b = unit.Quantity((0 * unit.angstrom, box_edge,        0 * unit.angstrom))
        c = unit.Quantity((0 * unit.angstrom, 0 * unit.angstrom, box_edge))
        system.setDefaultPeriodicBoxVectors(a, b, c)

        # Set up periodic nonbonded interactions with a cutoff.
        nb = openmm.NonbondedForce()
        nb.setNonbondedMethod(cutoff_type)
        nb.setCutoffDistance(cutoff)
        nb.setUseDispersionCorrection(dispersion_correction)
        if ewaldErrorTolerance is not None:
            nb.setEwaldErrorTolerance(ewaldErrorTolerance)

        nb.setUseSwitchingFunction(False)
        if (switch_width != None) and (not shift):
            nb.setUseSwitchingFunction(True)
            nb.setSwitchingDistance(cutoff - switch_width)

        for particle_index in range(nparticles):
            system.addParticle(mass)
            if cutoff_type == openmm.NonbondedForce.PME:
                charge_i = charge * ((particle_index % 2) * 2 - 1.)  # Alternate plus and minus
            else:
                charge_i = charge
            nb.addParticle(charge_i, sigma, epsilon)

        # Add shift if desired.
        if (shift):
            shift_potential = - 4 * epsilon * ((sigma / cutoff)**12 - (sigma / cutoff)**6)  # amount by which potential is to be shifted
            cnb = openmm.CustomNonbondedForce('%f' % in_openmm_units(shift_potential))
            cnb.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
            cnb.setUseSwitchingFunction(False)
            cnb.setCutoffDistance(cutoff)
            for particle_index in range(nparticles):
                cnb.addParticle([])
            system.addForce(cnb)

        if lattice:
            box_nm = box_edge / unit.nanometers
            xyz, box = build_lattice(nparticles)
            xyz *= (box_nm / box)
            traj = generate_dummy_trajectory(xyz, box_nm)
            positions = traj.openmm_positions(0)
        else:  # Create initial coordinates using subrandom positions.
            positions = subrandom_particle_positions(nparticles, system.getDefaultPeriodicBoxVectors())
        # Add the nonbonded force.
        system.addForce(nb)

        # Create topology.
        topology = app.Topology()
        element = app.Element.getBySymbol('Ar')
        chain = topology.addChain()
        for particle in range(system.getNumParticles()):
            residue = topology.addResidue('Ar', chain)
            topology.addAtom('Ar', element, residue)
        self.topology = topology

        self.system, self.positions = system, positions


class LennardJonesFluidTruncated(LennardJonesFluid):

    """
    Lennard-Jones fluid with truncated potential (instead of switched).

    """

    def __init__(self, *args, **kwargs):
        """
        Create a Lennard-Jones fluid with truncated potential.

        Parameters are inherited from LennardJonesFluid (except for 'switch_width').

        Examples
        --------

        >>> testsystem = LennardJonesFluidTruncated()
        >>> [system, positions] = [testsystem.system, testsystem.positions]

        """
        super(LennardJonesFluidTruncated, self).__init__(switch_width=None, *args, **kwargs)


class LennardJonesFluidSwitched(LennardJonesFluid):

    """
    Lennard-Jones fluid with switched potential (instead of truncated).

    """

    def __init__(self, *args, **kwargs):
        """
        Create a Lennard-Jones fluid with switched potential.

        Parameters are inherited from LennardJonesFluid (except for 'switch_width').

        Examples
        --------

        >>> testsystem = LennardJonesFluidSwitched()
        >>> [system, positions] = [testsystem.system, testsystem.positions]

        """
        super(LennardJonesFluidSwitched, self).__init__(switch_width=3.4 * unit.angstrom, *args, **kwargs)

#=============================================================================================
# Lennard-Jones grid
#=============================================================================================


class LennardJonesGrid(LennardJonesFluid):

    """Create a periodic fluid of Lennard-Jones particles on a grid.
    Initial positions are assigned using a subrandom grid to minimize steric interactions.

    Parameters
    ----------
    nx, ny, nz : int, optional, default=8
        Number of particles in x, y, and z dimensions.
    reduced_density : float, optional, default=0.86
        Reduced density (density * sigma**3); default is appropriate for liquid argon.
    mass : simtk.unit.Quantity, optional, default=39.9 * unit.amu
        mass of each particle; default is appropriate for argon
    sigma : simtk.unit.Quantity, optional, default=3.4 * unit.angstrom
        Lennard-Jones sigma parameter; default is appropriate for argon
    epsilon : simtk.unit.Quantity, optional, default=0.238 * unit.kilocalories_per_mole
        Lennard-Jones well depth; default is appropriate for argon
    cutoff : simtk.unit.Quantity, optional, default=None
        Cutoff for nonbonded interactions.  If None, defaults to 2.5 * sigma
    switch_width : simtk.unit.Quantity with units compatible with angstroms, optional, default=0.2*unit.angstroms
        switching function is turned on at cutoff - switch_width
        If None, no switch will be applied (e.g. hard cutoff).
    dispersion_correction : bool, optional, default=True
        if True, will use analytical dispersion correction (if not using switching function)

    Examples
    --------

    Create default-size Lennard-Jones fluid with initial positions on a grid.

    >>> fluid = LennardJonesGrid()
    >>> system, positions = fluid.system, fluid.positions

    Create a box of Lennard-Jones particles with unequal grid spacing.

    >>> fluid = LennardJonesGrid(nx=8, ny=9, nz=10)
    >>> system, positions = fluid.system, fluid.positions

    """

    def __init__(self,
                 nx=8, ny=8, nz=8,  # grid dimensions
                 *args,
                 **kwargs):

        # Create system with quasirandom particle positions.
        nparticles = nx * ny * nz
        super(LennardJonesGrid, self).__init__(nparticles, *args, **kwargs)

        # Compute volume per particle.
        box = self._system.getDefaultPeriodicBoxVectors()
        volume = box[0][0] * box[1][1] * box[2][2]
        volume_per_particle = volume / float(nparticles)
        delta = volume_per_particle**(1.0 / 3.0)

        # Adjust box vectors.
        box[0] = openmm.Vec3(nx * delta,  0 * delta,  0 * delta)
        box[1] = openmm.Vec3(0 * delta, ny * delta,  0 * delta)
        box[2] = openmm.Vec3(0 * delta,  0 * delta, nz * delta)
        self._system.setDefaultPeriodicBoxVectors(box[0], box[1], box[2])

        # Set positions.
        particle = 0
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    self._positions[particle, 0] = x * delta
                    self._positions[particle, 1] = y * delta
                    self._positions[particle, 2] = z * delta

        # Create topology.
        topology = app.Topology()
        element = app.Element.getBySymbol('Ar')
        chain = topology.addChain()
        for particle in range(nparticles):
            residue = topology.addResidue('Ar', chain)
            topology.addAtom('Ar', element, residue)
        self.topology = topology

        return

#=============================================================================================
# Custom Lennard-Jones fluid mixture of NonbondedForce and CustomNonbondedForce
#=============================================================================================


class CustomLennardJonesFluidMixture(TestSystem):

    """Create a periodic rectilinear grid of Lennard-Jones particles, but implemented via CustomBondForce and NonbondedForce.
    Parameters for argon are used by default. Cutoff is set to 3 sigma by default.

    Parameters
    ----------
    nparticles : int, optional, default=1000
        Number of Lennard-Jones particles.
    reduced_density : float, optional, default=0.05
        Reduced density (density * sigma**3); default is appropriate for gas
    mass : simtk.unit.Quantity, optional, default=39.9 * unit.amu
        mass of each particle.
    sigma : simtk.unit.Quantity, optional, default=3.4 * unit.angstrom
        Lennard-Jones sigma parameter
    epsilon : simtk.unit.Quantity, optional, default=0.238 * unit.kilocalories_per_mole
        Lennard-Jones well depth
    cutoff : simtk.unit.Quantity, optional, default=None
        Cutoff for nonbonded interactions.  If None, defaults to 3 * sigma
    switch_width : simtk.unit.Quantity with units compatible with angstroms, optional, default=None
        switching function is turned on at cutoff - switch_width
        If None, no switch will be applied (e.g. hard cutoff).
    dispersion_correction : bool, optional, default=True
        if True, will use analytical dispersion correction (if not using switching function)

    Notes
    -----

    No analytical dispersion correction is included here.

    Examples
    --------

    Create default-size Lennard-Jones fluid.

    >>> fluid = CustomLennardJonesFluidMixture()
    >>> system, positions = fluid.system, fluid.positions

    Create a larger box of Lennard-Jones particles.

    >>> fluid = CustomLennardJonesFluidMixture(nparticles=400)
    >>> system, positions = fluid.system, fluid.positions

    Create Lennard-Jones fluid using switched particle interactions (switched off betwee 7 and 9 A) and more particles.

    >>> fluid = CustomLennardJonesFluidMixture(nparticles=1000, switch=True, switch_width=7.0*unit.angstroms, cutoff=9.0*unit.angstroms)
    >>> system, positions = fluid.system, fluid.positions
    """

    def __init__(self,
                 nparticles=1000,
                 reduced_density=0.05,  # gas
                 mass=39.9 * unit.amu,  # argon
                 sigma=3.4 * unit.angstrom,  # argon,
                 epsilon=0.238 * unit.kilocalories_per_mole,  # argon,
                 cutoff=None,
                 switch_width=None,
                 dispersion_correction=True, **kwargs):

        TestSystem.__init__(self, **kwargs)

        charge = 0.0 * unit.elementary_charge

        # Determine Lennard-Jones cutoff.
        if cutoff is None:
            cutoff = 3.0 * sigma

        # Determine number of atoms that will be treated by CustomNonbondedForce
        ncustom = int(nparticles / 2)

        # Create an empty system object.
        system = openmm.System()

        # Determine volume and periodic box vectors.
        number_density = reduced_density / sigma**3
        volume = nparticles * (number_density ** -1)
        box_edge = volume ** (1. / 3.)
        a = unit.Quantity((box_edge,        0 * unit.angstrom, 0 * unit.angstrom))
        b = unit.Quantity((0 * unit.angstrom, box_edge,        0 * unit.angstrom))
        c = unit.Quantity((0 * unit.angstrom, 0 * unit.angstrom, box_edge))
        system.setDefaultPeriodicBoxVectors(a, b, c)

        # Set up periodic nonbonded interactions with a cutoff.
        nb = openmm.NonbondedForce()
        nb.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
        nb.setCutoffDistance(cutoff)
        nb.setUseDispersionCorrection(dispersion_correction)

        nb.setUseSwitchingFunction(False)
        if switch_width is not None:
            nb.setUseSwitchingFunction(True)
            nb.setSwitchingDistance(cutoff - switch_width)

        system.addForce(nb)

        # Set up periodic nonbonded interactions with a cutoff.
        energy_expression = '4*epsilon*((sigma/r)^12 - (sigma/r)^6);'
        energy_expression += 'sigma = %f;' % in_openmm_units(sigma)
        energy_expression += 'epsilon = %f;' % in_openmm_units(epsilon)
        cnb = openmm.CustomNonbondedForce(energy_expression)
        cnb.addPerParticleParameter('charge')
        cnb.addPerParticleParameter('sigma')
        cnb.addPerParticleParameter('epsilon')
        cnb.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
        cnb.setUseLongRangeCorrection(dispersion_correction)
        cnb.setCutoffDistance(cutoff)

        cnb.setUseSwitchingFunction(False)
        if switch_width is not None:
            cnb.setUseSwitchingFunction(True)
            cnb.setSwitchingDistance(cutoff - switch_width)

        system.addForce(cnb)

        # Add particles to system.
        for atom_index in range(nparticles):
            system.addParticle(mass)
            if (atom_index < ncustom):
                cnb.addParticle([charge, sigma, epsilon])
                nb.addParticle(0.0 * charge, sigma, 0.0 * epsilon)
            else:
                cnb.addParticle([0.0 * charge, sigma, 0.0 * epsilon])
                nb.addParticle(charge, sigma, epsilon)

        # Create initial coordinates using subrandom positions.
        positions = subrandom_particle_positions(nparticles, system.getDefaultPeriodicBoxVectors())

        # Create topology.
        topology = app.Topology()
        element = app.Element.getBySymbol('Ar')
        chain = topology.addChain()
        for particle in range(system.getNumParticles()):
            residue = topology.addResidue('Ar', chain)
            topology.addAtom('Ar', element, residue)
        self.topology = topology

        self.system, self.positions = system, positions


#=============================================================================================
# WCA Fluid
#=============================================================================================

class WCAFluid(TestSystem):

    def __init__(self, nparticles=216, density=0.96, mass=39.9 * unit.amu, epsilon=120.0 * unit.kelvin * kB, sigma=3.4 * unit.angstrom, **kwargs):
        """
        Create a Weeks-Chandler-Andersen system.

        Parameters:
        -----------
        npartocles : int, optional, default = 216
            Number of particles.
        density : float, optional, default = 0.96
            Reduced density, N sigma^3 / V.
        mass : simtk.unit.Quantity with units compatible with angstrom, optional, default=39.9 amu
            Particle mass.
        epsilon : simtk.unit.Quantity with units compatible with kilocalories_per_mole, optional, default=120K*kB
            WCA well depth.
        sigma : simtk.unit.Quantity
            WCA sigma.

        """

        TestSystem.__init__(self, **kwargs)

        # Create system
        system = openmm.System()

        # Compute total system volume.
        volume = nparticles / density

        # Make system cubic in dimension.
        length = volume**(1.0 / 3.0)
        # TODO: Can we change this to use tuples or 3x3 array?
        a = unit.Quantity(numpy.array([1.0, 0.0, 0.0], numpy.float32), unit.nanometer) * length / unit.nanometer
        b = unit.Quantity(numpy.array([0.0, 1.0, 0.0], numpy.float32), unit.nanometer) * length / unit.nanometer
        c = unit.Quantity(numpy.array([0.0, 0.0, 1.0], numpy.float32), unit.nanometer) * length / unit.nanometer
        system.setDefaultPeriodicBoxVectors(a, b, c)

        # Add particles to system.
        for n in range(nparticles):
            system.addParticle(mass)

        # Create nonbonded force term implementing Kob-Andersen two-component Lennard-Jones interaction.
        energy_expression = '4.0*epsilon*((sigma/r)^12 - (sigma/r)^6) + epsilon;'
        energy_expression += 'sigma = %f;' % in_openmm_units(sigma)
        energy_expression += 'epsilon = %f;' % in_openmm_units(epsilon)

        # Create force.
        force = openmm.CustomNonbondedForce(energy_expression)

        # Add particles
        for n in range(nparticles):
            force.addParticle([])

        # Set periodic boundary conditions with cutoff.
        force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
        rmin = 2.**(1. / 6.) * sigma  # distance of minimum energy for Lennard-Jones potential
        force.setCutoffDistance(rmin)

        # Add nonbonded force term to the system.
        system.addForce(force)

        # Create initial coordinates using subrandom positions.
        positions = subrandom_particle_positions(nparticles, system.getDefaultPeriodicBoxVectors())

        # Create topology.
        topology = app.Topology()
        element = app.Element.getBySymbol('Ar')
        chain = topology.addChain()
        for particle in range(system.getNumParticles()):
            residue = topology.addResidue('Ar', chain)
            topology.addAtom('Ar', element, residue)
        self.topology = topology

        # Store system.
        self.system, self.positions = system, positions

#=============================================================================================
# Ideal gas
#=============================================================================================


class IdealGas(TestSystem):

    """Create an 'ideal gas' of noninteracting particles in a periodic box.

    Parameters
    ----------
    nparticles : int, optional, default=216
        number of particles
    mass : int, optional, default=39.9 * unit.amu
    temperature : int, optional, default=298.0 * unit.kelvin
    pressure : int, optional, default=1.0 * unit.atmosphere
    volume : None
        if None, defaults to (nparticles * temperature * unit.BOLTZMANN_CONSTANT_kB / pressure).in_units_of(unit.nanometers**3)

    Examples
    --------

    Create an ideal gas system.

    >>> gas = IdealGas()
    >>> system, positions = gas.system, gas.positions

    Create a smaller ideal gas system containing 64 particles.

    >>> gas = IdealGas(nparticles=64)
    >>> system, positions = gas.system, gas.positions

    """

    def __init__(self, nparticles=216, mass=39.9 * unit.amu, temperature=298.0 * unit.kelvin, pressure=1.0 * unit.atmosphere, volume=None, **kwargs):

        TestSystem.__init__(self, **kwargs)

        if volume is None:
            volume = (nparticles * temperature * unit.BOLTZMANN_CONSTANT_kB / pressure).in_units_of(unit.nanometers**3)

        # Create an empty system object.
        system = openmm.System()

        # Compute box size.
        length = volume**(1.0 / 3.0)
        a = unit.Quantity((length,           0 * unit.nanometer, 0 * unit.nanometer))
        b = unit.Quantity((0 * unit.nanometer,           length, 0 * unit.nanometer))
        c = unit.Quantity((0 * unit.nanometer, 0 * unit.nanometer, length))
        system.setDefaultPeriodicBoxVectors(a, b, c)

        # Add particles.
        for index in range(nparticles):
            system.addParticle(mass)

        # Create initial coordinates using subrandom positions.
        positions = subrandom_particle_positions(nparticles, system.getDefaultPeriodicBoxVectors())

        # Create topology.
        topology = app.Topology()
        element = app.Element.getBySymbol('Ar')
        chain = topology.addChain()
        for particle in range(system.getNumParticles()):
            residue = topology.addResidue('Ar', chain)
            topology.addAtom('Ar', element, residue)
        self.topology = topology

        self.system, self.positions = system, positions
        self.ndof = 3 * nparticles

    def get_potential_expectation(self, state):
        """Return the expectation of the potential energy, computed analytically or numerically.

        Parameters
        ----------
        state : ThermodynamicState with temperature defined
            The thermodynamic state at which the property is to be computed.

        Returns
        -------
        potential_mean : simtk.unit.Quantity compatible with simtk.unit.kilojoules_per_mole
            The expectation of the potential energy.

        """

        return 0.0 * unit.kilojoules_per_mole

    def get_potential_standard_deviation(self, state):
        """Return the standard deviation of the potential energy, computed analytically or numerically.

        Parameters
        ----------
        state : ThermodynamicState with temperature defined
            The thermodynamic state at which the property is to be computed.

        Returns
        -------
        potential_stddev : simtk.unit.Quantity compatible with simtk.unit.kilojoules_per_mole
            potential energy standard deviation if implemented, or else None

        """

        return 0.0 * unit.kilojoules_per_mole

    def get_kinetic_expectation(self, state):
        """Return the expectation of the kinetic energy, computed analytically or numerically.

        Parameters
        ----------
        state : ThermodynamicState with temperature defined
            The thermodynamic state at which the property is to be computed.

        Returns
        -------
        potential_mean : simtk.unit.Quantity compatible with simtk.unit.kilojoules_per_mole
            The expectation of the potential energy.

        """

        return (3. / 2.) * kB * state.temperature

    def get_kinetic_standard_deviation(self, state):
        """Return the standard deviation of the kinetic energy, computed analytically or numerically.

        Parameters
        ----------
        state : ThermodynamicState with temperature defined
            The thermodynamic state at which the property is to be computed.

        Returns
        -------
        potential_stddev : simtk.unit.Quantity compatible with simtk.unit.kilojoules_per_mole
            potential energy standard deviation if implemented, or else None

        """

        return (3. / 2.) * kB * state.temperature

    def get_volume_expectation(self, state):
        """Return the expectation of the volume, computed analytically.

        Parameters
        ----------
        state : ThermodynamicState with temperature and pressure defined
            The thermodynamic state at which the property is to be computed.

        Returns
        -------
        volume_mean : simtk.unit.Quantity compatible with simtk.unit.nanometers**3
            The expectation of the volume at equilibrium.

        Notes
        -----
        The true mean volume is used, rather than the large-N limit.

        """

        if not state.pressure:
            box_vectors = self.system.getDefaultPeriodicBoxVectors()
            volume = box_vectors[0][0] * box_vectors[1][1] * box_vectors[2][2]
            return volume

        N = self._system.getNumParticles()
        return ((N + 1) * unit.BOLTZMANN_CONSTANT_kB * state.temperature / state.pressure).in_units_of(unit.nanometers**3)

    def get_volume_standard_deviation(self, state):
        """Return the standard deviation of the volume, computed analytically.

        Parameters
        ----------
        state : ThermodynamicState with temperature and pressure defined
            The thermodynamic state at which the property is to be computed.

        Returns
        -------
        volume_stddev : simtk.unit.Quantity compatible with simtk.unit.nanometers**3
            The standard deviation of the volume at equilibrium.

        Notes
        -----
        The true mean volume is used, rather than the large-N limit.

        """

        if not state.pressure:
            return 0.0 * unit.nanometers**3

        N = self._system.getNumParticles()
        return (numpy.sqrt(N + 1) * unit.BOLTZMANN_CONSTANT_kB * state.temperature / state.pressure).in_units_of(unit.nanometers**3)

#=============================================================================================
# Water box
#=============================================================================================


class WaterBox(TestSystem):

    """
    Create a water box test system.

    Examples
    --------

    Create a default (TIP3P) waterbox.

    >>> waterbox = WaterBox()

    Control the cutoff.

    >>> waterbox = WaterBox(box_edge=3.0*unit.nanometers, cutoff=1.0*unit.nanometers)

    Use a different water model.

    >>> waterbox = WaterBox(model='tip4pew')

    Don't use constraints.

    >>> waterbox = WaterBox(constrained=False)

    """

    def __init__(self, box_edge=2.5 * unit.nanometers, cutoff=0.9 * unit.nanometers, model='tip3p', switch_width=0.5 * unit.angstroms, constrained=True, dispersion_correction=True, nonbondedMethod=app.PME, ewaldErrorTolerance=5E-4, **kwargs):
        """
        Create a water box test system.

        Parameters
        ----------

        box_edge : simtk.unit.Quantity with units compatible with nanometers, optional, default = 2.5 nm
           Edge length for cubic box [should be greater than 2*cutoff]
        cutoff : simtk.unit.Quantity with units compatible with nanometers, optional, default = 0.9 nm
           Nonbonded cutoff
        model : str, optional, default = 'tip3p'
           The name of the water model to use ['tip3p', 'tip4p', 'tip4pew', 'tip5p', 'spce']
        switch_width : simtk.unit.Quantity with units compatible with nanometers, optional, default = 0.5 A
           Sets the width of the switch function for Lennard-Jones.
        constrained : bool, optional, default=True
           Sets whether water geometry should be constrained (rigid water implemented via SETTLE) or flexible.
        dispersion_correction : bool, optional, default=True
           Sets whether the long-range dispersion correction should be used.
        nonbondedMethod : simtk.openmm.app nonbonded method, optional, default=app.PME
           Sets the nonbonded method to use for the water box (one of app.CutoffPeriodic, app.Ewald, app.PME).
        ewaldErrorTolerance : float, optional, default=5E-4
           The Ewald or PME tolerance.  Used only if nonbondedMethod is Ewald or PME.

        Examples
        --------

        Create a default waterbox.

        >>> waterbox = WaterBox()
        >>> [system, positions] = [waterbox.system, waterbox.positions]

        Use reaction-field electrostatics instead.

        >>> waterbox = WaterBox(nonbondedMethod=app.CutoffPeriodic)

        Control the cutoff.

        >>> waterbox = WaterBox(box_edge=3.0*unit.nanometers, cutoff=1.0*unit.nanometers)

        Use a different water model.

        >>> waterbox = WaterBox(model='spce')

        Use a five-site water model.

        >>> waterbox = WaterBox(model='tip5p')

        Turn off the switch function.

        >>> waterbox = WaterBox(switch_width=None)

        Set the switch width.

        >>> waterbox = WaterBox(switch_width=0.8*unit.angstroms)

        Turn of long-range dispersion correction.

        >>> waterbox = WaterBox(dispersion_correction=False)

        """

        TestSystem.__init__(self, **kwargs)

        supported_models = ['tip3p', 'tip4pew', 'tip5p', 'spce']
        if model not in supported_models:
            raise Exception("Specified water model '%s' is not in list of supported models: %s" % (model, str(supported_models)))

        # Load forcefield for solvent model.
        ff = app.ForceField(model + '.xml')

        # Create empty topology and coordinates.
        top = app.Topology()
        pos = unit.Quantity((), unit.angstroms)

        # Create new Modeller instance.
        m = app.Modeller(top, pos)

        # Add solvent to specified box dimensions.
        boxSize = unit.Quantity(numpy.ones([3]) * box_edge / box_edge.unit, box_edge.unit)
        m.addSolvent(ff, boxSize=boxSize, model=model)

        # Get new topology and coordinates.
        newtop = m.getTopology()
        newpos = m.getPositions()

        # Convert positions to numpy.
        positions = unit.Quantity(numpy.array(newpos / newpos.unit), newpos.unit)

        # Create OpenMM System.
        system = ff.createSystem(newtop, nonbondedMethod=nonbondedMethod, nonbondedCutoff=cutoff, constraints=None, rigidWater=constrained, removeCMMotion=False)

        # Set switching function and dispersion correction.
        forces = {system.getForce(index).__class__.__name__: system.getForce(index) for index in range(system.getNumForces())}

        forces['NonbondedForce'].setUseSwitchingFunction(False)
        if switch_width is not None:
            forces['NonbondedForce'].setUseSwitchingFunction(True)
            forces['NonbondedForce'].setSwitchingDistance(cutoff - switch_width)

        forces['NonbondedForce'].setUseDispersionCorrection(dispersion_correction)
        forces['NonbondedForce'].setEwaldErrorTolerance(ewaldErrorTolerance)

        self.ndof = 3 * system.getNumParticles() - 3 * constrained

        self.topology = m.getTopology()
        self.system = system
        self.positions = positions


class FlexibleWaterBox(WaterBox):

    """
    Flexible water box.

    """

    def __init__(self, *args, **kwargs):
        """
        Create a flexible water box.

        Parameters are inherited from WaterBox (except for 'constrained').

        Examples
        --------

        Create a default flexible waterbox.

        >>> waterbox = FlexibleWaterBox()
        >>> [system, positions] = [waterbox.system, waterbox.positions]

        """
        super(FlexibleWaterBox, self).__init__(constrained=False, *args, **kwargs)


class FourSiteWaterBox(WaterBox):

    """
    Four-site water box (TIP4P-Ew).

    """

    def __init__(self, *args, **kwargs):
        """
        Create a water box test systemm using a four-site water model (TIP4P-Ew).

        Parameters are inherited from WaterBox (except for 'model').

        Examples
        --------

        Create a default waterbox.

        >>> waterbox = FourSiteWaterBox()
        >>> [system, positions] = [waterbox.system, waterbox.positions]

        Control the cutoff.

        >>> waterbox = FourSiteWaterBox(box_edge=3.0*unit.nanometers, cutoff=1.0*unit.nanometers)

        """
        super(FourSiteWaterBox, self).__init__(model='tip4pew', *args, **kwargs)


class FiveSiteWaterBox(WaterBox):

    """
    Five-site water box (TIP5P).

    """

    def __init__(self, *args, **kwargs):
        """
        Create a water box test systemm using a five-site water model (TIP5P).

        Parameters are inherited from WaterBox (except for 'model').

        Examples
        --------

        Create a default waterbox.

        >>> waterbox = FiveSiteWaterBox()
        >>> [system, positions] = [waterbox.system, waterbox.positions]

        Control the cutoff.

        >>> waterbox = FiveSiteWaterBox(box_edge=3.0*unit.nanometers, cutoff=1.0*unit.nanometers)

        """
        super(FiveSiteWaterBox, self).__init__(model='tip5p', *args, **kwargs)


class DischargedWaterBox(WaterBox):

    """
    Water box test system with zeroed charges.

    """

    def __init__(self, *args, **kwargs):
        """
        Create a water box test systemm using a four-site water model (TIP4P-Ew).

        Parameters are inherited from WaterBox.

        Examples
        --------

        Create a default waterbox.

        >>> waterbox = DischargedWaterBox()
        >>> [system, positions] = [waterbox.system, waterbox.positions]

        Control the cutoff.

        >>> waterbox = DischargedWaterBox(box_edge=3.0*unit.nanometers, cutoff=1.0*unit.nanometers)

        """
        super(DischargedWaterBox, self).__init__(*args, **kwargs)

        # Zero charges.
        system = self.system
        forces = {system.getForce(index).__class__.__name__: system.getForce(index) for index in range(system.getNumForces())}
        force = forces['NonbondedForce']
        for index in range(force.getNumParticles()):
            [charge, sigma, epsilon] = force.getParticleParameters(index)
            force.setParticleParameters(index, 0 * charge, sigma, epsilon)
        for index in range(force.getNumExceptions()):
            [particle1, particle2, chargeProd, sigma, epsilon] = force.getExceptionParameters(index)
            force.setExceptionParameters(index, particle1, particle2, 0 * chargeProd, sigma, epsilon)

        return


class DischargedWaterBoxHsites(WaterBox):

    """
    Water box test system with zeroed charges and Lennard-Jones sites on hydrogens.

    """

    def __init__(self, *args, **kwargs):
        """
        Create a water box with zeroed charges and Lennard-Jones sites on hydrogens.

        Parameters are inherited from WaterBox.

        Examples
        --------

        Create a default waterbox.

        >>> waterbox = DischargedWaterBox()
        >>> [system, positions] = [waterbox.system, waterbox.positions]

        Control the cutoff.

        >>> waterbox = DischargedWaterBox(box_edge=3.0*unit.nanometers, cutoff=1.0*unit.nanometers)

        """
        super(DischargedWaterBoxHsites, self).__init__(*args, **kwargs)

        # Zero charges.
        system = self.system
        forces = {system.getForce(index).__class__.__name__: system.getForce(index) for index in range(system.getNumForces())}
        force = forces['NonbondedForce']
        for index in range(force.getNumParticles()):
            [charge, sigma, epsilon] = force.getParticleParameters(index)
            charge *= 0
            if epsilon == 0.0 * unit.kilojoules_per_mole:
                # Add LJ site to hydrogens.
                epsilon = 0.0157 * unit.kilojoules_per_mole
                sigma = 0.06 * unit.angstroms
            force.setParticleParameters(index, charge, sigma, epsilon)
        for index in range(force.getNumExceptions()):
            [particle1, particle2, chargeProd, sigma, epsilon] = force.getExceptionParameters(index)
            chargeProd *= 0
            epsilon *= 0
            force.setExceptionParameters(index, particle1, particle2, chargeProd, sigma, epsilon)

        return

#=============================================================================================
# Alanine dipeptide in vacuum.
#=============================================================================================


class AlanineDipeptideVacuum(TestSystem):

    """Alanine dipeptide ff96 in vacuum.

    Parameters
    ----------
    constraints : optional, default=simtk.openmm.app.HBonds

    Examples
    --------

    Create alanine dipeptide with constraints on bonds to hydrogen
    >>> alanine = AlanineDipeptideVacuum()
    >>> (system, positions) = alanine.system, alanine.positions
    """

    def __init__(self, constraints=app.HBonds, **kwargs):

        TestSystem.__init__(self, **kwargs)

        prmtop_filename = get_data_filename("data/alanine-dipeptide-gbsa/alanine-dipeptide.prmtop")
        crd_filename = get_data_filename("data/alanine-dipeptide-gbsa/alanine-dipeptide.crd")

        prmtop = app.AmberPrmtopFile(prmtop_filename)
        system = prmtop.createSystem(implicitSolvent=None, constraints=constraints, nonbondedCutoff=None)

        # Extract topology
        self.topology = prmtop.topology

        # Read positions.
        inpcrd = app.AmberInpcrdFile(crd_filename)
        positions = inpcrd.getPositions(asNumpy=True)

        self.system, self.positions = system, positions

#=============================================================================================
# Alanine dipeptide in implicit solvent.
#=============================================================================================


class AlanineDipeptideImplicit(TestSystem):

    """Alanine dipeptide ff96 in OBC GBSA implicit solvent.

    Parameters
    ----------
    constraints : optional, default=simtk.openmm.app.HBonds

    Examples
    --------

    Create alanine dipeptide with constraints on bonds to hydrogen
    >>> alanine = AlanineDipeptideImplicit()
    >>> (system, positions) = alanine.system, alanine.positions
    """

    def __init__(self, constraints=app.HBonds, **kwargs):

        TestSystem.__init__(self, **kwargs)

        prmtop_filename = get_data_filename("data/alanine-dipeptide-gbsa/alanine-dipeptide.prmtop")
        crd_filename = get_data_filename("data/alanine-dipeptide-gbsa/alanine-dipeptide.crd")

        # Initialize system.
        prmtop = app.AmberPrmtopFile(prmtop_filename)
        system = prmtop.createSystem(implicitSolvent=app.OBC1, constraints=constraints, nonbondedCutoff=None)

        # Extract topology
        self.topology = prmtop.topology

        # Read positions.
        inpcrd = app.AmberInpcrdFile(crd_filename)
        positions = inpcrd.getPositions(asNumpy=True)

        self.system, self.positions = system, positions

#=============================================================================================
# Alanine dipeptide in explicit solvent
#=============================================================================================


class AlanineDipeptideExplicit(TestSystem):

    """Alanine dipeptide ff96 in TIP3P explicit solvent..

    Parameters
    ----------
    constraints : optional, default=simtk.openmm.app.HBonds
    rigid_water : bool, optional, default=True
    nonbondedCutoff : Quantity, optional, default=9.0 * unit.angstroms
    use_dispersion_correction : bool, optional, default=True
        If True, the long-range disperson correction will be used.
    nonbondedMethod : simtk.openmm.app nonbonded method, optional, default=app.PME
       Sets the nonbonded method to use for the water box (one of app.CutoffPeriodic, app.Ewald, app.PME).
    hydrogenMass : unit, optional, default=None
        If set, will pass along a modified hydrogen mass for OpenMM to
        use mass repartitioning.

    Examples
    --------

    >>> alanine = AlanineDipeptideExplicit()
    >>> (system, positions) = alanine.system, alanine.positions
    """

    def __init__(self, constraints=app.HBonds, rigid_water=True, nonbondedCutoff=9.0 * unit.angstroms, use_dispersion_correction=True, nonbondedMethod=app.PME, hydrogenMass=None, **kwargs):

        TestSystem.__init__(self, **kwargs)

        prmtop_filename = get_data_filename("data/alanine-dipeptide-explicit/alanine-dipeptide.prmtop")
        crd_filename = get_data_filename("data/alanine-dipeptide-explicit/alanine-dipeptide.crd")

        # Initialize system.
        prmtop = app.AmberPrmtopFile(prmtop_filename)
        system = prmtop.createSystem(constraints=constraints, nonbondedMethod=nonbondedMethod, rigidWater=rigid_water, nonbondedCutoff=nonbondedCutoff, hydrogenMass=hydrogenMass)

        # Extract topology
        self.topology = prmtop.topology

        # Set dispersion correction use.
        forces = {system.getForce(index).__class__.__name__: system.getForce(index) for index in range(system.getNumForces())}
        forces['NonbondedForce'].setUseDispersionCorrection(use_dispersion_correction)

        # Read positions.
        inpcrd = app.AmberInpcrdFile(crd_filename)
        positions = inpcrd.getPositions(asNumpy=True)

        # Set box vectors.
        box_vectors = inpcrd.getBoxVectors(asNumpy=True)
        system.setDefaultPeriodicBoxVectors(box_vectors[0], box_vectors[1], box_vectors[2])

        self.system, self.positions = system, positions


#=============================================================================================
# Alanine dipeptide in explicit solvent
#=============================================================================================

class DHFRExplicit(TestSystem):

    """Joint Amber CHARMM (JAC) DHFR / TIP3P benchmark system with 23558 atoms.

    Parameters
    ----------
    constraints : optional, default=simtk.openmm.app.HBonds
    rigid_water : bool, optional, default=True
    nonbondedCutoff : Quantity, optional, default=8.0 * unit.angstroms
    use_dispersion_correction : bool, optional, default=True
        If True, the long-range disperson correction will be used.
    nonbondedMethod : simtk.openmm.app nonbonded method, optional, default=app.PME
       Sets the nonbonded method to use for the water box (one of app.CutoffPeriodic, app.Ewald, app.PME).
    hydrogenMass : unit, optional, default=None
        If set, will pass along a modified hydrogen mass for OpenMM to
        use mass repartitioning.

    """

    def __init__(self, constraints=app.HBonds, rigid_water=True, nonbondedCutoff=8.0 * unit.angstroms, use_dispersion_correction=True, nonbondedMethod=app.PME, hydrogenMass=None, **kwargs):

        TestSystem.__init__(self, **kwargs)

        try:
            from chemistry.amber import AmberParm
        except ImportError as e:
            print("DHFR test system requires Parmed (`import chemistry`).")
            raise(e)

        prmtop_filename = get_data_filename("data/dhfr/prmtop")
        crd_filename = get_data_filename("data/dhfr/inpcrd")

        # Initialize system.
        self.prmtop = AmberParm(prmtop_filename, crd_filename)
        system = self.prmtop.createSystem(constraints=constraints, nonbondedMethod=nonbondedMethod, rigidWater=rigid_water, nonbondedCutoff=nonbondedCutoff, hydrogenMass=hydrogenMass)

        # Extract topology
        self.topology = self.prmtop.topology

        # Set dispersion correction use.
        forces = {system.getForce(index).__class__.__name__: system.getForce(index) for index in range(system.getNumForces())}
        forces['NonbondedForce'].setUseDispersionCorrection(use_dispersion_correction)

        positions = self.prmtop.positions

        # Set box vectors.
        box_vectors = self.prmtop.box_vectors
        system.setDefaultPeriodicBoxVectors(box_vectors[0], box_vectors[1], box_vectors[2])

        self.system, self.positions = system, positions


#=============================================================================================
# T4 lysozyme L99A mutant with p-xylene ligand.
#=============================================================================================

class LysozymeImplicit(TestSystem):

    """T4 lysozyme L99A (AMBER ff96) with p-xylene ligand (GAFF + AM1-BCC) in implicit OBC GBSA solvent.

    Parameters
    ----------
    constraints : simtk.openmm.app constraints (None, HBonds, HAngles, AllBonds)
       constraints to be imposed

    Examples
    --------

    >>> lysozyme = LysozymeImplicit()
    >>> (system, positions) = lysozyme.system, lysozyme.positions
    """

    def __init__(self, constraints=app.HBonds, implicitSolvent=app.OBC1, **kwargs):

        TestSystem.__init__(self, **kwargs)

        prmtop_filename = get_data_filename("data/T4-lysozyme-L99A-implicit/complex.prmtop")
        crd_filename = get_data_filename("data/T4-lysozyme-L99A-implicit/complex.crd")

        # Initialize system.
        prmtop = app.AmberPrmtopFile(prmtop_filename)
        system = prmtop.createSystem(implicitSolvent=app.OBC1, constraints=app.HBonds, nonbondedCutoff=None)

        # Extract topology
        self.topology = prmtop.topology

        # Read positions.
        inpcrd = app.AmberInpcrdFile(crd_filename)
        positions = inpcrd.getPositions(asNumpy=True)

        self.system, self.positions = system, positions


class SrcImplicit(TestSystem):

    """Src kinase in implicit AMBER 99sb-ildn with OBC GBSA solvent.

    Examples
    --------
    >>> src = SrcImplicit()
    >>> system, positions = src.system, src.positions
    """

    def __init__(self, **kwargs):

        TestSystem.__init__(self, **kwargs)

        pdb_filename = get_data_filename("data/src-implicit/2src-minimized.pdb")
        pdbfile = app.PDBFile(pdb_filename)

        # Construct system.
        forcefields_to_use = ['amber99sbildn.xml', 'amber99_obc.xml']  # list of forcefields to use in parameterization
        forcefield = app.ForceField(*forcefields_to_use)
        system = forcefield.createSystem(pdbfile.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)

        # Get positions.
        positions = pdbfile.getPositions()

        self.system, self.positions, self.topology = system, positions, pdbfile.topology

#=============================================================================================
# Src kinase in explicit solvent.
#=============================================================================================


class SrcExplicit(TestSystem):

    """Src kinase (AMBER 99sb-ildn) in explicit TIP3P solvent using PME electrostatics.

    Parameters
    ----------
    nonbondedMethod : simtk.openmm.app nonbonded method, optional, default=app.PME
       Sets the nonbonded method to use for the water box (CutoffPeriodic, app.Ewald, app.PME).

    Examples
    --------
    >>> src = SrcExplicit()
    >>> system, positions = src.system, src.positions

    """

    def __init__(self, nonbondedMethod=app.PME, **kwargs):

        TestSystem.__init__(self, **kwargs)

        pdb_filename = get_data_filename("data/src-explicit/2src-minimized.pdb")
        pdbfile = app.PDBFile(pdb_filename)

        # Construct system.
        forcefields_to_use = ['amber99sbildn.xml', 'tip3p.xml']  # list of forcefields to use in parameterization
        forcefield = app.ForceField(*forcefields_to_use)
        system = forcefield.createSystem(pdbfile.topology, nonbondedMethod=nonbondedMethod, constraints=app.HBonds)

        # Get positions.
        positions = pdbfile.getPositions()

        self.system, self.positions, self.topology = system, positions, pdbfile.topology


class SrcExplicitReactionField(SrcExplicit):

    """
    Flexible water box.

    """

    def __init__(self, *args, **kwargs):
        """Src kinase (AMBER 99sb-ildn) in explicit TIP3P solvent using reaction field electrostatics.

        Parameters are inherited from SrcExplicit (except for 'nonbondedMethod').

        Examples
        --------

        >>> src = SrcExplicitReactionField()
        >>> system, positions = src.system, src.positions

        """
        super(SrcExplicitReactionField, self).__init__(nonbondedMethod=app.CutoffPeriodic, *args, **kwargs)

#=============================================================================================
# Methanol box.
#=============================================================================================


class MethanolBox(TestSystem):

    """Methanol box.

    Parameters
    ----------
    shake : string, optional, default="h-bonds"
    nonbondedCutoff : Quantity, optional, default=7.0 * unit.angstroms
    nonbondedMethod : simtk.openmm.app nonbonded method, optional, default=app.PME
       Sets the nonbonded method to use for the water box (one of app.CutoffPeriodic, app.Ewald, app.PME).

    Examples
    --------

    >>> methanol_box = MethanolBox()
    >>> system, positions = methanol_box.system, methanol_box.positions
    """

    def __init__(self, constraints=app.HBonds, nonbondedCutoff=7.0 * unit.angstroms, nonbondedMethod=app.CutoffPeriodic, **kwargs):

        TestSystem.__init__(self, **kwargs)

        system_name = 'methanol-box'
        prmtop_filename = get_data_filename("data/%s/%s.prmtop" % (system_name, system_name))
        crd_filename = get_data_filename("data/%s/%s.crd" % (system_name, system_name))

        # Initialize system.
        prmtop = app.AmberPrmtopFile(prmtop_filename)
        system = prmtop.createSystem(constraints=constraints, nonbondedMethod=nonbondedMethod, rigidWater=True, nonbondedCutoff=0.9 * unit.nanometer)

        # Read positions.
        inpcrd = app.AmberInpcrdFile(crd_filename)
        positions = inpcrd.getPositions(asNumpy=True)

        # Set box vectors.
        box_vectors = inpcrd.getBoxVectors(asNumpy=True)
        system.setDefaultPeriodicBoxVectors(box_vectors[0], box_vectors[1], box_vectors[2])

        self.system, self.positions, self.topology = system, positions, prmtop.topology

#=============================================================================================
# Molecular ideal gas (methanol box).
#=============================================================================================


class MolecularIdealGas(TestSystem):

    """Molecular ideal gas (methanol box).

    Parameters
    ----------
    shake : string, optional, default=None
    nonbondedCutoff : Quantity, optional, default=7.0 * unit.angstroms
    nonbondedMethod : simtk.openmm.app nonbonded method, optional, default=app.PME
       Sets the nonbonded method to use for the water box (one of app.CutoffPeriodic, app.Ewald, app.PME).

    Examples
    --------

    >>> methanol_box = MolecularIdealGas()
    >>> system, positions = methanol_box.system, methanol_box.positions
    """

    def __init__(self, shake=None, nonbondedCutoff=7.0 * unit.angstroms, nonbondedMethod=app.CutoffPeriodic, **kwargs):

        TestSystem.__init__(self, **kwargs)

        system_name = 'methanol-box'
        prmtop_filename = get_data_filename("data/%s/%s.prmtop" % (system_name, system_name))
        crd_filename = get_data_filename("data/%s/%s.crd" % (system_name, system_name))

        # Initialize system.
        prmtop = app.AmberPrmtopFile(prmtop_filename)
        reference_system = prmtop.createSystem(constraints=app.HBonds, nonbondedMethod=nonbondedMethod, rigidWater=True, nonbondedCutoff=0.9 * unit.nanometer)

        # Make a new system that contains no intermolecular interactions.
        system = openmm.System()

        # Add atoms.
        for atom_index in range(reference_system.getNumParticles()):
            mass = reference_system.getParticleMass(atom_index)
            system.addParticle(mass)

        # Add constraints
        for constraint_index in range(reference_system.getNumConstraints()):
            [iatom, jatom, r0] = reference_system.getConstraintParameters(constraint_index)
            system.addConstraint(iatom, jatom, r0)

        # Copy only intramolecular forces.
        nforces = reference_system.getNumForces()
        for force_index in range(nforces):
            reference_force = reference_system.getForce(force_index)
            if isinstance(reference_force, openmm.HarmonicBondForce):
                # HarmonicBondForce
                force = openmm.HarmonicBondForce()
                for bond_index in range(reference_force.getNumBonds()):
                    [iatom, jatom, r0, K] = reference_force.getBondParameters(bond_index)
                    force.addBond(iatom, jatom, r0, K)
                system.addForce(force)
            elif isinstance(reference_force, openmm.HarmonicAngleForce):
                # HarmonicAngleForce
                force = openmm.HarmonicAngleForce()
                for angle_index in range(reference_force.getNumAngles()):
                    [iatom, jatom, katom, theta0, Ktheta] = reference_force.getAngleParameters(angle_index)
                    force.addAngle(iatom, jatom, katom, theta0, Ktheta)
                system.addForce(force)
            elif isinstance(reference_force, openmm.PeriodicTorsionForce):
                # PeriodicTorsionForce
                force = openmm.PeriodicTorsionForce()
                for torsion_index in range(reference_force.getNumTorsions()):
                    [particle1, particle2, particle3, particle4, periodicity, phase, k] = reference_force.getTorsionParameters(torsion_index)
                    force.addTorsion(particle1, particle2, particle3, particle4, periodicity, phase, k)
                system.addForce(force)
            else:
                # Don't add any other forces.
                pass

        # Read positions.
        inpcrd = app.AmberInpcrdFile(crd_filename)
        positions = inpcrd.getPositions(asNumpy=True)

        # Set box vectors.
        box_vectors = inpcrd.getBoxVectors(asNumpy=True)
        system.setDefaultPeriodicBoxVectors(box_vectors[0], box_vectors[1], box_vectors[2])

        self.topology = prmtop.topology
        self.system, self.positions = system, positions

#=============================================================================================
# System of particles with CustomGBForce
#=============================================================================================


class CustomGBForceSystem(TestSystem):

    """A system of particles with a CustomGBForce.

    Notes
    -----

    This example comes from TestReferenceCustomGBForce.cpp from the OpenMM distribution.

    Examples
    --------

    >>> gb_system = CustomGBForceSystem()
    >>> system, positions = gb_system.system, gb_system.positions
    """

    def __init__(self, **kwargs):

        TestSystem.__init__(self, **kwargs)

        numMolecules = 70
        numParticles = numMolecules * 2
        boxSize = 10.0 * unit.nanometers

        # Default parameters
        mass = 39.9 * unit.amu
        sigma = 3.350 * unit.angstrom
        epsilon = 0.001603 * unit.kilojoule_per_mole
        cutoff = 2.0 * unit.nanometers

        system = openmm.System()
        for i in range(numParticles):
            system.addParticle(mass)

        system.setDefaultPeriodicBoxVectors(openmm.Vec3(boxSize, 0.0, 0.0), openmm.Vec3(0.0, boxSize, 0.0), openmm.Vec3(0.0, 0.0, boxSize))

        # Create NonbondedForce.
        nonbonded = openmm.NonbondedForce()
        nonbonded.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
        nonbonded.setCutoffDistance(cutoff)

        # Create CustomGBForce.
        custom = openmm.CustomGBForce()
        custom.setNonbondedMethod(openmm.CustomGBForce.CutoffPeriodic)
        custom.setCutoffDistance(cutoff)

        custom.addPerParticleParameter("charge")
        custom.addPerParticleParameter("radius")
        custom.addPerParticleParameter("scale")

        custom.addGlobalParameter("testsystems_CustomGBForceSystem_solventDielectric", 80.0)
        custom.addGlobalParameter("testsystems_CustomGBForceSystem_soluteDielectric", 1.0)

        custom.addComputedValue("I", "step(r+sr2-or1)*0.5*(1/L-1/U+0.25*(1/U^2-1/L^2)*(r-sr2*sr2/r)+0.5*log(L/U)/r+C);"
                                "U=r+sr2;"
                                "C=2*(1/or1-1/L)*step(sr2-r-or1);"
                                "L=max(or1, D);"
                                "D=abs(r-sr2);"
                                "sr2 = scale2*or2;"
                                "or1 = radius1-0.009; or2 = radius2-0.009", openmm.CustomGBForce.ParticlePairNoExclusions)
        custom.addComputedValue("B", "1/(1/or-tanh(1*psi-0.8*psi^2+4.85*psi^3)/radius);"
                                "psi=I*or; or=radius-0.009", openmm.CustomGBForce.SingleParticle)

        energy_expression = '28.3919551*(radius+0.14)^2*(radius/B)^6-0.5*138.935485*(1/soluteDielectric-1/solventDielectric)*charge^2/B;'
        energy_expression += 'solventDielectric = testsystems_CustomGBForceSystem_solventDielectric;'
        energy_expression += 'soluteDielectric = testsystems_CustomGBForceSystem_soluteDielectric;'
        custom.addEnergyTerm(energy_expression, openmm.CustomGBForce.SingleParticle)

        energy_expression = '-138.935485*(1/soluteDielectric-1/solventDielectric)*charge1*charge2/f;'
        energy_expression += 'f=sqrt(r^2+B1*B2*exp(-r^2/(4*B1*B2)));'
        energy_expression += 'solventDielectric = testsystems_CustomGBForceSystem_solventDielectric;'
        energy_expression += 'soluteDielectric = testsystems_CustomGBForceSystem_soluteDielectric;'
        custom.addEnergyTerm(energy_expression, openmm.CustomGBForce.ParticlePairNoExclusions)

        # Add particles.
        for i in range(numMolecules):
            if (i < numMolecules / 2):
                charge = 1.0 * unit.elementary_charge
                radius = 0.2 * unit.nanometers
                scale = 0.5
                nonbonded.addParticle(charge, sigma, epsilon)
                custom.addParticle([charge, radius, scale])

                charge = -1.0 * unit.elementary_charge
                radius = 0.1 * unit.nanometers
                scale = 0.5
                nonbonded.addParticle(charge, sigma, epsilon)
                custom.addParticle([charge, radius, scale])
            else:
                charge = 1.0 * unit.elementary_charge
                radius = 0.2 * unit.nanometers
                scale = 0.8
                nonbonded.addParticle(charge, sigma, epsilon)
                custom.addParticle([charge, radius, scale])

                charge = -1.0 * unit.elementary_charge
                radius = 0.1 * unit.nanometers
                scale = 0.8
                nonbonded.addParticle(charge, sigma, epsilon)
                custom.addParticle([charge, radius, scale])

        system.addForce(nonbonded)
        system.addForce(custom)

        # Create initial coordinates using subrandom positions.
        positions = subrandom_particle_positions(numParticles, system.getDefaultPeriodicBoxVectors())

        # Create topology.
        topology = app.Topology()
        element = app.Element.getBySymbol('Ar')
        chain = topology.addChain()
        for index in range(numParticles):
            residue = topology.addResidue('OSC', chain)
            topology.addAtom('Ar', element, residue)
        self.topology = topology

        self.system, self.positions = system, positions

#=============================================================================================
# AMOEBA SYSTEMS
#=============================================================================================


class AMOEBAIonBox(TestSystem):

    """A single Ca2 ion in a water box.

    >>> testsystem = AMOEBAIonBox()
    >>> system, positions = testsystem.system, testsystem.positions

    """

    def __init__(self, **kwargs):
        TestSystem.__init__(self, **kwargs)

        pdb_filename = get_data_filename("data/amoeba/ion-in-water.pdb")
        pdbfile = app.PDBFile(pdb_filename)

        ff = app.ForceField("amoeba2009.xml")
        # TODO: 7A is a hack
        system = ff.createSystem(pdbfile.topology, nonbondedMethod=app.PME, constraints=app.HBonds, useDispersionCorrection=True, nonbondedCutoff=7.0 * unit.angstroms)

        positions = pdbfile.getPositions()
        self.topology = pdbfile.topology
        self.system, self.positions = system, positions


class AMOEBAProteinBox(TestSystem):

    """PDB 1AP4 in water box.

    >>> testsystem = AMOEBAProteinBox()
    >>> system, positions = testsystem.system, testsystem.positions

    """

    def __init__(self, **kwargs):
        TestSystem.__init__(self, **kwargs)

        pdb_filename = get_data_filename("data/amoeba/1AP4_14_wat.pdb")
        pdbfile = app.PDBFile(pdb_filename)

        ff = app.ForceField("amoeba2009.xml")
        system = ff.createSystem(pdbfile.topology, nonbondedMethod=app.PME, constraints=app.HBonds, useDispersionCorrection=True)

        positions = pdbfile.getPositions()
        self.topology = pdbfile.topology
        self.system, self.positions = system, positions

#=============================================================================================
# ALCHEMICALLY MODIFIED SYSTEMS
#=============================================================================================


class AlchemicalState(object):

    """
    Alchemical state description.

    These parameters describe the parameters that affect computation of the energy.

    Attributes
    ----------
    relativeRestraints : float
        Scaling factor for remaining receptor-ligand relative restraint terms (to help keep ligand near protein).
    ligandElectrostatics : float
        Scaling factor for ligand charges, intrinsic Born radii, and surface area term.
    ligandSterics : float
        Scaling factor for ligand sterics (Lennard-Jones and Halgren) interactions.
    ligandTorsions : float
        Scaling factor for ligand non-ring torsions.
    annihilateElectrostatics : bool
        If True, electrostatics should be annihilated, rather than decoupled.
    annihilateSterics : bool
        If True, sterics (Lennard-Jones or Halgren potential) will be annihilated, rather than decoupled.

    TODO
    ----
    * Rework these structure members into something more general and flexible?
    * Add receptor modulation back in?
    """

    def __init__(self, relativeRestraints=0.0, ligandElectrostatics=1.0, ligandSterics=1.0, ligandTorsions=1.0, annihilateElectrostatics=True, annihilateSterics=False):
        """
        Create an Alchemical state.

        Parameters
        ----------
        relativeRestraints : float, optional, default = 0.0
            Scaling factor for remaining receptor-ligand relative restraint terms (to help keep ligand near protein).
        ligandElectrostatics : float, optional, default = 1.0
            Scaling factor for ligand charges, intrinsic Born radii, and surface area term.
        ligandSterics : float, optional, default = 1.0
            Scaling factor for ligand sterics (Lennard-Jones or Halgren) interactions.
        ligandTorsions : float, optional, default = 1.0
            Scaling factor for ligand non-ring torsions.
        annihilateElectrostatics : bool, optional, default = True
            If True, electrostatics should be annihilated, rather than decoupled.
        annihilateSterics : bool, optional, default = False
            If True, sterics (Lennard-Jones or Halgren potential) will be annihilated, rather than decoupled.

        Examples
        --------

        Create a fully-interacting, unrestrained alchemical state.

        >>> alchemical_state = AlchemicalState(relativeRestraints=0.0, ligandElectrostatics=1.0, ligandSterics=1.0, ligandTorsions=1.0)
        >>> # This is equivalent to
        >>> alchemical_state = AlchemicalState()


        Annihilate electrostatics.

        >>> alchemical_state = AlchemicalState(annihilateElectrostatics=True, ligandElectrostatics=0.0)

        """

        self.relativeRestraints = relativeRestraints
        self.ligandElectrostatics = ligandElectrostatics
        self.ligandSterics = ligandSterics
        self.ligandTorsions = ligandTorsions
        self.annihilateElectrostatics = annihilateElectrostatics
        self.annihilateSterics = annihilateSterics

        return


class AlchemicalTestSystem(object):

    def _alchemicallyModifyLennardJones(cls, system, nonbonded_force, alchemical_atom_indices, alchemical_state, alpha=0.50, a=1, b=1, c=6):
        """
        Alchemically modify the Lennard-Jones force terms.

        This version uses the new group-based restriction capabilities of CustomNonbondedForce.


        Parameters
        ----------
        system : simtk.openmm.System
        System to modify.
        nonbonded_force : simtk.openmm.NonbondedForce
        The NonbondedForce to modify (will be changed).
        alchemical_atom_indices : list of int
        Atom indices to be alchemically modified.
        alchemical_state : AlchemicalState
        The alchemical state specification to be used in modifying Lennard-Jones terms.
        alpha : float, optional, default = 0.5
        Alchemical softcore parameter.
        a, b, c : float, optional, default a=1, b=1, c=6
        Parameters describing softcore force.

        """

        import simtk.openmm as openmm

        # Create CustomNonbondedForce to handle softcore interactions between alchemically-modified system and rest of system.

        energy_expression = "4*epsilon*(lambda^a)*x*(x-1.0);"
        energy_expression += "x = (1.0/(alpha*(1.0-lambda)^b + (r/sigma)^c))^(6/c);"
        energy_expression += "epsilon = sqrt(epsilon1*epsilon2);"  # mixing rule for epsilon
        energy_expression += "sigma = 0.5*(sigma1 + sigma2);"  # mixing rule for sigma
        energy_expression += "lambda = testsystems_AlchemicalTestSystem_lennard_jones_lambda;"  # lambda

        # Create atom groups.
        atomset1 = set(alchemical_atom_indices)  # only alchemically-modified atoms
        atomset2 = set(range(system.getNumParticles())) - atomset1  # all atoms minus intra-alchemical region

        # Create alchemically modified nonbonded force.
        # TODO: Create a _createCustomNonbondedForce method to duplicate parameters?
        energy_expression += "alpha = %f;" % alpha
        energy_expression += "a = %f; b = %f; c = %f;" % (a, b, c)
        custom_nonbonded_force = openmm.CustomNonbondedForce(energy_expression)
        custom_nonbonded_force.setNonbondedMethod(nonbonded_force.getNonbondedMethod())  # TODO: Make sure these method indices are identical.
        custom_nonbonded_force.setUseSwitchingFunction(nonbonded_force.getUseSwitchingFunction())
        custom_nonbonded_force.setSwitchingDistance(nonbonded_force.getSwitchingDistance())
        custom_nonbonded_force.setUseLongRangeCorrection(nonbonded_force.getUseDispersionCorrection())
        custom_nonbonded_force.addGlobalParameter("testsystems_AlchemicalTestSystem_lennard_jones_lambda", alchemical_state.ligandSterics)
        custom_nonbonded_force.addPerParticleParameter("sigma")  # Lennard-Jones sigma
        custom_nonbonded_force.addPerParticleParameter("epsilon")  # Lennard-Jones epsilon

        # Restrict interaction evaluation to be between alchemical atoms and rest of environment.
        # Only add custom nonbonded force if interacting groups are both nonzero in size.
        if (len(atomset1) != 0) and (len(atomset2) != 0):
            custom_nonbonded_force.addInteractionGroup(atomset1, atomset2)
            system.addForce(custom_nonbonded_force)

        # Create CustomBondedForce to handle softcore exceptions if alchemically annihilating ligand.
        if alchemical_state.annihilateSterics:
            energy_expression = "4*epsilon*(lambda^a)*x*(x-1.0);"
            energy_expression += "x = (1.0/(alpha*(1.0-lambda)^b + (r/sigma)^c))^(6/c);"
            energy_expression += "alpha = %f;" % alpha
            energy_expression += "a = %f; b = %f; c = %f;" % (a, b, c)
            energy_expression += "lambda = testsystems_AlchemicalTestSystem_lennard_jones_lambda;"
            custom_bond_force = openmm.CustomBondForce(energy_expression)
            custom_bond_force.addGlobalParameter("lennard_jones_lambda", alchemical_state.ligandSterics)
            custom_bond_force.addPerBondParameter("sigma")  # Lennard-Jones sigma
            custom_bond_force.addPerBondParameter("epsilon")  # Lennard-Jones epsilon
            system.addForce(custom_bond_force)
        else:
            # Decoupling of sterics.
            # Add a second CustomNonbondedForce to restore "intra-alchemical" interactions to full strength.
            energy_expression = "4*epsilon*((sigma/r)^12 - (sigma/r)^6);"
            energy_expression += "epsilon = sqrt(epsilon1*epsilon2);"  # mixing rule for epsilon
            energy_expression += "sigma = 0.5*(sigma1 + sigma2);"  # mixing rule for sigma
            custom_nonbonded_force2 = openmm.CustomNonbondedForce(energy_expression)
            custom_nonbonded_force2.setUseSwitchingFunction(nonbonded_force.getUseSwitchingFunction())
            custom_nonbonded_force2.setSwitchingDistance(nonbonded_force.getSwitchingDistance())
            custom_nonbonded_force2.setUseLongRangeCorrection(nonbonded_force.getUseDispersionCorrection())
            custom_nonbonded_force2.addPerParticleParameter("sigma")  # Lennard-Jones sigma
            custom_nonbonded_force2.addPerParticleParameter("epsilon")  # Lennard-Jones epsilon
            system.addForce(custom_nonbonded_force2)
            # Restrict interaction evaluation to be between alchemical atoms and rest of environment.
            atomset1 = set(alchemical_atom_indices)  # only alchemically-modified atoms
            atomset2 = set(alchemical_atom_indices)  # only alchemically-modified atoms
            custom_nonbonded_force2.addInteractionGroup(atomset1, atomset2)

        # Copy Lennard-Jones particle parameters.
        for particle_index in range(nonbonded_force.getNumParticles()):
            # Retrieve parameters.
            [charge, sigma, epsilon] = nonbonded_force.getParticleParameters(particle_index)
            # Add corresponding particle to softcore interactions.
            if particle_index in alchemical_atom_indices:
                # Turn off Lennard-Jones contribution from alchemically-modified particles.
                nonbonded_force.setParticleParameters(particle_index, charge, sigma, epsilon * 0.0)
            # Add contribution back to custom force.
            custom_nonbonded_force.addParticle([sigma, epsilon])
            if not alchemical_state.annihilateSterics:
                custom_nonbonded_force2.addParticle([sigma, epsilon])

        # Create an exclusion for each exception in the reference NonbondedForce, assuming that NonbondedForce will handle them.
        for exception_index in range(nonbonded_force.getNumExceptions()):
            # Retrieve parameters.
            [iatom, jatom, chargeprod, sigma, epsilon] = nonbonded_force.getExceptionParameters(exception_index)
            # Exclude this atom pair in CustomNonbondedForce.
            custom_nonbonded_force.addExclusion(iatom, jatom)
            if not alchemical_state.annihilateSterics:
                custom_nonbonded_force2.addExclusion(iatom, jatom)

            # If annihilating Lennard-Jones, move intramolecular interactions to custom_bond_force.
            if alchemical_state.annihilateSterics and (iatom in alchemical_atom_indices) and (jatom in alchemical_atom_indices):
                # Remove Lennard-Jones exception.
                nonbonded_force.setExceptionParameters(exception_index, iatom, jatom, chargeprod, sigma, epsilon * 0.0)
                # Add special CustomBondForce term to handle alchemically-modified Lennard-Jones exception.
                custom_bond_force.addBond(iatom, jatom, [sigma, epsilon])

        # Set periodicity and cutoff parameters corresponding to reference Force.
        if nonbonded_force.getNonbondedMethod() in [openmm.NonbondedForce.Ewald, openmm.NonbondedForce.PME]:
            # Convert Ewald and PME to CutoffPeriodic.
            custom_nonbonded_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
            if not alchemical_state.annihilateSterics:
                custom_nonbonded_force2.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
        else:
            custom_nonbonded_force.setNonbondedMethod(nonbonded_force.getNonbondedMethod())
            if not alchemical_state.annihilateSterics:
                custom_nonbonded_force2.setNonbondedMethod(nonbonded_force.getNonbondedMethod())

        custom_nonbonded_force.setCutoffDistance(nonbonded_force.getCutoffDistance())
        if not alchemical_state.annihilateSterics:
            custom_nonbonded_force2.setCutoffDistance(nonbonded_force.getCutoffDistance())

        return


class AlchemicalLennardJonesCluster(TestSystem, AlchemicalTestSystem):

    """Create an alchemically-perturbed version of LennardJonesCluster.


    Parameters
    ----------
    nx : int, optional, default=3
        number of particles in the x direction
    ny : int, optional, default=3
        number of particles in the y direction
    nz : int, optional, default=3
        number of particles in the z direction
    K : simtk.unit.Quantity, optional, default=1.0 * unit.kilojoules_per_mole/unit.nanometer**2
        harmonic restraining potential

    Examples
    --------

    Create Lennard-Jones cluster.

    >>> cluster = AlchemicalLennardJonesCluster()
    >>> system, positions = cluster.system, cluster.positions

    Create default 3x3x3 Lennard-Jones cluster in a harmonic restraining potential.

    >>> cluster = AlchemicalLennardJonesCluster(nx=10, ny=10, nz=10)
    >>> system, positions = cluster.system, cluster.positions

    """

    def __init__(self, nx=3, ny=3, nz=3, K=1.0 * unit.kilojoules_per_mole / unit.nanometer**2, **kwargs):

        TestSystem.__init__(self, **kwargs)

        # Default parameters
        mass_Ar = 39.9 * unit.amu
        q_Ar = 0.0 * unit.elementary_charge
        sigma_Ar = 3.350 * unit.angstrom
        epsilon_Ar = 0.001603 * unit.kilojoule_per_mole

        scaleStepSizeX = 1.0
        scaleStepSizeY = 1.0
        scaleStepSizeZ = 1.0

        # Determine total number of atoms.
        natoms = nx * ny * nz

        # Create an empty system object.
        system = openmm.System()

        # Create a NonbondedForce object with no cutoff.
        nb = openmm.NonbondedForce()
        nb.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)

        positions = unit.Quantity(np.zeros([natoms, 3], np.float32), unit.angstrom)

        atom_index = 0
        for ii in range(nx):
            for jj in range(ny):
                for kk in range(nz):
                    system.addParticle(mass_Ar)
                    nb.addParticle(q_Ar, sigma_Ar, epsilon_Ar)
                    x = sigma_Ar * scaleStepSizeX * (ii - nx / 2.0)
                    y = sigma_Ar * scaleStepSizeY * (jj - ny / 2.0)
                    z = sigma_Ar * scaleStepSizeZ * (kk - nz / 2.0)

                    positions[atom_index, 0] = x
                    positions[atom_index, 1] = y
                    positions[atom_index, 2] = z
                    atom_index += 1

        # Add the nonbonded force.
        system.addForce(nb)

        # Add a restrining potential centered at the origin.
        energy_expression = "(K/2.0) * (x^2 + y^2 + z^2);"
        energy_expression += "K = testsystems_AlchemicalLennardJonesCluster_K;"
        force = openmm.CustomExternalForce(energy_expression)
        force.addGlobalParameter('testsystems_AlchemicalLennardJonesCluster_K', K)
        for particle_index in range(natoms):
            force.addParticle(particle_index, [])
        system.addForce(force)

        # Alchemically modify system.
        alchemical_atom_indices = [0]
        delta = 1.0e-5
        alchemical_state = AlchemicalState(0, 0, 1 - delta, 1, annihilateElectrostatics=True, annihilateSterics=False)
        self._alchemicallyModifyLennardJones(system, nb, alchemical_atom_indices, alchemical_state)

        # Create topology.
        topology = app.Topology()
        element = app.Element.getBySymbol('Ar')
        chain = topology.addChain()
        for particle in range(system.getNumParticles()):
            residue = topology.addResidue('Ar', chain)
            topology.addAtom('Ar', element, residue)

        self.system, self.positions, self.topology = system, positions, topology


#=============================================================================================
# BINDING FREE ENERGY TESTS
#=============================================================================================

#=============================================================================================
# Lennard-Jones pair
#=============================================================================================

class LennardJonesPair(TestSystem):

    """Create a pair of Lennard-Jones particles.

    Parameters
    ----------
    mass : simtk.unit.Quantity with units compatible with amu, optional, default=39.9*amu
       The mass of each particle.
    epsilon : simtk.unit.Quantity with units compatible with kilojoules_per_mole, optional, default=1.0*kilocalories_per_mole
       The effective Lennard-Jones sigma parameter.
    sigma : simtk.unit.Quantity with units compatible with nanometers, optional, default=3.350*angstroms
       The effective Lennard-Jones sigma parameter.

    Examples
    --------

    Create Lennard-Jones pair.

    >>> test = LennardJonesPair()
    >>> system, positions = test.system, test.positions
    >>> thermodynamic_state = ThermodynamicState(temperature=300.0*unit.kelvin)
    >>> binding_free_energy = test.get_binding_free_energy(thermodynamic_state)

    Create Lennard-Jones pair with different well depth.

    >>> test = LennardJonesPair(epsilon=11.0*unit.kilocalories_per_mole)
    >>> system, positions = test.system, test.positions
    >>> thermodynamic_state = ThermodynamicState(temperature=300.0*unit.kelvin)
    >>> binding_free_energy = test.get_binding_free_energy(thermodynamic_state)

    Create Lennard-Jones pair with different well depth and sigma.

    >>> test = LennardJonesPair(epsilon=7.0*unit.kilocalories_per_mole, sigma=4.5*unit.angstroms)
    >>> system, positions = test.system, test.positions
    >>> thermodynamic_state = ThermodynamicState(temperature=300.0*unit.kelvin)
    >>> binding_free_energy = test.get_binding_free_energy(thermodynamic_state)

    """

    def __init__(self, mass=39.9 * unit.amu, sigma=3.350 * unit.angstrom, epsilon=10.0 * unit.kilocalories_per_mole, **kwargs):

        TestSystem.__init__(self, **kwargs)

        # Store parameters
        self.mass = mass
        self.sigma = sigma
        self.epsilon = epsilon

        # Charge must be zero.
        charge = 0.0 * unit.elementary_charge

        # Create an empty system object.
        system = openmm.System()

        # Create a NonbondedForce object with no cutoff.
        force = openmm.NonbondedForce()
        force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)

        # Create positions.
        positions = unit.Quantity(np.zeros([2, 3], np.float32), unit.angstrom)
        # Move the second particle along the x axis to be at the potential minimum.
        positions[1, 0] = 2.0**(1.0 / 6.0) * sigma

        # Create first particle.
        system.addParticle(mass)
        force.addParticle(charge, sigma, epsilon)

        # Create second particle.
        system.addParticle(mass)
        force.addParticle(-charge, sigma, epsilon)

        # Add the nonbonded force.
        system.addForce(force)

        # Store system and positions.
        self.system, self.positions = system, positions

        # Store ligand and receptor particle indices.
        self.ligand_indices = [0]
        self.receptor_indices = [1]

        # Create topology.
        topology = app.Topology()
        element = app.Element.getBySymbol('Ar')
        chain = topology.addChain()
        residue = topology.addResidue('Ar', chain)
        topology.addAtom('Ar', element, residue)
        residue = topology.addResidue('Ar', chain)
        topology.addAtom('Ar', element, residue)
        self.topology = topology

    def get_binding_free_energy(self, thermodynamic_state):
        """
        Compute the binding free energy of the two particles at the given thermodynamic state.

        Parameters
        ----------
        thermodynamic_state : ThermodynamicState
           The thermodynamic state specifying the temperature for which the binding free energy is to be computed.

        This is currently computed by numerical integration.

        """

        # Compute thermal energy.
        kT = kB * thermodynamic_state.temperature

        # Form the integrand function for integration in reduced units (r/sigma).
        platform = openmm.Platform.getPlatformByName('Reference')
        integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
        context = openmm.Context(self.system, integrator, platform)
        context.setPositions(self.positions)

        def integrand_openmm(xvec, args):
            """OpenMM implementation of integrand (for sanity checks)."""
            [context] = args
            positions = unit.Quantity(np.zeros([2, 3], np.float32), unit.angstrom)
            integrands = 0.0 * xvec
            for (i, x) in enumerate(xvec):
                positions[1, 0] = x * self.sigma
                context.setPositions(positions)
                state = context.getState(getEnergy=True)
                u = state.getPotentialEnergy() / kT  # effective energy
                integrand = 4.0 * pi * (x**2) * np.exp(-u)
                integrands[i] = integrand

            return integrands

        def integrand_numpy(x, args):
            """NumPy implementation of integrand (for speed)."""
            u = 4.0 * (self.epsilon) * (x**(-12) - x**(-6)) / kT
            integrand = 4.0 * pi * (x**2) * np.exp(-u)
            return integrand

        # Compute standard state volume
        V0 = (unit.liter / (unit.AVOGADRO_CONSTANT_NA * unit.mole)).in_units_of(unit.angstrom**3)

        # Integrate the free energy of binding in unitless coordinate system.
        xmin = 0.15  # in units of sigma
        xmax = 6.0  # in units of sigma
        from scipy.integrate import quadrature
        [integral, abserr] = quadrature(integrand_numpy, xmin, xmax, args=[context], maxiter=500)
        # correct for performing unitless integration
        integral = integral * (self.sigma ** 3)

        # Correct for actual integration volume (which exceeds standard state volume).
        rmax = xmax * self.sigma
        Vint = (4.0 / 3.0) * pi * (rmax**3)
        integral = integral * (V0 / Vint)

        # Clean up.
        del context, integrator

        # Compute standard state binding free energy.
        binding_free_energy = -kT * np.log(integral / V0)

        return binding_free_energy
