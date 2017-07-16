"""Hamiltonian Monte Carlo Integrators

Notes
-----
The code in this module is considered EXPERIMENTAL until further notice.
"""
import logging

import simtk.unit as u
import simtk.openmm as mm

from .ghmc import GHMCIntegrator
from .utils import warn_experimental

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

        stepsPerParentStep = substeps // parentSubsteps

        if stepsPerParentStep < 1 or stepsPerParentStep != int(stepsPerParentStep):
            raise ValueError("The number for substeps for each group must be a multiple of the number for the previous group")

        stepsPerParentStep = int(stepsPerParentStep)  # needed for Python 3.x

        if group < 0 or group > 31:
            raise ValueError("Force group must be between 0 and 31")

        for i in range(stepsPerParentStep):
            self.addComputePerDof("v", "v+0.5*(dt/%s)*f%s/m" % (str_sub, str_group))
            if len(groups) == 1:
                self.addComputePerDof("x", "x+(dt/%s)*v" % str_sub)
                self.addComputePerDof("x1", "x")
                self.addConstrainPositions()
                self.addComputePerDof("v", "v+(x-x1)/(dt/%s)" % str_sub)
                self.addConstrainVelocities()
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
        warn_experimental()
        self.groups = check_groups(groups)
        super(GHMCRESPAIntegrator, self).__init__(temperature=temperature, steps_per_hmc=steps_per_hmc, timestep=timestep, collision_rate=collision_rate)
        #self.steps_per_hmc = steps_per_hmc
        #self.collision_rate = collision_rate
        #self.timestep = timestep

        #self.add_compute_steps()
