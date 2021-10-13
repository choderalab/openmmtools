from math import pi
import simtk.unit as u

kB = u.BOLTZMANN_CONSTANT_kB * u.AVOGADRO_CONSTANT_NA

# OpenMM constant for Coulomb interactions in OpenMM units
# (openmm/platforms/reference/include/SimTKOpenMMRealType.h)
# TODO: Replace this with an import from simtk.openmm.constants once available
E_CHARGE = 1.602176634e-19 * u.coulomb
EPSILON0 = 8.8541878128e-12 * u.farad/u.meter
ONE_4PI_EPS0 = 1e6*(u.AVOGADRO_CONSTANT_NA*E_CHARGE**2)/(4*pi*EPSILON0)

# Standard-state volume for a single molecule in a box of size (1 L) / (avogadros number).
LITER = 1000.0 * u.centimeters**3
STANDARD_STATE_VOLUME = LITER / (u.AVOGADRO_CONSTANT_NA*u.mole)
