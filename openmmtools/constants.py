from math import pi
try:
    import openmm.unit as u
except ImportError:  # OpenMM < 7.6
    import simtk.unit as u

kB = u.BOLTZMANN_CONSTANT_kB * u.AVOGADRO_CONSTANT_NA

# OpenMM constant for Coulomb interactions in OpenMM units
# (openmm/platforms/reference/include/SimTKOpenMMRealType.h)
# TODO: Replace this with an import from openmm.constants once available
E_CHARGE = 1.602176634e-19 * u.coulomb
EPSILON0 = 1e-6*8.8541878128e-12/(u.AVOGADRO_CONSTANT_NA*E_CHARGE**2) * u.farad/u.meter
ONE_4PI_EPS0 = 1/(4*pi*EPSILON0) * EPSILON0.unit  # we need it unitless

# Standard-state volume for a single molecule in a box of size (1 L) / (avogadros number).
LITER = 1000.0 * u.centimeters**3
STANDARD_STATE_VOLUME = LITER / (u.AVOGADRO_CONSTANT_NA*u.mole)
