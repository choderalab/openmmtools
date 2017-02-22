import simtk.unit as u

kB = u.BOLTZMANN_CONSTANT_kB * u.AVOGADRO_CONSTANT_NA

# OpenMM constant for Coulomb interactions in OpenMM units
# (openmm/platforms/reference/include/SimTKOpenMMRealType.h)
# TODO: Replace this with an import from simtk.openmm.constants once available
ONE_4PI_EPS0 = 138.935456
