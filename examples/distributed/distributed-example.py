from simtk import openmm, unit
from openmmtools import testsystems, distributed

testsystem = testsystems.AlanineDipeptideVacuum()
[system, positions] = [testsystem.system, testsystem.positions]

from celery import group, chain, chord
from openmmtools.distributed.tasks import propagate, compute_energy, mix_replicas
print('Start group...')
#group(chain(propagate.s(system, positions) | compute_energy.s(system)) for i in range(10))().get()
niterations = 10
for iteration in range(niterations):
    print('iteration %5d / %5d' % (iteration, niterations))
    c = chain(propagate.s(system, positions) | compute_energy.s(system=system))
    c().get()
print('Done.')
