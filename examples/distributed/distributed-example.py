from simtk import openmm, unit
from openmmtools import testsystems, distributed

testsystem = testsystems.AlanineDipeptideVacuum()
[system, positions] = [testsystem.system, testsystem.positions]

from celery import group, chain, chord
from openmmtools.distributed.tasks import propagate, compute_energy, mix_replicas
print('Start group...')

# Run 10 iterations of 10 parallel propagations, mixing replicas after each propagation.
# Efficiency could be improved by constructing the DAG for many iterations up front and then submitting it to celery all at once.
niterations = 10
nreplicas = 10
for iteration in range(niterations):
    print('iteration %5d / %5d' % (iteration, niterations))
    propagate_replicas_stage = group( chain(propagate.s(system, positions) | compute_energy.s(system=system)) for replica_index in range(nreplicas) )
    mix_replicas_stage = mix_replicas.s()
    c = ( propagate_replicas_stage | mix_replicas_stage )
    c().get()
print('Done.')
