.. _scripts:

:mod:`openmmtools.scripts` contains scripts that may be useful in testing your OpenMM installation is functioning correctly:

Command-line scripts
--------------------

``./test-openmm-platforms`` will test the various platforms available to OpenMM to ensure that all systems in :mod:`openmmtools.testsystems` give consistent potential energies.
If differences in energies in excess of ``ENERGY_TOLERANCE`` (default: 0.06 kcal/mol) are detected, these systems will be serialized to XML for further debugging.
