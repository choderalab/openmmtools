# Platform tests

These scripts run a battery of tests to compare the behavior of the various OpenMM platforms on test systems from `openmmtools.testsystems`.

To run:
```
python test_platforms.py
```
This will compute the potential energy on all available OpenMM platforms and flag failures where the potential energy difference between platforms is greater than `ENERGY_TOLERANCE`, which by default is 0.06 kcal/mol (0.1 kT for T ~ 300 K).

For systems that fail, the script will serialize these systems out to XML for further study.
