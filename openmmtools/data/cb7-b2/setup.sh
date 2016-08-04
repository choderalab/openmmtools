#!/bin/bash

# Generate any missing parameters
parmchk2 -i cb7_am1-bcc.mol2 -f mol2 -o cb7_am1-bcc.frcmod
parmchk2 -i b2_am1-bcc.mol2 -f mol2 -o b2_am1-bcc.frcmod

# Create benzene-toluene system.
rm -f leap.log {complex,vacuum}*.{crd,prmtop,pdb}
tleap -f setup.leap.in
