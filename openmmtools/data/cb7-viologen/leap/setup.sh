#!/bin/bash

# Regenerate molecules (requires OpenEye toolkit)
rm -f *.mol2

# Parameterize CB7 from Tripos mol2.
rm -f CB7.gaff.mol2
cp ../molecules/CB7.tripos.mol2 .
antechamber -fi mol2 -i CB7.tripos.mol2 -fo mol2 -o CB7.gaff.mol2
parmchk -i CB7.gaff.mol2 -o CB7.frcmod -f mol2

# Parameterize viologen from Tripos mol2.
rm -f viologen.gaff.mol2
cp ../molecules/viologen.tripos.mol2 .
antechamber -fi mol2 -i viologen.tripos.mol2 -fo mol2 -o viologen.gaff.mol2
parmchk -i viologen.gaff.mol2 -o viologen.frcmod -f mol2

# Create benzene-toluene system.
rm -f leap.log {complex,vacuum}*.{crd,prmtop,pdb}
tleap -f setup.leap.in
