#!/bin/bash

# Regenerate molecules (requires OpenEye toolkit)
rm -f *.mol2
python generate-molecules.py

# Parameterize benzene from Tripos mol2.
rm -f benzene.gaff.mol2 benzene.frcmod
antechamber -fi mol2 -i benzene.tripos.mol2 -fo mol2 -o benzene.gaff.mol2
parmchk2 -i benzene.gaff.mol2 -o benzene.frcmod -f mol2

# Parameterize toluene from Tripos mol2.
rm -f toluene.gaff.mol2 toluene.frcmod
antechamber -fi mol2 -i toluene.tripos.mol2 -fo mol2 -o toluene.gaff.mol2
parmchk2 -i toluene.gaff.mol2 -o toluene.frcmod -f mol2

# Create benzene-toluene system.
rm -f leap.log {complex,solvent}.{crd,prmtop,pdb}
tleap -f setup.leap.in
