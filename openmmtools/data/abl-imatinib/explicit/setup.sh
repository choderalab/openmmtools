#!/bin/bash

# Set up complex using AmberTools tleap

cp 2HYY-pdbfixer.pdb receptor.pdb

# Parameterize ligand from Tripos mol2.
echo "Parameterizing ligand with GAFF and AM1-BCC charges..."
antechamber -fi mol2 -i STI02.mol2 -fo mol2 -o ligand.gaff.mol2
parmchk -i ligand.gaff.mol2 -o ligand.gaff.frcmod -f mol2

# Create AMBER prmtop/inpcrd files.
echo "Creating AMBER prmtop/inpcrd files..."
rm -f leap.log {vacuum,solvent,complex}.{inpcrd,prmtop,pdb}
tleap -f setup.leap.in > setup.leap.out
