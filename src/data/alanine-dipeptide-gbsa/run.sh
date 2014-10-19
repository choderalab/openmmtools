#!/bin/tcsh

# Name of system
setenv SYSTEM alanine-dipeptide

# Clean up old files, if present.
rm -f leap.log ${SYSTEM}.{crd,prmtop,pdb}

# Create prmtop/crd files.
tleap -f setup.leap.in

# Create PDB file.
cat ${SYSTEM}.crd | ambpdb -p ${SYSTEM}.prmtop > ${SYSTEM}.pdb

