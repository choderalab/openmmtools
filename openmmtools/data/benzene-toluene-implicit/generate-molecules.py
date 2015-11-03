#!/usr/bin/env python

"""
Generate molecules for test system using OpenEye tools.

"""

molecules = { 'BEN' : 'benzene',
              'TOL' : 'toluene' }

from openeye import oechem
from openeye import oeomega
from openeye import oeiupac
from openeye import oequacpac

# Create molecules.
for resname in molecules:
    name = molecules[resname]
    print name

    # Create molecule from IUPAC name.
    molecule = oechem.OEMol()
    oeiupac.OEParseIUPACName(molecule, name)
    molecule.SetTitle(name)

    # Normalize molecule.
    oechem.OEAddExplicitHydrogens(molecule)
    oechem.OETriposAtomNames(molecule)
    oechem.OEAssignAromaticFlags(molecule, oechem.OEAroModelOpenEye)

    # Create configuration.
    omega = oeomega.OEOmega()
    omega.SetStrictStereo(True)
    omega.SetIncludeInput(False)
    omega(molecule)

    # Create charges.
    oequacpac.OEAssignPartialCharges(molecule, oequacpac.OECharges_AM1BCCSym)

    # Write molecule.
    filename = '%s.tripos.mol2' % name
    print filename
    ofs = oechem.oemolostream()
    ofs.open(filename)
    oechem.OEWriteMolecule(ofs, molecule)
    ofs.close()

    # Replace <0> with resname.
    infile = open(filename, 'r')
    lines = infile.readlines()
    infile.close()
    newlines = [line.replace('<0>', resname) for line in lines]
    outfile = open(filename, 'w')
    outfile.writelines(newlines)
    outfile.close()

