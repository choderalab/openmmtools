"""
Create mol2 file for substituted viologen.

"""

from openeye.oechem import *
from openeye.oeomega import *
from openeye.oequacpac import *

smiles = "OC(=O)CCCCC[n+](cc1)ccc1-c2cc[n+](cc2)CCCCCC(=O)O" # substituted viologen
output_filename = 'viologen.tripos.mol2'


def assign_am1bcc_charges(mol):
    """
    Assign canonical AM1BCC charges.

    Parameters
    ----------
    mol : OEMol
       The molecule to assign charges for.

    Returns:
    charged_mol : OEMol
       The charged molecule.

    """

    omega = OEOmega()
    omega.SetIncludeInput(True)
    omega.SetCanonOrder(False)
    omega.SetSampleHydrogens(True)
    eWindow = 15.0
    omega.SetEnergyWindow(eWindow)
    omega.SetMaxConfs(800)
    omega.SetRMSThreshold(1.0)

    if omega(mol):
        OEAssignPartialCharges(mol, OECharges_AM1BCCSym)
        charged_mol = mol.GetConf(OEHasConfIdx(0))
        absFCharge = 0
        sumFCharge = 0
        sumPCharge = 0.0
        for atm in mol.GetAtoms():
            sumFCharge += atm.GetFormalCharge()
            absFCharge += abs(atm.GetFormalCharge())
            sumPCharge += atm.GetPartialCharge()
        OEThrow.Info("%s: %d formal charges give total charge %d ; Sum of Partial Charges %5.4f"
                     % (mol.GetTitle(), absFCharge, sumFCharge, sumPCharge))

        return charged_mol

    else:
        OEThrow.Warning("Failed to generate conformation(s) for molecule %s" % mol.GetTitle())



##
# MAIN
##

# Create molecule.
mol = OEMol()
OESmilesToMol(mol, smiles)

# Generate conformation.
print("Generating conformation...")
omega = OEOmega()
omega.SetMaxConfs(1)
omega(mol)

# Assign aromaticity.
OEAssignAromaticFlags(mol, OEAroModelOpenEye)

# Add explicit hydrogens.
OEAddExplicitHydrogens(mol)

# Set title
mol.SetTitle('protonated viologen')

# Assign charges.
print("Assigning canonical AM1-BCC charges...")
charged_mol = assign_am1bcc_charges(mol)

# Write conformation.
ofs = oechem.oemolostream()
ofs.open(output_filename)
oechem.OEWriteMolecule(ofs, charged_mol)

print("Done.")
