# Drew-Dickenson B-DNA Dodecamer 
## Preparation
* Download PDB structure with accession code 4C64
* Removed Mg ion to create file 4c64_no_mg.pdb
* Ran tleap on the crystal structure with the input file leap.in.
    * This created prmtop and inpcrd
* Briefly minimized and thermalized the structure with openmm using minimize.py
*   * This created `minimized_dna_dodecamer.pdb`.

### Notes
* Crystal structure was resolved to a 1.3 Angs. level of resolution
* Reference of crystal structure: Chem.Commun.(Camb.), 50, page 1794, 2014.
* Crystallographic water molecules were retained.
* Solvated using the TIP3P water model in a cuboidal box with at least a 10 Angstrom
clearance between the edge of the box and the DNA.
* The DNA is parametrized with the AMBER OL15 DNA forcefield
* Neutralizing counter-ions were not added, such that the system a charge of -22e.
