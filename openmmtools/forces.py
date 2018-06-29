#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Custom OpenMM Forces classes and utilities.

"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

from simtk import openmm, unit

from openmmtools.constants import ONE_4PI_EPS0


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def find_nonbonded_force(system):
    """Find the first OpenMM `NonbondedForce` in the system.

    Parameters
    ----------
    system : simtk.openmm.System
        The system to search.

    Returns
    -------
    nonbonded_force : simtk.openmm.NonbondedForce
        The first `NonbondedForce` object in `system`.

    Raises
    ------
    ValueError
        If the system contains multiple `NonbondedForce`s

    """
    nonbonded_force = None
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            if nonbonded_force is not None:
                raise ValueError('The System has multiple NonbondedForces')
            nonbonded_force = force
    return nonbonded_force


def iterate_nonbonded_forces(system):
    """Iterate over all OpenMM ``NonbondedForce``s in an OpenMM system.

    Parameters
    ----------
    system : simtk.openmm.System
        The system to search.

    Yields
    ------
    force : simtk.openmm.NonbondedForce
        A `NonbondedForce` object in `system`.

    """
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            yield force


# =============================================================================
# REACTION FIELD
# =============================================================================

class UnshiftedReactionFieldForce(openmm.CustomNonbondedForce):
    """A force modelling switched reaction-field electrostatics.

    Contrarily to a normal `NonbondedForce` with `CutoffPeriodic` nonbonded
    method, this force sets the `c_rf` to 0.0 and uses a switching function
    to avoid forces discontinuities at the cutoff distance.

    Parameters
    ----------
    cutoff_distance : simtk.unit.Quantity, default 15*angstroms
        The cutoff distance (units of distance).
    switch_width : simtk.unit.Quantity, default 1.0*angstrom
        Switch width for electrostatics (units of distance).
    reaction_field_dielectric : float
        The dielectric constant used for the solvent.

    """

    def __init__(self, cutoff_distance=15*unit.angstroms, switch_width=1.0*unit.angstrom,
                 reaction_field_dielectric=78.3):
        k_rf = cutoff_distance**(-3) * (reaction_field_dielectric - 1.0) / (2.0*reaction_field_dielectric + 1.0)

        # Energy expression omits c_rf constant term.
        energy_expression = "ONE_4PI_EPS0*chargeprod*(r^(-1) + k_rf*r^2);"
        energy_expression += "chargeprod = charge1*charge2;"
        energy_expression += "k_rf = {:f};".format(k_rf.value_in_unit_system(unit.md_unit_system))
        energy_expression += "ONE_4PI_EPS0 = {:f};".format(ONE_4PI_EPS0)  # already in OpenMM units

        # Create CustomNonbondedForce.
        super(UnshiftedReactionFieldForce, self).__init__(energy_expression)

        # Add parameters.
        self.addPerParticleParameter("charge")

        # Configure force.
        self.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
        self.setCutoffDistance(cutoff_distance)
        self.setUseLongRangeCorrection(False)
        if switch_width is not None:
            self.setUseSwitchingFunction(True)
            self.setSwitchingDistance(cutoff_distance - switch_width)
        else:  # Truncated
            self.setUseSwitchingFunction(False)

    @classmethod
    def from_nonbonded_force(cls, nonbonded_force, switch_width=1.0*unit.angstrom):
        """Copy constructor from an OpenMM `NonbondedForce`.

        The returned force has same cutoff distance and dielectric, and
        its particles have the same charges. Exclusions corresponding to
        `nonbonded_force` exceptions are also added.

        .. warning
            This only creates the force object. The electrostatics in
            `nonbonded_force` remains unmodified. Use the function
            `replace_reaction_field` to correctly convert a system to
            use an unshifted reaction field potential.

        Parameters
        ----------
        nonbonded_force : simtk.openmm.NonbondedForce
            The nonbonded force to copy.
        switch_width : simtk.unit.Quantity
            Switch width for electrostatics (units of distance).

        Returns
        -------
        reaction_field_force : UnshiftedReactionFieldForce
            The reaction field force with copied particles.

        """
        # OpenMM gives unitless values.
        cutoff_distance = nonbonded_force.getCutoffDistance()
        reaction_field_dielectric = nonbonded_force.getReactionFieldDielectric()
        reaction_field_force = cls(cutoff_distance, switch_width, reaction_field_dielectric)

        # Set particle charges.
        for particle_index in range(nonbonded_force.getNumParticles()):
            charge, sigma, epsilon = nonbonded_force.getParticleParameters(particle_index)
            reaction_field_force.addParticle([charge])

        # Add exclusions to CustomNonbondedForce.
        for exception_index in range(nonbonded_force.getNumExceptions()):
            iatom, jatom, chargeprod, sigma, epsilon = nonbonded_force.getExceptionParameters(exception_index)
            reaction_field_force.addExclusion(iatom, jatom)

        return reaction_field_force

    @classmethod
    def from_system(cls, system, switch_width=1.0*unit.angstrom):
        """Copy constructor from the first OpenMM `NonbondedForce` in `system`.

        If multiple `NonbondedForce`s are found, an exception is raised.

        .. warning
            This only creates the force object. The electrostatics in
            `nonbonded_force` remains unmodified. Use the function
            `replace_reaction_field` to correctly convert a system to
            use an unshifted reaction field potential.

        Parameters
        ----------
        system : simtk.openmm.System
            The system containing the nonbonded force to copy.
        switch_width : simtk.unit.Quantity
            Switch width for electrostatics (units of distance).

        Returns
        -------
        reaction_field_force : UnshiftedReactionFieldForce
            The reaction field force.

        Raises
        ------
        ValueError
            If multiple `NonbondedForce`s are found in the system.

        See Also
        --------
        UnshiftedReactionField.from_nonbonded_force

        """
        nonbonded_force = find_nonbonded_force(system)
        return cls.from_nonbonded_force(nonbonded_force, switch_width)

class SwitchedReactionFieldForce(openmm.CustomNonbondedForce):
    """A force modelling switched reaction-field electrostatics.

    Parameters
    ----------
    cutoff_distance : simtk.unit.Quantity, default 15*angstroms
        The cutoff distance (units of distance).
    switch_width : simtk.unit.Quantity, default 1.0*angstrom
        Switch width for electrostatics (units of distance).
    reaction_field_dielectric : float
        The dielectric constant used for the solvent.

    """

    def __init__(self, cutoff_distance=15*unit.angstroms, switch_width=1.0*unit.angstrom,
                 reaction_field_dielectric=78.3):
        k_rf = cutoff_distance**(-3) * (reaction_field_dielectric - 1.0) / (2.0*reaction_field_dielectric + 1.0)
        c_rf = cutoff_distance**(-1) * (3*reaction_field_dielectric) / (2.0*reaction_field_dielectric + 1.0)

        # Energy expression omits c_rf constant term.
        energy_expression = "ONE_4PI_EPS0*chargeprod*(r^(-1) + k_rf*r^2 - c_rf);"
        energy_expression += "chargeprod = charge1*charge2;"
        energy_expression += "k_rf = {:f};".format(k_rf.value_in_unit_system(unit.md_unit_system))
        energy_expression += "c_rf = {:f};".format(c_rf.value_in_unit_system(unit.md_unit_system))
        energy_expression += "ONE_4PI_EPS0 = {:f};".format(ONE_4PI_EPS0)  # already in OpenMM units

        # Create CustomNonbondedForce.
        super(SwitchedReactionFieldForce, self).__init__(energy_expression)

        # Add parameters.
        self.addPerParticleParameter("charge")

        # Configure force.
        self.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
        self.setCutoffDistance(cutoff_distance)
        self.setUseLongRangeCorrection(False)
        if switch_width is not None:
            self.setUseSwitchingFunction(True)
            self.setSwitchingDistance(cutoff_distance - switch_width)
        else:  # Truncated
            self.setUseSwitchingFunction(False)

    @classmethod
    def from_nonbonded_force(cls, nonbonded_force, switch_width=1.0*unit.angstrom):
        """Copy constructor from an OpenMM `NonbondedForce`.

        The returned force has same cutoff distance and dielectric, and
        its particles have the same charges. Exclusions corresponding to
        `nonbonded_force` exceptions are also added.

        .. warning
            This only creates the force object. The electrostatics in
            `nonbonded_force` remains unmodified. Use the function
            `replace_reaction_field` to correctly convert a system to
            use an unshifted reaction field potential.

        Parameters
        ----------
        nonbonded_force : simtk.openmm.NonbondedForce
            The nonbonded force to copy.
        switch_width : simtk.unit.Quantity
            Switch width for electrostatics (units of distance).

        Returns
        -------
        reaction_field_force : UnshiftedReactionFieldForce
            The reaction field force with copied particles.

        """
        # OpenMM gives unitless values.
        cutoff_distance = nonbonded_force.getCutoffDistance()
        reaction_field_dielectric = nonbonded_force.getReactionFieldDielectric()
        reaction_field_force = cls(cutoff_distance, switch_width, reaction_field_dielectric)

        # Set particle charges.
        for particle_index in range(nonbonded_force.getNumParticles()):
            charge, sigma, epsilon = nonbonded_force.getParticleParameters(particle_index)
            reaction_field_force.addParticle([charge])

        # Add exclusions to CustomNonbondedForce.
        for exception_index in range(nonbonded_force.getNumExceptions()):
            iatom, jatom, chargeprod, sigma, epsilon = nonbonded_force.getExceptionParameters(exception_index)
            reaction_field_force.addExclusion(iatom, jatom)

        return reaction_field_force

    @classmethod
    def from_system(cls, system, switch_width=1.0*unit.angstrom):
        """Copy constructor from the first OpenMM `NonbondedForce` in `system`.

        If multiple `NonbondedForce`s are found, an exception is raised.

        .. warning
            This only creates the force object. The electrostatics in
            `nonbonded_force` remains unmodified. Use the function
            `replace_reaction_field` to correctly convert a system to
            use an unshifted reaction field potential.

        Parameters
        ----------
        system : simtk.openmm.System
            The system containing the nonbonded force to copy.
        switch_width : simtk.unit.Quantity
            Switch width for electrostatics (units of distance).

        Returns
        -------
        reaction_field_force : UnshiftedReactionFieldForce
            The reaction field force.

        Raises
        ------
        ValueError
            If multiple `NonbondedForce`s are found in the system.

        See Also
        --------
        UnshiftedReactionField.from_nonbonded_force

        """
        nonbonded_force = find_nonbonded_force(system)
        return cls.from_nonbonded_force(nonbonded_force, switch_width)

# =============================================================================
# EWALD AND PME
# =============================================================================

class PMEDirectForce(openmm.CustomNonbondedForce):
    """A force modelling direct-space component of PME electrostatics.

    Parameters
    ----------
    sign : float, optional, default=1.0
        The sign applied to direct-space force.

    """

    def __init__(self, nonbonded_force, include_particles=None, sign=+1.0):

        # Determine PME parameters from nonbonded_force
        [alpha_ewald, nx, ny, nz] = nonbonded_force.getPMEParameters()
        if (alpha_ewald/alpha_ewald.unit) == 0.0:
            # If alpha is 0.0, alpha_ewald is computed by OpenMM from from the error tolerance.
            tol = reference_force.getEwaldErrorTolerance()
            alpha_ewald = (1.0/reference_force.getCutoffDistance()) * np.sqrt(-np.log(2.0*tol))

        alpha_ewald = alpha_ewald.value_in_unit_system(unit.md_unit_system)
        energy_expression = ("sign*erfc(alpha_ewald*reff_electrostatics)/reff_electrostatics;"
                             "alpha_ewald = {};").format(alpha_ewald)

        # Copy particles
        for particle_index in range(nonbonded_force.getNumParticles()):
            [charge, sigma, epsilon] = nonbonded_force.getParticleParameters(particle_index)

            if (include_particles is not None) and (particle_index not in include_particles):
                charge *= 0.0
                epsilon *= 0.0

            force.addParticle([charge, sigma, epsilon])

        # Copy exceptions
        for exception_index in range(nonbonded_force.getNumExceptions()):
            # Retrieve parameters.
            [iatom, jatom, chargeprod, sigma, epsilon] = nonbonded_force.getExceptionParameters(exception_index)

            if (include_particles is not None) and ((iatom not in include_particles) or (jatom not in include_particles)):
                chargeprod *= 0.0
                epsilon *= 0.0

            force.addExclusion(iatom, jatom)

            # Check how many alchemical atoms we have
            both_alchemical = iatom in alchemical_atomset and jatom in alchemical_atomset
            only_one_alchemical = (iatom in alchemical_atomset) != (jatom in alchemical_atomset)

            # Check if this is an exception or an exclusion
            is_exception_epsilon = abs(epsilon.value_in_unit_system(unit.md_unit_system)) > 0.0
            is_exception_chargeprod = abs(chargeprod.value_in_unit_system(unit.md_unit_system)) > 0.0

            # If exception (and not exclusion), add special CustomBondForce terms to
            # handle alchemically-modified Lennard-Jones and electrostatics exceptions
            if both_alchemical:
                if is_exception_epsilon:
                    aa_sterics_custom_bond_force.addBond(iatom, jatom, [sigma, epsilon])
                if is_exception_chargeprod:
                    aa_electrostatics_custom_bond_force.addBond(iatom, jatom, [chargeprod, sigma])
            elif only_one_alchemical:
                if is_exception_epsilon:
                    na_sterics_custom_bond_force.addBond(iatom, jatom, [sigma, epsilon])
                if is_exception_chargeprod:
                    na_electrostatics_custom_bond_force.addBond(iatom, jatom, [chargeprod, sigma])
            # else: both particles are non-alchemical, leave them in the unmodified NonbondedForce

        # Turn off all exception contributions from alchemical atoms in the NonbondedForce
        # modelling non-alchemical atoms only
        for exception_index in range(nonbonded_force.getNumExceptions()):
            [iatom, jatom, chargeprod, sigma, epsilon] = nonbonded_force.getExceptionParameters(exception_index)
            if iatom in alchemical_atomset or jatom in alchemical_atomset:
                nonbonded_force.setExceptionParameters(exception_index, iatom, jatom,
                                                       abs(0.0*chargeprod), sigma, abs(0.0*epsilon))



        # Fix any NonbondedForce issues with Lennard-Jones sigma = 0 (epsilon = 0), which should have sigma > 0
        for particle_index in range(nonbonded_force.getNumParticles()):
            # Retrieve parameters.
            [charge, sigma, epsilon] = nonbonded_force.getParticleParameters(particle_index)
            # Check particle sigma is not zero.
            if (sigma == 0.0 * unit.angstrom):
                logger.warning("particle %d has Lennard-Jones sigma = 0 (charge=%s, sigma=%s, epsilon=%s); setting sigma=1A" % (particle_index, str(charge), str(sigma), str(epsilon)))
                sigma = 1.0 * unit.angstrom
                # Fix it.
                nonbonded_force.setParticleParameters(particle_index, charge, sigma, epsilon)
        for exception_index in range(nonbonded_force.getNumExceptions()):
            # Retrieve parameters.
            [iatom, jatom, chargeprod, sigma, epsilon] = nonbonded_force.getExceptionParameters(exception_index)
            # Check particle sigma is not zero.
            if (sigma == 0.0 * unit.angstrom):
                logger.warning("exception %d has Lennard-Jones sigma = 0 (iatom=%d, jatom=%d, chargeprod=%s, sigma=%s, epsilon=%s); setting sigma=1A" % (exception_index, iatom, jatom, str(chargeprod), str(sigma), str(epsilon)))
                sigma = 1.0 * unit.angstrom
                # Fix it.
                nonbonded_force.setExceptionParameters(exception_index, iatom, jatom, chargeprod, sigma, epsilon)


        # Energy expression omits c_rf constant term.
        energy_expression = "ONE_4PI_EPS0*chargeprod*(r^(-1) + k_rf*r^2);"
        energy_expression += "chargeprod = charge1*charge2;"
        energy_expression += "k_rf = {:f};".format(k_rf.value_in_unit_system(unit.md_unit_system))
        energy_expression += "ONE_4PI_EPS0 = {:f};".format(ONE_4PI_EPS0)  # already in OpenMM units

        # Create CustomNonbondedForce.
        super(UnshiftedReactionFieldForce, self).__init__(energy_expression)

        # Add parameters.
        self.addPerParticleParameter("charge")

        # Configure force.
        self.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
        self.setCutoffDistance(cutoff_distance)
        self.setUseLongRangeCorrection(False)
        if switch_width is not None:
            self.setUseSwitchingFunction(True)
            self.setSwitchingDistance(cutoff_distance - switch_width)
        else:  # Truncated
            self.setUseSwitchingFunction(False)

    @classmethod
    def from_nonbonded_force(cls, nonbonded_force, included_particles=None, sign=+1.0):
        """Copy constructor from an OpenMM `NonbondedForce`.

        Parameters
        ----------
        nonbonded_force : simtk.openmm.NonbondedForce
            The nonbonded force to copy.
        included_particles : list or set of int, optional, default=None
            If not None, particles for which interactions are to be included.
        sign : float, optional, default=+1.0
            Multiplier applied to energy expression.

        Returns
        -------
        force : PMEDirectForce
            The PME direct space force with copied particles.

        """

        # OpenMM gives unitless values.
        cutoff_distance = nonbonded_force.getCutoffDistance()
        reaction_field_dielectric = nonbonded_force.getReactionFieldDielectric()
        reaction_field_force = cls(cutoff_distance, switch_width, reaction_field_dielectric)

        # Set particle charges.
        for particle_index in range(nonbonded_force.getNumParticles()):
            charge, sigma, epsilon = nonbonded_force.getParticleParameters(particle_index)
            reaction_field_force.addParticle([charge])

        # Add exclusions to CustomNonbondedForce.
        for exception_index in range(nonbonded_force.getNumExceptions()):
            iatom, jatom, chargeprod, sigma, epsilon = nonbonded_force.getExceptionParameters(exception_index)
            reaction_field_force.addExclusion(iatom, jatom)

        return reaction_field_force

    @classmethod
    def from_system(cls, system, switch_width=1.0*unit.angstrom):
        """Copy constructor from the first OpenMM `NonbondedForce` in `system`.

        If multiple `NonbondedForce`s are found, an exception is raised.

        .. warning
            This only creates the force object. The electrostatics in
            `nonbonded_force` remains unmodified. Use the function
            `replace_reaction_field` to correctly convert a system to
            use an unshifted reaction field potential.

        Parameters
        ----------
        system : simtk.openmm.System
            The system containing the nonbonded force to copy.
        switch_width : simtk.unit.Quantity
            Switch width for electrostatics (units of distance).

        Returns
        -------
        reaction_field_force : UnshiftedReactionFieldForce
            The reaction field force.

        Raises
        ------
        ValueError
            If multiple `NonbondedForce`s are found in the system.

        See Also
        --------
        UnshiftedReactionField.from_nonbonded_force

        """
        nonbonded_force = find_nonbonded_force(system)
        return cls.from_nonbonded_force(nonbonded_force, switch_width)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
