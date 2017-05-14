.. _testsystems::

Test Systems
============

:mod:`openmmtools.testsystems` contains a variety of test systems useful for benchmarking, validation, testing, and debugging.

.. TODO:
   Categories these tests into useful groupings, such as physical systems, artificial systems for testing OpenMM, easy to equilibrate systems, benchmarking systems, protein systems, protein-ligand systems, fluids, solids, etc.
   Can we tag the classes in the code with certain keywords and do this automatically?

Test systems
------------

.. currentmodule:: openmmtools.testsystems
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    TestSystem
    CustomExternalForcesTestSystem
    HarmonicOscillator
    PowerOscillator
    ConstraintCoupledHarmonicOscillator
    HarmonicOscillatorArray
    Diatom
    DiatomicFluid
    UnconstrainedDiatomicFluid
    ConstrainedDiatomicFluid
    DipolarFluid
    UnconstrainedDipolarFluid
    ConstrainedDipolarFluid
    SodiumChlorideCrystal
    LennardJonesCluster
    LennardJonesFluid
    LennardJonesFluidTruncated
    LennardJonesFluidSwitched
    LennardJonesGrid
    CustomLennardJonesFluidMixture
    WCAFluid
    IdealGas
    WaterBox
    FlexibleWaterBox
    FlexibleReactionFieldWaterBox
    FlexiblePMEWaterBox
    PMEWaterBox
    GiantFlexibleWaterBox
    FourSiteWaterBox
    FiveSiteWaterBox
    DischargedWaterBox
    FlexibleDischargedWaterBox
    GiantFlexibleDischargedWaterBox
    DischargedWaterBoxHsites
    AlchemicalWaterBox
    AlanineDipeptideVacuum
    AlanineDipeptideImplicit
    AlanineDipeptideExplicit
    TolueneVacuum
    TolueneImplicit
    TolueneImplicitHCT
    TolueneImplicitOBC1
    TolueneImplicitOBC2
    TolueneImplicitGBn
    TolueneImplicitGBn2
    HostGuestVacuum
    HostGuestImplicit
    HostGuestImplicitHCT
    HostGuestImplicitOBC1
    HostGuestImplicitOBC2
    HostGuestImplicitGBn
    HostGuestImplicitGBn2
    HostGuestExplicit
    DHFRExplicit
    LysozymeImplicit
    SrcImplicit
    SrcExplicit
    SrcExplicitReactionField
    MethanolBox
    MolecularIdealGas
    CustomGBForceSystem
    AMOEBAIonBox
    AMOEBAProteinBox
    LennardJonesPair
