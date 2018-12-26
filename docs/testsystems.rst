.. _testsystems:

Test Systems
============

:mod:`openmmtools.testsystems` contains a variety of test systems useful for benchmarking, validation, testing, and debugging.

.. TODO:
   Categories these tests into useful groupings, such as physical systems, artificial systems for testing OpenMM, easy to equilibrate systems, benchmarking systems, protein systems, protein-ligand systems, fluids, solids, etc.
   Can we tag the classes in the code with certain keywords and do this automatically?

Analytically tractable systems
------------------------------

These test systems are simple test systems where some properties are analytically tractable.

.. currentmodule:: openmmtools.testsystems
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    HarmonicOscillator
    PowerOscillator
    ConstraintCoupledHarmonicOscillator
    HarmonicOscillatorArray
    IdealGas
    MolecularIdealGas
    Diatom
    CustomExternalForcesTestSystem
    CustomGBForceSystem
    LennardJonesPair

Clusters and simple fluids
--------------------------

.. currentmodule:: openmmtools.testsystems
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    DiatomicFluid
    UnconstrainedDiatomicFluid
    ConstrainedDiatomicFluid
    DipolarFluid
    UnconstrainedDipolarFluid
    ConstrainedDipolarFluid
    LennardJonesCluster
    LennardJonesFluid
    LennardJonesFluidTruncated
    LennardJonesFluidSwitched
    LennardJonesGrid
    CustomLennardJonesFluidMixture
    WCAFluid
    DoubleWellDimer_WCAFluid
    DoubleWellChain_WCAFluid
    TolueneVacuum
    TolueneImplicit
    TolueneImplicitHCT
    TolueneImplicitOBC1
    TolueneImplicitOBC2
    TolueneImplicitGBn
    TolueneImplicitGBn2
    MethanolBox
    WaterCluster

Solids
------

.. currentmodule:: openmmtools.testsystems
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    SodiumChlorideCrystal

Water boxes
-----------

.. currentmodule:: openmmtools.testsystems
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

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
    WaterCluster

Peptide and protein systems
---------------------------

.. currentmodule:: openmmtools.testsystems
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    AlanineDipeptideVacuum
    AlanineDipeptideImplicit
    AlanineDipeptideExplicit
    DHFRExplicit
    LysozymeImplicit
    SrcImplicit
    SrcExplicit
    SrcExplicitReactionField

Complexes
---------

.. currentmodule:: openmmtools.testsystems
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    HostGuestVacuum
    HostGuestImplicit
    HostGuestImplicitHCT
    HostGuestImplicitOBC1
    HostGuestImplicitOBC2
    HostGuestImplicitGBn
    HostGuestImplicitGBn2
    HostGuestExplicit

Polarizable test systems
------------------------

.. currentmodule:: openmmtools.testsystems
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    AMOEBAIonBox
    AMOEBAProteinBox

Test system base classes
------------------------

These are base classes you can inherit from to develop new test systems.

.. currentmodule:: openmmtools.testsystems
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    TestSystem
