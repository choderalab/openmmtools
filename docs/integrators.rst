.. _integrators:

Integrators
===========

:mod:`openmmtools.integrators` provides a number of high quality integrators implemented using OpenMM's `CustomIntegrator <http://docs.openmm.org/7.1.0/api-python/generated/simtk.openmm.openmm.CustomIntegrator.html>`_ facility.

The integrators provided in the :mod:`openmmtools.integrators` package subclass the OpenMM `CustomIntegrator <http://docs.openmm.org/7.1.0/api-python/generated/simtk.openmm.openmm.CustomIntegrator.html>`_, providing a more full-featured Pythonic class wrapping the Swig-wrapped `CustomIntegrator <http://docs.openmm.org/7.1.0/api-python/generated/simtk.openmm.openmm.CustomIntegrator.html>`_.

.. warning::
   OpenMM's `CompoundIntegrator <http://docs.openmm.org/7.1.0/api-python/generated/simtk.openmm.openmm.CompoundIntegrator.html>`_ caches OpenMM ``Integrator`` objects, but can only return the SWIG-wrapped base integrator object if you call ``CompoundIntegrator.getIntegrator()`` or ``CompoundIntegrator.getCurrentIntegrator()``.
   If you want to hold onto one of the Python subclasses we make available in :mod:`openmmtools.integrators`, you will need to cache the original Python integrator you create.
   You can still use either ``integrator.step()`` call, but you *MUST MAKE SURE THAT INTEGRATOR IS SELECTED* currently in ``CompoundIntegrator.setCurrentIntegrator(index)`` before calling ``integrator.step()`` or else the behavior is undefined.

Langevin integrators
--------------------

The entire family of Langevin integrators described by Trotter splittings of the propagator is available.
These integrators also support multiple-timestep force splittings and Metropolization.
In addition, we provide special subclasses for several popular classes of Langevin integrators.

.. NOTE::
   We highly recommend the excellent geodesic BAOAB (g-BAOAB) integrator of Leimkuhler and Matthews for all equilibrium simulations where only equilibrium configurational properties are of interest.
   This integrator (`g-BAOAB <http://rspa.royalsocietypublishing.org/content/472/2189/20160138>`_) has extraordinarily good properties for biomolecular simulation.

.. currentmodule:: openmmtools.integrators
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    LangevinIntegrator
    VVVRIntegrator
    BAOABIntegrator
    GeodesicBAOABIntegrator
    GHMCIntegrator

Nonequilibrium integrators
--------------------------

These integrators are available for nonequilibrium switching simulations, and provide additional features for measuring protocol, shadow, and total work.

.. currentmodule:: openmmtools.integrators
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    NonequilibriumLangevinIntegrator
    AlchemicalNonequilibriumLangevinIntegrator
    ExternalPerturbationLangevinIntegrator

|

Miscellaneous integrators
-------------------------

Other miscellaneous integrators are available.

.. currentmodule:: openmmtools.integrators
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    MTSIntegrator
    DummyIntegrator
    GradientDescentMinimizationIntegrator
    VelocityVerletIntegrator
    AndersenVelocityVerletIntegrator
    NoseHooverChainVelocityVerletIntegrator
    MetropolisMonteCarloIntegrator
    HMCIntegrator

|

Mix-ins
-------

A number of useful mix-ins are provided to endow integrators with additional features.

.. currentmodule:: openmmtools.integrators
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    PrettyPrintableIntegrator
    ThermostatedIntegrator

Base classes
------------

New integrators can inherit from these base classes to inherit extra features

.. currentmodule:: openmmtools.integrators
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    ThermostatedIntegrator
    NonequilibriumLangevinIntegrator
