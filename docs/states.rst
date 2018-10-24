.. _states:

Thermodynamic and Sampler States
================================

The module :mod:`openmmtools.states` contains classes to maintain a consistent state of the simulation.

 - :class:`ThermodynamicState`: Represent and manipulate the thermodynamic state of OpenMM `System <http://docs.openmm.org/7.1.0/api-python/generated/simtk.openmm.openmm.System.html>`_ and `Context <http://docs.openmm.org/7.1.0/api-python/generated/simtk.openmm.openmm.Context.html>`_ objects.
 - :class:`SamplerState`: Represent and cache the state of the simulation that changes when the `System <http://docs.openmm.org/7.1.0/api-python/generated/simtk.openmm.openmm.System.html>`_ is integrated.
 - :class:`CompoundThermodynamicState`: Extend the :class:`ThermodynamicState` to handle parameters other than temperature and pressure through the implementations of the :class:`IComposableState` abstract class.
 - :class:`GlobalParameterState`: An implementation of :class:`IComposableState` specialized to control OpenMM forces' global parameters.
 - :class:`GlobalParameterFunction`: Allow enslaving a global parameter to arbitrary variables and mathematical expressions.

States
------

.. currentmodule:: openmmtools.states
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    ThermodynamicState
    SamplerState
    CompoundThermodynamicState
    IComposableState
    GlobalParameterFunction
    GlobalParameterState
