.. _utils:

Miscellaneous Utilities
=======================

:mod:`openmmtools.utils` provides a number of useful utilities for working with OpenMM.

Timing functions
----------------

.. currentmodule:: openmmtools.utils
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    time_it
    with_timer
    Timer

Logging functions
-----------------

.. currentmodule:: openmmtools.utils
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    is_terminal_verbose
    config_root_logger

Temporary directories
---------------------

.. currentmodule:: openmmtools.utils
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    temporary_directory

Symbolic mathematics
--------------------

.. currentmodule:: openmmtools.utils
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    sanitize_expression
    math_eval

Quantity functions
------------------

.. currentmodule:: openmmtools.utils
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    is_quantity_close

OpenMM Platform utilities
-------------------------

.. currentmodule:: openmmtools.utils
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    get_available_platforms
    get_fastest_platform

Serialization utilities
-----------------------

.. currentmodule:: openmmtools.utils
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    serialize
    deserialize

Metaclass utilities
-------------------

.. currentmodule:: openmmtools.utils
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    with_metaclass
    SubhookedABCMeta

Python 2/3 compatibility
-------------------------

.. currentmodule:: openmmtools.utils
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    wraps_py2
    unwraps_py2
