#!/usr/local/bin/env python

"""
Storage submodule

This submodule is a user-friendly storage driver which relies on two major classes from the user perspective:
StorageIODriver and StorageInterface

StorageIODriver
---------------

The StorageIODriver is the abstract base class which handles IO operations on disk with the real data. Derived classes
from this handle the specific storage medium, like NetCDF. This class tracks all the known variables, and where they
are on the disk. However, because the abstract class cannot know how the derived class actually interacts with the
disk, it is up to the derived class to know how each variable writes to disk.

The NetCDFIODriver is the derived StorageIODriver for NetCDF storage. The NetCDFIODriver handles the top level file
operations and keeps track of where each variable and group (equivalent to a directory) is on the disk. Read/Write
operations are handed off to the individual NCVariableCodec classes which interpret and write to file.

The NCVariableCodec is an abstract base class which defines how data is passed to and from the disk. Its derived
classes handled interpreting the specific types of data we want to store and read from disk, e.g. ints, lists np.array,
etc. Each derived NCVariableCodec enacts is own codec to know how to format the data type for storage on disk,
and how to read that data back from disk, converting it to the correct type.

The StorageIODriver's and the StorageInterface work on the principal of not knowing or caring what is on the disk
until the user first attempt to access it, the process of initial interaction with the disk is called "binding."
Variables and directories are considered "unbound" if they have not accessed the disk yet, and "bound" if they have.
This bound/unbound mechanism is to reduce the amount of IO actions to disk, which is a slow process relative to the main
code.

Binding
-------

Unbound variables and directories do not know what type of data they will handle, and only store where on the disk
data will be accessed. Upon the first attempt to read/write/append, a binding action occurs. The variables check if
there is already data on the disk at the known location, what happens next depends on what operation was called:
- If read and on disk:
    Determine the codec the variable will use.
    Fetch data, only accept data the codec can interpret.
- If read and NOT on disk:
    Raise error.
- If write/append and on disk:
    Ensure data to write is compatible with codec that was used to store data.
    Ensure data to store is of the same shape (for non-scalar data)
    Store new data.
- If write/append and not on disk:
    Allocate storage on disk
    Store new data
The variable is now considered "bound" and there are some checks which ensure new data can now be stored on this
variable.

StorageInterface
----------------

StorageInterface (SI) is a layer which runs on top of a provided StorageIODriver to create an way for users to interface
with the disk with as minimal effort as possible. Variables and directories are treated as user defined properties of
the SI, which then those properties can also be given user defined properties to point to other variables below it.
E.g. `SI.mydir.myvar` creates a directory object called "mydir" at the top level of the SI object on disk, then "myvar"
is the variable inside "mydir" on disk. The depth of this can be arbitrary. None of the user defined properties are
bound until the first read/write/append operation, which is done with `.read()` `.write()` and `.append()` functions
respectively.

StorageInterfaceDirVar
----------------------

StorageInterfaceDirVar (SIDV) is the class which is assigned to each of the user defined properties in the SI are
attached to. This class is what hooks into the StorageIODriver and passes the instructions to create/manage variables
and handle any other sub-directories/variables attached to it.

"""

from .iodrivers import NetCDFIODriver
from .storageinterface import StorageInterface
