#!/usr/bin/env/python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Classes that store arbitrary NetCDF (or other) options and describe how to handle them

"""

# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import os
import abc

from .iodrivers import NetCDFIODriver

ABC = abc.ABCMeta('ABC', (object,), {})  # compatible with Python 2 *and* 3

# =============================================================================
# GENERIC SAVABLE DATA
# =============================================================================


class StorageInterfaceDirVar(object):
    """
    Storage Interface Directory/Variable (SIDV) class is a versatile, dynamic class which gives structure to the
    variables stored on disk by representing them as methods of the StorageInterface and other instances of itself.
    New variables and folders are created by simply trying to use them as properties of the class itself.
    The data stored and read is not kept in memory, the SIDV passes the data onto the writer or tells the fetcher to
    read from disk on demand.

    The API only shows the protected and internal methods of the SIDV class which cannot be used for variable and
    directory names.

    This class is NEVER meant to be used as its own object and should never be invoked by itself. The class is currently
    housed outside of the main StorageInterface class to show the API to users.

    Examples are located in the StorageInterface main class

    TODO: Move this class as an internal class of StorageInterface (NOT a subclass)

    """

    def __init__(self, name, storage_interface, predecessor=None):
        """
        Parameters
        ----------
        name : string
            Specify the name of the storage variable on the disk. Full path is determined from the predecessor chain
        storage_interface : StorageInterface instance
            Acting interface which is handling top level IO operations on the file itself
            The storage driver which handles all the commands is derived from this interface
        predecessor : StorageInterfaceDirVar
            Directory-like SIDV above this instance

        """
        self._name = name
        self._storage_interface = storage_interface
        self._storage_driver = self._storage_interface.storage_driver
        self._predecessor = predecessor
        # Define the object this instance is attached to
        self._variable = None
        # initially undetermined if var or dir
        self._directory = None
        self._metadata_buffer = {}

    """
    USER FUNCTIONS
    Methods the user calls on SIDV variables to actually store and read data from disk. These cannot be names of child
    SIDV instances and will raise an error if you try to use them as such.
    """

    def write(self, data, at_index=None):
        """
        Write data to a variable which cannot be appended to or write data at a specific index of an appendable
        variable. This method is typically called when you do not want the value stored to change. The at_index flag
        should be an integer to specify that the appendable variable that this is should overwrite the data at the
        specific index.

        This method raises an error if this instance of SIDV is DIRECTORY

        Parameters
        ----------
        data
            This is what will be written to disk, the data will be processed by the STORAGESYSTEM as to how to actually
            handle the data once the units have been stripped out and assigned to this instance of SIDV.
        at_index : int or None, default None
            Change the behavior or "write" to act on an appendable variable instead
            Replace the data at the specific index of at_index with data
            Checks for compatible data should be/are handled at the storage_driver level

        Examples
        --------
        Store a numpy "eye" array to the variable "my_arr"
        >>> import numpy as np
        >>> my_driver = NetCDFIODriver('my_store.nc')
        >>> my_store = StorageInterface(my_driver)
        >>> x = np.eye(3)
        >>> my_store.my_arr.write(x)

        Save 2 entries of a list, then update the frist
        >>> my_driver = NetCDFIODriver('my_store.nc')
        >>> my_store = StorageInterface(my_driver)
        >>> x = [0,0,1]
        >>> my_store.the_list.append(x)
        >>> my_store.the_list.append(x)
        >>> y = [1,1,1]
        >>> my_store.the_list.write(y, at_index=0)

        """
        if not self.bound_target:
            self._bind_to_variable_with_write_or_append()
        dump_metadata = False  # Flag to dump metadata after write-protection check
        if not self._variable:
            path = self.path
            # Try to get already on disk variable
            try:
                self._variable = self._storage_driver.get_storage_variable(path)
            except KeyError:  # Trap the "not present" case, AttributeErrors are larger problems
                self._variable = self._storage_driver.create_storage_variable(path, type(data))
            self._directory = False
            dump_metadata = True
        # Protect variable if already written
        if dump_metadata:
            self._dump_metadata_buffer()
        self._variable.write(data, at_index=at_index)

    def append(self, data):
        """
        Write data to a variable whose size changes every time this function is called. The first dimension of the
        variable on disk will be appended with data after checking the size. If no variable exists, one is created with
        a dynamic variable size for its first dimension, then the other dimensions are inferred from the data.

        This method raises an error if this instance of SIDV is DIRECTORY

        Parameters
        ----------
        data : any data type
            This is what will be written to disk, the data will be processed by the STORAGESYSTEM as to how to actually
            handle the data once the units have been stripped out and assigned to this instance of SIDV.

        Examples
        --------
        Store a single int several times in a variable that is inside a directory
        >>> my_driver = NetCDFIODriver('my_store.nc')
        >>> my_store = StorageInterface(my_driver)
        >>> x = 756
        >>> my_store.IAmADir.AnInt.append(x)
        >>> my_store.IAmADir.AnInt.append(x+1)
        >>> my_store.IAmADir.AnInt.append(x+2)
        """
        if not self.bound_target:
            self._bind_to_variable_with_write_or_append()
        if not self._variable:
            path = self.path
            # Try to get already on disk variable
            try:
                self._variable = self._storage_driver.get_storage_variable(path)
            except KeyError:  # Trap the "not present" case, AttributeErrors are larger problems
                self._variable = self._storage_driver.create_storage_variable(path, type(data))
            self._directory = False
            self._dump_metadata_buffer()
        self._variable.append(data)

    def read(self):
        """
        Read the variable and its data from disk.

        If this instance is unbound to an object in the storage_handler or is DIRECTORY, then this method raises an
        error.

        Returns
        -------
        data
            Data stored on VARIABLE read from disk and processed through the STORAGESYSTEM back into a Python type
            and possibly through the UNIT logic to recast into Quantity before being handed to the user.
        """
        if not self.bound_target:
            self._bind_to_variable_with_read()
        if not self._variable:
            path = self.path
            # Try to get variable on disk
            try:
                self._variable = self._storage_driver.get_storage_variable(path)
            except KeyError as e:
                raise e
            self._directory = False
        return self._variable.read()

    def add_metadata(self, name, data):
        """
        Attempt to add metadata to the variable/directory. Usually a string input to include additional information
        about the variable.

        Because this can act on DIRECTORY or VARIABLE types, all metadata is buffered until this instance of SIDV is
        bound to an object on the file itself.

        This does not check if the metadata already exists and simply overwrites it if present.


        Parameters
        ----------
        name : string
            Name of the metadata variable. Since this is data we are directly attaching to the variable or directory
            itself, we do not make this another SIDV instance.
        data: What data you wish to attach to the `name`d metadata pointer.

        """
        bound_target = self.bound_target
        if bound_target is None:
            self._metadata_buffer[name] = data
        else:
            self._storage_driver.add_metadata(name, data, path=self.path)

    """
    PROTECTED FLAGS
    Used for functionality checks that will likely never be called by the user, but exposed to show
    what names can NOT be used in the dynamic directory/variable namespace a user will generate.
    """
    @property
    def variable(self):
        """
        Checks if the object can be used in the .write, .append, .read functions can be used. Once this is set, this
        instance cannot be converted to a directory type.

        Returns
        -------
        variable_pointer : None or storage_driver specific unit of storage
            Returns None if this instance is a directory, or if its functionality has not been determined yet (this
            includes the variable not being assigned yet).
            Returns the storage_driver specific variable that this instance is bound to once assigned

        """
        return self._variable

    @property
    def directory(self):
        """
        Checks if the object can be used as a directory for accepting other SIDV objects. Once this is True or bound,
        the ``.write``, ``.append``, and ``.read`` functions are locked out.

        Returns
        -------
        directory_pointer : None, True, or storage_driver specific directory of storage
            Returns None if this instance is a variable, or if its functionality has not been determined yet.
            Returns True if this SIDV will be a directory, but is not yet bound. i.e. has additional SIDV children
            spawned by ``__getattr__`` method.
            Returns the storage_driver specific directory that this instance is bound to once assigned

        """
        return self._directory

    @property
    def path(self):
        """
        Generate the complete path of this instance by getting its predecessor's path + itself. This is a cascading
        function up the stack of SIDV's until the top level attached to the main SI instance is reached, then
        reassembled on the way down.

        Returns
        -------
        full_path : string
            The complete path of this variable as it is seen on the storage file_name, returned as / separated values.

        """
        # Cascade the path
        if self.predecessor is not None:
            path = self.predecessor.path + ('/' + self.name)  # the parenthesis just makes it a little faster
        else:
            path = self.name
        return path

    @property
    def predecessor(self):
        """
        Give the parent SIDV to construct the full path on the way down

        Returns
        -------
        predecessor : None or StorageInterfaceDirVar instance
            Returns this instance's parent SIDV instance or None if it is the top level SIDV.
        """
        return self._predecessor

    @property
    def name(self):
        """
        Pointer to this directory or variable as it will appear on the disk

        Returns
        -------
        name : string

        """
        return self._name

    @property
    def bound_target(self):
        """
        Fetch the handler for the bound target, either the directory or the variable so that the user can directly
        manipulate the handler has they need.

        Returns
        -------
        handler : bound handler of the storage_driver's variable or None
            returns None if the target is not bound
        """
        if self._variable:
            target = self._variable
        elif self._directory is not None and self._directory is not True:
            target = self._directory
        else:
            target = None
        return target

    def _check_variable(self):
        # Check if we are a variable or directory
        if self._directory:  # None and False will both ignore this
            raise AttributeError("Cannot write/append/read on a directory-like object!")

    def _check_directory(self):
        # Check that we can be made into a directory and are not already a variable
        if self._variable:  # None and False will both ignore this
            raise AttributeError("Cannot take directory actions on a variable object!")

    def _bind_to_variable_with_write_or_append(self):
        self._check_variable()
        # Check instanced storage driver, accessing its protected ability to bind write/append (user should not do this)
        if self._predecessor is not None:
            self._predecessor._set_predecessor_as_directory(create_if_missing=True)

    def _check_read_file(self):
        """Check that the file exists before trying to read"""
        file_name = self._storage_interface.file_name
        if not os.path.isfile(file_name):
            raise NameError("No such file exists at {}! Cannot read from non-existent file!".format(file_name))

    def _bind_to_variable_with_read(self):
        """Check that we are not a directory and all predecessors can read as well"""
        self._check_variable()
        if self._predecessor is not None:
            self._predecessor._set_predecessor_as_directory(create_if_missing=False)

    def _dump_metadata_buffer(self):
        """Dump the metadata buffer to file, this is only ever called once bound"""
        for key in self._metadata_buffer.keys():
            data = self._metadata_buffer[key]
            self.add_metadata(key, data)
        self._metadata_buffer = {}

    def _set_predecessor_as_directory(self, create_if_missing=None):
        """Special function executed by the child on the predecessor to cast predecessor as directory"""
        if create_if_missing is None:
            # create_if_missing is a kwarg instead of arg because _set_predecessor_as_directory(False) is confusing
            raise KeyError("create_if_missing must be a bool!")
        self._check_directory()
        if not create_if_missing:
            self._check_read_file()
        if self._directory is None or self._directory is True:
            self._directory = self._storage_driver.get_directory(self.path, create=create_if_missing)
        if create_if_missing:
            self._dump_metadata_buffer()

    def __getattr__(self, name):
        """This method is only called if __getattribute__ fails, meaning that the attribute is not already defined"""
        if self._variable:
            raise AttributeError("Cannot convert this object to a directory as its already bound to a variable!")
        if not self._directory:
            # Assign directory features
            self._directory = True
        setattr(self, name, StorageInterfaceDirVar(name, self._storage_interface, predecessor=self))
        return getattr(self, name)

# =============================================================================
# STORAGE INTERFACE
# =============================================================================


class StorageInterface(object):
    """
    This class interfaces with a StorageIODriver class to internally hold what folders and variables are known to the
    file on the disk, and dynamically creates them on the fly. Any attempt to reference a property which is not
    explicitly listed below implies that you wish to define either a storage directory or variable, and creates a new
    StorageInterfaceVarDir (SIVD) with that name. The SIVD is what handles the read/write operations on this disk by
    interfacing with the StorageIODriver object you provide StorageInterface class's __init__.

    See StorageInterfaceVarDir for how the dynamic interfacing works.

    Examples
    --------

    Create a basic storage interface and write new my_data to disk
    >>> my_driver = NetCDFIODriver('my_store.nc')
    >>> my_data = [4,2,1,6]
    >>> my_storage = StorageInterface(my_driver)
    >>> my_storage.my_variable.write(my_data)

    Create a folder called "vardir" with two variables in it, "var1" and "var2" holding DATA1 and DATA2 respectively
    Then, fetch data from the same folders and variables as above
    >>> my_driver = NetCDFIODriver('my_store.nc')
    >>> DATA1 = "some string"
    >>> DATA2 = (4.5, 7.0, -23.1)
    >>> my_storage = StorageInterface(my_driver)
    >>> my_storage.vardir.var1.write(DATA1)
    >>> my_storage.vardir.var2.write(DATA2)
    >>> var1 = my_storage.vardir.var1.read()
    >>> var2 = my_storage.vardir.var2.read()

    Run some_function() to generate data over an iteration, store each value in a dynamically sized var called "looper"
    >>> my_driver = NetCDFIODriver('my_store.nc')
    >>> mydata = [-1, 24, 5]
    >>> my_storage = StorageInterface(my_driver)
    >>> for i in range(10):
    ...     mydata = i**2
    ...     my_storage.looper.append(mydata)
    """
    def __init__(self, storage_driver):
        """
        Initialize the class by reading in the StorageIODriver in storage_driver. The file name is inferred from the
        storage driver and the read/write/append actions are handled by the SIVD class which also act on the
        storage_driver.

        Parameters
        ----------
        storage_driver : StorageIODriver object
            What type of storage to use. Requires fully implemented and instanced StorageIODriver class to use.

        """
        self._storage_driver = storage_driver
        # Used for logic checks

    def add_metadata(self, name, data):
        """
        Write additional meta data to attach to the storage_driver file itself.

        Parameters
        ----------
        name : string
            Name of the attribute you wish to assign
        data : any, but preferred string
            Extra meta data to add to the variable

        Examples
        --------

        Create a storage interface and add meta data
        >>> my_driver = NetCDFIODriver('my_store.nc')
        >>> my_storage = StorageInterface(my_driver)
        >>> my_storage.add_metadata('my_index', 4)
        """
        # Instance if not done
        self.storage_driver.add_metadata(name, data)

    @property
    def file_name(self):
        """
        Returns the protected _file_name variable, mentioned as an explicit method as it is one of the protected names
        that cannot be a directory or variable pointer

        Returns
        -------
        file_name : string
            Name of the file on the disk

        """
        return self.storage_driver.file_name

    @property
    def storage_driver(self):
        """
        Pointer to the object which actually handles read/write operations

        Returns
        -------
        storage_driver :
            Instance of the module which handles IO actions to specific storage type requested by storage_driver
            string at initialization.

        """
        return self._storage_driver

    def __getattr__(self, name):
        """
        Workhorse function to handle all the auto-magical path and variable assignments
        This method is only called if __getattribute__ fails, meaning that the attribute is not already defined

        Parameters
        ----------
        name : string
            Name of the group or variable that will instance a StorageInterfaceDirVar

        """
        # Instance Storage Interface Directory/Variable object
        setattr(self, name, StorageInterfaceDirVar(name, self))
        return getattr(self, name)
