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
import warnings

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

    This class is never meant to be used as its own object and should never be invoked by itself. The class is currently
    housed outside of the main StorageInterface class to show the API to users.

    Examples are located in the StorageInterface main class

    TODO: Move this class as an internal class of StorageInterface (NOT a subclass)

    """

    def __init__(self, name, storage_driver, storage_interface, predecessor=None):
        """
        Parameters
        ----------
        name : string
            Specify the name of the storage variable on the disk. Full path is determined from the predecessor chain
        storage_driver : instanced storage driver of top level StorageInterface object
            This is the driver where commands are issued to fetch/create the variables as needed
        storage_interface : StorageInterface instance
            Acting interface which is handling top level IO operations on the file itself
        predecessor : StorageInterfaceDirVar
            Directory-like SIDV above this instance

        """
        self._name = name
        self._storage_driver = storage_driver
        self._storage_interface = storage_interface
        self._predecessor = predecessor
        # Define the object this instance is attached to
        self._variable = None
        # initially undetermined type
        self._directory = None
        self._variable = None
        self._metadata_buffer = {}

    """
    USER FUNCTIONS
    Methods the user calls on SIDV variables to actually store and read data from disk. These cannot be names of child
    SIDV instances and will raise an error if you try to use them as such.
    """

    def write(self, data, protected_write=True):
        """
        Write data to a variable which cannot be appended to, nor overwritten (unless specified). This method should
        be called when you do not want the value stored to change, typically written only once. The protected_write
        variable will let you overwrite this behavior if you want to use something on disk that is a toggleable flag,
        but does not change in shape.

        Metadata is added to this variable if possible to indicate this is a write protectable variable.

        This method raises an error if this instance of SIDV is DIRECTORY

        Parameters
        ----------
        data
            This is what will be written to disk, the data will be processed by the STORAGESYSTEM as to how to actually
            handle the data once the units have been stripped out and assigned to this instance of SIDV.
        protected_write : boolean, default True
            Decide whether to check if the variable is already on file if present.
            If True, no overwrite is allowed and an error is raised if the variable already exists.
            If False, overwriting is allowed but the variable will still not be able to be appended to.

        """
        self._bind_write_append()
        previously_written = True  # Assume this is true until proven otherwise
        if not self._variable:
            path = self.path
            # Try to get already on disk variable
            try:
                self._variable = self._storage_driver.get_variable_handler(path)
            except KeyError:  # Trap the "not present" case, AttributeErrors are larger problems
                self._variable = self._storage_driver.create_storage_variable(path, type(data))
                previously_written = False
            self._directory = False
            self._dump_metadata_buffer()
        # Protect variable if already written
        if previously_written and protected_write:
            # Lock ability to protect the variable
            raise IOError("Cannot write to protected object on disk! Set 'protected_write = False` to overwrite "
                          "this protection.")
        self._variable.write(data)

    def append(self, data):
        """
        Write data to a variable whose size changes every time this function is called. The first dimension of the
        variable on disk will be appended with data after checking the size. If no variable exists, one is created with
        a dynamic variable size for its first dimension, then the other dimensions are inferred from the data.

        This method raises an error if this instance of SIDV is DIRECTORY

        Parameters
        ----------
        data
            This is what will be written to disk, the data will be processed by the STORAGESYSTEM as to how to actually
            handle the data once the units have been stripped out and assigned to this instance of SIDV.

        """
        self._bind_write_append()
        if not self._variable:
            path = self.path
            # Try to get already on disk variable
            try:
                self._variable = self._storage_driver.get_variable_handler(path)
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
        self._bind_read()
        if not self._variable:
            path = self.path
            # Try to get variable on disk
            try:
                self._variable = self._storage_driver.get_variable_handler(path)
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
        if self._variable:
            self._variable.add_metadata(name, data)
        elif self._directory:
            self._variable.add_metadata(name, data)
        else:
            self._metadata_buffer[name] = data

    """
    PROTECTED FLAGS
    Used for functionality checks that will likely never be called by the user, but exposed to show
    what names can NOT be used in the dynamic directory/variable namespace a user will generate.
    """
    @property
    def variable(self):
        """
        Checks if the object can be used in the .write, .append, .fetch functions can be used. Once this is set, this
        instance cannot be converted to a directory type.

        Returns
        -------
        variable_pointer : None or storage_system specific unit of storage
            Returns None if this instance is a directory, or if its functionality has not been determined yet (this
                includes the variable not being assigned yet)
            Returns the storage_system specific variable that this instance is bound to once assigned

        """
        return self._variable

    @property
    def directory(self):
        """
        Checks if the object can be used as a directory for accepting other SIDV objects. Once this is set, the
        .write, .append, and .fetch functions are locked out.

        Returns
        -------
        directory_pointer : None, or storage_system specific directory of storage
            Returns None if this instance is a variable, or if its functionality has not been determined yet (this
                includes the directory not being assigned yet)
            Returns the storage_system specific directory that this instance is bound to once assigned

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
        path = []
        # Cascade the path
        if self.predecessor is not None:
            path = self.predecessor.path.split('/')  # Break path-like string to list to process
        # Add self to the end
        path.extend([self.name])  # Wrap in list or it iterates over the name chars
        # Reduce to a path-like string
        return '/'.join(path)

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
    def storage_driver(self):
        """
        Pointer to the object which actually handles read/write operations

        Returns
        -------
        storage_driver :
            Instance of the module which handles IO actions to specific storage type requested by storage_system
            string at initialization.

        """
        return self._storage_driver

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

    def _bind_write_append(self):
        self._check_variable()
        # Check instanced storage driver, accessing its protected ability to bind write/append (user should not do this)
        self._storage_interface._instance_write_append()
        if self._predecessor is not None:
            self._predecessor._write_append_directory()

    def _bind_read(self):
        self._check_variable()
        # Check instanced storage driver, accessing its protected ability to bind reading (user should not do this)
        self._storage_interface._instance_read()
        if self._predecessor is not None:
            self._predecessor._read_directory()

    def _dump_metadata_buffer(self):
        """
        Dump the metadata buffer to file, this is only ever called once bound
        """
        for key in self._metadata_buffer.keys():
            data = self._metadata_buffer[key]
            self.add_metadata(key, data)
        self._metadata_buffer = {}

    def _read_directory(self):
        """Special function of a predecessor to try and fetch the directory from file"""
        self._storage_interface._instance_read()
        self._check_directory()
        if self._directory is None or self._directory is True:
            self._directory = self._storage_driver.get_directory(self.path, create=False)

    def _write_append_directory(self):
        """Special function of a predecessor to try and fetch the directory from file"""
        self._storage_interface._instance_read()
        self._check_directory()
        if self._directory is None or self._directory is True:
            self._directory = self._storage_driver.get_directory(self.path, create=True)

    def __getattr__(self, name):
        if self._variable:
            raise AttributeError("Cannot convert this object to a directory as its already bound to a variable!")
        if not self._directory:
            # Assign directory features
            self._directory = True
        setattr(self, name, StorageInterfaceDirVar(name, self._storage_driver, self._storage_interface,
                                                   predecessor=self))
        return getattr(self, name)

# =============================================================================
# STORAGE INTERFACE
# =============================================================================


class StorageInterface(object):
    """
    This class holds what folders and variables are known to the file on the disk, and dynamically
    creates them on the fly. Any attempt to reference a property which is not explicitly listed below implies that you
    wish to define either a storage directory or variable, and creates a new StorageInterfaceVarDir (SIVD) with that
    name. The SIVD is what handles the read/write operations on this disk by interfacing with the uninstanced
    StorageIODriver you provide to storage_system keyword in StorageInterface class's __init__.

    See StorageInterfaceVarDir for how the dynamic interfacing works.

    Examples
    --------

    Create a basic storage system and write new my_data to disk
    >>> my_data = [4,2,1,6]
    >>> my_storage = StorageInterface('my_store.nc')
    >>> my_storage.my_variable.write(my_data)

    Create a folder called "vardir" with two variables in it, "var1" and "var2" holding DATA1 and DATA2 respectively
    Then, fetch data from the same folders and variables as above
    >>> DATA1 = "some string"
    >>> DATA2 = (4.5, 7.0, -23.1)
    >>> my_storage = StorageInterface('my_store.nc')
    >>> my_storage.vardir.var1.write(DATA1)
    >>> my_storage.vardir.var2.write(DATA2)
    >>> var1 = my_storage.vardir.var1.fetch()
    >>> var2 = my_storage.vardir.var2.fetch()

    Run some_function() to generate data over an iteration, store each value in a dynamically sized var called "looper"
    >>> mydata = [-1, 24, 5]
    >>> my_storage = StorageInterface('my_store.nc')
    >>> for i in range(10):
    >>>     mydata = some_function()
    >>>     my_storage.looper.append(mydata)
    """
    def __init__(self, file_name, storage_system=NetCDFIODriver):
        """

        Parameters
        ----------
        file_name : string
            Full path to the location of the save file on the operating system. A new file will be generated if one
            cannot be found, or this class will load and read the file it does find in that location.
        storage_system : un-instanced StorageIODriver, default: NetCDFIODriver
            What type of storage to use. Requires fully implemented StorageIODriver class to use.

        """
        self._file_name = file_name
        self._storage_system = storage_system
        # Used for logic checks
        self._base_storage_type = storage_system

    def add_metadata(self, name, data):
        """
        Write additional meta data to attach to the storage_system file itself.
        Parameters
        ----------
        name : string
            Name of the attribute you wish to assign
        data : any, but prefered string
            Extra meta data to add to the variable

        """
        # Instance if not done
        self._instance_write_append()
        self.storage_system.add_metadata(name, data)

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
        return self._file_name

    @property
    def storage_system(self):
        """
        Pointer to the object which actually handles read/write operations
        Returns
        -------
        storage_system :
            Instance of the module which handles IO actions to specific storage type requested by storage_system
            string at initialization.

        """
        # Check if instanced yet
        if not self.instanced:
            warnings.warn("Storage system currently not instanced", RuntimeWarning)
        return self._storage_system

    def instanced(self):
        """Helper function to see if we are instanced"""
        return isinstance(self._storage_system, self._base_storage_type)

    def _instance_read(self):
        """
        Try to instance if file on disk in append, fail if not there
        Don't create empty file that then cannot read
        """
        if not os.path.isfile(self._file_name):
            raise NameError("No such file exists at {}! Cannot read from non-existent file!".format(self._file_name))
        if not self.instanced():
            self._storage_system = self._base_storage_type(self._file_name, access_mode='a')
        return

    def _instance_write_append(self):
        """
        Try to instance if file on disk in append, create file if not there in write
        """
        if not self.instanced():
            if os.path.isfile(self._file_name):
                self._storage_system = self._base_storage_type(self._file_name, access_mode='a')
            else:
                self._storage_system = self._base_storage_type(self._file_name, access_mode='w')

    def __getattr__(self, name):
        """
        Workhorse function to handle all the automagical path and variable assignments

        Parameters
        ----------
        name : string
            Name of the group or variable that will instance a StorageInterfaceDirVar

        """
        # Instance storage system
        self._instance_write_append()
        setattr(self, name, StorageInterfaceDirVar(name, self.storage_system, self))
        return getattr(self, name)
