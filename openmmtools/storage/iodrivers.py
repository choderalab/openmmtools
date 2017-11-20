#!/usr/bin/env/python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Module which houses all the handling instructions for reading and writing to netCDF files for a given type.

This exists as its own module to keep the main storage module file smaller since any number of types may need to be
saved which special instructions for each.

"""

# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import os
import abc
import yaml
import warnings
import importlib
import collections
import numpy as np
import netCDF4 as nc
from sys import getsizeof

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from simtk import unit

from ..utils import typename, quantity_from_string

# TODO: Use the `with_metaclass` from .utils when we merge it in
ABC = abc.ABCMeta('ABC', (object,), {})  # compatible with Python 2 *and* 3


# =============================================================================
# MODULE VARIABLES
# =============================================================================

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================


def decompose_path(path):
    """
    Break a path down into individual parts
    Parameters
    ----------
    path : string
        Path to variable on the

    Returns
    -------
    structure : tuple of strings
        Tuple of split apart path
    """
    return tuple((path_entry for path_entry in path.split('/') if path_entry != ''))


def normalize_path(path):
    """
    Remove trailing/leading slashes from each part of the path and combine them into a clean, normalized path
    Similar to os.path.normpath, but just its own function

    Parameters
    ----------
    path : string
        Path variable to normalize

    Returns
    -------
    normalized_path : string
        Normalized path as a single string

    """
    split_path = decompose_path(path)
    return '/'.join([path_part.strip('/ ') for path_part in split_path if path_part is not ''])


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================


# =============================================================================
# ABSTRACT DRIVER
# =============================================================================

class StorageIODriver(ABC):
    """
    Abstract class to define the basic functions any storage driver needs to read/write to the disk.
    The specific driver for a type of storage should be a subclass of this with its own
    encoders and decoders for specific file types.

    Each type of variable codec should subclass :class:`Codec` which has the minimum ``write``, ``read``, and ``append``
    methods

    Parameters
    ----------
    file_name : string
        Name of the file to read/write to of a given storage type
    access_mode : string or None, Default None, accepts 'w', 'r', 'a'
        Define how to access the file in either write, read, or append mode
        None should behave like Python "a+" in which a file is created if not present, or opened in append if it is.
        How this is implemented is up to the subclass

    """
    def __init__(self, file_name, access_mode=None):
        # Internal map from Python Type <-> De/Encoder which handles the actual encoding and decoding of the data
        self._codec_type_maps = {}
        self._variables = {}
        self._file_name = file_name
        self._access_mode = access_mode

    def set_codec(self, type_key, codec):
        """
        Add new codifier to the specific driver class. This coder must know how to read/write and append to disk.

        This method also acts to overwrite any existing type <-> codec map, however, will not overwrite any codec
        already in use by a variable. E.g. Variable X of type T has codec A as the codecs have {T:A}. The maps is
        changed by set_codec(T,B) so now {T:B}, but X will still be on codec A. Unloading X and then reloading X will
        bind it to codec B.

        Parameters
        ----------
        type_key : Unique immutable object
            Unique key that will be added to identify this de_encoder as part of the class
        codec : Specific codifier class
            Class to handle all of the encoding of decoding of the variables

        """
        self._codec_type_maps[type_key] = codec

    @abc.abstractmethod
    def create_storage_variable(self, path, type_key):
        """
        Create a new variable on the disk and at the path location and store it as the given type.

        Parameters
        ----------
        path : string
            The way to identify the variable on the storage system. This can be either a variable name or a full path
            (such as in NetCDF files)
        type_key : Immutable object
            Type specifies the key identifier in the _codec_type_maps added by the set_codec function. If type is not in
            _codec_type_maps variable, an error is raised.

        Returns
        -------
        bound_codec : Codec which is linked to a specific reference on the disk.

        """
        raise NotImplementedError("create_variable has not been implemented!")

    @abc.abstractmethod
    def get_storage_variable(self, path):
        """
        Get a variable IO object from disk at path. Raises a KeyError or AttributeError if no storage object exists at
        that level

        Parameters
        ----------
        path : string
            Path to the variable/storage object on disk

        Returns
        -------
        bound_codec : Codec which is linked to a specific reference on the disk.

        """
        raise NotImplementedError("get_storage_variable has not been implemented!")

    @abc.abstractmethod
    def get_directory(self, path, create=True):
        """
        Get a directory-like object located at path from disk.

        Parameters
        ----------
        path : string
            Path to directory-like object on disk
        create: boolean, default: True
            Should create the stack of directories on the way down, similar function to `mkdir -p` in shell

        Returns
        -------
        directory_handler : directory object as its stored on disk

        """
        raise NotImplementedError("get_directory method has not been implemented!")

    @abc.abstractmethod
    def close(self):
        """
        Instruct how to safely close down the file.

        """
        raise NotImplementedError("close method has not been implemented!")

    @abc.abstractmethod
    def add_metadata(self, name, value, path=''):
        """
        Function to add metadata to the file. This can be treated as optional and can simply be a `pass` if you do not
        want your storage system to handle additional metadata

        Parameters
        ----------
        name : string
            Name of the attribute you wish to assign
        value : any, but preferred string
            Extra meta data to add to the variable
        path : string, Default: ''
            Extra path pointer to add metadata to a specific location if platform allows it

        """
        raise NotImplementedError("add_metadata has not been implemented!")

    @property
    def file_name(self):
        """File name of on hard drive"""
        return self._file_name

    @property
    def access_mode(self):
        """Access mode of file on disk"""
        return self._access_mode


# =============================================================================
# NetCDF IO Driver
# =============================================================================


class NetCDFIODriver(StorageIODriver):
    """
    Driver to handle all NetCDF IO operations, variable creation, and other operations.
    Can be extended to add new or modified type codecs
    """
    def get_directory(self, path, create=True):
        """
        Get the group (directory) on the NetCDF file, create the full path if not present

        Parameters
        ----------
        path : string
            Path to group on the disk
        create: boolean, default: True
            Should create the directory/ies on the way down, similar function to `mkdir -p` in shell
            If False, raise KeyError if not in the stack

        Returns
        -------
        group : NetCDF Group
            Group object requested from file. All subsequent groups are created on the way down and can be accessed
            the same way.
        """
        self._check_bind_to_file()
        path = normalize_path(path)
        try:
            group = self._groups[path]
        except KeyError:
            if create:
                group = self._bind_group(path)
            else:
                split_path = decompose_path(path)
                target = self.ncfile
                for index, fragment in enumerate(split_path):
                    target = target.groups[fragment]
                # Do a proper bind group now since all other fragments now exist
                group = self._bind_group(path)
        finally:
            return group

    def get_storage_variable(self, path):
        """
        Get a variable IO object from disk at path. Raises an error if no storage object exists at that level

        Parameters
        ----------
        path : string
            Path to the variable/storage object on disk

        Returns
        -------
        codec : Subclass of NCVariableCodec
            The codec tied to a specific variable and bound to it on the disk

        """
        self._check_bind_to_file()
        path = normalize_path(path)
        try:
            # Check if the codec is already known to this instance
            codec = self._variables[path]
        except KeyError:
            try:
                # Attempt to read the disk and bind to that variable
                # Navigate the path down from top NC file to last entry
                head_group = self.ncfile
                split_path = decompose_path(path)
                for header in split_path[:-1]:
                    head_group = head_group.groups[header]
                # Check if this is a group type
                is_group = False
                if split_path[-1] in head_group.groups:
                    # Check if storage object IS a group (e.g. dict)
                    try:
                        obj = head_group.groups[split_path[-1]]
                        store_type = obj.getncattr('IODriver_Storage_Type')
                        if store_type == 'groups':
                            variable = obj
                            is_group = True
                    except AttributeError:  # Trap the case of no group name in head_group, non-fatal
                        pass
                if not is_group:
                    # Bind to the specific variable instead since its not a group
                    variable = head_group.variables[split_path[-1]]
            except KeyError:
                raise KeyError("No variable found at {} on file!".format(path))
            try:
                # Bind to the storage type by mapping IODriver_Type -> Known Codec
                data_type = variable.getncattr('IODriver_Type')
                head_path = '/'.join(split_path[:-1])
                target_name = split_path[-1]
                # Remember the group for the future while also getting storage binder
                if head_path == '':
                    storage_object = self.ncfile
                else:
                    storage_object = self._bind_group(head_path)
                uninstanced_codec = self._IOMetaDataReaders[data_type]
                self._variables[path] = uninstanced_codec(self, target_name, storage_object=storage_object)
                codec = self._variables[path]
            except AttributeError:
                raise AttributeError("Cannot auto-detect variable type, ensure that 'IODriver_Type' is a set ncattr")
            except KeyError:
                raise KeyError("No mapped type codecs known for 'IODriver_Type' = '{}'".format(data_type))
        return codec

    def create_storage_variable(self, path, type_key):
        self._check_bind_to_file()
        path = normalize_path(path)
        try:
            codec = self._codec_type_maps[type_key]
        except KeyError:
            raise KeyError("No known Codec for given type!")
        split_path = decompose_path(path)
        # Bind groups as needed, splitting off the last entry
        head_path = '/'.join(split_path[:-1])
        target_name = split_path[-1]
        if head_path == '':
            storage_object = self.ncfile
        else:
            storage_object = self._bind_group(head_path)
        self._variables[path] = codec(self, target_name, storage_object=storage_object)
        return self._variables[path]

    def check_scalar_dimension(self):
        """
        Check that the `scalar` dimension exists on file and create it if not

        """
        self._check_bind_to_file()
        if 'scalar' not in self.ncfile.dimensions:
            self.ncfile.createDimension('scalar', 1)  # scalar dimension

    def check_infinite_dimension(self, name='iteration'):
        """
        Check that the arbitrary infinite dimension exists on file and create it if not.

        Parameters
        ----------
        name : string, optional, Default: 'iteration'
            Name of the dimension

        """
        self._check_bind_to_file()
        if name not in self.ncfile.dimensions:
            self.ncfile.createDimension(name, 0)

    def check_iterable_dimension(self, length=0):
        """
        Check that the dimension of appropriate size for a given iterable exists on file and create it if not

        Parameters
        ----------
        length : int, Default: 0
            Length of the dimension, leave as 0 for infinite length

        """
        if type(length) is not int:
            raise TypeError("length must be an integer, not {}!".format(type(length)))
        if length < 0:
            raise ValueError("length must be >= 0")
        name = 'iterable{}'.format(length)
        if name not in self.ncfile.dimensions:
            self.ncfile.createDimension(name, length)

    def generate_infinite_dimension(self):
        """
        Generate a new infinite dimension and return the name of that dimension

        Returns
        -------
        infinite_dim_name : string
            Name of the new infinite dimension on file
        """
        self._check_bind_to_file()
        created_dim = False
        while not created_dim:
            infinite_dim_name = 'unlimited{}'.format(self._auto_iterable_count)
            if infinite_dim_name not in self.ncfile.dimensions:
                self.ncfile.createDimension(infinite_dim_name, 0)
                created_dim = True
            else:
                self._auto_iterable_count += 1
        return infinite_dim_name

    def add_metadata(self, name, value, path='/'):
        """
        Add metadata to self on disk, extra bits of information that can be used for flags or other variables

        Parameters
        ----------
        name : string
            Name of the attribute you wish to assign
        value : any, but preferred string
            Extra meta data to add to the variable
        path : string, optional, Default: '/'
            Path to the object to assign metadata. If the object does not exist, an error is raised
            Not passing a path in attaches the data to the top level file
        """
        self._check_bind_to_file()
        path = normalize_path(path)
        split_path = decompose_path(path)
        if len(split_path) == 0:
            self.ncfile.setncattr(name, value)
        elif split_path[0].strip() == '':  # Split this into its own elif since if the first is true this will fail
            self.ncfile.setncattr(name, value)
        elif path in self._groups:
            self._groups[path].setncattr(name, value)
        elif path in self._variables:
            self._variables[path].add_metadata(name, value)
        else:
            raise KeyError("Cannot assign metadata at path {} since no known object exists there! "
                           "Try get_directory or get_storage_variable first.".format(path))

    def _bind_group(self, path):
        """
        Bind a group to a particular path on the nc file. Note that this method creates the cascade of groups all the
        way to the final object if it can.

        Parameters
        ----------
        path : string
            Absolute path to the group as it appears on the NetCDF file.

        Returns
        -------
        group : NetCDF Group
            The group that path points to. Can be accessed by path through the ._groups dictionary after binding

        """
        # NetCDF4 creates the cascade of groups automatically or returns the group if already present
        # To simplify code, the cascade of groups is not stored in this class until called
        self._check_bind_to_file()
        path = normalize_path(path)
        self._groups[path] = self.ncfile.createGroup(path)
        return self._groups[path]

    def sync(self):
        if self.ncfile is not None:
            self.ncfile.sync()

    def close(self):
        if self.ncfile is not None:
            # Ensure the netcdf file closes down
            self.sync()
            self.ncfile.close()
            self.ncfile = None

    def _check_bind_to_file(self):
        """
        Bind to and create the file if it does not already exist (depending on access_mode)

        """
        if self.ncfile is None:
            if self.access_mode is None:
                if os.path.isfile(self.file_name):
                    self.ncfile = nc.Dataset(self.file_name, 'a')
                else:
                    self.ncfile = nc.Dataset(self.file_name, 'w')
            else:
                self.ncfile = nc.Dataset(self.file_name, self.access_mode)

    def _update_IOMetaDataReaders(self):
        self._IOMetaDataReaders = {self._codec_type_maps[key].dtype_string(): self._codec_type_maps[key] for key in
                                   self._codec_type_maps}

    def set_codec(self, type_key, codec):
        super(NetCDFIODriver, self).set_codec(type_key, codec)
        self._update_IOMetaDataReaders()

    def __init__(self, file_name, access_mode=None):
        super(NetCDFIODriver, self).__init__(file_name, access_mode=access_mode)
        # Initialize the file bind variable. All actions involving files
        self.ncfile = None
        self._groups = {}
        # Bind all of the Type Codecs
        super_codec = super(NetCDFIODriver, self).set_codec  # Shortcut for this init to avoid excess loops
        super_codec(str, NCString)  # String
        super_codec(int, NCInt)  # Int
        super_codec(dict, NCDict)  # Dict
        super_codec(float, NCFloat)  # Float
        # List/tuple
        super_codec(list, NCIterable)
        super_codec(tuple, NCIterable)
        super_codec(np.ndarray, NCArray)  # Array
        super_codec(unit.Quantity, NCQuantity)  # Quantity
        # Bind the metadata reader types based on the dtype string of each class
        self._update_IOMetaDataReaders()
        # Counter for auto-creating infinite iterable dimensions
        self._auto_iterable_count = 0


# =============================================================================
# ABSTRACT TYPE Codecs
# =============================================================================

class Codec(ABC):
    """
    Basic abstract codec class laying out all the methods which must be implemented in every Codec.
    All codec need a ``write``, ``read``, and ``append`` method.

    Parameters
    ----------
    parent_driver : Parent StorageIODriver driver
        Driver this instance of the codec is bound to which can manipulate the top level file and possible meta
        data handling
    target : string
        String of the name of the object. Not explicitly a variable nor a group since the object could be either
    """

    def __init__(self, parent_driver, target):
        self._target = target
        # Target of the top level driver which houses all the variables
        self._parent_driver = parent_driver
        # Buffer to store metadata if assigned before binding
        self._metadata_buffer = {}

    @abc.abstractmethod
    def read(self):
        """
        Return the property read from the file

        Returns
        -------
        Given property read from the file and cast into the correct Python data type
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def write(self, data, at_index=None):
        """
        Tell this writer how to write to the file given the final object that it is bound to

        Alternately, tell a variable which is normally appended to to write a specific entry on the index at_index

        Parameters
        ----------
        data : any data you wish to write
        at_index : None or Int, optional, default=None
            Specify the index of a variable created by append to write specific data at the index entry.
            When None, this option is ignored
            The integer of at_index must be <= to the size of the appended data
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def append(self, data):
        """
        Tell this codec how to append to the file given the final object that it is bound to. This should allways write
        to the end of the currently existing data.

        Some :class:`StorageIODriver``'s may not be able to append due to the type of storage medium. In this case, this
        method should be implemented and raise a ``NotImplementedError`` or ``RuntimeError`` with an appropriate
        message

        To overwrite data at a specific index of the already appended data, use the :func:`write`` method with the
        ``at_index`` keyword.

        Parameters
        ----------
        data : any data you wish to append

        """
        raise NotImplementedError


class NCVariableCodec(Codec):
    """
    Pointer class which provides instructions on how to handle a given nc_variable

    Bind to a given nc_storage_object on ncfile with given final_target_name,
    If no nc_storage_object is None, it defaults to the top level ncfile

    Parameters
    ----------
    parent_driver : Parent NetCDF driver
        Class which can manipulate the NetCDF file at the top level for dimension creation and meta handling
    target : string
        String of the name of the object. Not explicitly a variable nor a group since the object could be either
    storage_object : NetCDF file or NetCDF group, optional, Default to ncfile on parent_driver
        Object the variable/object will be written onto

    """

    def __init__(self, parent_driver, target, storage_object=None):
        super(NCVariableCodec, self).__init__(parent_driver, target)
        # Eventual NetCDF object this class will be bound to
        self._bound_target = None
        # Target object where the data read/written to this instance resides
        # Similar to the "directory" in a file system
        if storage_object is None:
            storage_object = self._parent_driver.ncfile
        self._storage_object = storage_object

    @abc.abstractproperty  # TODO: Depreciate when we move to Python 3 fully with @abc.abstractmethod + @property
    def dtype(self):
        """
        Define the Python data type for this variable

        Returns
        -------
        dtype : type

        """
        raise NotImplementedError("dtype property has not been implemented in this subclass yet!")

    # @abc.abstractproperty
    @staticmethod
    def dtype_string():
        """
        Short name of variable for strings and errors

        Returns
        -------
        string

        """
        # TODO: Replace with @abstractstaticmethod when on Python 3
        raise NotImplementedError("dtype_string has not been implemented in this subclass yet!")

    @abc.abstractproperty
    def _encoder(self):
        """
        Define the encoder used to convert from Python Data -> netcdf

        Returns
        -------
        encoder : function
            Returns the encoder function
        """
        raise NotImplementedError("Encoder has not yet been set!")

    @abc.abstractproperty
    def _decoder(self):
        """
        Define the decoder used to convert from netCDF -> Python Data

        Returns
        -------
        decoder : function
            Returns the decoder function
        """
        raise NotImplementedError("Decoder has not yet been set!")

    def _bind_read(self):
        """
        A one time event that binds this class to the object on disk. This method should set self._bound_target
        This function is unique to the read() function in that no data is attempted to write to the disk.
        Should raise error if the object is not found on disk (i.e. no data has been written to this location yet)
        Should raise error if the object on disk is incompatible with this type of Codec.

        This is normally a common action among codecs, but can be redefined as needed in subclasses

        Returns
        -------
        None, but should set self._bound_target
        """
        self._attempt_storage_read()
        # Handle variable size objects
        # This line will not happen unless target is real, so output_mode will return the correct value
        if self._output_mode is 'a':
            self._save_shape = self._bound_target.shape[1:]
        else:
            self._save_shape = self._bound_target.shape

    @abc.abstractmethod
    def _bind_write(self, data):
        """
        A one time event that binds this class to the object on disk. This method should set self._bound_target
        This function is unique to the write() function in that the data passed in should help create the storage object
        if not already on disk and prepare it for a write operation.
        Last action of this method should always be dump_metadata_buffer.

        Parameters
        ----------
        data : Any type this Codec can process
            Data which will be stored to disk of type. The data should not be written at this stage, but inspected to
            configure the storage as needed. In some cases, you may not even need the data.

        Returns
        -------
        None, but should set self._bound_target
        """
        raise NotImplementedError("_bind_write function has not been implemented in this subclass yet!")

    @abc.abstractmethod
    def _bind_append(self, data):
        """
        A one time event that binds this class to the object on disk. This method should set self._bound_target
        This function is unique to the append() function in that the data passed in should append what is at
        the location, or should create the object, then write the data with the first dimension infinite in size.
        Last action of this method should always be dump_metadata_buffer.

        Parameters
        ----------
        data : Any type this Codec can process
            Data which will be stored to disk of type. The data should not be written at this stage, but inspected to
            configure the storage as needed. In some cases, you may not even need the data.

        Returns
        -------
        None, but should set self._bound_target
        """
        raise NotImplementedError("_bind_append function has not been implemented in this subclass yet!")

    def read(self):
        """
        Return the property read from the ncfile

        Returns
        -------
        Given property read from the nc file and cast into the correct Python data type
        """

        if self._bound_target is None:
            self._bind_read()
        return self._decoder(self._bound_target)

    def _common_bind_output_actions(self, type_string, append_mode, store_unit_string='NoneType'):
        """
        Method to handle the common NetCDF variable/group Metadata actions when binding a new variable/group to the
        disk in write/append mode. This code should be called in all the _bind_write and _bind_append blocks inside
        the trapped error when _bind_read fails to find the object (i.e. new variable on disk creation)

        Parameters
        ----------
        type_string : String
            Type of data being stored either as a single object, or the data being stored in the compound object.
            For simple objects like ints and floats, this should just be the typename(self.dtype) and will align
                with the codec's dtype_string
            For compound objects such as lists, tuples, and np.ndarray's, this should be the string of the data stored
                in the object and will be wholly different from the codec's dtype_string and dependent on what is being
                stored in the codec
        append_mode : Integer, 0 or 1
            Integer boolean representation of if this is appended data or not.
            _bind_write methods should pass a 0
            _bind_append methods should pass 1
        store_unit_string : String, optional, Default: 'NoneType'
            String representation of the simtk.unit attached to this data. This string should be able to be fed into
            quantity_from_string(store_unit_string) and return a valid simtk.Unit object. Typically generated from
                str(unit).
            If no unit is assigned to the data, then the default of 'NoneType' should be given.

        """
        if append_mode not in [0, 1]:
            raise ValueError('append_mode must be integer of 0 for _bind write, or 1 for _bind_append')
        self.add_metadata('IODriver_Type', self.dtype_string())
        self.add_metadata('type', type_string)
        self._unit = store_unit_string
        self.add_metadata('IODriver_Unit', self._unit)
        # Specify the type of storage object this should tie to
        self.add_metadata('IODriver_Storage_Type', self.storage_type)
        self.add_metadata('IODriver_Appendable', append_mode)

    def write(self, data, at_index=None):
        """
        Tell this writer how to write to the NetCDF file given the final object that it is bound to

        Alternately, tell a variable which is normally appended to to write a specific entry on the index at_index

        Parameters
        ----------
        data : any data you wish to write
        at_index : None or Int, optional, default=None
            Specify the index of a variable created by append to write specific data at the index entry.
            When None, this option is ignored
            The integer of at_index must be <= to the size of the appended data

        """
        # Check type
        if not isinstance(data, self.dtype):
            raise TypeError("Invalid data type on variable {}.".format(self._target))
        if at_index is not None:
            self._write_to_append_at_index(data, at_index)
            return
        # Bind
        if self._bound_target is None:
            self._bind_write(data)
        self._check_storage_mode('w')
        self._check_data_shape_matching(data)
        # Save data
        packaged_data = self._encoder(data)
        self._bound_target[:] = packaged_data
        return

    def append(self, data):
        """
        Tell this writer how to write to the NetCDF file given the final object that it is bound to

        To overwrite data at a specific index of the already appended data, use the .write(data, at_index=X) method

        Parameters
        ----------
        data :

        """
        # Check type
        if not isinstance(data, self.dtype):
            raise TypeError("Invalid data type on variable {}.".format(self._target))
        # Bind
        if self._bound_target is None:
            self._bind_append(data)
        self._check_storage_mode('a')
        self._check_data_shape_matching(data)
        # Determine current current length and therefore the last index
        length = self._bound_target.shape[0]
        # Save data
        self._bound_target[length, :] = self._encoder(data)

    @abc.abstractmethod
    def _check_data_shape_matching(self, data):
        """
        Check to make sure that the appendable data is the same shape/size/compatible with the other data on the
        appendable data.

        e.g. Lists should be the same length, NumPy arrays should be the same shape and dtype, etc

        For static shape objects such as Ints and Floats, the dtype alone is sufficient and this method can be
        implemented with a simple `pass`

        Parameters
        ----------
        data

        """
        raise NotImplementedError("I don't know how to compare data yet!")

    @abc.abstractproperty
    def storage_type(self):
        """
        Tell the Codec what NetCDF storage type this Codec treats the data as.
        This is explicitly either 'variables' or 'groups' so the driver knows which property to call on the NetCDF
        storage object

        Returns
        -------
        storage_type: string of either 'variables' or 'groups'

        """
        raise NotImplementedError("I have not been set to 'variables' or 'groups'")

    def add_metadata(self, name, value):
        """
        Add metadata to self on disk, extra bits of information that can be used for flags or other variables
        This is NOT a staticmethod of the top data set since you can buffer this before binding

        Parameters
        ----------
        name : string
            Name of the attribute you wish to assign
        value : any, but preferred string
            Extra meta data to add to the variable
        """
        if not self._bound_target:
            self._metadata_buffer[name] = value
        else:
            self._bound_target.setncattr(name, value)

    def _dump_metadata_buffer(self):
        """
        Dump the metadata buffer to file
        """
        if self._bound_target is None:
            raise UnboundLocalError("Cannot dump the metadata buffer to target since no target exists!")
        self._bound_target.setncatts(self._metadata_buffer)
        self._metadata_buffer = {}

    @staticmethod
    def _convert_netcdf_store_type(stored_type):
        """
        Convert the stored NetCDF data type from string to type without relying on unsafe eval() function

        Parameters
        ----------
        stored_type : string
            Read from ncfile.Variable.type

        Returns
        -------
        proper_type : type
            Python or module type

        """
        try:
            # Check if it's a builtin type
            try:  # Python 2
                module = importlib.import_module('__builtin__')
            except ImportError:  # Python 3
                module = importlib.import_module('builtins')
            proper_type = getattr(module, stored_type)
        except AttributeError:
            # if not, separate module and class
            module, stored_type = stored_type.rsplit(".", 1)
            module = importlib.import_module(module)
            proper_type = getattr(module, stored_type)
        return proper_type

    @property
    def _output_mode(self):
        """
        Set the write and append flags. Code should only call this after being bound to a variable

        Returns
        -------
        output_mode : string
            Either 'a' for append or 'w' for write
        """
        if self._bound_target.getncattr('IODriver_Appendable'):
            output_mode = 'a'
        else:
            output_mode = 'w'
        return output_mode

    def _attempt_storage_read(self):
        """
        This is a helper function to try and read the target from the disk then do some validation checks common to
        every _bind_read call. Helps cut down on recoding.

        Returns
        -------
        None, but should try to set _bound_target from disk
        """
        self._bound_target = getattr(self._storage_object, self.storage_type)[self._target]
        # Ensure that the target we bind to matches the type of driver
        try:
            if self._bound_target.getncattr('IODriver_Type') != self.dtype_string():
                raise TypeError("Storage target on NetCDF file is of type {} but this driver is designed to handle "
                                "type {}!".format(self._bound_target.getncattr('IODriver_Type'), self.dtype_string()))
        except AttributeError:
            warnings.warn("This Codec cannot detect storage type from on-disk variable. .write() and .append() "
                          "operations will not work and .read() operations may work", RuntimeWarning)

    def _check_storage_mode(self, expected_mode):
        """
        Check to see if the data stored at this codec is actually compatible with the type of write operation that was
        performed (write vs. append)

        Parameters
        ----------
        expected_mode : string, either "w' or "a"

        Raises
        ------
        TypeError if ._output_mode != expected mode
        """

        # String fill in, uses the opposite of expected mode to raise warnings
        saved_as = {'w': 'appendable', 'a': 'statically written'}
        cannot = {'w': 'write', 'a': 'append'}
        must_use = {'w': 'append() or the to_index keyword of write()', 'a': 'write()'}
        if self._output_mode != expected_mode:
            raise TypeError("{target} at {type} was saved as {saved_as} data! Cannot {cannot}, must use "
                            "{must_use}".format(target=self._target,
                                                type=self.dtype_string(),
                                                saved_as=saved_as[expected_mode],
                                                cannot=cannot[expected_mode],
                                                must_use=must_use[expected_mode])
                            )

    def _write_to_append_at_index(self, data, index):
        """
        Try to write data to a specific site on an append variable. This is a method which should be called in
        every `write` call if the index is defined by something other than None.

        Parameters
        ----------
        data : Data to write to location on a previously appended variable
        index : Int,
            Index to write the data at, replacing what is already there
            If index > size of written data, crash
        """
        if self._bound_target is None:
            try:
                self._bind_read()
            except KeyError:
                # Trap the NetCDF Key Error to raise an issue that data must exist first
                raise IOError("Cannot write to a specific index for data that does not exist!")
        if type(index) is not int:
            raise ValueError("to_index must be an integer!")
        self._check_storage_mode('a')  # We want this in append mode
        self._check_data_shape_matching(data)
        # Determine current current length and therefore if the index is too large
        length = self._bound_target.shape[0]
        # Must actually compare to full length so people don't fill an infinite variable with garbage that is just
        # masked from empty entries
        if index >= length or abs(index) > length:
            raise ValueError("Cannot choose an index beyond the maximum length of the "
                             "appended data of {}".format(length))
        self._bound_target[index, :] = self._encoder(data)


# =============================================================================
# NETCDF NON-COMPOUND TYPE CODECS
# =============================================================================

# Decoders: Convert from NC variable to python type
# Encoders: Decompose Python Type into something NC storable data

def nc_string_decoder(nc_variable):
    if nc_variable.shape == ():
        return str(nc_variable.getValue())
    elif nc_variable.shape == (1,):
        return str(nc_variable[0])
    else:
        return nc_variable[:].astype(str)


def nc_string_encoder(data):
    packed_data = np.empty(1, 'O')
    packed_data[0] = data
    return packed_data


# There really isn't anything that needs to happen here, arrays are the ideal type
# Leaving these as explicit codecs in case we need to change them later
def nc_numpy_array_decoder(nc_variable):
    return nc_variable[:]


# List and tuple iterables, assumes contents are the same type.
# Use dictionaries for compound types
def nc_iterable_decoder(nc_variable):
    shape = nc_variable.shape
    type_name = nc_variable.getncattr('type')
    output_type = NCVariableCodec._convert_netcdf_store_type(type_name)
    if len(shape) == 1:  # Determine if iterable
        output = output_type(nc_variable[:])
    else:  # Handle long form iterable by making an array of iterable type
        output = np.empty(shape[0], dtype=output_type)
        for i in range(shape[0]):
            output[i] = output_type(nc_variable[i])
    return output


# Encoder for float, int, iterable, and numpy arrays
def simple_encoder(data):
    return data


# Works for float and int
def scalar_decoder_generator(casting_type):
    def _scalar_decoder(nc_variable):
        data = nc_variable[:]
        if data.shape == (1,):
            data = casting_type(data[0])
        else:
            data = data.astype(casting_type)
        return data
    return _scalar_decoder


# =============================================================================
# HDF5 CHUNK SIZE ROUTINES
# =============================================================================

def determine_appendable_chunk_size(data, max_iteration=128, max_memory=104857600):
    """
    Determine the chunk size of the appendable dimension, it will either be max_iterations in count or max_memory in
    bytes where the function will try to reduce the number of iterations until it is under the max chunk size down to
    a single iteration.

    Parameters
    ----------
    data : Data that will be saved to disk of shape that will be saved
        This is a sample of what will be written at any one point in time.
    max_iteration : int, Default: 128
        Maximum number of iterations that will be chunked, either this limit or max_memory will be hit first, reducing
        the max iterations by a factor of 2 until we are below the memory limit, to a minimum of 1
    max_memory: int (bytes), Default: 104856700 (100MB)
        Maximum number of bytes the chunk is allowed to have, if the 100 iterations exceeds this size, then we
        reduce the number of iterations by half until we are below the memory limit

    Returns
    -------
    iteration_chunk : int
        Chunksize of the iteration dimension

    """
    if max_iteration < 1 or not isinstance(max_iteration, int):
        raise ValueError("max_iteration was {} but must be an integer greater than 1!".format(max_iteration))
    iteration_chunk = int(max_iteration)
    data_size = getsizeof(data)
    while iteration_chunk * data_size > max_memory and iteration_chunk > 1:
        iteration_chunk /= 2
    # Ceiling and int since np.ceil returns a float
    return int(np.ceil(iteration_chunk))


# =============================================================================
# REAL Codecs
# =============================================================================

# Generic codecs for non-compound data types: inf, float, string

class NCScalar(NCVariableCodec, ABC):

    """"
    This particular class is to minimize code duplication between some very basic data types such as int, str, float

    It is itself an abstract class and requires the following functions to be complete:
    dtype (@property)
    dtype_string (@staticmethod)
    """

    def _bind_write(self, data):
        try:
            self._bind_read()
        except KeyError:
            self._parent_driver.check_scalar_dimension()
            self._bound_target = self._storage_object.createVariable(self._target, self._on_disk_dtype,
                                                                     dimensions='scalar',
                                                                     chunksizes=(1,))
            self._common_bind_output_actions(typename(self.dtype), 0)
        self._dump_metadata_buffer()

    def _bind_append(self, data):
        try:
            self._bind_read()
        except KeyError:
            self._parent_driver.check_scalar_dimension()
            infinite_name = self._parent_driver.generate_infinite_dimension()
            appendable_chunk_size = determine_appendable_chunk_size(data)
            self._bound_target = self._storage_object.createVariable(self._target, self._on_disk_dtype,
                                                                     dimensions=[infinite_name, 'scalar'],
                                                                     chunksizes=(appendable_chunk_size, 1))
            self._common_bind_output_actions(typename(self.dtype), 1)
        self._dump_metadata_buffer()
        return

    def _check_data_shape_matching(self, data):
        pass

    @property
    def storage_type(self):
        return 'variables'

    @property
    def _on_disk_dtype(self):
        """
        Allow overwriting the dtype for storage for extending this method to cast data as a different type on disk
        This is the property to overwrite the cast dtype if it is different than the input/output dtype
        """
        return self.dtype


class NCInt(NCScalar):
    """
    NetCDF codec for Integers
    """

    @property
    def _encoder(self):
        return simple_encoder

    @property
    def _decoder(self):
        return scalar_decoder_generator(int)

    @property
    def dtype(self):
        return int

    @staticmethod
    def dtype_string():
        return "int"


class NCFloat(NCScalar):
    """
    NetCDF codec for Floats
    """

    @property
    def _encoder(self):
        return simple_encoder

    @property
    def _decoder(self):
        return scalar_decoder_generator(float)

    @property
    def dtype(self):
        return float

    @staticmethod
    def dtype_string():
        return "float"


class NCString(NCScalar):
    """
    NetCDF codec for String
    """

    @property
    def _encoder(self):
        return nc_string_encoder

    @property
    def _decoder(self):
        return nc_string_decoder

    @property
    def dtype(self):
        return str

    @staticmethod
    def dtype_string():
        return "str"


# Array

class NCArray(NCVariableCodec):
    """
    NetCDF Codec for numpy arrays
    """

    @property
    def _encoder(self):
        return simple_encoder

    @property
    def _decoder(self):
        return nc_numpy_array_decoder

    @property
    def dtype(self):
        return np.ndarray

    @staticmethod
    def dtype_string():
        return "numpy.ndarray"

    def _bind_write(self, data):
        try:
            self._bind_read()
        except KeyError:
            data_shape, data_base_type, data_type_name = self._determine_data_information(data)
            dims = []
            for length in data_shape:
                self._parent_driver.check_iterable_dimension(length=length)
                dims.append('iterable{}'.format(length))
            self._bound_target = self._storage_object.createVariable(self._target, data_base_type,
                                                                     dimensions=dims,
                                                                     chunksizes=data_shape)
            self._common_bind_output_actions(str(data_base_type), 0)
            self._save_shape = data_shape
        self._dump_metadata_buffer()

    def _bind_append(self, data):
        try:
            self._bind_read()
        except KeyError:
            data_shape, data_base_type, data_type_name = self._determine_data_information(data)
            infinite_name = self._parent_driver.generate_infinite_dimension()
            appendable_chunk_size = determine_appendable_chunk_size(data)
            dims = [infinite_name]
            for length in data_shape:
                self._parent_driver.check_iterable_dimension(length=length)
                dims.append('iterable{}'.format(length))
            self._bound_target = self._storage_object.createVariable(self._target, data_base_type,
                                                                     dimensions=dims,
                                                                     chunksizes=(appendable_chunk_size,) + data_shape)
            self._common_bind_output_actions(str(data_base_type), 1)
            self._save_shape = data_shape
        self._dump_metadata_buffer()

    def _check_data_shape_matching(self, data):
        if self._save_shape != data.shape:
            raise ValueError("Input data must be of shape {} but is instead of shape {}!".format(
                self._save_shape, data.shape)
            )

    @staticmethod
    def _determine_data_information(data):
        # Make common _bind functions a single function
        data_shape = data.shape
        data_base_type = data.dtype
        data_type_name = typename(type(data))
        return data_shape, data_base_type, data_type_name

    @property
    def storage_type(self):
        return 'variables'


class NCIterable(NCVariableCodec):
    """
    NetCDF codec for lists and tuples
    """
    @property
    def dtype(self):
        return collections.Iterable

    @staticmethod
    def dtype_string():
        return "iterable"

    @property
    def _encoder(self):
        return simple_encoder

    @property
    def _decoder(self):
        return nc_iterable_decoder

    def _bind_write(self, data):
        try:
            self._bind_read()
        except KeyError:
            data_shape, data_base_type, data_type_name = self._determine_data_information(data)
            self._parent_driver.check_iterable_dimension(length=data_shape)
            self._bound_target = self._storage_object.createVariable(self._target, data_base_type,
                                                                     dimensions='iterable{}'.format(data_shape),
                                                                     chunksizes=(data_shape,))
            self._common_bind_output_actions(data_type_name, 0)
            self._save_shape = data_shape
        self._dump_metadata_buffer()
        return

    def _bind_append(self, data):
        try:
            self._bind_read()
        except KeyError:
            data_shape, data_base_type, data_type_name = self._determine_data_information(data)
            infinite_name = self._parent_driver.generate_infinite_dimension()
            appendable_chunk_size = determine_appendable_chunk_size(data)
            self._parent_driver.check_iterable_dimension(length=data_shape)
            dims = [infinite_name, 'iterable{}'.format(data_shape)]
            self._bound_target = self._storage_object.createVariable(self._target, data_base_type,
                                                                     dimensions=dims,
                                                                     chunksizes=(appendable_chunk_size, data_shape))
            self._common_bind_output_actions(data_type_name, 1)
            self._save_shape = data_shape
        self._dump_metadata_buffer()
        return

    def _check_data_shape_matching(self, data):
        data_shape = len(data)
        if self._save_shape != data_shape:
            raise ValueError("Input data must be of shape {} but is instead of shape {}!".format(
                self._save_shape, data_shape)
            )

    @staticmethod
    def _determine_data_information(data):
        # Make common _bind functions a single function
        data_type_name = typename(type(data))
        data_base_type = type(data[0])
        data_shape = len(data)
        return data_shape, data_base_type, data_type_name

    @property
    def storage_type(self):
        return 'variables'


class NCQuantity(NCVariableCodec):
    """
    NetCDF codec for ALL simtk.unit.Quantity's
    """
    @property
    def dtype(self):
        return unit.Quantity

    @staticmethod
    def dtype_string():
        return "quantity"

    def _bind_read(self):
        # Method of this subclass as it calls extra data
        super(NCQuantity, self)._bind_read()
        self._unit = self._bound_target.getncattr('IODriver_Unit')
        self._set_codifiers(self._bound_target.getncattr('type'))

    def _bind_write(self, data):
        try:
            self._bind_read()
        except KeyError:
            data_shape, data_base_type, data_type_name = self._determine_data_information(data)
            if data_shape == 1:  # Single dimension quantity
                self._parent_driver.check_scalar_dimension()
                self._bound_target = self._storage_object.createVariable(self._target, data_base_type,
                                                                         dimensions='scalar',
                                                                         chunksizes=(1,))
            else:
                dims = []
                for length in data_shape:
                    self._parent_driver.check_iterable_dimension(length=length)
                    dims.append('iterable{}'.format(length))
                self._bound_target = self._storage_object.createVariable(self._target, data_base_type,
                                                                         dimensions=dims,
                                                                         chunksizes=data_shape)

            self._common_bind_output_actions(data_type_name, 0, store_unit_string=str(data.unit))
            self._save_shape = data_shape
            self._set_codifiers(data_type_name)
        self._dump_metadata_buffer()
        return

    def _bind_append(self, data):
        try:
            self._bind_read()
        except KeyError:
            data_shape, data_base_type, data_type_name = self._determine_data_information(data)
            appendable_chunk_size = determine_appendable_chunk_size(data)
            infinite_name = self._parent_driver.generate_infinite_dimension()
            if data_shape == 1:  # Single dimension quantity
                self._parent_driver.check_scalar_dimension()
                self._bound_target = self._storage_object.createVariable(self._target, data_base_type,
                                                                         dimensions=[infinite_name, 'scalar'],
                                                                         chunksizes=(appendable_chunk_size, 1))
            else:
                dims = [infinite_name]
                for length in data_shape:
                    self._parent_driver.check_iterable_dimension(length=length)
                    dims.append('iterable{}'.format(length))
                self._bound_target = self._storage_object.createVariable(self._target, data_base_type,
                                                                         dimensions=dims,
                                                                         chunksizes=(appendable_chunk_size,) + data_shape)
            self._common_bind_output_actions(data_type_name, 1, store_unit_string=str(data.unit))
            self._save_shape = data_shape
            self._set_codifiers(data_type_name)
        self._dump_metadata_buffer()
        return

    def _check_data_shape_matching(self, data):
        if self._save_shape != self._compare_shape(data):
            raise ValueError("Input data must be of shape {} but is instead of shape {}!".format(
                self._save_shape, self._compare_shape(data))
            )
        if self._unit != str(data.unit):
            raise ValueError("Input data must have units of {}, but instead is {}".format(self._unit,
                                                                                          str(data.unit)))

    def _determine_data_information(self, data):
        # Make common _bind functions a single function
        data_unit = data.unit
        data_value = data / data_unit
        data_type_name = typename(type(data_value))
        try:
            data_shape = data_value.shape
            data_base_type = type(data_value.flatten()[0])
            self._compare_shape = lambda x: x.shape
        except AttributeError:  # Trap not array
            try:
                data_shape = (len(data_value),)
                data_base_type = type(data_value[0])
                self._compare_shape = lambda x: (len(x),)
            except TypeError:  # Trap not iterable
                data_shape = 1
                data_base_type = type(data_value)
                self._compare_shape = lambda x: 1
        return data_shape, data_base_type, data_type_name

    def _set_codifiers(self, stype):
        # Assign the codecs in a single block
        if stype == 'int':
            self._value_encoder = simple_encoder
            self._value_decoder = scalar_decoder_generator(int)
        elif stype == 'float':
            self._value_encoder = simple_encoder
            self._value_decoder = scalar_decoder_generator(float)
        elif stype == 'list' or stype == 'tuple':
            self._value_encoder = simple_encoder
            self._value_decoder = nc_iterable_decoder
        elif 'ndarray' in stype:
            self._value_encoder = simple_encoder
            self._value_decoder = nc_numpy_array_decoder
        else:
            raise TypeError("NCQuantity does not know how to handle a quantity of type {}!".format(stype))

    @property
    def _encoder(self):
        return self._quantity_encoder

    @property
    def _decoder(self):
        return self._quantity_decoder

    def _quantity_encoder(self, data):
        # Strip Unit
        data_unit = data.unit
        data_value = data / data_unit
        return self._value_encoder(data_value)

    def _quantity_decoder(self, bound_target):
        data = self._value_decoder(bound_target)
        unit_name = bound_target.getncattr('IODriver_Unit')
        cast_unit = quantity_from_string(unit_name)
        if isinstance(cast_unit, unit.Quantity):
            cast_unit = cast_unit.unit
        return data * cast_unit

    @property
    def storage_type(self):
        return 'variables'


# =============================================================================
# NETCDF DICT YAML HANDLERS
# =============================================================================

class _DictYamlLoader(Loader):
    """PyYAML Loader that recognized !Quantity nodes, converts YAML output -> Python type"""
    def __init__(self, *args, **kwargs):
        super(_DictYamlLoader, self).__init__(*args, **kwargs)
        self.add_constructor(u'!Quantity', self.quantity_constructor)

    @staticmethod
    def quantity_constructor(loader, node):
        loaded_mapping = loader.construct_mapping(node)
        data_unit = quantity_from_string(loaded_mapping['QuantityUnit'])
        data_value = loaded_mapping['QuantityValue']
        return data_value * data_unit


class _DictYamlDumper(Dumper):
    """PyYAML Dumper that convert from Python -> YAML output"""
    def __init__(self, *args, **kwargs):
        super(_DictYamlDumper, self).__init__(*args, **kwargs)
        self.add_representer(unit.Quantity, self.quantity_representer)

    @staticmethod
    def quantity_representer(dumper, data):
        """YAML Quantity representer."""
        data_unit = data.unit
        data_value = data / data_unit
        data_dump = {'QuantityUnit': str(data_unit), 'QuantityValue': data_value}
        # Uses "self (DictYamlDumper)" as the dumper to allow nested !Quantity types
        return dumper.represent_mapping(u'!Quantity', data_dump)


class NCDict(NCScalar):
    """
    NetCDF codec for Dict, which we store in YAML as a glorified String with some extra processing
    """

    @staticmethod
    def _nc_dict_decoder(nc_variable):
        decoded_string = nc_string_decoder(nc_variable)
        # Handle array type
        try:
            output = yaml.load(decoded_string, Loader=_DictYamlLoader)
        except (AttributeError, TypeError):  # Appended data
            n_entries = decoded_string.shape[0]
            output = np.empty(n_entries, dtype=dict)
            for n in range(n_entries):
                output[n] = yaml.load(str(decoded_string[n, 0]), Loader=_DictYamlLoader)
        return output

    @staticmethod
    def _nc_dict_encoder(data):
        dump_options = {'Dumper': _DictYamlDumper, 'line_break': '\n', 'indent': 4}
        data_as_string = yaml.dump(data, **dump_options)
        packaged_string = nc_string_encoder(data_as_string)
        return packaged_string

    @property
    def _encoder(self):
        return self._nc_dict_encoder

    @property
    def _decoder(self):
        return self._nc_dict_decoder

    @property
    def dtype(self):
        return dict

    @staticmethod
    def dtype_string():
        return "dict"

    @property
    def _on_disk_dtype(self):
        return str
