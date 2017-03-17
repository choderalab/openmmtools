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

from simtk import unit

from ..utils import typename, quantity_from_string

# TODO: Use the `with_metaclass` from yank.utils when we merge it in
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
    """
    def __init__(self, file_name, access_mode=None):
        """

        Parameters
        ----------
        file_name : string
            Name of the file to read/write to of a given storage type
        access_mode : string or None, Default None, accepts 'w', 'r', 'a'
            Define how to access the file in either write, read, or append mode
            None should behave like Python "a+" in which a file is created if not present, or opened in append if it is.
            How this is implemented is up to the subclass
        """
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
        codec : Specific codifer class
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
    def close_down(self):
        """
        Instruct how to safely close down the file.

        """
        raise NotImplementedError("close_down method has not been implemented!")

    @abc.abstractmethod
    def add_metadata(self, name, value, path=''):
        """
        Function to add metadata to the file. This can be treated as optional and can simply be a `pass` if you do not
        want your storage system to handle additional metadata

        Parameters
        ----------
        name : string
            Name of the attribute you wish to assign
        value : any, but prefered string
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

    def close_down(self):
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
        super_codec = super(NetCDFIODriver, self).set_codec  # Shortcut for this init to avoid exces loops
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


class NCVariableCodec(ABC):
    """
    Pointer class which provides instructions on how to handle a given nc_variable
    """
    def __init__(self, parent_driver, target, storage_object=None):
        """
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
        self._target = target
        # Eventual NetCDF object this class will be bound to
        self._bound_target = None
        # Target of the top level driver which houses all the variables
        self._parent_driver = parent_driver
        # Target object where the data read/written to this instance resides
        # Similar to the "directory" in a file system
        if storage_object is None:
            storage_object = self._parent_driver.ncfile
        self._storage_object = storage_object
        # Buffer to store metadata if assigned before binding
        self._metadata_buffer = {}

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
    def dtype_string(self):
        """
        Short name of variable for strings and errors

        Returns
        -------
        string

        """
        # TODO: Replace with @abstractstaticmethod when on Python 3
        raise NotImplementedError("dtype_string has not been implemented in this subclass yet!")

    @abc.abstractmethod
    def _bind_read(self):
        """
        A one time event that binds this class to the object on disk. This method should set self._bound_target
        This function is unique to the read() function in that no data is attempted to write to the disk.
        Should raise error if the object is not found on disk (i.e. no data has been written to this location yet)
        Should raise error if the object on disk is incompatible with this type of Codec.

        Returns
        -------
        None, but should set self._bound_target
        """
        raise NotImplementedError("_bind_read function has not been implemented in this subclass yet!")

    @abc.abstractmethod
    def _bind_write(self, data):
        """
        A one time event that binds this class to the object on disk. This method should set self._bound_target
        This function is unique to the write() function in that the data passed in should help create the storage object
        if not already on disk and prepare it for a write operation

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
        the location, or should create the object, then write the data with the first dimension infinite in size

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

    @abc.abstractmethod
    def read(self):
        """
        Return the property read from the ncfile

        Returns
        -------
        Given property read from the nc file and cast into the correct Python data type
        """

        raise NotImplementedError("Extracting stored NetCDF data into Python data has not been implemented!")

    @abc.abstractmethod
    def write(self, data):
        """
        Tell this writer how to write to the NetCDF file given the final object that it is bound to

        Parameters
        ----------
        data

        Returns
        -------

        """
        raise NotImplementedError("Writing Python data to NetCDF data has not been implemented!")

    @abc.abstractmethod
    def append(self, data):
        """
        Tell this writer how to write to the NetCDF file given the final object that it is bound to

        Parameters
        ----------
        data

        Returns
        -------

        """
        raise NotImplementedError("Appending Python data to NetCDF data has not been implemented!")

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
        This is NOT a staticmethod of the top dataset since you can buffer this before binding

        Parameters
        ----------
        name : string
            Name of the attribute you wish to assign
        value : any, but prefered string
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
    _encoder (@property)
    _decoder (@property)
    dtype (@property)
    dtype_string (@staticmethod)
    """

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
        self._attempt_storage_read()
        # Handle variable size objects
        # This line will not happen unless target is real, so output mode should return correct value
        if self._output_mode is 'a':
            self._save_shape = self._bound_target.shape[1:]
        else:
            self._save_shape = self._bound_target.shape

    def _bind_write(self, data):
        try:
            self._bind_read()
        except KeyError:
            self._parent_driver.check_scalar_dimension()
            self._bound_target = self._storage_object.createVariable(self._target, self._on_disk_dtype,
                                                                     dimensions='scalar',
                                                                     chunksizes=(1,))
            # Specify a way for the IO Driver stores data
            self.add_metadata('IODriver_Type', self.dtype_string())
            self.add_metadata('type', typename(self.dtype))
            self._unit = 'NoneType'
            self.add_metadata('IODriver_Unit', self._unit)
            # Specify the type of storage object this should tie to
            self.add_metadata('IODriver_Storage_Type', self.storage_type)
            self.add_metadata('IODriver_Appendable', 0)
        self._dump_metadata_buffer()
        # Set the output mode by calling the variable
        self._output_mode

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
            # Specify a way for the IO Driver stores data
            self.add_metadata('IODriver_Type', self.dtype_string())
            self.add_metadata('type', typename(self.dtype))
            self._unit = 'NoneType'
            self.add_metadata('IODriver_Unit', self._unit)
            # Specify the type of storage object this should tie to
            self.add_metadata('IODriver_Storage_Type', self.storage_type)
            self.add_metadata('IODriver_Appendable', 1)
        self._dump_metadata_buffer()
        # Set the output mode by calling the variable
        self._output_mode
        return

    def read(self):
        if self._bound_target is None:
            self._bind_read()
        # Set the output mode by calling the variable
        self._output_mode
        return self._decoder(self._bound_target)

    def write(self, data):
        # Check type
        if type(data) is not self.dtype:
            raise TypeError("Invalid data type on variable {}.".format(self._target))
        # Bind
        if self._bound_target is None:
            self._bind_write(data)
        # Check writeable
        if self._output_mode != 'w':
            raise TypeError("{} at {} was saved as appendable data! Cannot overwrite, must use append()".format(
                self.dtype_string(), self._target)
            )
        # Save data
        self._bound_target[:] = self._encoder(data)
        return

    def append(self, data):
        # Check type
        if type(data) is not self.dtype:
            raise TypeError("Invalid data type on variable {}.".format(self._target))
        # Bind
        if self._bound_target is None:
            self._bind_append(data)
        # Check writeable
        if self._output_mode != 'a':
            raise TypeError("{} at {} was saved as appendable data! Cannot overwrite, must use append()".format(
                self.dtype_string(), self._target)
            )
        # Determine current current length and therefore the last index
        length = self._bound_target.shape[0]
        # Save data
        self._bound_target[length, :] = self._encoder(data)

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
    def dtype(self):
        return np.ndarray

    @staticmethod
    def dtype_string():
        return "numpy.ndarray"

    def _bind_read(self):
        self._attempt_storage_read()
        # Handle variable size objects
        # This line will not happen unless target is real, so output_mode should return correct value
        if self._output_mode is 'a':
            self._save_shape = self._bound_target.shape[1:]
        else:
            self._save_shape = self._bound_target.shape

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
            # Specify a way for the IO Driver stores data
            self.add_metadata('IODriver_Type', self.dtype_string())
            self.add_metadata('type', str(data_base_type))
            self._unit = 'NoneType'
            self.add_metadata('IODriver_Unit', self._unit)
            # Specify the type of storage object this should tie to
            self.add_metadata('IODriver_Storage_Type', self.storage_type)
            self.add_metadata('IODriver_Appendable', 0)
            self._save_shape = data_shape
        self._dump_metadata_buffer()
        # Set the output mode by calling the variable
        self._output_mode

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
            # Specify a way for the IO Driver stores data
            self.add_metadata('IODriver_Type', self.dtype_string())
            self.add_metadata('type', str(data_base_type))
            self._unit = 'NoneType'
            self.add_metadata('IODriver_Unit', self._unit)
            # Specify the type of storage object this should tie to
            self.add_metadata('IODriver_Storage_Type', self.storage_type)
            self.add_metadata('IODriver_Appendable', 1)
            self._save_shape = data_shape
        self._dump_metadata_buffer()
        # Set the output mode by calling the variable
        self._output_mode

    def read(self):
        if self._bound_target is None:
            self._bind_read()
        # Set the output mode by calling the variable
        self._output_mode
        return nc_numpy_array_decoder(self._bound_target)

    def write(self, data):
        # Check type
        if type(data) is not self.dtype:
            raise TypeError("Invalid data type on variable {}.".format(self._target))
        # Bind
        if self._bound_target is None:
            self._bind_write(data)
        # Check writeable
        if self._output_mode != 'w':
            raise TypeError("{} at {} was saved as appendable data! Cannot overwrite, must use append()".format(
                self.dtype_string(), self._target)
            )
        if self._save_shape != data.shape:
            raise ValueError("Input data must be of shape {} but is instead of shape {}!".format(
                self._compare_shape(data), self._save_shape)
            )
        # Save data
        packaged_data = simple_encoder(data)
        self._bound_target[:] = packaged_data
        return

    def append(self, data):
        # Check type
        if type(data) is not self.dtype:
            raise TypeError("Invalid data type on variable {}.".format(self._target))
        # Bind
        if self._bound_target is None:
            self._bind_append(data)
        # Check writeable
        if self._output_mode != 'a':
            raise TypeError("{} at {} was saved as appendable data! Cannot overwrite, must use append()".format(
                self.dtype_string(), self._target)
            )
        if self._save_shape != data.shape:
            raise ValueError("Input data must be of shape {} but is instead of shape {}!".format(
                self._compare_shape(data), self._save_shape)
            )
        # Save data
        packaged_data = simple_encoder(data)
        # Determine current current length and therefore the last index
        length = self._bound_target.shape[0]
        self._bound_target[length, :] = packaged_data

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

    def _bind_read(self):
        self._attempt_storage_read()
        # Handle variable size objects
        # This line will not happen unless target is real, so output_mode should return the correct value
        if self._output_mode is 'a':
            self._save_shape = self._bound_target.shape[1:]
        else:
            self._save_shape = self._bound_target.shape

    def _bind_write(self, data):
        try:
            self._bind_read()
        except KeyError:
            data_shape, data_base_type, data_type_name = self._determine_data_information(data)
            self._parent_driver.check_iterable_dimension(length=data_shape)
            self._bound_target = self._storage_object.createVariable(self._target, data_base_type,
                                                                     dimensions='iterable{}'.format(data_shape),
                                                                     chunksizes=(data_shape,))
            # Specify a way for the IO Driver stores data
            self.add_metadata('IODriver_Type', self.dtype_string())
            self.add_metadata('type', data_type_name)
            self._unit = "NoneType"
            self.add_metadata('IODriver_Unit', self._unit)
            # Specify the type of storage object this should tie to
            self.add_metadata('IODriver_Storage_Type', self.storage_type)
            self.add_metadata('IODriver_Appendable', 0)
            self._save_shape = data_shape
        self._dump_metadata_buffer()
        # Set the output mode by calling the variable
        self._output_mode
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
            # Specify a way for the IO Driver stores data
            self.add_metadata('IODriver_Type', self.dtype_string())
            self.add_metadata('type', data_type_name)
            self._unit = "NoneType"
            self.add_metadata('IODriver_Unit', self._unit)
            # Specify the type of storage object this should tie to
            self.add_metadata('IODriver_Storage_Type', self.storage_type)
            self.add_metadata('IODriver_Appendable', 1)
            self._save_shape = data_shape
        self._dump_metadata_buffer()
        # Set the output mode by calling the variable
        self._output_mode
        return

    def read(self):
        if self._bound_target is None:
            self._bind_read()
        # Set the output mode by calling the variable
        self._output_mode
        return nc_iterable_decoder(self._bound_target)

    def write(self, data):
        # Check type
        if not isinstance(data, self.dtype):
            raise TypeError("Invalid data type on variable {}.".format(self._target))
        # Bind
        if self._bound_target is None:
            self._bind_write(data)
        # Check writeable
        if self._output_mode != 'w':
            raise TypeError("{} at {} was saved as appendable data! Cannot overwrite, must use append()".format(
                self.dtype_string(), self._target)
            )
        if self._save_shape != len(data):
            raise ValueError("Input data must be of shape {} but is instead of shape {}!".format(
                self._compare_shape(data), self._save_shape)
            )
        # Save data
        packaged_data = simple_encoder(data)
        self._bound_target[:] = packaged_data
        return

    def append(self, data):
        # Check type
        if not isinstance(data, self.dtype):
            raise TypeError("Invalid data type on variable {}.".format(self._target))
        # Bind
        if self._bound_target is None:
            self._bind_append(data)
        # Check writeable
        if self._output_mode != 'a':
            raise TypeError("{} at {} was saved as appendable data! Cannot overwrite, must use append()".format(
                self.dtype_string(), self._target)
            )
        if self._save_shape != len(data):
            raise ValueError("Input data must be of shape {} but is instead of shape {}!".format(
                self._compare_shape(data), self._save_shape)
            )
        # Save data
        packaged_data = simple_encoder(data)
        # Determine current current length and therefore the last index
        length = self._bound_target.shape[0]
        self._bound_target[length, :] = packaged_data

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
        self._attempt_storage_read()
        # Handle variable size objects
        # This line will not happen unless target is real, so output_mode will return the correct value
        if self._output_mode is 'a':
            self._save_shape = self._bound_target.shape[1:]
        else:
            self._save_shape = self._bound_target.shape
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

            # Specify a way for the IO Driver stores data
            self.add_metadata('IODriver_Type', self.dtype_string())
            self.add_metadata('type', data_type_name)
            self._unit = str(data.unit)
            self.add_metadata('IODriver_Unit', self._unit)
            # Specify the type of storage object this should tie to
            self.add_metadata('IODriver_Storage_Type', self.storage_type)
            self.add_metadata('IODriver_Appendable', 0)
            self._save_shape = data_shape
            self._set_codifiers(data_type_name)
        self._dump_metadata_buffer()
        # Set the output mode by calling the variable
        self._output_mode
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
            # Specify a way for the IO Driver stores data
            self.add_metadata('IODriver_Type', self.dtype_string())
            self.add_metadata('type', data_type_name)
            self._unit = str(data.unit)
            self.add_metadata('IODriver_Unit', self._unit)
            # Specify the type of storage object this should tie to
            self.add_metadata('IODriver_Storage_Type', self.storage_type)
            self.add_metadata('IODriver_Appendable', 1)
            self._save_shape = data_shape
            self._set_codifiers(data_type_name)
        self._dump_metadata_buffer()
        # Set the output mode by calling the variable
        self._output_mode
        return

    def read(self):
        if self._bound_target is None:
            self._bind_read()
        # Set the output mode by calling the variable
        self._output_mode
        data = self._decoder(self._bound_target)
        unit_name = self._bound_target.getncattr('IODriver_Unit')
        cast_unit = quantity_from_string(unit_name)
        if isinstance(cast_unit, unit.Quantity):
            cast_unit = cast_unit.unit
        return data * cast_unit

    def write(self, data):
        # Check type
        if type(data) is not self.dtype:
            raise TypeError("Invalid data type on variable {}.".format(self._target))
        # Bind
        if self._bound_target is None:
            self._bind_write(data)
        # Check writeable
        if self._output_mode != 'w':
            raise TypeError("{} at {} was saved as appendable data! Cannot overwrite, must use append()".format(
                self.dtype_string(), self._target)
            )
        if self._save_shape != self._compare_shape(data):
            raise ValueError("Input data must be of shape {} but is instead of shape {}!".format(
                self._compare_shape(data), self._save_shape)
            )
        if self._unit != str(data.unit):
            raise ValueError("Input data must have units of {}, but instead is {}".format(self._unit,
                                                                                          str(data.unit)))
        # Save data
        # Strip Unit
        data_unit = data.unit
        data_value = data / data_unit
        packaged_data = self._encoder(data_value)
        self._bound_target[:] = packaged_data
        return

    def append(self, data):
        # Check type
        if type(data) is not self.dtype:
            raise TypeError("Invalid data type on variable {}.".format(self._target))
        # Bind
        if self._bound_target is None:
            self._bind_append(data)
        # Check writeable
        if self._output_mode != 'a':
            raise TypeError("{} at {} was saved as appendable data! Cannot overwrite, must use append()".format(
                self.dtype_string(), self._target)
            )
        if self._save_shape != self._compare_shape(data):
            raise ValueError("Input data must be of shape {} but is instead of shape {}!".format(
                self._compare_shape(data), self._save_shape)
            )
        if self._unit != str(data.unit):
            raise ValueError("Input data must have units of {}, but instead is {}".format(self._unit,
                                                                                          str(data.unit)))
        # Determine current current length and therefore the last index
        length = self._bound_target.shape[0]
        # Save data
        # Strip Unit
        data_unit = data.unit
        data_value = data / data_unit
        packaged_data = self._encoder(data_value)
        self._bound_target[length, :] = packaged_data

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
            self._encoder = simple_encoder
            self._decoder = scalar_decoder_generator(int)
        elif stype == 'float':
            self._encoder = simple_encoder
            self._decoder = scalar_decoder_generator(float)
        elif stype == 'list' or stype == 'tuple':
            self._encoder = simple_encoder
            self._decoder = nc_iterable_decoder
        elif 'ndarray' in stype:
            self._encoder = simple_encoder
            self._decoder = nc_numpy_array_decoder
        else:
            raise TypeError("NCQuantity does not know how to handle a quantity of type {}!".format(stype))

    @property
    def storage_type(self):
        return 'variables'


# =============================================================================
# NETCDF DICT YAML HANDLERS
# =============================================================================

class DictYamlLoader(yaml.Loader):
    """PyYAML Loader that recognized !Quantity nodes, converts YAML output -> Python type"""
    def __init__(self, *args, **kwargs):
        super(DictYamlLoader, self).__init__(*args, **kwargs)
        self.add_constructor(u'!Quantity', self.quantity_constructor)

    @staticmethod
    def quantity_constructor(loader, node):
        loaded_mapping = loader.construct_mapping(node)
        data_unit = quantity_from_string(loaded_mapping['QuantityUnit'])
        data_value = loaded_mapping['QuantityValue']
        return data_value * data_unit


class DictYamlDumper(yaml.Dumper):
    """PyYAML Dumper that convert from Python -> YAML output"""
    def __init__(self, *args, **kwargs):
        super(DictYamlDumper, self).__init__(*args, **kwargs)
        self.add_representer(unit.Quantity, self.quantity_representer)

    @staticmethod
    def quantity_representer(dumper, data):
        """YAML Quantity representer."""
        data_unit = data.unit
        data_value = data / data_unit
        data_dump = {'QuantityUnit': str(data_unit), 'QuantityValue': data_value}
        # Uses "self (DictYamlDumper)" as the dumper to allow nested !Quantitity types
        return yaml.Dumper.represent_mapping(dumper, u'!Quantity', data_dump)


class NCDict(NCScalar):
    """
    NetCDF codec for Dict, which we store in YAML as a glorified String with some extra processing
    """

    @staticmethod
    def _nc_dict_decoder(nc_variable):
        decoded_string = nc_string_decoder(nc_variable)
        # Handle array type
        try:
            output = yaml.load(decoded_string, Loader=DictYamlLoader)
        except AttributeError:  # Appended data
            n_entries = decoded_string.shape[0]
            output = np.empty(n_entries, dtype=dict)
            for n in range(n_entries):
                output[n] = yaml.load(decoded_string[n, 0], Loader=DictYamlLoader)
        return output

    @staticmethod
    def _nc_dict_encoder(data):
        dump_options = {'Dumper': DictYamlDumper, 'line_break': '\n', 'indent': 4}
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
