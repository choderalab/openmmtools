#!/usr/local/bin/env python

"""
Test storageinterface.py facility.

The tests are written around the netcdf storage handler for its asserts (its default)
Testing the storage handlers themselves should be left to the test_storage_iodrivers.py file

"""

# =============================================================================================
# GLOBAL IMPORTS
# =============================================================================================

import numpy as np
from simtk import unit
import contextlib
import tempfile
import shutil

from nose import tools

from openmmtools.storage import StorageInterface, NetCDFIODriver


# =============================================================================================
# TEST HELPER FUNCTIONS
# =============================================================================================

def spawn_driver(path):
    """Create a driver that is used to test the StorageInterface class at path location"""
    return NetCDFIODriver(path)


@contextlib.contextmanager
def temporary_directory():
    """Context for safe creation of temporary directories."""
    tmp_dir = tempfile.mkdtemp()
    try:
        yield tmp_dir
    finally:
        shutil.rmtree(tmp_dir)

# =============================================================================================
# STORAGE INTERFACE TESTING FUNCTIONS
# =============================================================================================


def test_storage_interface_creation():
    """Test that the storage interface can create a top level file and read from it"""
    with temporary_directory() as tmp_dir:
        test_store = tmp_dir + '/teststore.nc'
        driver = spawn_driver(test_store)
        si = StorageInterface(driver)
        si.add_metadata('name', 'data')
        assert si.storage_driver.ncfile.getncattr('name') == 'data'


@tools.raises(Exception)
def test_read_trap():
    """Test that attempting to read a non-existent file fails"""
    with temporary_directory() as tmp_dir:
        test_store = tmp_dir + '/teststore.nc'
        driver = spawn_driver(test_store)
        si = StorageInterface(driver)
        si.var1.read()


def test_variable_write_read():
    """Test that a variable can be create and written to file"""
    with temporary_directory() as tmp_dir:
        test_store = tmp_dir + '/teststore.nc'
        driver = spawn_driver(test_store)
        si = StorageInterface(driver)
        input_data = 4
        si.four.write(input_data)
        output_data = si.four.read()
        assert output_data == input_data


def test_variable_append_read():
    """Test that a variable can be create and written to file"""
    with temporary_directory() as tmp_dir:
        test_store = tmp_dir + '/teststore.nc'
        driver = spawn_driver(test_store)
        si = StorageInterface(driver)
        input_data = np.eye(3) * 4.0
        si.four.append(input_data)
        si.four.append(input_data)
        output_data = si.four.read()
        assert np.all(output_data[0] == input_data)
        assert np.all(output_data[1] == input_data)


def test_at_index_write():
    """Test that writing at a specific index of appended data works"""
    with temporary_directory() as tmp_dir:
        test_store = tmp_dir + '/teststore.nc'
        driver = spawn_driver(test_store)
        si = StorageInterface(driver)
        input_data = 4
        overwrite_data = 5
        for i in range(3):
            si.four.append(input_data)
        si.four.write(overwrite_data, at_index=1)  # Sacrilege, I know -LNN
        output_data = si.four.read()
        assert np.all(output_data[0] == input_data)
        assert np.all(output_data[2] == input_data)
        assert np.all(output_data[1] == overwrite_data)


def test_unbound_read():
    """Test that a variable can read from the file without previous binding"""
    with temporary_directory() as tmp_dir:
        test_store = tmp_dir + '/teststore.nc'
        driver = spawn_driver(test_store)
        si = StorageInterface(driver)
        input_data = 4*unit.kelvin
        si.four.write(input_data)
        si.storage_driver.close()
        del si
        driver = spawn_driver(test_store)
        si = StorageInterface(driver)
        output_data = si.four.read()
        assert input_data == output_data


def test_directory_creation():
    """Test that automatic directory-like objects are created on the fly"""
    with temporary_directory() as tmp_dir:
        test_store = tmp_dir + '/teststore.nc'
        driver = spawn_driver(test_store)
        si = StorageInterface(driver)
        input_data = 'four'
        si.dir0.dir1.dir2.var.write(input_data)
        ncfile = si.storage_driver.ncfile
        target = ncfile
        for i in range(3):
            my_dir = 'dir{}'.format(i)
            assert my_dir in target.groups
            target = target.groups[my_dir]
        si.storage_driver.close()
        del si
        driver = spawn_driver(test_store)
        si = StorageInterface(driver)
        target = si
        for i in range(3):
            my_dir = 'dir{}'.format(i)
            target = getattr(target, my_dir)
        assert target.var.read() == input_data


def test_multi_variable_creation():
    """Test that multiple variables can be created in a single directory structure"""
    with temporary_directory() as tmp_dir:
        test_store = tmp_dir + '/teststore.nc'
        driver = spawn_driver(test_store)
        si = StorageInterface(driver)
        input_data = [4.0, 4.0, 4.0]
        si.dir0.var0.write(input_data)
        si.dir0.var1.append(input_data)
        si.dir0.var1.append(input_data)
        si.storage_driver.close()
        del si, driver
        driver = spawn_driver(test_store)
        si = StorageInterface(driver)
        assert si.dir0.var0.read() == input_data
        app_data = si.dir0.var1.read()
        assert app_data[0] == input_data
        assert app_data[1] == input_data


def test_metadata_creation():
    """Test that metadata can be added to variables and directories"""
    with temporary_directory() as tmp_dir:
        test_store = tmp_dir + '/teststore.nc'
        driver = spawn_driver(test_store)
        si = StorageInterface(driver)
        input_data = 4
        si.dir0.var1.write(input_data)
        si.dir0.add_metadata('AmIAGroup', 'yes')
        si.dir0.var1.add_metadata('AmIAGroup', 'no')
        dir0 = si.storage_driver.ncfile.groups['dir0']
        var1 = dir0.variables['var1']
        assert dir0.getncattr('AmIAGroup') == 'yes'
        assert var1.getncattr('AmIAGroup') == 'no'
