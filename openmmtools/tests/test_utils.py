#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test utility functions in utils.py.

"""
import abc
import copy

# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import nose
from nose.tools import nottest
import numpy as np

try:
    import openmm
    from openmm import unit
except ImportError:  # OpenMM < 7.6
    from simtk import openmm, unit

from openmmtools.utils import _RESERVED_WORDS_PATTERNS
from openmmtools.utils import *


# =============================================================================
# TEST CONTEXT UTILITIES
# =============================================================================

def test_platform_supports_precision():
    """Test that platform_supports_precision works correctly."""

    for platform_index in range(openmm.Platform.getNumPlatforms()):
        platform = openmm.Platform.getPlatform(platform_index)
        platform_name = platform.getName()
        supported_precisions = { precision for precision in ['single', 'mixed', 'double'] if platform_supports_precision(platform, precision) }
        if platform_name == 'Reference':
            if supported_precisions != {'double'}:
                raise Exception(f"'Reference' platform should only support 'double' precision, but platform_supports_precision reports {supported_precisions}")
        if platform_name == 'CUDA':
            if supported_precisions != {'single', 'mixed', 'double'}:
                raise Exception(f"'CUDA' platform should support 'mixed' precision, but platform_supports_precision reports {supported_precisions}")
        if platform_name == 'CPU':
            if supported_precisions != {'mixed'}:
                raise Exception(f"'CPU' platform should support 'mixed' precision, but platform_supports_precision reports {supported_precisions}")


def test_string_platform_supports_precision():
    """Test that if we use a string for the platform name, it works"""
    assert platform_supports_precision("CPU", "mixed")

# =============================================================================
# TEST STRING MATHEMATICAL EXPRESSION PARSING UTILITIES
# =============================================================================

def test_sanitize_expression():
    """Test that reserved keywords are substituted correctly."""
    prefix = '_sanitized__'

    # Generate a bunch of test cases for each supported reserved keyword.
    test_cases = {}
    for word in _RESERVED_WORDS_PATTERNS:
        s_word = prefix + word  # sanitized word
        test_cases[word] = [(word, s_word),
                            ('(' + word + ')', '(' + s_word + ')'),  # parenthesis
                            ('( ' + word + ' )', '( ' + s_word + ' )'),  # parenthesis w/ spaces
                            (word + '_suffix', word + '_suffix'),  # w/ suffix
                            ('prefix_' + word, 'prefix_' + word),  # w/ prefix
                            ('2+' + word + '-' + word + '_suffix/(' + word + ' - 3)',  # expression
                             '2+' + s_word + '-' + word + '_suffix/(' + s_word + ' - 3)')]

    # Run test cases.
    for word in _RESERVED_WORDS_PATTERNS:
        variables = {word: 5.0}
        for expression, result in test_cases[word]:
            sanitized_expression, sanitized_variables = sanitize_expression(expression, variables)
            assert sanitized_expression == result, f'{sanitized_expression}, {result}'
            assert word not in sanitized_variables
            assert sanitized_variables[prefix + word] == 5.0


def test_math_eval():
    """Test math_eval method."""
    test_cases = [('1 + 3', None, 4),
                  ('x + y', {'x': 1.5, 'y': 2}, 3.5),
                  ('(x + lambda) / z * 4', {'x': 1, 'lambda': 2, 'z': 3}, 4.0),
                  ('-((x + y) / z * 4)**2', {'x': 1, 'y': 2, 'z': 3}, -16.0),
                  ('ceil(0.8) + acos(x) + step(0.5 - x) + step(0.5)', {'x': 1}, 2),
                  ('step_hm(x)', {'x': 0}, 0.5),
                  ('myset & myset2', {'myset': {1,2,3}, 'myset2': {2,3,4}}, {2, 3}),
                  ('myset or myset2', {'myset': {1,2,3}, 'myset2': {2,3,4}}, {1, 2, 3, 4}),
                  ('(myset or my2set) & myset3', {'myset': {1, 2}, 'my2set': {3, 4}, 'myset3': {2, 3}}, {2, 3})]
    for expression, variables, result in test_cases:
        evaluated_expression = math_eval(expression, variables)
        assert evaluated_expression == result, '{}, {}, {}'.format(
            expression, evaluated_expression, result)


# =============================================================================
# TEST QUANTITY UTILITIES
# =============================================================================

def test_tracked_quantity():
    """Test TrackedQuantity objects."""
    def reset(q):
        assert tracked_quantity.has_changed is True
        tracked_quantity.has_changed = False

    test_cases = [
        np.array([10.0, 20.0, 30.0]) * unit.kelvin,
        [1.0, 2.0, 3.0] * unit.nanometers,
    ]
    for quantity in test_cases:
        tracked_quantity = TrackedQuantity(quantity)
        u = tracked_quantity.unit
        assert tracked_quantity.has_changed is False

        tracked_quantity[0] = 5.0 * u
        assert tracked_quantity[0] == 5.0 * u
        reset(tracked_quantity)

        tracked_quantity[0:2] = [5.0, 6.0] * u
        assert np.all(tracked_quantity[0:2] == [5.0, 6.0] * u)
        reset(tracked_quantity)

        if isinstance(tracked_quantity._value, list):
            del tracked_quantity[0]
            assert len(tracked_quantity) == 2
            reset(tracked_quantity)

            tracked_quantity.append(10.0*u)
            assert len(tracked_quantity) == 3
            reset(tracked_quantity)

            tracked_quantity.extend([11.0, 12.0]*u)
            assert len(tracked_quantity) == 5
            reset(tracked_quantity)

            element = 15.0*u
            tracked_quantity.insert(1, element)
            assert len(tracked_quantity) == 6
            reset(tracked_quantity)

            tracked_quantity.remove(element.value_in_unit(u))
            assert len(tracked_quantity) == 5
            reset(tracked_quantity)

            assert tracked_quantity.pop().unit == u
            assert len(tracked_quantity) == 4
            reset(tracked_quantity)
        else:
            # Check that numpy views are handled correctly.
            view = tracked_quantity[:3]
            view[0] = 20.0*u
            assert tracked_quantity[0] == 20.0*u
            reset(tracked_quantity)

            view2 = view[1:]
            view2[0] = 30.0*u
            assert tracked_quantity[1] == 30.0*u
            reset(tracked_quantity)


def test_is_quantity_close():
    """Test is_quantity_close method."""
    # (quantity1, quantity2, test_result)
    test_cases = [(300.0*unit.kelvin, 300.000000004*unit.kelvin, True),
                  (300.0*unit.kelvin, 300.00000004*unit.kelvin, False),
                  (1.01325*unit.bar, 1.01325000006*unit.bar, True),
                  (1.01325*unit.bar, 1.0132500006*unit.bar, False)]

    err_msg = 'obtained: {}, expected: {} (quantity1: {}, quantity2: {})'
    for quantity1, quantity2, test_result in test_cases:
        msg = f"Test failed: ({quantity1}, {quantity2}, {test_result})"
        assert is_quantity_close(quantity1, quantity2) == test_result, msg

    # Passing quantities with different units raise an exception.
    with nose.tools.assert_raises(TypeError):
        is_quantity_close(300*unit.kelvin, 1*unit.atmosphere)


def test_quantity_from_string():
    """Test that quantities can be derived from strings"""
    test_strings = [
        ('3', 3.0),  # Handle basic float
        ('meter', unit.meter),  # Handle basic unit object
        ('300 * kelvin', 300 * unit.kelvin),  # Handle standard Quantity
        ('" 0.3 * kilojoules_per_mole / watt**3"', 0.3 * unit.kilojoules_per_mole / unit.watt ** 3), # Handle division, exponent, nested string
        ('1*meter / (4*second)', 0.25 * unit.meter / unit.second),  # Handle compound math and parenthesis
        ('1 * watt**2 /((1* kelvin)**3 / gram)', 1 * (unit.watt ** 2) * (unit.gram) / (unit.kelvin ** 3)), # Handle everything
        ('/watt', unit.watt ** -1)  # Handle special "inverse unit" case
    ]

    for test_string in test_strings:
        input_string, expected_result = test_string
        assert quantity_from_string(input_string) == expected_result


# =============================================================================
# TEST SERIALIZATION UTILITIES
# =============================================================================

class MyClass:
    """Example of serializable class used by test_serialize_deserialize."""
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __getstate__(self):
        serialization = dict()
        serialization['a'] = self.a
        serialization['b'] = self.b
        return serialization

    def __setstate__(self, serialization):
        self.a = serialization['a']
        self.b = serialization['b']

    def add(self):
        return self.a + self.b


def test_serialize_deserialize():
    """Test serialize method."""

    my_instance = MyClass(a=4, b=5)

    # Test serialization.
    serialization = serialize(my_instance)
    expected_serialization = {'_serialized__module_name': 'test_utils',
                              '_serialized__class_name': 'MyClass',
                              'a': 4, 'b': 5}
    assert serialization == expected_serialization

    # Test deserialization.
    deserialized_instance = deserialize(serialization)
    assert deserialized_instance is not my_instance  # this is a new instantiation
    assert isinstance(deserialized_instance, MyClass)
    assert deserialized_instance.a == 4
    assert deserialized_instance.b == 5
    assert deserialized_instance.add() == 9


# =============================================================================
# TEST METACLASS UTILITIES
# =============================================================================

def test_subhooked_abcmeta():
    """Test class SubhookedABCMeta."""
    # Define an interface
    class IInterface(SubhookedABCMeta):
        @abc.abstractmethod
        def my_method(self): pass

        @abc.abstractproperty
        def my_property(self): pass

        @staticmethod
        @abc.abstractmethod
        def my_static_method(): pass

    # Define object implementing the interface with duck typing
    class InterfaceImplementation:
        def my_method(self): pass

        def my_property(self): pass

        @staticmethod
        def my_static_method(): pass

    implementation_instance = InterfaceImplementation()
    assert isinstance(implementation_instance, IInterface)

    # Define incomplete implementation
    class WrongInterfaceImplementation:
        def my_method(self): pass

    implementation_instance = WrongInterfaceImplementation()
    assert not isinstance(implementation_instance, IInterface)


def test_find_all_subclasses():
    """Test find_all_subclasses() function."""
    # Create Python2-3 compatible abstract classes.
    ABC = abc.ABCMeta('ABC', (), {})

    # Diamond inheritance.
    class A:
        pass

    class B(A):
        pass

    class C(A, ABC):
        @abc.abstractmethod
        def m(self):
            pass

    class D(B, C, ABC):
        @abc.abstractmethod
        def m(self):
            pass

    class E(D):
        def m(self):
            pass

    assert find_all_subclasses(B) == {B, D, E}
    assert find_all_subclasses(B, discard_abstract=True, include_parent=False) == {E}
    assert find_all_subclasses(A) == {A, B, D, E, C}
    assert find_all_subclasses(A, discard_abstract=True, include_parent=False) == {B, E}


# =============================================================================
# RESTORABLE OPENMM OBJECT
# =============================================================================

class TestRestorableOpenMMObject:
    """Test the RestorableOpenMMObject utility class."""

    @classmethod
    def setup_class(cls):
        """Example restorable classes for tests."""
        class DummyRestorableCustomForce(RestorableOpenMMObject, openmm.CustomBondForce):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

        class DummyRestorableCustomIntegrator(RestorableOpenMMObject, openmm.CustomIntegrator):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

        cls.dummy_force = DummyRestorableCustomForce('0.0;')
        cls.dummier_force = DummyRestorableCustomForce('0.0;')
        cls.dummy_integrator= DummyRestorableCustomIntegrator(2.0*unit.femtoseconds)

    def test_restorable_openmm_object(self):
        """Test RestorableOpenMMObject classes can be serialized and copied correctly."""

        # Each test case is a pair (object, is_restorable).
        test_cases = [
            (copy.deepcopy(self.dummy_force), True),
            (copy.deepcopy(self.dummy_integrator), True),
            (openmm.CustomBondForce('K'), False)
        ]

        for openmm_object, is_restorable in test_cases:
            assert RestorableOpenMMObject.is_restorable(openmm_object) is is_restorable
            err_msg = f'{openmm_object}: {RestorableOpenMMObject.restore_interface(openmm_object)}, {is_restorable}'
            assert RestorableOpenMMObject.restore_interface(openmm_object) is is_restorable, err_msg

            # Serializing/deserializing restore the class correctly.
            serialization = openmm.XmlSerializer.serialize(openmm_object)
            deserialized_object = RestorableOpenMMObject.deserialize_xml(serialization)
            if is_restorable:
                assert type(deserialized_object) is type(openmm_object)

            # Copying keep the Python class and attributes.
            deserialized_object._monkey_patching = True
            copied_object = copy.deepcopy(deserialized_object)
            if is_restorable:
                assert type(copied_object) is type(openmm_object)
                assert hasattr(copied_object, '_monkey_patching')

    def test_multiple_object_context_creation(self):
        """Test that it is possible to create contexts with multiple restorable objects.

        The aim of this test is to make sure we can restore the force objects using to create the context;
        after serialization.

        Notes
        -----
        As of Openmm 8 having the same type of openmm objects with different default values for global parameters is
        not allowed.
        """
        system = openmm.System()
        for i in range(4):
            system.addParticle(1.0*unit.atom_mass_units)
        system.addForce(copy.deepcopy(self.dummy_force))
        system.addForce(copy.deepcopy(self.dummier_force))
        context = openmm.Context(system, copy.deepcopy(self.dummy_integrator))

        # Try modifying the parameter global variable.
        force_hash_parameter_name = self.dummy_force._hash_parameter_name
        context.setParameter(force_hash_parameter_name, 1.0)

        # Check that two forces keep independent global variables.
        system_serialization = openmm.XmlSerializer.serialize(context.getSystem())
        system = openmm.XmlSerializer.deserialize(system_serialization)
        force1, force2 = system.getForce(0), system.getForce(1)
        assert RestorableOpenMMObject.restore_interface(force1)
        assert RestorableOpenMMObject.restore_interface(force2)
        assert isinstance(force1, self.dummy_force.__class__)
        assert isinstance(force2, self.dummier_force.__class__)

    def test_context_from_restorable_with_different_globals(self):
        """
        Test that you cannot create a context from restorable objects with different default values for
        global parameters.

        Creates a system with two forces that have different default values for the hash global parameter and
        expects an OpenmmException when trying to create a Context using these forces.

        Notes
        -----
        As of Openmm 8 having the same type of openmm objects with different default values for global parameters is
        not allowed.
        """
        dummy_force = copy.deepcopy(self.dummy_force)
        dummier_force = copy.deepcopy(self.dummier_force)

        # Change the global parameter default value for one of the forces
        force_hash_parameter_name = dummy_force._hash_parameter_name
        dummier_force.addGlobalParameter(force_hash_parameter_name, 3.141592)

        system = openmm.System()
        for i in range(4):
            system.addParticle(1.0 * unit.atom_mass_units)
        system.addForce(copy.deepcopy(dummy_force))
        system.addForce(copy.deepcopy(dummier_force))

        # TODO: Change this once we migrate to pytest -- using skipif as needed
        # Skip assertion for openmm < 8
        if int(openmm.__version__[0]) < 8:
            pass
        else:
            with nose.tools.assert_raises(openmm.OpenMMException):
                openmm.Context(system, copy.deepcopy(self.dummy_integrator))

    def test_restorable_openmm_object_failure(self):
        """An exception is raised if the class has a restorable hash but the class can't be found."""
        force = openmm.CustomBondForce('0.0')
        force_hash_parameter_name = self.dummy_force._hash_parameter_name
        force.addGlobalParameter(force_hash_parameter_name, 15.0)
        with nose.tools.assert_raises(RestorableOpenMMObjectError):
            RestorableOpenMMObject.restore_interface(force)

    def test_restorable_openmm_object_hash_collisions(self):
        """Check hash collisions between all objects inheriting from RestorableOpenMMObject."""
        restorable_classes = find_all_subclasses(RestorableOpenMMObject)

        # Test pre-condition: make sure that our custom forces and integrators are loaded.
        restorable_classes_names = {restorable_cls.__name__ for restorable_cls in restorable_classes}
        assert 'ThermostatedIntegrator' in restorable_classes_names
        assert 'RadiallySymmetricRestraintForce' in restorable_classes_names

        # Test hash collisions.
        all_hashes = set()
        for restorable_cls in restorable_classes:
            hash_float = RestorableOpenMMObject._compute_class_hash(restorable_cls)
            all_hashes.add(hash_float)
        assert len(all_hashes) == len(restorable_classes)


class TestEquilibrationUtils:
    """
    Class for testing equilibration utility functions in openmmtools.utils.equilibration
    """
    def test_gentle_equilibration_setup(self):
        """
        Test gentle equilibration implementation using the Alanine dipeptide in explicit solvent
        system found in `openmmtools.testsystems.AlanineDipeptideExplicit`

        This only tests the gentle equilibration can be run with this system, only one iteration
        of each stage is run.
        """
        from openmmtools.testsystems import AlanineDipeptideExplicit
        from openmmtools.utils import run_gentle_equilibration

        test_system = AlanineDipeptideExplicit()

        # Retrieve positions, system and topology from the test_system object
        positions = np.array(test_system.positions.value_in_unit(unit.nanometer))
        system = test_system.system
        topology = test_system.topology

        stages = [
            {'EOM': 'minimize', 'n_steps': 1, 'temperature': 300 * unit.kelvin, 'ensemble': None,
             'restraint_selection': 'protein and not type H',
             'force_constant': 100 * unit.kilocalories_per_mole / unit.angstrom ** 2,
             'collision_rate': 2 / unit.picoseconds,
             'timestep': 1 * unit.femtoseconds},
            {'EOM': 'MD_interpolate', 'n_steps': 1, 'temperature': 100 * unit.kelvin,
             'temperature_end': 300 * unit.kelvin,
             'ensemble': 'NVT', 'restraint_selection': 'protein and not type H',
             'force_constant': 100 * unit.kilocalories_per_mole / unit.angstrom ** 2,
             'collision_rate': 10 / unit.picoseconds,
             'timestep': 1 * unit.femtoseconds},
            {'EOM': 'MD', 'n_steps': 1, 'temperature': 300, 'ensemble': 'NPT',
             'restraint_selection': 'protein and not type H',
             'force_constant': 100 * unit.kilocalories_per_mole / unit.angstrom ** 2,
             'collision_rate': 10 / unit.picoseconds,
             'timestep': 1 * unit.femtoseconds},
            {'EOM': 'MD', 'n_steps': 1, 'temperature': 300 * unit.kelvin, 'ensemble': 'NPT',
             'restraint_selection': 'protein and not type H',
             'force_constant': 10 * unit.kilocalories_per_mole / unit.angstrom ** 2,
             'collision_rate': 2 / unit.picoseconds,
             'timestep': 1 * unit.femtoseconds},
            {'EOM': 'minimize', 'n_steps': 1, 'temperature': 300 * unit.kelvin, 'ensemble': None,
             'restraint_selection': 'protein and backbone',
             'force_constant': 10 * unit.kilocalories_per_mole / unit.angstrom ** 2,
             'collision_rate': 2 / unit.picoseconds,
             'timestep': 1 * unit.femtoseconds},
            {'EOM': 'MD', 'n_steps': 1, 'temperature': 300 * unit.kelvin, 'ensemble': 'NPT',
             'restraint_selection': 'protein and backbone',
             'force_constant': 10 * unit.kilocalories_per_mole / unit.angstrom ** 2,
             'collision_rate': 2 / unit.picoseconds,
             'timestep': 1 * unit.femtoseconds},
            {'EOM': 'MD', 'n_steps': 1, 'temperature': 300 * unit.kelvin, 'ensemble': 'NPT',
             'restraint_selection': 'protein and backbone',
             'force_constant': 1 * unit.kilocalories_per_mole / unit.angstrom ** 2,
             'collision_rate': 2 / unit.picoseconds,
             'timestep': 1 * unit.femtoseconds},
            {'EOM': 'MD', 'n_steps': 1, 'temperature': 300 * unit.kelvin, 'ensemble': 'NPT',
             'restraint_selection': 'protein and backbone',
             'force_constant': 0.1 * unit.kilocalories_per_mole / unit.angstrom ** 2,
             'collision_rate': 2 / unit.picoseconds,
             'timestep': 1 * unit.femtoseconds},
            {'EOM': 'MD', 'n_steps': 1, 'temperature': 300 * unit.kelvin, 'ensemble': 'NPT',
             'restraint_selection': None,
             'force_constant': 0 * unit.kilocalories_per_mole / unit.angstrom ** 2,
             'collision_rate': 2 / unit.picoseconds,
             'timestep': 2 * unit.femtoseconds},
        ]

        with temporary_directory() as tmp_path:
            outfile_path = f"{tmp_path}/outfile.cif"
            run_gentle_equilibration(topology, positions, system, stages, outfile_path, platform_name="CPU",
                                 save_box_vectors=False)

    # TODO: Marking as not a test until we solve our GPU CI
    @nottest
    def test_gentle_equilibration_cuda(self):
        """
        Test gentle equilibration implementation using the Alanine dipeptide in explicit solvent
        system found in `openmmtools.testsystems.AlanineDipeptideExplicit`

        To date it is meant to just test that the protocol can run with a test system and
        do a quick comparison of the energies (this latter part not implemented yet).

        Meant to be run using CUDA platform, similar to a production-ready environment.
        """
        # TODO: Perform the energy comparison part
        from openmmtools.testsystems import AlanineDipeptideExplicit
        from openmmtools.utils import run_gentle_equilibration

        test_system = AlanineDipeptideExplicit()

        # Retrieve positions, system and topology from the test_system object
        positions = np.array(test_system.positions.value_in_unit(unit.nanometer))
        system = test_system.system
        topology = test_system.topology

        stages = [
            {'EOM': 'minimize', 'n_steps': 10000, 'temperature': 300 * unit.kelvin, 'ensemble': None,
             'restraint_selection': 'protein and not type H',
             'force_constant': 100 * unit.kilocalories_per_mole / unit.angstrom ** 2,
             'collision_rate': 2 / unit.picoseconds,
             'timestep': 1 * unit.femtoseconds},
            {'EOM': 'MD_interpolate', 'n_steps': 100000, 'temperature': 100 * unit.kelvin,
             'temperature_end': 300 * unit.kelvin,
             'ensemble': 'NVT', 'restraint_selection': 'protein and not type H',
             'force_constant': 100 * unit.kilocalories_per_mole / unit.angstrom ** 2,
             'collision_rate': 10 / unit.picoseconds,
             'timestep': 1 * unit.femtoseconds},
            {'EOM': 'MD', 'n_steps': 100000, 'temperature': 300, 'ensemble': 'NPT',
             'restraint_selection': 'protein and not type H',
             'force_constant': 100 * unit.kilocalories_per_mole / unit.angstrom ** 2,
             'collision_rate': 10 / unit.picoseconds,
             'timestep': 1 * unit.femtoseconds},
            {'EOM': 'MD', 'n_steps': 250000, 'temperature': 300 * unit.kelvin, 'ensemble': 'NPT',
             'restraint_selection': 'protein and not type H',
             'force_constant': 10 * unit.kilocalories_per_mole / unit.angstrom ** 2,
             'collision_rate': 2 / unit.picoseconds,
             'timestep': 1 * unit.femtoseconds},
            {'EOM': 'minimize', 'n_steps': 10000, 'temperature': 300 * unit.kelvin, 'ensemble': None,
             'restraint_selection': 'protein and backbone',
             'force_constant': 10 * unit.kilocalories_per_mole / unit.angstrom ** 2,
             'collision_rate': 2 / unit.picoseconds,
             'timestep': 1 * unit.femtoseconds},
            {'EOM': 'MD', 'n_steps': 100000, 'temperature': 300 * unit.kelvin, 'ensemble': 'NPT',
             'restraint_selection': 'protein and backbone',
             'force_constant': 10 * unit.kilocalories_per_mole / unit.angstrom ** 2,
             'collision_rate': 2 / unit.picoseconds,
             'timestep': 1 * unit.femtoseconds},
            {'EOM': 'MD', 'n_steps': 100000, 'temperature': 300 * unit.kelvin, 'ensemble': 'NPT',
             'restraint_selection': 'protein and backbone',
             'force_constant': 1 * unit.kilocalories_per_mole / unit.angstrom ** 2,
             'collision_rate': 2 / unit.picoseconds,
             'timestep': 1 * unit.femtoseconds},
            {'EOM': 'MD', 'n_steps': 100000, 'temperature': 300 * unit.kelvin, 'ensemble': 'NPT',
             'restraint_selection': 'protein and backbone',
             'force_constant': 0.1 * unit.kilocalories_per_mole / unit.angstrom ** 2,
             'collision_rate': 2 / unit.picoseconds,
             'timestep': 1 * unit.femtoseconds},
            {'EOM': 'MD', 'n_steps': 2500000, 'temperature': 300 * unit.kelvin, 'ensemble': 'NPT',
             'restraint_selection': None,
             'force_constant': 0 * unit.kilocalories_per_mole / unit.angstrom ** 2,
             'collision_rate': 2 / unit.picoseconds,
             'timestep': 2 * unit.femtoseconds},
        ]

        run_gentle_equilibration(topology, positions, system, stages, "outfile.cif", platform_name="CUDA",
                             save_box_vectors=False)
