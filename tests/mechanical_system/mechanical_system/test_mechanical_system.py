import unittest
from collections import OrderedDict
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from mechanical_system.mechanical_system import MechanicalSystem
from mechanical_system.spring.magnetic_spring import MagneticSpring
from mechanical_system.model import ode_decoupled

from tests.mechanical_system.utils import build_test_mechanical_system_model
from tests.mechanical_system.mechanical_system import test_data


class TestMechanicalSystem(unittest.TestCase):
    """
    Tests the MechanicalSystem class
    """

    def setUp(self):
        """
        Run before every test.
        """

        self.test_mechanical_system = MechanicalSystem()
        self.test_model = ode_decoupled
        self.test_initial_conditions = [1, 2, 3, 4]
        self.test_mechanical_system.raw_output = test_data.TEST_RAW_OUTPUT
        self.test_mechanical_system.output_time_steps = test_data.TEST_TIME_STEPS

    def test_set_model(self):
        """
        Tests if the model is set correctly
        """
        self.test_mechanical_system.set_model(self.test_model, initial_conditions=self.test_initial_conditions)

        # Tests
        self.assertEqual(self.test_mechanical_system.model, ode_decoupled)

    def test_set_initial_conditions(self):
        """
        Tests if the initial conditions can be set correctly
        """
        self.test_mechanical_system.set_initial_conditions(self.test_initial_conditions)
        self.assertEqual(self.test_mechanical_system.initial_conditions, self.test_initial_conditions)

    def test_set_spring(self):
        """
        Tests if the spring can be set correctly.
        """
        test_spring = MagneticSpring('../test_data/test_magnetic_spring_fea.csv', model='coulombs_modified')
        self.test_mechanical_system.set_spring(test_spring)

        self.assertEqual(self.test_mechanical_system.spring, test_spring)

    def test_set_damper(self):
        """
        Tests if the damper can be set correctly.
        :return:
        """
        pass

    def test_set_input(self):
        """
        Tests if the mechanical input can be set correctly.
        """
        pass

    def test_set_magnet_assembly(self):
        """
        Tests if the magnet assembly can be set correctly.
        """
        pass

    def test_build_model_kwargs(self):
        """
        Tests if the model kwarg dictionary is generated correctly.
        """
        pass

    def test_solve(self):
        """
        Test if the mechanical system is able to be solved.
        """
        complete_mechanical_system = build_test_mechanical_system_model()

        test_t_array = np.arange(0, 5, 0.001)
        complete_mechanical_system.solve(test_t_array)

        # Tests
        self.assertFalse(complete_mechanical_system.raw_output is None)
        self.assertEqual(len(complete_mechanical_system.raw_output), len(test_t_array))

    def test_get_output(self):
        """
        Tests if the simulation output is returned correctly.
        """
        actual_output_dataframe = self.test_mechanical_system.get_output()
        expected_data = OrderedDict(
            time=[1., 2., 3., 4., 5.],
            tube_displacement=[1., 1., 1., 1., 1.],
            tube_velocity=[2., 2., 2., 2., 2.],
            assembly_displacement=[3., 3., 3., 3., 3.],
            assembly_velocity=[4., 4., 4., 4., 4.],
            assembly_relative_displacement=[2., 2., 2., 2., 2.],
            assembly_relative_velocity=[2., 2., 2., 2., 2.]
        )
        expected_output_dataframe = pd.DataFrame(expected_data)
        assert_frame_equal(actual_output_dataframe, expected_output_dataframe)
