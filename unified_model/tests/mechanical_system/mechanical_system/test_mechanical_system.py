import unittest
from collections import OrderedDict
import numpy as np

from pandas.testing import assert_frame_equal
import pandas as pd
from scipy import integrate

# Local imports
from unified_model.mechanical_model import MechanicalModel
from unified_model.mechanical_system.spring.magnetic_spring import MagneticSpring
from unified_model.mechanical_system.model import ode_decoupled

# Local test helpers
from unified_model.tests.mechanical_system.utils import build_test_mechanical_system_model
from unified_model.tests.mechanical_system.mechanical_system import test_data
from unified_model.tests.mechanical_system.test_data import TEST_MAGNET_SPRING_FEA_PATH

from mockito import ANY, when, verify


class TestMechanicalModel(unittest.TestCase):
    """
    Tests the MechanicalModel class
    """

    def setUp(self):
        """
        Run before every test.
        """

        self.test_mechanical_system = MechanicalModel()
        self.test_model = 'ode_decoupled'
        self.test_initial_conditions = [1, 2, 3, 4]
        self.test_mechanical_system.raw_output = test_data.TEST_RAW_OUTPUT
        self.test_mechanical_system.t = test_data.TEST_TIME_STEPS

    def test_set_model(self):
        """
        Tests if the model is set correctly
        """
        self.test_mechanical_system.set_model(self.test_model,
                                              initial_conditions=self.test_initial_conditions)

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
        test_spring = MagneticSpring(TEST_MAGNET_SPRING_FEA_PATH, model='coulombs_modified')
        self.test_mechanical_system.set_spring(test_spring)

        self.assertEqual(self.test_mechanical_system.spring, test_spring)

    def test_build_model_kwargs(self):
        """
        Tests if the model kwarg dictionary is generated correctly.
        """
        test_spring = 'test_spring'
        test_damper = 'test_damper'
        test_input = 'test_input'
        test_magnet_assembly = 'test_magnet_assembly'
        test_model = 'ode_decoupled'

        self.test_mechanical_system.set_spring(test_spring)
        self.test_mechanical_system.set_damper(test_damper)
        self.test_mechanical_system.set_input(test_input)
        self.test_mechanical_system.set_magnet_assembly(test_magnet_assembly)
        self.test_mechanical_system.set_model(test_model)

        created_kwargs = self.test_mechanical_system._build_model_kwargs()

        self.assertIn('spring', created_kwargs)
        self.assertEqual(created_kwargs['spring'], test_spring)
        self.assertIn('damper', created_kwargs)
        self.assertEqual(created_kwargs['damper'], test_damper)
        self.assertIn('input', created_kwargs)
        self.assertEqual(created_kwargs['input'], test_input)
        self.assertIn('magnet_assembly', created_kwargs)
        self.assertEqual(created_kwargs['magnet_assembly'], test_magnet_assembly)

    def test_solve(self):
        """
        Test if the mechanical system will call the appropriate solver and produce the correct output.
        """
        test_mechanical_system = MechanicalModel()
        test_mechanical_system.additional_model_kwargs = {}
        test_t_start = 0
        test_t_end = 1

        class MockPsoln:
            def __init__(self):
                self.t = np.array([10., 20., 30.])
                self.y = np.array([[1., 2., 3.], [4., 5., 6.]])

        expected_raw_output = np.array([[1., 2., 3.], [4., 5., 6.]])
        expected_t = np.array([10., 20., 30.])

        mock_result = MockPsoln()
        when(integrate).solve_ivp(fun=ANY, t_span=ANY, y0=ANY, max_step=ANY).thenReturn(mock_result)
        test_mechanical_system.solve(test_t_start, test_t_end)  # execute

        # test
        self.assertEqual(expected_raw_output.tolist(), test_mechanical_system.raw_output.tolist())
        self.assertEqual(expected_t.tolist(), test_mechanical_system.t.tolist())

    def test_get_output(self):
        """
        Tests if the simulation output is returned correctly.
        """
        actual_output_dataframe = self.test_mechanical_system.get_output(time='t',
                                                                         x1='x1',
                                                                         x2='x2',
                                                                         relative_displacement='x3-x1')
        expected_data = OrderedDict(
            time=[1., 2., 3., 4., 5.],
            x1=[1., 1., 1., 1., 1.],
            x2=[2., 2., 2., 2., 2.],
            relative_displacement=[2., 2., 2., 2., 2.]
        )
        expected_output_dataframe = pd.DataFrame(expected_data)
        assert_frame_equal(actual_output_dataframe, expected_output_dataframe)
