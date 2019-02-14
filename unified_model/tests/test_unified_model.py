import numpy as np
import pandas as pd
import unittest
from mockito import mock, when, unstub, ANY
from scipy import integrate
from collections import OrderedDict
from pandas.testing import assert_frame_equal

from unified_model.unified import UnifiedModel
from unified_model.coupling import ConstantCoupling
from unified_model.mechanical_model import MechanicalModel
from unified_model.electrical_system import ElectricalModel


class TestUnifiedModel(unittest.TestCase):
    """Test the UnifiedModel class."""

    def test_add_mechanical_model(self):
        """Test the add_mechanical_model class method"""

        test_unified_model = UnifiedModel(name='test_unified_model')
        test_mechanical_system = mock(MechanicalModel)

        self.assertTrue(test_unified_model.mechanical_model is None)  # before
        test_unified_model.add_mechanical_model(test_mechanical_system)
        self.assertIsInstance(test_unified_model.mechanical_model, type(test_mechanical_system))

        unstub()

    def test_add_electrical_model(self):
        """Test the add_electrical_model class method"""

        test_unified_model = UnifiedModel(name='test_unified_model')
        test_electrical_system = mock(ElectricalModel)

        self.assertTrue(test_unified_model.electrical_model is None)
        test_unified_model.add_electrical_model(test_electrical_system)
        self.assertIsInstance(test_unified_model.electrical_model, type(test_electrical_system))

        unstub()

    def test_add_coupling_model(self):
        """Test the add_coupling_model class method."""

        test_unified_model = UnifiedModel(name='test_unified_model')
        test_coupling_model = mock()

        self.assertTrue(test_unified_model.coupling_model is None)
        test_unified_model.add_coupling_model(test_coupling_model)
        self.assertIsInstance(test_unified_model.coupling_model, type(test_coupling_model))

    def test_add_governing_equations(self):
        """Test the add_governing_equations class method"""
        test_unified_model = UnifiedModel(name='test_unified_model')
        test_governing_equations = mock()

        self.assertTrue(test_unified_model.governing_equations is None)
        test_unified_model.add_governing_equations(test_governing_equations)
        self.assertIsInstance(test_unified_model.governing_equations, type(test_governing_equations))

    def test_solve(self):
        """Test that he unified model will attempt to find a solution"""

        test_time = np.array([10., 20., 30.])
        test_y_raw_output = np.array([[1., 2., 3.], [4., 5., 6.]])

        class MockPsoln:
            """A mock solution that is returned by the scipy.integrate.solve_ivp method."""
            def __init__(self):
                self.t = test_time
                self.y = test_y_raw_output

        class TestGoverningEquations:
            def __init__(self):
                pass

        # setup
        test_t_start = 0
        test_t_end = 0
        test_t_span = [test_t_start, test_t_end]
        test_initial_conditions = [0, 0, 0]
        test_max_step = 1e-5

        test_mechanical_model = mock(MechanicalModel)
        test_electrical_model = mock(ElectricalModel)
        test_coupling_model = mock(ConstantCoupling)
        test_governing_equations = TestGoverningEquations()

        test_unified_model = UnifiedModel(name='test_unified_model')
        test_unified_model.mechanical_model = test_mechanical_model
        test_unified_model.electrical_model = test_electrical_model
        test_unified_model.coupling_model = test_coupling_model
        test_unified_model.governing_equations = test_governing_equations

        when(integrate).solve_ivp(
            fun=ANY,
            t_span=test_t_span,
            y0=test_initial_conditions,
            max_step=test_max_step).thenReturn(MockPsoln())

        expected_t = test_time
        expected_raw_solution = np.array(test_y_raw_output)
        test_unified_model.solve(test_t_start,
                                 test_t_end,
                                 test_initial_conditions,
                                 test_max_step)

        # test
        actual_raw_solution = test_unified_model.raw_solution
        actual_t = test_unified_model.t

        self.assertEqual(expected_t.tolist(), actual_t.tolist())
        self.assertEqual(expected_raw_solution.tolist(), actual_raw_solution.tolist())

    def test_get_result(self):
        """
        Test if the simulation result query and be parsed and returned correctly
        """
        expected_data = OrderedDict(
            time=[1., 2., 3., 4., 5.],
            x1=[1., 1., 1., 1., 1.],
            x2=[2., 2., 2., 2., 2.],
            relative_displacement=[3., 3., 3., 3., 3.]
        )
        expected_output_dataframe = pd.DataFrame(expected_data)

        test_time = np.array([1., 2., 3., 4., 5.])
        test_y_raw_output = np.array([[1., 1., 1., 1., 1.], [2., 2., 2., 2., 2.]])

        test_unified_model = UnifiedModel(name='test_unified_model')
        test_unified_model.raw_solution = test_y_raw_output
        test_unified_model.t = test_time

        actual_result = test_unified_model.get_result(time='t',
                                                      x1='x1',
                                                      x2='x2',
                                                      relative_displacement='x1+x2')

        assert_frame_equal(expected_output_dataframe, actual_result)
