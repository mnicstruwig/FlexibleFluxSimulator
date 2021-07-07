import numpy as np
import pandas as pd
import unittest
from numpy.testing import assert_array_equal
from mockito import mock, when, unstub, ANY
from scipy import integrate
from collections import OrderedDict
from pandas.testing import assert_frame_equal

from unified_model.unified import UnifiedModel
from unified_model.coupling import CouplingModel
from unified_model.mechanical_model import MechanicalModel
from unified_model.electrical_model import ElectricalModel


class TestUnifiedModel(unittest.TestCase):
    """Test the UnifiedModel class."""

    def test_set_mechanical_model(self):
        """Test the set_mechanical_model class method"""

        test_unified_model = UnifiedModel()
        test_mechanical_system = mock(MechanicalModel)

        self.assertTrue(test_unified_model.mechanical_model is None)  # before
        test_unified_model.set_mechanical_model(test_mechanical_system)
        self.assertIsInstance(
            test_unified_model.mechanical_model, type(test_mechanical_system)  # after
        )

        unstub()

    def test_set_electrical_model(self):
        """Test the set_electrical_model class method"""

        test_unified_model = UnifiedModel()
        test_electrical_system = mock(ElectricalModel)

        self.assertTrue(test_unified_model.electrical_model is None)
        test_unified_model.set_electrical_model(test_electrical_system)
        self.assertIsInstance(
            test_unified_model.electrical_model, type(test_electrical_system)
        )

        unstub()

    def test_set_coupling_model(self):
        """Test the set_coupling_model class method."""

        test_unified_model = UnifiedModel()
        test_coupling_model = mock()

        self.assertTrue(test_unified_model.coupling_model is None)
        test_unified_model.set_coupling_model(test_coupling_model)
        self.assertIsInstance(
            test_unified_model.coupling_model, type(test_coupling_model)
        )

    def test_set_governing_equations(self):
        """Test the set_governing_equations class method"""
        test_unified_model = UnifiedModel()
        test_governing_equations = mock()

        self.assertTrue(test_unified_model.governing_equations is None)
        test_unified_model.set_governing_equations(test_governing_equations)
        self.assertIsInstance(
            test_unified_model.governing_equations, type(test_governing_equations)
        )

    def test_solve(self):
        """Test that the unified model will attempt to find a solution"""

        test_time = np.array([10.0, 20.0, 30.0])
        test_y_raw_output = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        class MockPsoln:
            """
            A mock solution that is returned by the
            scipy.integrate.solve_ivp method.
            """

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
        test_coupling_model = mock(CouplingModel)
        test_governing_equations = TestGoverningEquations()

        test_unified_model = UnifiedModel()
        test_unified_model.mechanical_model = test_mechanical_model
        test_unified_model.electrical_model = test_electrical_model
        test_unified_model.coupling_model = test_coupling_model
        test_unified_model.governing_equations = test_governing_equations

        when(integrate).solve_ivp(
            fun=ANY,
            t_span=test_t_span,
            y0=test_initial_conditions,
            max_step=test_max_step,
        ).thenReturn(MockPsoln())

        expected_t = test_time
        expected_raw_solution = np.array(test_y_raw_output)
        test_unified_model.solve(
            test_t_start, test_t_end, test_initial_conditions, test_max_step
        )

        # test
        actual_raw_solution = test_unified_model.raw_solution
        actual_t = test_unified_model.time

        self.assertEqual(expected_t.tolist(), actual_t.tolist())
        self.assertEqual(expected_raw_solution.tolist(), actual_raw_solution.tolist())

    def test_set_post_processing_pipeline(self):
        """
        Test adding a post-processing pipeline
        """

        def mock_pipeline(y):
            x1, x2, x3, x4, x5 = y
            return [99, 99, 99, 99, 99]

        test_unified_model = UnifiedModel()

        # Pipeline should be empty before adding to it
        self.assertTrue(len(test_unified_model.post_processing_pipeline) == 0)

        test_unified_model.set_post_processing_pipeline(
            mock_pipeline, name="test_pipeline_a"
        )

        self.assertTrue(
            test_unified_model.post_processing_pipeline["test_pipeline_a"]
            is mock_pipeline
        )

        test_unified_model.set_post_processing_pipeline(
            mock_pipeline, name="test_pipeline_b"
        )

        self.assertTrue(len(test_unified_model.post_processing_pipeline) == 2)
        self.assertTrue(
            "test_pipeline_a" in test_unified_model.post_processing_pipeline
        )
        self.assertTrue(
            "test_pipeline_b" in test_unified_model.post_processing_pipeline
        )

    def test_apply_pipeline_single_pipeline(self):
        """
        Test the execution of the post-processing pipeline.
        """
        test_unified_model = UnifiedModel()

        def square_pipeline(y):
            x1, x2, x3 = y
            return [x1 * x1, x2 * x2, x3 * x3]

        test_unified_model.raw_solution = np.array(
            [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]]
        ).T

        test_unified_model.set_post_processing_pipeline(
            square_pipeline, name="square_pipeline"
        )

        expected_result = np.array([[0, 0, 0], [1, 1, 1], [4, 4, 4], [9, 9, 9]]).T

        test_unified_model._apply_pipeline()
        actual_result = test_unified_model.raw_solution

        assert_array_equal(expected_result, actual_result)

    def test_apply_pipeline_multiple_pipelines(self):
        """Test the execution of multiple sequential pipelines"""

        def plus_one_pipeline(y):
            x1, x2, x3 = y
            return [x1 + 1, x2 + 1, x3 + 1]

        test_unified_model = UnifiedModel()
        test_unified_model.set_post_processing_pipeline(
            plus_one_pipeline, "test_pipeline_1"
        )
        test_unified_model.set_post_processing_pipeline(
            plus_one_pipeline, "test_pipeline_2"
        )

        test_unified_model.raw_solution = np.array(
            [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]]
        ).T
        test_unified_model._apply_pipeline()
        expected_result = np.array([[2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]]).T
        actual_result = test_unified_model.raw_solution

        assert_array_equal(expected_result, actual_result)

    def test_get_result(self):
        """
        Test if the simulation result query and be parsed and returned correctly
        """
        expected_data = OrderedDict(
            time=[1.0, 2.0, 3.0, 4.0, 5.0],
            x1=[1.0, 1.0, 1.0, 1.0, 1.0],
            x2=[2.0, 2.0, 2.0, 2.0, 2.0],
            relative_displacement=[3.0, 3.0, 3.0, 3.0, 3.0],
        )
        expected_output_dataframe = pd.DataFrame(expected_data)

        test_time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        test_y_raw_output = np.array(
            [[1.0, 1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0, 2.0]]
        )

        test_unified_model = UnifiedModel()
        test_unified_model.raw_solution = test_y_raw_output
        test_unified_model.time = test_time

        actual_result = test_unified_model.get_result(
            time="t", x1="x1", x2="x2", relative_displacement="x1+x2"
        )

        assert_frame_equal(expected_output_dataframe, actual_result)

    def test_calculate_metric_scalar_function(self):
        """
        Test the `calculate_metric` function where metric is a scalar output
        """
        test_max_metric = lambda x: np.max(x)

        test_y_raw_output = np.array(
            [[1.0, 1.0, 1.0, 1.0, 1.0], [2.0, 3.0, 4.0, 5.0, 6.0]]  # x1  # x2
        )

        test_unified_model = UnifiedModel()

        # patch our model's results
        test_unified_model.raw_solution = test_y_raw_output

        actual_result = test_unified_model.calculate_metric(test_max_metric, "x2")
        expected_result = 6.0

        assert actual_result == expected_result

    def test_calculate_metric_array_function(self):
        """
        Test the `calculate_metric` function where metric is a array output
        """

        # A metric that adds one to each value in the array
        add_one = lambda x_arr: x_arr + 1

        test_y_raw_output = np.array(
            [[1.0, 1.0, 1.0, 1.0, 1.0], [2.0, 3.0, 4.0, 5.0, 6.0]]  # x1  # x2
        )

        test_unified_model = UnifiedModel()

        # patch our model's results
        test_unified_model.raw_solution = test_y_raw_output
        actual_result = test_unified_model.calculate_metric(add_one, "x2")
        # Reshape to have fixed dimension
        expected_result = np.array([3.0, 4.0, 5.0, 6.0, 7.0]).reshape(1, -1)

        assert_array_equal(actual_result, expected_result)

    def test_calculate_metric_multiple_expressions(self):
        """
        Test the `calculate_metric` function for multiple prediction expressions
        """

        # A metric that takes multiple expressions as input
        add_together = lambda x_arr: x_arr[0] + x_arr[1]

        test_y_raw_output = np.array(
            [[1.0, 1.0, 1.0, 1.0, 1.0], [2.0, 3.0, 4.0, 5.0, 6.0]]  # x1  # x2
        )

        test_unified_model = UnifiedModel()

        # patch our model's results
        test_unified_model.raw_solution = test_y_raw_output

        actual_result = test_unified_model.calculate_metric(add_together, ["x1", "x2"])
        expected_result = np.array([3.0, 4.0, 5.0, 6.0, 7.0])

        assert_array_equal(actual_result, expected_result)
