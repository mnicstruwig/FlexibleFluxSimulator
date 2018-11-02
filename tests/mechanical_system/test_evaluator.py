import unittest
import numpy as np
import pandas as pd

from numpy.testing import assert_array_equal

# Local imports
from unified_model.mechanical_system.evaluator import LabeledProcessor, Evaluator


class TestLabeledProcessor(unittest.TestCase):
    """Test the `LabeledProcessor` class."""

    def setUp(self):
        """Run before every test"""

        self.test_groundtruth_df = pd.DataFrame({'start_x': [10, 11],
                                                 'start_y': [10, 20],
                                                 'end_x': [10, 11],
                                                 'end_y': [20, 30],
                                                 'top_of_magnet': [0, 1],
                                                 'y_pixel_scale': [0.2, 0.2]})

    def test_fit_transform(self):
        """Test the `fit_transform method`."""
        test_lp = LabeledProcessor(L=120, mf=10, mm=10, seconds_per_frame=0.1)
        actual_displacement, actual_timesteps = test_lp.fit_transform(self.test_groundtruth_df)

        expected_displacement = np.array([0.108, 0.098])
        expected_timesteps = np.array([0, 0.1])

        assert_array_equal(actual_displacement, expected_displacement)
        assert_array_equal(actual_timesteps, expected_timesteps)


# TODO: Write tests for Evaluator class
class TestEvaluator(unittest.TestCase):
    pass
