import unittest
import numpy as np
import pandas as pd

from numpy.testing import assert_array_equal

# Local imports
from unified_model.evaluate import LabeledVideoProcessor


class TestLabeledProcessor(unittest.TestCase):
    """Test the `LabeledVideoProcessor` class."""

    def setUp(self):
        """Run before every test"""

        self.test_groundtruth_df = pd.DataFrame(
            {
                "start_x": [10, 11],
                "start_y": [10, 20],
                "end_x": [10, 11],
                "end_y": [30, 40],
                "top_of_magnet": [0, 1],
                "y_pixel_scale": [1, 1],
            }
        )

    def test_fit_transform(self):
        """Test the `fit_transform method`."""
        test_lp = LabeledVideoProcessor(L=120, mm=10, seconds_per_frame=0.1)
        actual_displacement, actual_timesteps = test_lp.fit_transform(
            self.test_groundtruth_df
        )

        expected_displacement = np.array([0.02, 0.01])
        expected_timesteps = np.array([0, 0.1])

        assert_array_equal(actual_displacement, expected_displacement)
        assert_array_equal(actual_timesteps, expected_timesteps)


class TestMechanicalSystemEvaluator(unittest.TestCase):
    pass
