import numpy as np
import unittest

from unified_model.utils.utils import warp_signals, apply_scalar_functions


class TestUtils(unittest.TestCase):
    """Test the utils.py module."""

    def test_warp_signals(self):
        test_signal_1 = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])
        test_signal_2 = np.array([2, 5, 5, 5, 5, 5, 5, 5, 2])

        warped_signal_1, warped_signal_2 = warp_signals(test_signal_1,
                                                        test_signal_2)
        self.assertEqual(len(warped_signal_1), len(warped_signal_2))
        self.assertEqual(np.min(test_signal_1), np.min(warped_signal_1))
        self.assertEqual(np.max(test_signal_1), np.max(warped_signal_1))
        self.assertEqual(np.min(test_signal_2), np.min(warped_signal_2))
        self.assertEqual(np.max(test_signal_2), np.max(warped_signal_2))

    def test_apply_scalar_functions(self):
        """Test the apply_scalar_functions method"""

        def test_mean_func(x1, x2):
            return np.mean(x1+x2)

        def test_max_func(x1, x2):
            return np.max(np.concatenate([x1, x2]))

        def test_min_func(x1, x2):
            return np.min(np.concatenate([x1, x2]))

        test_x1 = [1., 2., 3.]
        test_x2 = [5., 6., 7.]

        actual_result = apply_scalar_functions(x1=test_x1,
                                               x2=test_x2,
                                               mean_val=test_mean_func,
                                               max_val=test_max_func,
                                               min_val=test_min_func)
        expected_result = dict(mean_val=4.,
                               max_val=7.,
                               min_val=1.)

        self.assertEqual(actual_result, expected_result)
