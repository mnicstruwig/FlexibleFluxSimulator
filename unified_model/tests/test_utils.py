import numpy as np

import unittest

from unified_model.utils.utils import (
    warp_signals,
    apply_scalar_functions,
    find_signal_limits,
)


class TestUtils(unittest.TestCase):
    """Test the utils.py module."""

    def test_warp_signals(self):
        """Test the `warp_signals` function."""

        test_signal_1 = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])
        test_signal_2 = np.array([2, 5, 5, 5, 5, 5, 5, 5, 2])

        warped_signal_1, warped_signal_2 = warp_signals(test_signal_1, test_signal_2)
        self.assertEqual(len(warped_signal_1), len(warped_signal_2))
        self.assertEqual(np.min(test_signal_1), np.min(warped_signal_1))
        self.assertEqual(np.max(test_signal_1), np.max(warped_signal_1))
        self.assertEqual(np.min(test_signal_2), np.min(warped_signal_2))
        self.assertEqual(np.max(test_signal_2), np.max(warped_signal_2))

    def test_find_signal_limits(self):
        """Test the `find_signal_limits` function."""
        sampling_period = 0.001
        test_time_arr = np.arange(0, 10, sampling_period)
        test_wave = np.sin(2 * np.pi * 50 * test_time_arr)
        padding = np.zeros(1000)

        test_signal = np.concatenate([padding, test_wave, padding])

        actual_result = find_signal_limits(test_signal, sampling_period)
        expected_result = (1.0, 11.0)

        self.assertAlmostEqual(actual_result[0], expected_result[0], delta=0.1)
        self.assertAlmostEqual(actual_result[1], expected_result[1], delta=0.1)

    def test_apply_scalar_functions(self):
        """Test the apply_scalar_functions function."""

        def test_mean_func(x1, x2):
            return np.mean(x1 + x2)

        def test_max_func(x1, x2):
            return np.max(np.concatenate([x1, x2]))

        def test_min_func(x1, x2):
            return np.min(np.concatenate([x1, x2]))

        test_x1 = [1.0, 2.0, 3.0]
        test_x2 = [5.0, 6.0, 7.0]

        actual_result = apply_scalar_functions(
            x1=test_x1,
            x2=test_x2,
            mean_val=test_mean_func,
            max_val=test_max_func,
            min_val=test_min_func,
        )
        expected_result = dict(mean_val=4.0, max_val=7.0, min_val=1.0)

        self.assertEqual(actual_result, expected_result)
