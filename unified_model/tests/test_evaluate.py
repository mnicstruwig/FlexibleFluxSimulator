import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd

from unified_model.evaluate import AdcProcessor, ElectricalSystemEvaluator


class TestAdcProcessor(unittest.TestCase):
    """Test the `AdcProcessor` class."""

    def test_fit_transform_smooth(self):
        """Test the fit_transform method when the signal must be smoothed"""
        test_adc_processor = AdcProcessor(voltage_division_ratio=1,
                                          smooth=True)

        test_time = np.array([1., 2., 3., 4., 5.])
        test_signal = np.array([1., 2., 3., 2., 1.])
        test_groundtruth_df = pd.DataFrame({'time': test_time,
                                            'test_voltage': test_signal})

        test_smooth_signal = np.array([1, 2., 2., 2., 1.])

        expected_voltage_readings = test_smooth_signal - np.mean(test_smooth_signal)
        expected_time_values = test_time / 1000

        with patch('unified_model.evaluate.smooth_butterworth', return_value=test_smooth_signal) as _:
            actual_voltage_readings, actual_time_values = test_adc_processor.fit_transform(groundtruth_dataframe=test_groundtruth_df,
                                                                                           voltage_col='test_voltage',
                                                                                           time_col='time')

            assert_array_equal(expected_voltage_readings, actual_voltage_readings)
            assert_array_equal(expected_time_values, actual_time_values)

    def test_fit_transform_smooth_kwargs(self):
        """Test the fit_transform method with smoothing kwargs supplied."""
        test_critical_frequency = 1/6
        test_adc_processor = AdcProcessor(voltage_division_ratio=1,
                                          smooth=True,
                                          critical_frequency=test_critical_frequency)

        test_time = np.array([1., 2., 3., 4., 5.])
        test_signal = np.array([1., 2., 3., 2., 1.])
        test_groundtruth_df = pd.DataFrame({'time': test_time,
                                            'test_voltage': test_signal})

        test_smooth_signal = np.array([1, 2., 2., 2., 1.])

        expected_voltage_readings = test_smooth_signal - np.mean(test_smooth_signal)
        expected_time_values = test_time / 1000

        with patch('unified_model.evaluate.smooth_butterworth', return_value=test_smooth_signal) as mock_smooth_butterworth:
            actual_voltage_readings, actual_time_values = test_adc_processor.fit_transform(groundtruth_dataframe=test_groundtruth_df,
                                                                                           voltage_col='test_voltage',
                                                                                           time_col='time')
            assert_array_equal(mock_smooth_butterworth.call_args[0][0], test_signal)
            self.assertEqual(mock_smooth_butterworth.call_args[0][1], test_critical_frequency)
            assert_array_equal(expected_voltage_readings, actual_voltage_readings)
            assert_array_equal(expected_time_values, actual_time_values)

    def test_fit_transform_no_smooth(self):
        """Test the fit_transform method when signal must not be smoothed."""
        test_voltage_division_ratio = 2
        test_adc_processor = AdcProcessor(voltage_division_ratio=test_voltage_division_ratio,
                                          smooth=False)

        test_time = np.array([1., 2., 3., 4., 5.])
        test_signal = np.array([1., 2., 3., 2., 1.])
        test_groundtruth_df = pd.DataFrame({'time': test_time,
                                            'test_voltage': test_signal})

        expected_voltage_readings = test_signal * test_voltage_division_ratio
        expected_voltage_readings = expected_voltage_readings - np.mean(expected_voltage_readings)
        expected_time_values = test_time / 1000

        actual_voltage_readings, actual_time_values = test_adc_processor.fit_transform(groundtruth_dataframe=test_groundtruth_df,
                                                                                       voltage_col='test_voltage',
                                                                                       time_col='time')

        assert_array_equal(expected_voltage_readings, actual_voltage_readings)
        assert_array_equal(expected_time_values, actual_time_values)


class TestElectricalSystemEvaluator(unittest.TestCase):
    """Test the ElectricalSystemEvaluator class."""
    pass
