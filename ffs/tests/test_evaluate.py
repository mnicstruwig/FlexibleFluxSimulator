import unittest
from unittest.mock import call, patch

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from unified_model.evaluate import (
    AdcProcessor,
    ElectricalSystemEvaluator,
    LabeledVideoProcessor,
    impute_missing,
)


class TestAdcProcessor(unittest.TestCase):
    """Test the `AdcProcessor` class."""

    def test_fit_transform_smooth(self):
        """Test the fit_transform method when the signal must be smoothed"""
        test_adc_processor = AdcProcessor(voltage_division_ratio=1, smooth=True)

        test_time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        test_signal = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        test_groundtruth_df = pd.DataFrame(
            {"time": test_time, "test_voltage": test_signal}
        )

        test_smooth_signal = np.array([1, 2.0, 2.0, 2.0, 1.0])

        expected_voltage_readings = test_smooth_signal - np.mean(test_smooth_signal)
        expected_time_values = test_time / 1000

        with patch(
            "unified_model.evaluate.smooth_butterworth", return_value=test_smooth_signal
        ) as _:
            (
                actual_voltage_readings,
                actual_time_values,
            ) = test_adc_processor.fit_transform(
                groundtruth_dataframe=test_groundtruth_df,
                voltage_col="test_voltage",
                time_col="time",
            )

            assert_array_equal(expected_voltage_readings, actual_voltage_readings)
            assert_array_equal(expected_time_values, actual_time_values)

    def test_fit_transform_smooth_kwargs(self):
        """Test the fit_transform method with smoothing kwargs supplied."""
        test_critical_frequency = 1 / 6
        test_adc_processor = AdcProcessor(
            voltage_division_ratio=1,
            smooth=True,
            critical_frequency=test_critical_frequency,
        )

        test_time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        test_signal = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        test_groundtruth_df = pd.DataFrame(
            {"time": test_time, "test_voltage": test_signal}
        )

        test_smooth_signal = np.array([1, 2.0, 2.0, 2.0, 1.0])

        expected_voltage_readings = test_smooth_signal - np.mean(test_smooth_signal)
        expected_time_values = test_time / 1000

        with patch(
            "unified_model.evaluate.smooth_butterworth", return_value=test_smooth_signal
        ) as mock_smooth_butterworth:
            (
                actual_voltage_readings,
                actual_time_values,
            ) = test_adc_processor.fit_transform(
                groundtruth_dataframe=test_groundtruth_df,
                voltage_col="test_voltage",
                time_col="time",
            )
            assert_array_equal(mock_smooth_butterworth.call_args[0][0], test_signal)
            self.assertEqual(
                mock_smooth_butterworth.call_args[0][1], test_critical_frequency
            )
            assert_array_equal(expected_voltage_readings, actual_voltage_readings)
            assert_array_equal(expected_time_values, actual_time_values)

    def test_fit_transform_no_smooth(self):
        """Test the fit_transform method when signal must not be smoothed."""
        test_voltage_division_ratio = 2
        test_adc_processor = AdcProcessor(
            voltage_division_ratio=test_voltage_division_ratio, smooth=False
        )

        test_time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        test_signal = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        test_groundtruth_df = pd.DataFrame(
            {"time": test_time, "test_voltage": test_signal}
        )

        expected_voltage_readings = test_signal * test_voltage_division_ratio
        expected_voltage_readings = expected_voltage_readings - np.mean(
            expected_voltage_readings
        )
        expected_time_values = test_time / 1000

        actual_voltage_readings, actual_time_values = test_adc_processor.fit_transform(
            groundtruth_dataframe=test_groundtruth_df,
            voltage_col="test_voltage",
            time_col="time",
        )

        assert_array_equal(expected_voltage_readings, actual_voltage_readings)
        assert_array_equal(expected_time_values, actual_time_values)


class TestElectricalSystemEvaluator(unittest.TestCase):
    """Test the ElectricalSystemEvaluator class."""

    def test_fit(self):
        """Test the fit method."""
        test_emf_target = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])
        test_time_target = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        test_emf_predict = np.array([1, 2, 3, 4, 5, 6, 5, 4, 3])
        test_time_predict = test_time_target

        test_electrical_system_evaluator = ElectricalSystemEvaluator(
            emf_target=test_emf_target, time_target=test_time_target
        )

        # Before fit
        self.assertTrue(test_electrical_system_evaluator.emf_target_ is None)
        self.assertTrue(test_electrical_system_evaluator.emf_predict_ is None)
        self.assertTrue(test_electrical_system_evaluator.time_ is None)

        test_electrical_system_evaluator.fit(
            emf_predict=test_emf_predict, time_predict=test_time_target
        )

        assert_array_equal(
            test_emf_predict, test_electrical_system_evaluator.emf_predict
        )
        assert_array_equal(
            test_time_predict, test_electrical_system_evaluator.time_predict
        )

        self.assertTrue(
            isinstance(test_electrical_system_evaluator.emf_target_, np.ndarray)
        )
        self.assertTrue(len(test_electrical_system_evaluator.emf_target_) > 1)
        self.assertTrue(
            isinstance(test_electrical_system_evaluator.emf_predict_, np.ndarray)
        )
        self.assertTrue(len(test_electrical_system_evaluator.emf_predict_) > 1)
        self.assertTrue(isinstance(test_electrical_system_evaluator.time_, np.ndarray))
        self.assertTrue(len(test_electrical_system_evaluator.time_) > 1)

    def test_fit_transform(self):
        """Test the fit_transform method."""
        test_emf_target = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])
        test_time_target = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        test_emf_predict = np.array([1, 2, 3, 4, 5, 6, 5, 4, 3])
        test_time_predict = test_time_target

        expected_emf_predict = np.array([1, 2, 3, 4, 5, 6, 5, 4, 3])
        expected_time = test_time_target

        test_electrical_system_evaluator = ElectricalSystemEvaluator(
            emf_target=test_emf_target, time_target=test_time_target
        )

        # We test the fit method separately. So let's mock it out
        # here for testing purposes.
        def mock_fit(self, *args, **kwargs):
            self.time_ = expected_time
            self.emf_predict_ = expected_emf_predict

        with patch(
            "unified_model.evaluate.ElectricalSystemEvaluator._fit", new=mock_fit
        ):
            (
                actual_time,
                actual_emf_predict,
            ) = test_electrical_system_evaluator.fit_transform(
                emf_predict=test_emf_predict, time_predict=test_time_predict
            )
            assert_array_equal(expected_time, actual_time)
            assert_array_equal(expected_emf_predict, actual_emf_predict)

    def test_score(self):
        """Test the score method."""
        test_emf_target = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])
        test_time_target = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        test_emf_predict = np.array([1, 2, 3, 4, 5, 6, 5, 4, 3])

        test_electrical_system_evaluator = ElectricalSystemEvaluator(
            emf_target=test_emf_target, time_target=test_time_target
        )

        # We test the fit method separately. So let's mock it out
        # here for testing purposes.
        def mock_fit(self, emf_predict, time_predict):
            self.emf_predict_ = emf_predict  # Fake resampled predicted values
            self.time_ = time_predict  # Fake resampled timestamps
            self.emf_target_ = self.emf_target  # Fake resampled target values

        with patch(
            "unified_model.evaluate.ElectricalSystemEvaluator.fit", new=mock_fit
        ):
            # Mock `fit` method
            test_electrical_system_evaluator.fit(
                emf_predict=test_emf_predict, time_predict=test_time_target
            )

            def test_metric_mean(x, y):
                return np.mean([x, y])

            def test_metric_max(x, y):
                return np.max([x, y])

            expected_mean = test_metric_mean(
                test_electrical_system_evaluator.emf_predict_,
                test_electrical_system_evaluator.emf_target_,
            )

            expected_max = test_metric_max(
                test_electrical_system_evaluator.emf_predict_,
                test_electrical_system_evaluator.emf_target_,
            )

            actual_result = test_electrical_system_evaluator.score(
                mean=test_metric_mean, max_value=test_metric_max
            )

            self.assertEqual(expected_mean, actual_result.mean)
            self.assertEqual(expected_max, actual_result.max_value)

    def test_poof_no_dtw(self):
        """Test the poof method."""

        test_electrical_system_evaluator = ElectricalSystemEvaluator(None, None)

        test_time_ = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        test_emf_target_ = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        test_emf_predict_ = np.array([1, 2, 3, 4, 5, 6, 5, 4, 3])

        test_electrical_system_evaluator.time_ = test_time_
        test_electrical_system_evaluator.emf_predict_ = test_emf_predict_
        test_electrical_system_evaluator.emf_target_ = test_emf_target_

        with patch("unified_model.evaluate.plt", return_value=None) as mock_pyplot:
            test_electrical_system_evaluator.poof(include_dtw=False)

            expected_call_target = call(
                test_electrical_system_evaluator.time_,
                test_electrical_system_evaluator.emf_target_,
                label="Target",
            )
            expected_call_predictions = call(
                test_electrical_system_evaluator.time_,
                test_electrical_system_evaluator.emf_predict_,
                label="Predictions",
            )

            expected_calls = [expected_call_target, expected_call_predictions]

            mock_pyplot.plot.assert_has_calls(expected_calls, any_order=True)

    def test_poof_dtw(self):
        """Test the poof with the dtw flag enabled."""

        test_electrical_system_evaluator = ElectricalSystemEvaluator(None, None)

        test_time_ = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        test_emf_target_ = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        test_emf_predict_ = np.array([1, 2, 3, 4, 5, 6, 5, 4, 3])

        test_emf_target_warped_ = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
        test_emf_predict_warped_ = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2])

        test_electrical_system_evaluator.time_ = test_time_
        test_electrical_system_evaluator.emf_predict_ = test_emf_predict_
        test_electrical_system_evaluator.emf_target_ = test_emf_target_
        test_electrical_system_evaluator.emf_predict_warped_ = test_emf_predict_warped_
        test_electrical_system_evaluator.emf_target_warped_ = test_emf_target_warped_

        with patch("unified_model.evaluate.plt", return_value=None) as mock_pyplot:
            test_electrical_system_evaluator.poof(include_dtw=True)

            expected_call_target = call(
                test_electrical_system_evaluator.time_,
                test_electrical_system_evaluator.emf_target_,
                label="Target",
            )
            expected_call_predictions = call(
                test_electrical_system_evaluator.time_,
                test_electrical_system_evaluator.emf_predict_,
                label="Predictions",
            )

            expected_call_target_warped = call(
                test_electrical_system_evaluator.emf_target_warped_,
                label="Target, time-warped",
            )

            expected_call_predictions_warped = call(
                test_electrical_system_evaluator.emf_predict_warped_,
                label="Predictions, time-warped",
            )

            expected_calls = [
                expected_call_target,
                expected_call_predictions,
                expected_call_target_warped,
                expected_call_predictions_warped,
            ]

            mock_pyplot.plot.assert_has_calls(expected_calls, any_order=True)


class TestImputeMissing(unittest.TestCase):
    """Test various helper functions related to imputing missing values
    in evaluate module."""

    def test_impute_missing(self):
        """Test the `impute_missing` method."""
        test_df_missing = pd.DataFrame(
            {
                "start_y": [
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                ],  # Used to indicate where values are missing
                "y_prime_mm": [11, 12, 13, 14, 15, 16, 17],  # Target column
            }
        )

        expected_result = test_df_missing["y_prime_mm"].values

        # Create a missing value
        missing_index = 3
        test_df_missing.loc[missing_index, "start_y"] = -1  # Create a missing index
        test_df_missing.loc[missing_index, "y_prime_mm"] = 99  # Insert bad calculation

        imputed_df = impute_missing(df_missing=test_df_missing, indexes=[missing_index])
        actual_result = imputed_df["y_prime_mm"].values

        assert_array_equal(expected_result, actual_result)

    def test_impute_missing_raises_value_error(self):
        """Ensure a value error is raised when values cannot be imputed."""
        test_df_missing = pd.DataFrame(
            {
                "start_y": [
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                ],  # Used to indicate where values are missing
                "y_prime_mm": [11, 12, 13, 14, 15, 16, 17],  # Target column
            }
        )

        # Create a missing value
        missing_indexes = [2, 3]  # Cannot impute if two values missing in a row
        test_df_missing.loc[missing_indexes, "start_y"] = -1  # Create a missing index
        test_df_missing.loc[
            missing_indexes, "y_prime_mm"
        ] = 99  # Insert bad calculation

        with self.assertRaises(ValueError):
            impute_missing(df_missing=test_df_missing, indexes=missing_indexes)

    def test_impute_missing_raises_index_error(self):
        """Ensure a index error is raised when required values are out of range"""
        test_df_missing = pd.DataFrame(
            {
                "start_y": [
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                ],  # Used to indicate where values are missing
                "y_prime_mm": [11, 12, 13, 14, 15, 16, 17],  # Target column
            }
        )

        # Create a missing value
        missing_indexes = [
            0,
            1,
        ]  # Cannot impute if missing values are at the first indices
        test_df_missing.loc[missing_indexes, "start_y"] = -1  # Create a missing index
        test_df_missing.loc[
            missing_indexes, "y_prime_mm"
        ] = 99  # Insert bad calculation

        with self.assertRaises(IndexError):
            impute_missing(df_missing=test_df_missing, indexes=missing_indexes)


class TestLabeledVideoProcessor(unittest.TestCase):
    """Test the `LabeledVideoProcessor` class"""

    def test_fit_transform_pixel_scale_not_specified(self):
        """Test the `fit_transform` method when the pixel scale is not
        specified in both the groundtruth dataframe or as a parameter."""
        test_L = 125
        test_mm = 10
        test_seconds_per_frame = 1 / 60
        test_pixel_scale = None

        test_lvp = LabeledVideoProcessor(
            L=test_L,
            mm=test_mm,
            seconds_per_frame=test_seconds_per_frame,
            pixel_scale=test_pixel_scale,
        )

        test_groundtruth_df = pd.DataFrame()
        test_groundtruth_df["start_y"] = [1, 2, 3, 4, 5]
        test_groundtruth_df["end_y"] = [5, 6, 7, 8, 9]
        test_groundtruth_df["y_pixel_scale"] = -1  # scale unspecified

        with self.assertRaises(ValueError):
            test_lvp.fit_transform(
                groundtruth_dataframe=test_groundtruth_df, impute_missing_values=True
            )

    def test_fit_transform_groundtruth_df_is_none(self):
        """Test the `fit_transform` method when the groundtruth dataframe is
        `None`.

        This case occurs when the groundtruth file doesn't exist, or is parsed
        incorrectly.
        """

        test_L = 125
        test_mm = 10
        test_seconds_per_frame = 1 / 60
        test_pixel_scale = None

        test_lvp = LabeledVideoProcessor(
            L=test_L,
            mm=test_mm,
            seconds_per_frame=test_seconds_per_frame,
            pixel_scale=test_pixel_scale,
        )

        test_groundtruth_df = None

        with self.assertRaises(AssertionError):
            test_lvp.fit_transform(
                groundtruth_dataframe=test_groundtruth_df, impute_missing_values=True
            )
