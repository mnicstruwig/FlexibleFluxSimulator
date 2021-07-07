import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

# Local imports
from unified_model.mechanical_components.input_excitation import footstep
from unified_model.mechanical_components.input_excitation.accelerometer import (
    _find_nearest_acc_value,
    _parse_raw_accelerometer_input,
    _preprocess_acceleration_dataframe,
    AccelerometerInput,
)  # noqa

from unified_model.tests.mechanical_components.test_data.test_data import (
    TEST_ACCELEROMETER_FILE_PATH,
)

SIMULATION_TIME_COL = "simulation_time_seconds"


class TestAccelerometerInput(unittest.TestCase):
    """
    Test the `AccelerometerInput` class and helper functions.
    """

    def setUp(self):
        """
        Set-up.
        """
        self.test_accelerometer_file_path = TEST_ACCELEROMETER_FILE_PATH
        self.test_raw_accelerometer_df = pd.DataFrame(
            {"time": [1.0, 2.0, 3.0, 4.0, 5.0, 6], "z_G": [0, 1, 0, 1, -2, 0]}
        )

        self.test_accel_column = "z_G"
        self.test_time_column = "time"
        self.test_accel_unit = "g"

        self.test_simulation_time_array = np.array([1, 2, 3, 4, 5, 6])
        self.test_acceleration_array = np.array([0, 1, 0, 1, -2, 0])

    def test_find_nearest_acc_value_exact_value(self):
        """
        Test that the nearest acceleration value for a given timestamp is found
        for the case where the timestamp exactly matches the acceleration
        dataframe.
        """
        expected_acceleration = -2
        test_time = 5
        returned_acceleration = _find_nearest_acc_value(
            test_time, self.test_simulation_time_array, self.test_acceleration_array
        )

        self.assertEqual(expected_acceleration, returned_acceleration)

    def test_find_nearest_acc_value_larger_value(self):
        """
        Test that the nearest acceleration value for a given timestamp is found
        for the case where the timestamp is slightly larger than a timestamp in
        the acceleration dataframe.
        """
        expected_acceleration = -2
        test_time = 5.1
        returned_acceleration = _find_nearest_acc_value(
            test_time, self.test_simulation_time_array, self.test_acceleration_array
        )
        self.assertEqual(expected_acceleration, returned_acceleration)

    def test_find_nearest_acc_value_smaller_value(self):
        """
        Test that the nearest acceleration value for a given timestamp is found
        for the case where the timestamp is slightly smaller than a timestamp
        in the acceleration dataframe.
        """
        expected_acceleration = -2
        test_time = 4.9
        returned_acceleration = _find_nearest_acc_value(
            test_time, self.test_simulation_time_array, self.test_acceleration_array
        )
        self.assertEqual(expected_acceleration, returned_acceleration)

    def test_parse_raw_accelerometer_input_df(self):
        """
        Test that the accelerometer input is parsed correctly if it is a
        dataframe.
        """

        expected_accelerometer_input = self.test_raw_accelerometer_df
        actual_accelerometer_input = _parse_raw_accelerometer_input(
            expected_accelerometer_input
        )

        assert_frame_equal(expected_accelerometer_input, actual_accelerometer_input)

    def test_parse_raw_accelerometer_input_file(self):
        """
        Test that the accelerometer input is parsed correctly if it is a
        filename.
        """
        test_accelerometer_file = self.test_accelerometer_file_path

        loaded_accelerometer_input = _parse_raw_accelerometer_input(
            test_accelerometer_file
        )

        self.assertTrue(isinstance(loaded_accelerometer_input, pd.DataFrame))

    def test_preprocess_acceleration_dataframe_accel_g(self):
        """
        Test that the accelerometer dataframe is preprocessed correctly
        if acceleration units are in Gs.
        """

        expected_acceleration_values = (
            (self.test_raw_accelerometer_df[self.test_accel_column] - 1) * 9.81
        ).tolist()
        processed_df = _preprocess_acceleration_dataframe(
            self.test_raw_accelerometer_df,
            self.test_accel_column,
            self.test_time_column,
            accel_unit="g",
            time_unit="s",
            smooth=False,
        )
        actual_acceleration_values = processed_df[self.test_accel_column].tolist()
        self.assertEqual(expected_acceleration_values, actual_acceleration_values)

    def test_preprocess_acceleration_dataframe_accel_ms(self):
        """
        Test that the accelerometer dataframe is preprocessed correctly
        if acceleration units are in m/s^2.
        """

        expected_acceleration_values = (
            self.test_raw_accelerometer_df[self.test_accel_column] - 9.81
        ).tolist()
        processed_df = _preprocess_acceleration_dataframe(
            self.test_raw_accelerometer_df,
            self.test_accel_column,
            self.test_time_column,
            accel_unit="ms2",
            time_unit="s",
            smooth=False,
        )
        actual_acceleration_values = processed_df[self.test_accel_column].tolist()
        self.assertEqual(expected_acceleration_values, actual_acceleration_values)

    def test_preprocess_acceleration_dataframe_exception_wrong_accel(self):
        """
        Test that the `_preprocess_acceleration_dataframe` method throws an exception
        if the incorrect acceleration unit is specified.
        """

        with self.assertRaises(KeyError):
            processed_df = _preprocess_acceleration_dataframe(  # noqa
                self.test_raw_accelerometer_df,
                self.test_accel_column,
                self.test_time_column,
                accel_unit="not a unit",
                time_unit="s",
                smooth=False,
            )

    def test_preprocess_acceleration_dataframe_time_in_sec(self):
        """
        Test that the accelerometer dataframe is preprocessed correctly
        if time is in seconds.
        """
        expected_time_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        processed_df = _preprocess_acceleration_dataframe(
            self.test_raw_accelerometer_df,
            self.test_accel_column,
            self.test_time_column,
            self.test_accel_unit,
            time_unit="s",
            smooth=False,
        )
        actual_time_values = processed_df[SIMULATION_TIME_COL].tolist()

        self.assertEqual(expected_time_values, actual_time_values)

    def test_preprocess_acceleration_dataframe_time_in_msec(self):
        """
        Test that the accelerometer dataframe is preprocessed correctly
        if time is in seconds.
        """
        expected_time_values = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006]
        processed_df = _preprocess_acceleration_dataframe(
            self.test_raw_accelerometer_df,
            self.test_accel_column,
            self.test_time_column,
            self.test_accel_unit,
            time_unit="ms",
            smooth=False,
        )
        actual_time_values = processed_df[SIMULATION_TIME_COL].values.tolist()

        self.assertEqual(expected_time_values, actual_time_values)

    def test_constructor_raw_dataframe(self):
        """
        Test the constructor of the `AccelerometerInput` class for a dataframe
        as input.
        """
        test_accel_input = AccelerometerInput(
            self.test_raw_accelerometer_df,
            self.test_accel_column,
            self.test_time_column,
            self.test_accel_unit,
            time_unit="s",
            smooth=False,
        )

        self.assertEqual(test_accel_input._accel_column, self.test_accel_column)
        self.assertEqual(test_accel_input._time_unit, "s")
        self.assertEqual(test_accel_input._time_column, self.test_time_column)
        self.assertTrue(isinstance(test_accel_input.acceleration_df, pd.DataFrame))

    def test_constructor_raw_file(self):
        """
        Test the constructor of the `AccelerometerInput` class for a file as input.
        """
        test_accel_input = AccelerometerInput(
            self.test_accelerometer_file_path,
            self.test_accel_column,
            self.test_time_column,
            time_unit="s",
            smooth=False,
        )

        self.assertEqual(test_accel_input._accel_column, self.test_accel_column)
        self.assertEqual(test_accel_input._time_unit, "s")
        self.assertEqual(test_accel_input._time_column, self.test_time_column)
        self.assertTrue(isinstance(test_accel_input.acceleration_df, pd.DataFrame))

    def test_get_acceleration(self):
        """
        Test the `get_acceleration` method of the `AccelerometerInput` class.
        """
        expected_acceleration_a = -3 * 9.81
        expected_acceleration_b = 0
        test_accel_input = AccelerometerInput(
            self.test_accelerometer_file_path,
            self.test_accel_column,
            self.test_time_column,
            time_unit="s",
            smooth=False,
        )

        self.assertEqual(test_accel_input.get_acceleration(5), expected_acceleration_a)
        self.assertEqual(
            test_accel_input.get_acceleration(5.4), expected_acceleration_a
        )
        self.assertEqual(
            test_accel_input.get_acceleration(4.7), expected_acceleration_a
        )
        self.assertEqual(
            test_accel_input.get_acceleration(4.2), expected_acceleration_b
        )


class TestFootstep(unittest.TestCase):
    """
    Tests the footstep class
    """

    def setUp(self):
        """
        Set-up for the `TestFootstep` class
        """

        self.accelerations = [1.0, -2.0, -3.0, 4.0]
        self.acceleration_couple_time_separation = 0.05
        self.positive_footstep_displacement = 0.15
        self.footstep = footstep.Footstep(
            self.accelerations,
            self.acceleration_couple_time_separation,
            self.positive_footstep_displacement,
        )

    def test_calculate_acceleration_couple_times(self):
        """
        Tests the `_calculate_acceleration_couple_times` function
        """

        acc_up = self.accelerations[0]
        acc_dec = self.accelerations[1]
        acc_down = self.accelerations[2]
        acc_impact = self.accelerations[3]

        t_acc_up, t_acc_dec = footstep.calculate_acceleration_couple_times(
            acc_up, acc_dec, self.positive_footstep_displacement
        )
        t_acc_down, t_acc_impact = footstep.calculate_acceleration_couple_times(
            acc_down, acc_impact, -self.positive_footstep_displacement
        )

        # Tests
        self.assertAlmostEqual(t_acc_up, 0.4472135954999579)
        self.assertAlmostEqual(t_acc_dec, 0.223606797749979)
        self.assertAlmostEqual(t_acc_down, 0.23904572186687872)

    def test_set_accelerations(self):
        """
        Tests if the acceleration values are set correctly
        """

        test_accelerations = [9, -8, 7, -6]
        self.footstep._set_accelerations(test_accelerations)
        self.assertEqual(self.footstep.acc_up, 9)
        self.assertEqual(self.footstep.acc_dec, -8)
        self.assertEqual(self.footstep.acc_down, 7)
        self.assertEqual(self.footstep.acc_impact, -6)

    def test_set_acceleration_times(self):
        """
        Integration test if the acceleration times are set correctly.
        """

        self.assertAlmostEqual(self.footstep.t_acc_up, 0.4472135954999579)
        self.assertAlmostEqual(self.footstep.t_acc_dec, 0.223606797749979)
        self.assertAlmostEqual(self.footstep.t_acc_down, 0.2390457218668787)
        self.assertAlmostEqual(self.footstep.t_acc_impact, 0.17928429140015903)

    def test_get_footstep_acceleration(self):
        """
        Tests the `get_footstep_acceleration` method
        """
        t_before = -1
        t_up = 0.1
        t_dec = 0.5
        t_delay = 0.7
        t_down = 0.8
        t_impact = 1
        t_after = 5

        acc_before = self.footstep.get_acceleration(t_before)
        acc_up = self.footstep.get_acceleration(t_up)
        acc_dec = self.footstep.get_acceleration(t_dec)
        acc_delay = self.footstep.get_acceleration(t_delay)
        acc_down = self.footstep.get_acceleration(t_down)
        acc_impact = self.footstep.get_acceleration(t_impact)
        acc_after = self.footstep.get_acceleration(t_after)

        self.assertEqual(acc_before, 0)
        self.assertEqual(acc_up, 1)
        self.assertEqual(acc_dec, -2)
        self.assertEqual(acc_delay, 0)
        self.assertEqual(acc_down, -3)
        self.assertEqual(acc_impact, 4)
        self.assertEqual(acc_after, 0)
