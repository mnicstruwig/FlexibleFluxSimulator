import unittest
import pandas as pd
from pandas.testing import assert_frame_equal

# Local imports
from unified_model.mechanical_system.input_excitation.accelerometer import _find_nearest_acc_value, _parse_raw_accelerometer_input, _preprocess_acceleration_dataframe, AccelerometerInput  # noqa


class TestAccelerometerInput(unittest.TestCase):
    """
    Tests for the `AccelerometerInput` class and helper functions.
    """
    def setUp(self):
        """
        Set-up.
        """
        self.test_accelerometer_file_path = '../test_data/test_accelerometer_file.csv'
        self.test_accelerometer_df = pd.DataFrame({'time': [1., 2., 3., 4., 5., 6],
                                                   'z_G': [0, 1, 0, 1, -2, 0]})

        self.test_accel_column = 'z_G'
        self.test_time_column = 'time'

    def test_find_nearest_acc_value_exact_value(self):
        """
        Test that the nearest acceleration value for a given timestamp is found
        for the case where the timestamp exactly matches the acceleration
        dataframe.
        """
        expected_acceleration = -2
        test_time = 5
        returned_acceleration = _find_nearest_acc_value(test_time,
                                                        self.test_time_column,
                                                        self.test_accel_column,
                                                        self.test_accelerometer_df)

        self.assertEqual(expected_acceleration, returned_acceleration)

    def test_find_nearest_acc_value_larger_value(self):
        """
        Test that the nearest acceleration value for a given timestamp is found
        for the case where the timestamp is slightly larger than a timestamp in
        the acceleration dataframe.
        """
        expected_acceleration = -2
        test_time = 5.1
        returned_acceleration = _find_nearest_acc_value(test_time,
                                                        self.test_time_column,
                                                        self.test_accel_column,
                                                        self.test_accelerometer_df)

        self.assertEqual(expected_acceleration, returned_acceleration)

    def test_find_nearest_acc_value_smaller_value(self):
        """
        Test that the nearest acceleration value for a given timestamp is found
        for the case where the timestamp is slightly smaller than a timestamp
        in the acceleration dataframe.
        """
        expected_acceleration = -2
        test_time = 4.9
        returned_acceleration = _find_nearest_acc_value(test_time,
                                                        self.test_time_column,
                                                        self.test_accel_column,
                                                        self.test_accelerometer_df)

        self.assertEqual(expected_acceleration, returned_acceleration)

    def test_parse_raw_accelerometer_input_df(self):
        """
        Test that the accelerometer input is parsed correctly if it is a
        dataframe.
        """

        expected_accelerometer_input = self.test_accelerometer_df
        actual_accelerometer_input = _parse_raw_accelerometer_input(expected_accelerometer_input)

        assert_frame_equal(expected_accelerometer_input, actual_accelerometer_input)

    def test_parse_raw_accelerometer_input_file(self):
        """
        Test that the accelerometer input is parsed correctly if it is a
        filename.
        """
        test_accelerometer_file = self.test_accelerometer_file_path

        loaded_accelerometer_input = _parse_raw_accelerometer_input(test_accelerometer_file)

        self.assertTrue(isinstance(loaded_accelerometer_input, pd.DataFrame))

    def test_preprocess_acceleration_dataframe_time_in_sec(self):
        """
        Test that the accelerometer dataframe is preprocessed correctly.
        """
        expected_time_values = [1., 2., 3., 4., 5., 6.]
        processed_df = _preprocess_acceleration_dataframe(self.test_accelerometer_df,
                                                          self.test_time_column,
                                                          time_unit='s')
        actual_time_values = processed_df[self.test_time_column].values.tolist()

        self.assertEqual(expected_time_values, actual_time_values)

    def test_constructor_raw_dataframe(self):
        """
        Test the constructor of the `AccelerometerInput` class for a dataframe
        as input.
        """
        test_accel_input = AccelerometerInput(self.test_accelerometer_df,
                                              self.test_accel_column,
                                              self.test_time_column,
                                              time_unit='s')

        self.assertEqual(test_accel_input._accel_column, self.test_accel_column)
        self.assertEqual(test_accel_input._time_unit, 's')
        self.assertEqual(test_accel_input._time_column, self.test_time_column)
        self.assertTrue(isinstance(test_accel_input.acceleration_df, pd.DataFrame))

    def test_constructor_raw_file(self):
        """
        Test the constructor of the `AccelerometerInput` class for a file as input.
        """
        test_accel_input = AccelerometerInput(self.test_accelerometer_file_path,
                                              self.test_accel_column,
                                              self.test_time_column,
                                              time_unit='s')

        self.assertEqual(test_accel_input._accel_column, self.test_accel_column)
        self.assertEqual(test_accel_input._time_unit, 's')
        self.assertEqual(test_accel_input._time_column, self.test_time_column)
        self.assertTrue(isinstance(test_accel_input.acceleration_df, pd.DataFrame))

    def test_get_acceleration(self):
        """
        Test the `get_acceleration` method of the `AccelerometerInput` class.
        """
        expected_acceleration_a = -2
        expected_acceleration_b = 1
        test_accel_input = AccelerometerInput(self.test_accelerometer_file_path,
                                              self.test_accel_column,
                                              self.test_time_column,
                                              time_unit='s')

        self.assertEqual(test_accel_input.get_acceleration(5), expected_acceleration_a)
        self.assertEqual(test_accel_input.get_acceleration(5.4), expected_acceleration_a)
        self.assertEqual(test_accel_input.get_acceleration(4.7), expected_acceleration_a)
        self.assertEqual(test_accel_input.get_acceleration(4.2), expected_acceleration_b)
