import unittest
import pandas as pd
from pandas.testing import assert_frame_equal
import os
# Local imports
from unified_model.mechanical_system.spring.utils import get_model_function, read_raw_file
from unified_model.tests.mechanical_system.test_data import TEST_RAW_CSV_FILE_PATH


class TestMagneticSpringUtils(unittest.TestCase):
    """
    Tests the magnetic spring utility functions.
    """

    def test_get_model_function(self):
        """
        Tests the get_model_function
        """
        test_model_dict = {'model_a': 'model_object',
                           'model_b': 'a_different_model_object'}

        model_object = get_model_function(test_model_dict, 'model_a')
        another_model_object = get_model_function(test_model_dict, 'model_b')
        self.assertEqual(model_object, 'model_object')
        self.assertEqual(another_model_object, 'a_different_model_object')

        with self.assertRaises(KeyError):
            get_model_function(test_model_dict, 'not_in_dictionary')

    def test_read_raw_file(self):
        """
        Tests the read_raw_file function
        """
        path = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(path, '../test_data/test_raw_file.csv')
        test_df = read_raw_file(TEST_RAW_CSV_FILE_PATH, z_unit=None)

        expected_dataframe = pd.DataFrame()
        expected_dataframe['z'] = [1, 2, 3, 4, 5]
        expected_dataframe['force'] = [1, 4, 9, 16, 25]

        assert_frame_equal(test_df, expected_dataframe)
