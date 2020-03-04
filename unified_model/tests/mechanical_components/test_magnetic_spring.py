import unittest
from mockito import when, ANY, mock

import pandas as pd
from pandas.testing import assert_frame_equal
from scipy.interpolate import interp1d

from unified_model.mechanical_components.spring import magnetic_spring

class TestMagnetSpringPrivateMethods:
    '''Test the private methods of the magnetic_spring module'''
    def test_preprocess(self):
        '''Test the _preprocess helper.'''
        test_dataframe = pd.DataFrame({
            'z': [1,2,3,4,5],
            'force': [1,2,3,4,5]
        })

        expected_result = test_dataframe.copy()
        expected_result['force'] = 1

        actual_result = magnetic_spring._preprocess(
            test_dataframe,
            filter_obj=lambda x: [1]*len(x)
        )

        assert_frame_equal(expected_result, actual_result)


class TestMagneticSpringInterp:
    """Test the MagneticSpringInterp class."""

    def test_fit_model_default_model_kwargs(self):
        '''Test the _fit_model helper for no passed model kwargs'''

        test_fea_data_file = 'path/to/file'
        test_fea_dataframe = pd.DataFrame({
            'z': [1,2,3,4,5],
            'force': [1,2,3,4,5]
        })
        when(pd).read_csv(test_fea_data_file).thenReturn(test_fea_dataframe)
        x = mock(
            {'_fit_model': magnetic_spring.MagneticSpringInterp._fit_model},
            spec=magnetic_spring.MagneticSpringInterp
        )

        actual_result = x._fit_model(test_fea_dataframe)
        assert actual_result.fill_value == 0
        assert actual_result.bounds_error is False
        assert isinstance(actual_result, interp1d)

    def test_fit_model_set_model_kwargs(self):
        '''Test the _fit_model helper for passed model kwargs'''

        test_fea_data_file = 'path/to/file'
        test_fea_dataframe = pd.DataFrame({
            'z': [1,2,3,4,5],
            'force': [1,2,3,4,5]
        })
        test_model_kwargs = {'fill_value': 99, 'bounds_error': True}
        when(pd).read_csv(test_fea_data_file).thenReturn(test_fea_dataframe)
        x = mock(
            {'_fit_model': magnetic_spring.MagneticSpringInterp._fit_model},
            spec=magnetic_spring.MagneticSpringInterp
        )

        actual_result = x._fit_model(test_fea_dataframe, **test_model_kwargs)
        assert actual_result.fill_value == 99
        assert actual_result.bounds_error is True
        assert isinstance(actual_result, interp1d)
