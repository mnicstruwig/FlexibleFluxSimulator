import numpy as np
import pandas as pd
from numba import jit
from scipy.signal import savgol_filter

import warnings

@jit(nopython=True)
def _find_nearest_acc_value(t, simulation_time_arr, accel_arr):
    """
    Return the acceleration in `accel_column` corresponding with the closest
    value of `t` in `simulation_time_column` within the dataframe `df`.
    :param t:
    :param simulation_time_column:
    :param accel_column:
    :param df:
    :return:
    """
    # TODO: Add sanity check when t is out of bounds.
    index = np.abs(simulation_time_arr - t).argmin()
    acceleration = accel_arr[index]
    return acceleration


def _parse_raw_accelerometer_input(raw_accelerometer_input):
    """
    Parse `raw_accelerometer_input` correctly if it is a string specifying a
    .csv file or a passed dataframe.
    """
    if isinstance(raw_accelerometer_input, str):
        return pd.read_csv(raw_accelerometer_input)
    if isinstance(raw_accelerometer_input, pd.DataFrame):
        return raw_accelerometer_input
    return None

# TODO: Add test
def _smooth(values, **kwargs):
    """Smooth `values` using a savgol filter"""
    try:
        if 'window_length' in kwargs and 'polyorder' in kwargs:
            return savgol_filter(values, **kwargs)
        return savgol_filter(values, 101, 2)
    except ValueError:
        warnings.warn('Filter window length exceeds signal length. No filtering is being applied.', RuntimeWarning)
        return values


def _preprocess_acceleration_dataframe(df, accel_column, time_column, accel_unit, time_unit, smooth=True):
    """
    Perform pre-processing operations on the raw acceleration dataframe.
    """
    time_conversion_table = {'us': 1 / 1000000,
                             'ms': 1 / 1000,
                             's': 1}

    df['simulation_time_seconds'] = df[time_column] * time_conversion_table[time_unit]

    if accel_unit not in ['g', 'ms2']:
        raise KeyError('Acceleration unit must be specified as "g" or "ms2".')
    else:
        if accel_unit is 'g':
            df[accel_column] = (df[accel_column]-1)*9.81
        if accel_unit is 'ms2':
            df[accel_column] = df[accel_column] - 9.81

    if smooth:
        df[accel_column] = _smooth(df[accel_column])  # Apply smoothing filter

    return df


# TODO: Reformat docstring
class AccelerometerInput(object):
    """
    Provide custom accelerometer input from a file or dataframe.
    """
    def __init__(self, raw_accelerometer_input, accel_column, time_column, accel_unit='g', time_unit='ms', smooth=True):
        """
        Constructor.

        :param raw_accelerometer_input:
        :param accel_column:
        :param time_column:
        :param time_unit:
        """
        self._accel_column = accel_column
        self._time_column = time_column
        self._accel_unit = accel_unit
        self._time_unit = time_unit
        self.smooth = smooth

        self.acceleration_df = _parse_raw_accelerometer_input(raw_accelerometer_input)
        self.acceleration_df = _preprocess_acceleration_dataframe(self.acceleration_df,
                                                                  self._accel_column,
                                                                  self._time_column,
                                                                  self._accel_unit,
                                                                  self._time_unit,
                                                                  self.smooth)

    def get_acceleration(self, t):
        """
        Get the acceleration at time `t`.
        :param t:
        :return:
        """
        accel = _find_nearest_acc_value(t,
                                        self.acceleration_df['simulation_time_seconds'].values,
                                        self.acceleration_df[self._accel_column].values)

        return accel
