import numpy as np
import pandas as pd


def _find_nearest_acc_value(t, time_column, accel_column, df):
    """
    Return the acceleration in `accel_column` corresponding with the closest
    value of `t` in `time_column` within the dataframe `df`.
    :param t:
    :param time_column:
    :param accel_column:
    :param df:
    :return:
    """
    # TODO: Add sanity check when t is out of bounds.
    index = np.abs(df[time_column] - t).sort_values().index[0]
    acceleration = df.loc[index, accel_column]
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


def _preprocess_acceleration_dataframe(df, time_column, time_unit):
    """
    Perform pre-processing operations on the raw acceleration dataframe.
    """
    conversion_table = {'us': 1 / 1000000,
                        'ms': 1 / 1000,
                        's': 1}

    df[time_column] = df[time_column] * conversion_table[time_unit]
    return df


class AccelerometerInput(object):
    """
    Provide custom accelerometer input.
    """
    def __init__(self, raw_accelerometer_input, accel_column, time_column, time_unit='ms'):
        """
        Constructor.

        :param raw_accelerometer_input:
        :param accel_column:
        :param time_column:
        :param time_unit:
        """
        self._accel_column = accel_column
        self._time_unit = time_unit
        self._time_column = time_column

        self.acceleration_df = _parse_raw_accelerometer_input(raw_accelerometer_input)
        self.acceleration_df = _preprocess_acceleration_dataframe(self.acceleration_df,
                                                                  self._time_column,
                                                                  self._time_unit)

    def get_acceleration(self, t):
        """
        Get the acceleration at time `t`.
        :param t:
        :return:
        """
        return _find_nearest_acc_value(t, self._time_column, self._accel_column, self.acceleration_df)
