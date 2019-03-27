import numpy as np
import pandas as pd
from numba import jit

from unified_model.utils.utils import smooth_savgol
from scipy.interpolate import UnivariateSpline


@jit(nopython=True)
def _find_nearest_acc_value(t, time_arr, accel_arr):
    """Find the nearest acceleration value at time `t` from discrete values.

    Return the acceleration in `accel_column` corresponding with the closest
    value of `t` in `simulation_time_column` within the dataframe `df`.

    Parameters
    ----------
    t : float
        Simulation time.
        This is the point of time where the acceleration value will be looked up
    time_arr : array_like
        Time-array containing the corresponding time-values to `accel_arr`
    accel_arr : array_like
        Acceleration-array containing accelerometer readings corresponding to
        `time_arr`

    Returns
    -------
    float
        Nearest acceleration to time `t`.

    """
    # TODO: Add sanity check when t is out of bounds.
    index = np.abs(time_arr - t).argmin()
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


def _preprocess_acceleration_dataframe(df,
                                       accel_column,
                                       time_column,
                                       accel_unit,
                                       time_unit,
                                       smooth=True):
    """Perform pre-processing operations on the raw acceleration dataframe.

    This includes converting the time to seconds, converting the acceleration
    to the m/s^2 and removing gravity, and smooth the acceleration curve in
    order to remove excessive noise.

    Parameters
    ----------
    df : pandas dataframe
        Dataframe containing an accelerometer column and a time column.
    accel_column : str
        Column in `df` containing the accelerometer values.
    time_column : str
        Column in `df` containing the time values.
    accel_unit : {'g', 'ms2'}
        Unit of the accelerometer values in `accel_column`
    time_unit : {'us', 'ms', 's}
        Unit of the time values in `time_column`.
    smooth : bool, optional
        Whether to smooth the accelerometer readings.
        Default is True.

    Returns
    -------
    pandas dataframe
        Dataframe with processed time and acceleration columns.

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
        df[accel_column] = smooth_savgol(df[accel_column])  # Apply smoothing filter

    return df


# TODO: Reformat docstring
# TODO: Include documentation for interpolation
class AccelerometerInput(object):
    """Provide custom accelerometer input from a file or dataframe.

    Parameters
    ----------
    raw_accelerometer_input : str or pandas dataframe
        Path to .csv file containing accelerometer data, or pandas
        dataframe containing accelerometer data.
    accel_column : str
        Column containing accelerometer values.
    time_column : str
        Column containing time values.
    accel_unit : {'g', 'ms2'}
        Unit of the accelerometer values in `accel_column`.
    time_unit : {'us', 'ms', 's'}
        Unit of the time values in `time_column`
    smooth : bool, optional
        Whether to smooth the accelerometer readings.
        Default is True.

    """

    def __init__(self,
                 raw_accelerometer_input,
                 accel_column,
                 time_column,
                 accel_unit='g',
                 time_unit='ms',
                 smooth=True,
                 interpolate=False):
        """Constructor"""

        self._accel_column = accel_column
        self._time_column = time_column
        self._accel_unit = accel_unit
        self._time_unit = time_unit
        self._smooth = smooth
        self.interpolate = interpolate
        self.interpolator = None

        self.acceleration_df = _parse_raw_accelerometer_input(raw_accelerometer_input)
        self.acceleration_df = _preprocess_acceleration_dataframe(self.acceleration_df,
                                                                  self._accel_column,
                                                                  self._time_column,
                                                                  self._accel_unit,
                                                                  self._time_unit,
                                                                  self._smooth)

        if self.interpolate:
            self.interpolator = UnivariateSpline(x=self.acceleration_df['simulation_time_seconds'].values,
                                                 y=self.acceleration_df[self._accel_column].values,
                                                 s=0,
                                                 ext='zeros')

    def get_acceleration(self, t):
        """Get the acceleration at time `t`.

        Parameters
        ----------
        t : float
            Time, in seconds.

        Returns
        -------
        float
            Acceleration at time `t` in m/s^2.

        """
        if self.interpolate:
            # if y[0] <0 and y[1] < 0:
            #     if self.interpolator(t) < 0:
            #         return -self.interpolator(t)
            #     elif self.interpolator(t) == 0:
            #         return 0.1
            return self.interpolator(t)

        return _find_nearest_acc_value(t,
                                       self.acceleration_df['simulation_time_seconds'].values,
                                       self.acceleration_df[self._accel_column].values)
