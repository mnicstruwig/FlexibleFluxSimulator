import os

import numpy as np
import pandas as pd
from numba import jit

from ...utils.utils import smooth_savgol, FastInterpolator


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


def _preprocess_acceleration_dataframe(
    df, accel_column, time_column, accel_unit, time_unit, smooth=True
):
    """Pre-process the raw acceleration dataframe.

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
    time_conversion_table = {"us": 1 / 1000000, "ms": 1 / 1000, "s": 1}

    df["simulation_time_seconds"] = df[time_column] * time_conversion_table[time_unit]

    if smooth:
        df[accel_column] = smooth_savgol(
            df[accel_column], window_length=101, polyorder=2
        )

    if accel_unit not in ["g", "ms2"]:
        raise KeyError('Acceleration unit must be specified as "g" or "ms2".')
    else:
        # subtract gravity, convert to m/s^2
        if accel_unit == "g":
            df[accel_column] = (df[accel_column] - 1) * 9.81
        if accel_unit == "ms2":
            df[accel_column] = df[accel_column] - 9.81

    return df


class AccelerometerInput:
    """Provide custom accelerometer input from a file or dataframe.

    Attributes
    ----------
    smooth : bool
        True if the the accelerometer input has been smoothed.
    interpolate : bool
        True if the accelerometer input is available as an interpolation.
    interpolator : UnivariateSpline
        If `interpolate` is True, this holds the interpolator object of the
        accelerometer readings. None otherwise.
    acceleration_df : dataframe
        The processed accelerometer readings in dataframe format.

    """

    def __init__(
        self,
        raw_accelerometer_data_path,
        accel_column,
        time_column,
        accel_unit="g",
        time_unit="ms",
        smooth=True,
        interpolate=False,
    ):
        """Constructor

        Parameters
        ----------
        raw_accelerometer_data_path : str
            Path to a CSV file containing the accelerometer data.
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
        interpolate : bool, optional
            Whether to model the accelerometer input as an interpolator object.
            The interpolation object is available under `self.interpolator`.

        """
        self.raw_accelerometer_data_path = raw_accelerometer_data_path
        self.acceleration_df = pd.read_csv(self.raw_accelerometer_data_path)
        self.accel_column = accel_column
        self.time_column = time_column
        self.accel_unit = accel_unit
        self.time_unit = time_unit
        self.smooth = smooth
        self.interpolate = interpolate
        self.interpolator = None

        # TODO: Separate processed accelerometer from raw accelerometer
        self.acceleration_df = _preprocess_acceleration_dataframe(
            self.acceleration_df,
            self.accel_column,
            self.time_column,
            self.accel_unit,
            self.time_unit,
            self.smooth,
        )

        if self.interpolate:
            self.interpolator = FastInterpolator(
                self.acceleration_df["simulation_time_seconds"].values,
                self.acceleration_df[self.accel_column].values,
            )

    def to_json(self):
        return {
            "raw_accelerometer_data_path": os.path.abspath(
                self.raw_accelerometer_data_path
            ),
            "accel_column": self.accel_column,
            "time_column": self.time_column,
            "accel_unit": self.accel_unit,
            "time_unit": self.time_unit,
            "smooth": self.smooth,
            "interpolate": self.interpolate,
        }

    # TODO: Add over attributes
    def __str__(self) -> str:
        """Return string representation of the Accelerometer"""
        return f"AccelerometerInput('{self.raw_accelerometer_data_path}')"

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
            return self.interpolator.get(t)

        return _find_nearest_acc_value(
            t,
            self.acceleration_df["simulation_time_seconds"].values,
            self.acceleration_df[self.accel_column].values,
        )
