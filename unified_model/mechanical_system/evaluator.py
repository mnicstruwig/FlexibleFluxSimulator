import numpy as np
import peakutils
import matplotlib.pyplot as plt
import warnings


class LabeledVideoProcessor(object):
    """Post-processor labeled magnet-assembly data.

    Attributes
    ----------
    L : float
        Height of the microgenerator tube in mm.
    mf : float
        Height of the fixed magnet at the bottom of the tube in mm.
    mm : float
        Height of the moving magnet assembly in mm.
    seconds_per_frame : float
        Number of seconds per frame / datapoint.
        This is typically found in the `subsampled_seconds_per_frame` key
        of the generated .DONE file when using the OpenCV-based CLI
        helper script.

    """

    def __init__(self, L, mf, mm, seconds_per_frame):
        """Constructor.

        Parameters
        ----------
        L : float
            Height of the microgenerator tube in mm.
        mf : float
            Height of the fixed magnet at the bottom of the tube in mm.
        mm : float
            Height of the moving magnet assembly in mm.
        seconds_per_frame : float
            Number of seconds per frame / datapoint.
            This is typically found in the `subsampled_seconds_per_frame` key
            of the generated .DONE file when using the OpenCV-based CLI
            helper script.

        """
        self.L = L
        self.mm = mm
        self.spf = seconds_per_frame
        self.processed_dataframe = None

    def fit_transform(self, groundtruth_dataframe, impute_missing_values=False, pixel_scale=None):
        """
        Process and transform the `groundtruth_dataframe` that is generated by
        the OpenCV-based CLI helper script and return the position of the moving
        magnet assembly in mm, relative to the top of the fixed magnet.

        Parameters
        ----------
        groundtruth_dataframe : dataframe
            Dataframe generated by the OpenCV-based CLI helper script.
        pixel_scale : float, optional
            Manually specify the pixel scale to use (in mm per pixel).

        Returns
        -------
        numpy array
            Position of the bottom of the magnet assembly relative to the
            top of the fixed magnet.

        numpy array
            Timestamp of each position.

        """
        df = groundtruth_dataframe

        # TODO: Add test for this case.
        # check if the pixel-scale has been defined
        if pixel_scale is None:
            missing_pixel_scale = np.any(df['y_pixel_scale'] == -1)
            if missing_pixel_scale:
                raise ValueError('Dataframe contains missing pixel scale values and the pixel scale is not been '
                                 'manually specified.')
        else:
            df['y_pixel_scale'] = pixel_scale

        df['y'] = np.abs(df['end_y'] - df['start_y'])  # Get position metric
        df['y_mm'] = df['y'] * df['y_pixel_scale']  # Adjust for pixel scale
#        df['y_prime_mm'] = self.L - self.mf - df['y_mm']  # Get actual position
        df['y_prime_mm'] = df['y_mm']  # Get actual position
        df.loc[df['top_of_magnet'] == 1, 'y_prime_mm'] = df['y_prime_mm'] - self.mm

        if impute_missing_values:
            missing_indexes = df.query('start_y == -1').index.values
            df = impute_missing(df, missing_indexes)

        self.processed_dataframe = df
        timestamps = np.arange(0, round(len(df)*self.spf, 8), self.spf)

        return df['y_prime_mm'].values/1000, timestamps


# TODO: Add docs
# TODO: Add tests
def impute_missing(df_missing, indexes):
    for index in indexes:
        start_velocity_calc = index - 2
        end_velocity_calc = index - 1

        # sanity check
        if df_missing.loc[start_velocity_calc, 'start_y'] == -1 or df_missing.loc[end_velocity_calc, 'start_y'] == -1:
            warnings.warn('Warning: unable to impute all missing values.')
            break

        velocity = df_missing.loc[end_velocity_calc, 'y_prime_mm'] - df_missing.loc[start_velocity_calc, 'y_prime_mm']
        df_missing.loc[index, 'y_prime_mm'] = df_missing.loc[index - 1, 'y_prime_mm'] + velocity
    return df_missing


# TODO : Add documentation to member functions
class MechanicalSystemEvaluator(object):
    """
    Evaluate the accuracy of the mechanical system model

    Attributes
    ----------
    y_target : array_like
        The target.
    time_target : array_like
        The timestamps for each element in `y_target`.
    y_predicted : array_like, optional
        The predicted values of the target.
    time_predicted : array_like, optional
        The timestamps for each element in `y_predicted`.
    """

    def __init__(self, y_target, time_target, y_predicted=None, time_predicted=None):
        if len(y_target) != len(time_target):
            raise ValueError('`y_target` and `time_target` must be equal in length.')
        self.y_target = y_target
        self.time_target = time_target
        self.y_predicted = y_predicted if not None else None
        self.time_predicted = time_predicted if not None else None

    def fit(self, y_predicted, time_predicted):
        self._fit(y_predicted, time_predicted)

    def _fit(self, y_predicted, time_predicted):
        self.y_predicted = y_predicted
        self.time_predicted = time_predicted

        peak_idx_target = self._find_peak_index(self.y_target)
        peak_idx_predic = self._find_peak_index(self.y_predicted)
        peak_time_target = self.time_target[peak_idx_target]
        peak_time_predic = self.time_predicted[peak_idx_predic]
        time_shift = peak_time_target - peak_time_predic

        self.time_predicted = self.time_predicted + time_shift

    def fit_transform(self, y_predicted, time_predicted):
        self._fit(y_predicted, time_predicted)
        return self.time_predicted, self.y_predicted

    def _find_peak_index(self, y, thres=0.95):
        peak_idx = peakutils.indexes(y, thres=thres)
        return peak_idx[0]

    def _find_closest_value(self, value, arr):
        temp = arr - value
        return arr[np.abs(temp).argmin]

    def score(self, y_predicted):
        """Score `y_predicted`."""
        pass

    # TODO: Use something like Plotnine
    def poof(self, **kwargs):
        """
        Plot y_target and y_predicted
        """
        plt.plot(self.time_target, self.y_target, label='Target')
        plt.plot(self.time_predicted, self.y_predicted, label='Prediction')
        plt.legend()
        plt.show()
