from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import correlate, detrend
from scipy.interpolate import UnivariateSpline

from unified_model.utils.utils import (apply_scalar_functions,
                                       get_sample_delay, find_signal_limits,
                                       smooth_butterworth, warp_signals,
                                       interpolate_and_resample)


class AdcProcessor:
    """Post-processor for measured ADC data using Teensy logger.

    Attributes
    ----------
    voltage_divison_ratio : float
        The ratio of the voltage divider that was used when measuring a
        voltage using the ADC. Equal to (input voltage) / (measured voltage)
    smooth : bool
        Whether to smooth the ADC readings using a low-pass filter in order
        to remove some noise.
        Default is True.
    **smooth_kwargs :
        Keyword arguments passed to the smooth_butterworth function.

    """

    def __init__(self, voltage_division_ratio=1., smooth=True, **smooth_kwargs):
        """Constructor

        Parameters
        ----------
        voltage_divison_ratio : float
            The ratio of the voltage divider that was used when measuring a
            voltage using the ADC. Equal to (input voltage) / (measured voltage)
        smooth : bool
            Whether to smooth the ADC readings using a low-pass filter in order
            to remove some noise.
            Default is True.
        **smooth_kwargs :
            Keyword arguments passed to the smooth_butterworth function.

        See Also
        --------
        unified_model.utils.utils.smooth_butterworth : function
            Function used to perform the smoothing

        """
        self.voltage_division_ratio = voltage_division_ratio
        self.smooth = smooth
        self.smooth_kwargs = smooth_kwargs if not None else None

    def fit_transform(self,
                      groundtruth_dataframe,
                      voltage_col='V',
                      time_col='time(ms)'):
        """
        Extract and transform the voltage in pandas dataframe
        `groundtruth_dataframe` into a clean signal and normalize to the correct
        voltage.

        Parameters
        ----------
        groundtruth_dataframe : pandas dataframe
            Dataframe generated by reading in the .csv file generated by
            parsing the log files on the SD card of the data logger.
        voltage_col : string
            Column in `groundtruth_dataframe` containing the voltage readings.
        time_col : string
            Column in `groundtruth_dataframe` containing the timestamps of the
            voltage readings. Note: this is not used for any calculations.

        Returns
        -------
        numpy array
            Timestamps of each voltage measurement.

        numpy array
            True measured output voltage.
        """
        if isinstance(groundtruth_dataframe, str):
            groundtruth_dataframe = pd.read_csv(groundtruth_dataframe)

        voltage_readings = groundtruth_dataframe[voltage_col].values

        if self.smooth:
            critical_frequency = self.smooth_kwargs.pop('critical_frequency', 1/5)
            self.critical_frequency = critical_frequency
            voltage_readings = smooth_butterworth(voltage_readings,
                                                  critical_frequency)

        voltage_readings = detrend(voltage_readings)
        voltage_readings = voltage_readings * self.voltage_division_ratio
        return voltage_readings, groundtruth_dataframe[time_col].values / 1000


# TODO: Update attributes
# TODO: Write tests
class ElectricalSystemEvaluator:
    """Evaluate the accuracy of an electrical system model's output.

    Attributes
    ----------
    emf_target : ndarray
        The target or "groundtruth" emf values produced by the electrical
        system.
    time_target : ndarray
        The corresponding timestamps of `emf_target`.
    emf_predict : ndarray
        The predicted emf values produced by the electrical system.
    time_predict : ndarray
        The corresponding timestamps of `emf_predict`.
    emf_target_ : ndarray
        The resampled target emf values. This attribute is used for producing
        the final plot using the `poof` method.
    emf_predict_ : ndarray
        The resampled predicted emf values. This attribute is used for
        producing the final plot using the `poof` method.
    time_ : ndarray
        The corresponding, shared timestamps of `emf_target_` and
        `emf_predict_`. This attribute is used for producing the final plot
        using the `poof` method.

    """

    def __init__(self,
                 emf_target,
                 time_target,
                 warp=False,
                 clip_threshold=1e-4):
        """Constructor.

        Parameters
        ----------
        emf_target : ndarray
            The target EMF values that will serve as the groundtruth.
        time_target : ndarray
            The corresponding timestamps of `emf_target` that will serve
            as the groundtruth.
        warp : bool
            Set to True to score after dynamically time-warping the target and
            prediction signals. Default value is False.
        clip_threshold : float
            If greater than 0., clip the leading and trailing emf target
            samples that don't include signal information by taking a
            spectrogram. Default value is 1e-4.

        Additional Notes
        ----------------
        The `clip` argument is important, since comparing aggregate scores of
        the prediction and target signals (say, for example, the RMS) will
        yield inaccurate results if one of the signals contains a lot of noise
        (usually this is the `target` signal).

        Thus, if the target signal is significantly longer than the prediction
        signal, and contains noise, the aggregate value will be substantially
        elevated due to all the additional noise.

        Just be careful not to set the value too high, which will result in
        clipping of the *actual* signal.

        """

        self.warp = warp
        self.clip_threshold = clip_threshold

        # Holds original values
        self.emf_target = emf_target
        self.time_target = time_target
        self.emf_predict = None
        self.time_predict = None

        # Holds resampled values
        # Trailing underscores indicate resampled values were used.
        self.emf_target_ = None
        self.emf_predict_ = None
        self.time_ = None

        # Holds clipped resampled values (using spectrogram)
        self.time_clipped_ = None
        self.emf_target_clipped_ = None
        self.emf_predict_clipped_ = None

        # Holds dynamic time-warped values
        self.emf_target_warped_ = None
        self.emf_predict_warped_ = None

    def fit(self,
            emf_predict,
            time_predict):
        """Align `emf_predict` with `emf_target` in time.

        This allows the target and prediction to later be compared by plotting
        them on the same time axis.

        The alignment is performed using signal correlation and resampling of
        both the target and predicted emf waveforms using interpolation.

        Note: As a result of the interpolation method being a univariate spline,
        if the emf signals contain discontinuities, this may cause some
        interpolation errors and low-frequency deviations from the original
        signals. This very rarely occurs with these kinds of signals, but
        it is something to be aware of if unexpected results are obtained.

        Parameters
        ----------
        emf_predict : ndarray
            The predicted emf values produced by the electrical system.
        time_predict : ndarray
            The corresponding time values assosciated with `emf_predicted`.

        """

        self._fit(emf_predict, time_predict, self.clip_threshold)

    def _fit(self,
             emf_predict,
             time_predict,
             clip_threshold):
        """Implement the functionality of the `fit` method."""
        self.emf_predict = emf_predict
        self.time_predict = time_predict

        # Normalize
        emf_predict = np.abs(emf_predict)

        # Resample (must calculate cross-correlation with same sampling rate!)
        stop_time = np.max([self.time_target[-1], time_predict[-1]])

        # Target
        resampled_time, resampled_emf_target = interpolate_and_resample(
            self.time_target,
            self.emf_target,
            new_x_range=(0, stop_time)
        )

        # Predicted
        _, resampled_emf_predicted = interpolate_and_resample(
            self.time_predict,
            self.emf_predict,
            new_x_range=(0, stop_time)
        )

        # Calculate sample delay
        sample_delay = get_sample_delay(resampled_emf_target,
                                        resampled_emf_predicted)

        # Remove delay between signals by interpolating again
        time_delay = resampled_time[sample_delay]
        _, resampled_emf_predicted = interpolate_and_resample(
            resampled_time - time_delay,
            resampled_emf_predicted,
            new_x_range=(0, stop_time)
        )

        self.emf_target_ = resampled_emf_target
        self.emf_predict_ = resampled_emf_predicted
        self.time_ = resampled_time
        self._clip_signals(clip_threshold)

    def fit_transform(self, emf_predict, time_predict):
        """Align `emf_predict` with `emf_target` in time and return the
        result.

        Parameters
        ----------
        emf_predict : ndarray
            The predicted emf values produced by the electrical system.
        time_predict : ndarray
            The corresponding time values assosciated with `emf_predicted`.

        Returns
        -------
        time_ : array
            Common timestamps of both the resampled emf predicted signal and
            emf target signal.
        emf_predict_ : array
            Resampled and interpolated values of the emf predicted signal values.

        See Also
        --------
        ElectricalSystemEvaluator.fit : function
            The underlying method that is called.

        """
        self._fit(emf_predict, time_predict)
        return self.time_, self.emf_predict_

    def _calc_dtw(self):
        """Perform dynamic time warping on prediction and targets."""

        # Exclude trailing (i.e. steady state) portion of the predicted waveform.
        self.emf_predict_warped_, self.emf_target_warped_ = warp_signals(self.emf_predict_,
                                                                         self.emf_target_)

    def score(self, **metrics):
        """Evaluate the electrical model using a selection of metrics.

        A `Score` object is returned containing the results.

        Parameters
        ----------
        **metrics : Metrics to compute on the interpolated predicted and target
        electrical data. Keys will be used to set the attributes of the Score
        object. Values must be the function used to compute the metric. Each
        function must accept arguments (arr_predict, arr_target) as input,
        where `arr_predict` and `arr_target` are numpy arrays that contain the
        predicted values and target values, respectively. The return value of
        the functions can have any shape.

        Returns
        -------
        Instance of `Score`
            Score object that contains the results of the computed metrics.
            Attributes names are the keys passed to `score`, and their values
            are the outputs of the passed metric functions.

        Example
        -------
        >>> emf_target = np.array([1, 2, 3, 4, 3, 2, 1])
        >>> time_target = np.array([1, 2, 3, 4, 5, 6, 7])
        >>> emf_predict = np.array([1, 2, 3, 4, 5, 2, 1])
        >>> time_predict = np.array([1, 2, 3, 4, 5, 6, 7])
        >>> es_evaluator = ElectricalSystemEvaluator(emf_target, time_target)
        >>> es_evaluator.fit(emf_predict, time_predict)
        Calculate the score using any function of your choice
        >>> es_evaluator.score(mean_difference=(lambda x, y: np.mean(x-y)), max_value=(lambda x,y: np.max([x, y])))
        Score(mean_difference=0.21169084032224028, max_value=5.078793981160988)

        """

        results = self._score(**metrics)
        return results

    def _clip_signals(self, clip_threshold):
        """Clip the target and predicted signals to only the active parts."""

        if clip_threshold == 0.:
            self.clipped_indexes = (None, None)
            self.time_clipped_ = self.time_
            self.emf_predict_clipped_ = self.emf_predict_
            self.emf_target_clipped_ = self.emf_target_
            return

        start_index, end_index = find_signal_limits(target=self.emf_predict_,
                                                    sampling_period=1,
                                                    threshold=clip_threshold)

        # Convert to integer indices, since `find_signal_limits` actually
        # returns the "time" of the signal, but we have a sampling frequency of
        # 1, so we can directly convert to integer indexes.
        start_index = int(start_index)
        end_index = int(end_index)
        self.clipped_indexes = (start_index, end_index)
        self.time_clipped_ = self.time_[start_index:end_index]
        self.emf_predict_clipped_ = self.emf_predict_[start_index:end_index]
        self.emf_target_clipped_ = self.emf_target_[start_index:end_index]

    def _score(self, **metrics):
        """Implement the underlying functionality of the `score` method."""

        if self.warp:
            self._calc_dtw()
            metric_results = apply_scalar_functions(self.emf_predict_warped_,
                                                    self.emf_target_warped_,
                                                    **metrics)
        else:
            metric_results = apply_scalar_functions(self.emf_predict_clipped_,
                                                    self.emf_target_clipped_,
                                                    **metrics)
        Results = namedtuple('Score', metric_results.keys())

        return Results(*metric_results.values())

    def poof(self, include_dtw=False, **kwargs):
        """Plot the aligned target and predicted values.

        Parameters
        ----------
        include_dtw : bool, optional
            Set to `True` to also plot the dynamic-time-warped signals.
            Default value is False.
        kwargs:
            Kwargs passed to matplotlib.pyplot.plot function.

        """
        plt.plot(self.time_clipped_, self.emf_target_clipped_, label='Target', **kwargs)
        plt.plot(self.time_clipped_, self.emf_predict_clipped_, label='Predictions', **kwargs)
        plt.legend()

        if include_dtw:
            if self.emf_target_warped_ is None:
                self._calc_dtw()
            plt.figure()
            plt.plot(self.emf_target_warped_, label='Target, time-warped')
            plt.plot(self.emf_predict_warped_, label='Predictions, time-warped')
            plt.legend()
        plt.show()


class LabeledVideoProcessor:
    """Post-processor for labeled magnet-assembly data.

    Attributes
    ----------
    L : float
        Height of the microgenerator tube in mm.
    mm : float
        Height of the moving magnet assembly in mm.
    seconds_per_frame : float
        Number of seconds per frame / datapoint.
        This is typically found in the `subsampled_seconds_per_frame` key
        of the generated .DONE file when using the OpenCV-based CLI
        helper script.
    pixel_scale : float, optional
        The pixel scale to use (in mm per pixel). This value will override
        the recorded pixel values in the groundtruth_dataframe.
        Default value is None.

    """

    def __init__(self, L, mm, seconds_per_frame, pixel_scale=None):
        """Constructor.

        Parameters
        ----------
        L : float
            Height of the microgenerator tube in mm.
        mm : float
            Height of the moving magnet assembly in mm.
        seconds_per_frame : float
            Number of seconds per frame / datapoint.
            This is typically found in the `subsampled_seconds_per_frame` key
            of the generated .DONE file when using the OpenCV-based CLI
            helper script.
        pixel_scale : float, optional
            The pixel scale to use (in mm per pixel). This value will override
            the recorded pixel values in the groundtruth_dataframe.
            Default value is None.

        """
        self.L = L
        self.mm = mm
        self.spf = seconds_per_frame
        self.pixel_scale = pixel_scale

    def fit_transform(self, groundtruth_dataframe, impute_missing_values=False):
        """
        Process and transform the `groundtruth_dataframe` that is generated by
        the OpenCV-based CLI helper script and return the position of the moving
        magnet assembly in mm, relative to the top of the fixed magnet.

        Parameters
        ----------
        groundtruth_dataframe : dataframe
            Dataframe generated by the OpenCV-based CLI helper script.
        impute_missing_values : bool
            If values were unlabelled (usually due to shaking of the camera),
            attempt to impute them by calculating the moving magnet's
            velocity at the previous time step.
            Default value is False.

        Returns
        -------
        mag_bottom_pos : array
            Position of the bottom of the magnet assembly relative to the
            top of the fixed magnet. Unit is metres.
        timestamps : array
            Timestamp of each position. Unit is seconds.

        """
        df = groundtruth_dataframe

        # TODO: Add test for this case.
        if self.pixel_scale is None:  # If we don't manually set pixel scale...
            if np.any(df['y_pixel_scale'] == -1):  # ... and it isn't in the parsed file.
                raise ValueError('Dataframe contains missing pixel scale values and the pixel scale is not been '
                                 'manually specified.')
        else:
            df['y_pixel_scale'] = self.pixel_scale

        df['y'] = np.abs(df['end_y'] - df['start_y'])  # Calculate position
        df['y_mm'] = df['y'] * df['y_pixel_scale']  # Adjust with pixel scale
        df['y_prime_mm'] = df['y_mm']  # Get actual position
        # Adjust for top / bottom of magnet during labeling process
        df.loc[df['top_of_magnet'] == 1, 'y_prime_mm'] = df['y_prime_mm'] - self.mm

        # Correct calculations made with missing values, if they exist.
        if impute_missing_values:
            missing_indexes = df.query('start_y == -1').index.values
            df = impute_missing(df, missing_indexes)

        timestamps = np.linspace(0, (len(df)-1)*self.spf, len(df))

        return df['y_prime_mm'].values / 1000, timestamps


def impute_missing(df_missing, indexes):
    """Impute missing values from the labeled video data.

    The missing values are imputed by calculating the velocity of the magnet
    assembly from the previous two timestamps and inferring a displacement
    based on that.

    Parameters
    ----------
    df_missing : dataframe
        Dataframe containing all the measurements, including missing
        measurements. Must at least contain columns `start_y`, which is used to
        indicate missing values, and column `y_prime_mm` which is the target
        column for the corrections.
    indexes : array
        Indexes in `df_missing` where there are missing values.


    Returns
    -------
    dataframe
        Updated dataframe with missing values replaced by imputed values.

    Raises
    ------
    ValueError
        If there are too many sequential missing values to be able to impute
        the missing values. This can typically occurs when two or more
        subsequent readings are missing.

    Notes
    -----
    The `start_y` column is used to determine whether there are skipped /
    missing values in `df_missing.` Values of -1 are considered "missing".
    Note, however, that the `y_prime_mm` column contains the actual target
    values, and so *this* is the column that contains the imputed missing
    values.

    The reason for this is that the `start_y` and `end_y` values are not
    relative to any fixed reference point, and so corrections must be made
    relative to a fixed reference point, which is the case for the values of
    `y_prime_mm`.

    """
    for index in indexes:
        start_velocity_calc = index - 2
        end_velocity_calc = index - 1

        # Check we have enough points to calculate velocity
        try:
            # `-1` indicates missing values. Check if our points used for
            # imputing are _also_ unlabelled.
            # TODO: Turn this check into a function w/ tests
            if df_missing.loc[end_velocity_calc, 'start_y'] == -1:
                raise ValueError('Too few many sequential missing values to be able to impute all missing values.')
            if df_missing.loc[start_velocity_calc, 'start_y'] == -1:
                start_velocity_calc = start_velocity_calc - 1
                if df_missing.loc[start_velocity_calc, 'start_y'] == -1:
                    raise ValueError('Too many sequential missing values to be able to impute all missing values.')
        except KeyError:
            raise IndexError('Too few points available to calculate velocity and impute missing values.')

        velocity = df_missing.loc[end_velocity_calc, 'y_prime_mm'] - df_missing.loc[start_velocity_calc, 'y_prime_mm']
        df_missing.loc[index, 'y_prime_mm'] = df_missing.loc[index - 1, 'y_prime_mm'] + velocity
    return df_missing


class MechanicalSystemEvaluator(object):
    """Evaluate the accuracy of a mechanical system model's output

    Attributes
    ----------
    y_target : ndarray
        The target.
    time_target : ndarray
        The timestamps for each element in `y_target`.
    y_predict : ndarray
        The predicted values of the target.
    time_predict : ndarray
        The timestamps for each element in `y_predict`.
    y_target_ : ndarray
        The resampled target values. This attribute is used for producing
        the final plot using the `poof` method.
    y_predict_ : ndarray
        The resampled predicted values. This attribute is used for producing
        the final plot using the `poof` method.
    time_ : ndarray
        The corresponding, shared, timestamps of `y_target_` and `y_predict_`.
        This attribute is used for producing the final plot using the `poof`
        method.
    y_target_warped_ : ndarray
        The dynamically time-warped points of the target values. Each point
        matches with the corresponding points in `self.y_predict_warped_`.
        Generated by calling the `score` method.
    y_predict_warped_ : ndarray
        The dynamically time-warped points of the predicted values. Each point
        matches with the corresponding points in `self.y_target_warped_`.
        Generated by calling the `score` method.

    """

    def __init__(self, y_target, time_target, warp=False):
        """Constructor

        Parameters
        ----------
        y_target : ndarray
            The target values that the mechanical system model is
            attempting to predict. This is the "ground truth" that the
            prediction will be compared against.
        time_target : ndarray
            The corresponding timestamps of the values in `y_target`
        warp : bool
            Set to True to score after dynamically time-warping the target and
            prediction signals. Default value is False.

        """
        self.warp = warp

        if len(y_target) != len(time_target):
            raise ValueError('`y_target` and `time_target` must be equal in length.')

        # Holds original values
        self.y_target = y_target
        self.time_target = time_target
        self.y_predict = None
        self.time_predict = None

        # Holds resampled values
        self.y_target_ = None
        self.y_predict_ = None
        self.time_ = None

        # Holds dynamic time-warped values
        self.y_target_warped_ = None
        self.y_predict_warped_ = None

    def fit(self, y_predict, time_predict):
        """Align `y_predicted` and `y_target` in time.

        This allows the target and prediction to later be plotted overlaying
        one another.

        Parameters
        ----------
        y_predict : ndarray
            The predicted values from the mechanical system model.
        time_predict : ndarray
            The corresponding timestamps of the values in `y_predicted`.

        """
        self._fit(y_predict, time_predict)

    def _fit(self, y_predict, time_predict):
        """Execute routine called with by the `fit` class method."""

        self.y_predict = y_predict
        self.time_predict = time_predict

        # Resample the signals to the same sampling frequency
        # This is required for calculating the sample delay between them.
        stop_time = np.max([self.time_target[-1], time_predict[-1]])
        self._clip_time = np.min([self.time_target[-1], time_predict[-1]])

        resampled_time, resampled_y_target = interpolate_and_resample(
            self.time_target,
            self.y_target,
            new_x_range=(0, stop_time)
        )

        _, resampled_y_predicted = interpolate_and_resample(
            time_predict,
            y_predict,
            new_x_range=(0, stop_time)
        )

        # Calculate the sample delay
        sample_delay = get_sample_delay(resampled_y_target,
                                        resampled_y_predicted)

        # Remove the delay between the signals
        time_delay = resampled_time[sample_delay]
        _, resampled_y_predicted = interpolate_and_resample(
            resampled_time - time_delay,
            resampled_y_predicted,
            new_x_range=(0, stop_time)
        )

        # Clip signals
        clip_index = np.argmin(np.abs(resampled_time - self._clip_time))

        # Store results
        self.y_target_ = resampled_y_target
        self.y_predict_ = resampled_y_predicted
        self.time_ = resampled_time

    def _calc_dtw(self):
        """Perform dynamic time warping on prediction and targets."""

        # Exclude trailing (i.e. steady state) portion of the predicted waveform.
        clip_index = np.argmin(np.abs(self.time_ - self._clip_time))
        self.y_predict_warped_, self.y_target_warped_ = warp_signals(self.y_predict_[:clip_index],
                                                                     self.y_target_[:clip_index])

    def fit_transform(self, y_predict, time_predict):
        """Align `y_predicted` and `y_target` in time.

        This allows the target and prediction to later be plotted overlaying
        one another.

        The `fit_transform` class method returns the `y_predicted` values with
        its shifted timestamps.

        Parameters
        ----------
        y_predict : ndarray, optional
            The predicted values from the mechanical system model.
        time_predict : ndarray, optional
            The corresponding timestamps of the values in `y_predicted`.

        Returns
        -------
        time_ : array
            Common timestamps of both the resampled y prediction and
            y target values.
        y_predict_ : array
            Resampled and interpolated values of the y predicted signal values

        See Also
        --------
        MechanicalSystemEvaluator.fit : function
            The underlying method that is called.

        """
        self._fit(y_predict, time_predict)
        return self.time_, self.y_predict_

    def score(self, **metrics):
        """Evaluate the mechanical model using a selection of metrics.

        A `Score` object is returned containing the results.

        Parameters
        ----------
        **metrics :
            Metrics to compute on the interpolated predicted and target
            electrical data. Keys will be used to set the attributes of the
            Score object. Values must be the function used to compute the
            metric. Each function must accept arguments (arr_predict,
            arr_target) as input, where `arr_predict` and `arr_target` are
            numpy arrays that contain the predicted values and target values,
            respectively. The return value of the functions can have any shape.

        Returns
        -------
        Instance of `Score`
            Score object that contains the results of the computed metrics.
            Attributes names are the keys passed to `score`, and their values
            are the outputs of the passed metric functions.

        Example
        -------
        >>> y_target = np.array([1, 2, 3, 4, 3, 2, 1])
        >>> time_target = np.array([1, 2, 3, 4, 5, 6, 7])
        >>> y_predict = np.array([1, 2, 3, 4, 5, 2, 1])
        >>> time_predict = np.array([1, 2, 3, 4, 5, 6, 7])
        >>> ms_evaluator = MechanicalSystemEvaluator(y_target, time_target)
        >>> ms_evaluator.fit(y_predict, time_predict)
        Calculate the score using any function of your choice
        >>> ms_evaluator.score(mean_difference=(lambda x, y: np.mean(x-y)),
        ... max_value=(lambda x,y: np.max([x, y])))
        Score(mean_difference=0.05925911092493251, max_value=5.07879398116099)

        """
        results = self._score(**metrics)
        return results

    def _score(self, **metrics):
        """Calculate the score of the predicted y values."""

        if self.warp:
            self._calc_dtw()
            metric_results = apply_scalar_functions(self.y_predict_warped_,
                                                    self.y_target_warped_,
                                                    **metrics)
        else:
            metric_results = apply_scalar_functions(self.y_predict_,
                                                    self.y_target_,
                                                    **metrics)
        # Output "Score" class
        Results = namedtuple('Score', metric_results.keys())

        return Results(*metric_results.values())

    def poof(self, include_dtw=False, **kwargs):
        """Plot the aligned target and predicted values.

        Parameters
        ----------
        include_dtw : bool, optional
            Set to `True` to also plot the dynamic-time-warped signals.
            Default value is False.
        kwargs:
            Kwargs passed to matplotlib.pyplot.plot function.

        """
        plt.plot(self.time_, self.y_target_, label='Target', **kwargs)
        plt.plot(self.time_, self.y_predict_, label='Prediction', **kwargs)
        plt.legend()

        if include_dtw:
            if self.y_predict_warped_ is None:
                self._calc_dtw()

            plt.figure()
            plt.plot(self.y_target_warped_, label='Target, time-warped')
            plt.plot(self.y_predict_warped_, label='Prediction, time-warped')
            plt.legend()

        plt.show()
