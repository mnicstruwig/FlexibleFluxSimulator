from typing import Any, Callable, Dict, Optional, Union, List, cast
from dataclasses import dataclass
from unified_model import mechanical_components

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.signal import savgol_filter

from unified_model.utils.utils import (align_signals_in_time,
                                       apply_scalar_functions,
                                       find_signal_limits, smooth_butterworth,
                                       warp_signals,
                                       Sample)


@dataclass
class MechanicalGroundtruth:
    y_diff: Any
    time: Any


@dataclass
class ElectricalGroundtruth:
    emf: Any
    time: Any


@dataclass
class Groundtruth:
    mech: MechanicalGroundtruth
    elec: ElectricalGroundtruth


class GroundTruthFactory:
    """Collect groundtruth data and prepare evaluators in a batch."""
    def __init__(self,
                 samples_list: np.ndarray,
                 lvp_kwargs: Dict,
                 adc_kwargs: Dict) -> None:
        """Constructor.

        Parameters
        ----------
        samples_list : np.ndarray[Sample]
            An array of `Sample` objects containing processed groundtruth data.
            See the `unified_model.utils.utils.collect_samples` function, which
            is intended to be used to collect and build `samples_list`.
        lvp_kwargs : Dict
            Kwargs used for the LabeledVideoProcessor objects to process the
            groundtruth data.
        adc_kwargs : Dict
            Kwargs used for the AdcProcessor objects to process the groundtruth data.

        """

        self.samples_list = samples_list
        self.lvp_kwargs = lvp_kwargs
        self.adc_kwargs = adc_kwargs

        self.lvp = LabeledVideoProcessor(**lvp_kwargs)
        self.adc = AdcProcessor(**adc_kwargs)

    def _make_mechanical_groundtruth(self, sample):
        y_target, y_time_target = self.lvp.fit_transform(
            sample.video_labels_df,
        )
        y_target = savgol_filter(y_target, 9, 3)

        return MechanicalGroundtruth(y_target,
                                     y_time_target)

    def _make_electrical_groundtruth(self, sample):
        emf_target, emf_time_target = self.adc.fit_transform(sample.adc_df)
        return ElectricalGroundtruth(emf_target,
                                     emf_time_target)

    def make(self) -> List[Groundtruth]:
        """Make the Groundtruth objects.

        Returns
        -------
        List[Groundtruth]
            List of `Groundtruth` objects for each sample in `samples_list`.
        """
        groundtruths = []
        for sample in self.samples_list:
            try:
                mech_groundtruth = self._make_mechanical_groundtruth(sample)
                elec_groundtruth = self._make_electrical_groundtruth(sample)

                groundtruths.append(
                    Groundtruth(mech_groundtruth, elec_groundtruth)
                )
            except AttributeError:
                pass

        return groundtruths


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

    def __init__(self,
                 voltage_division_ratio: float=1.,
                 smooth: bool=True,
                 **smooth_kwargs: dict) -> None:
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
            True measured output voltage.
        numpy array
            Timestamps of each voltage measurement.

        """
        if isinstance(groundtruth_dataframe, str):
            groundtruth_dataframe = cast(pd.DataFrame, pd.read_csv(groundtruth_dataframe))

        voltage_readings = groundtruth_dataframe[voltage_col].values

        if self.smooth:
            critical_frequency = self.smooth_kwargs.get('critical_frequency', 1/5)  # noqa
            self.critical_frequency = critical_frequency
            voltage_readings = smooth_butterworth(voltage_readings,
                                                  critical_frequency)

        voltage_readings = voltage_readings * self.voltage_division_ratio  # type: ignore
        voltage_readings = self._detrend(voltage_readings, (100, 700))
        return voltage_readings, groundtruth_dataframe[time_col].values / 1000

    @staticmethod
    def _detrend(x, noise_window):
        noise_mean = np.mean(x[noise_window[0]: noise_window[1]])
        return x - noise_mean


class MechanicalSystemEvaluator:
    """Evaluate the accuracy of a mechanical system model's output

    Parameters
    ----------
    y_target : ndarray
        The target values that the mechanical system model is
        attempting to predict. This is the "ground truth" that the
        prediction will be compared against.
    time_target : ndarray
        The corresponding timestamps of the values in `y_target`
    metrics : Dict[str, Callable]
        Metrics to compute on the interpolated predicted and target data.
        Keys determine the "name" of the metric and values must be a
        Callable that is used to compute the metric. Each Callable must
        accept arguments (arr_predict, arr_target) as input, where
        `arr_predict` and `arr_target` are numpy arrays that contain the
        predicted values and target values, respectively. The return value
        of the Callable can have any shape.
    clip : bool
        Whether to clip or trim the target and predicted arrays after
        alignment in order to restrict scoring to the "active" part of the
        signal.
    warp : bool
        Set to True to score after dynamically time-warping the target and
        prediction signals. Default value is False.

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

    def __init__(self,  # pylint: disable=too-many-arguments
                 y_target: np.ndarray,
                 time_target: np.ndarray,
                 metrics: Dict[str, Callable],
                 clip: bool = True,
                 warp: bool = False) -> None:
        """Constructor


        """
        self.metrics = metrics
        self.clip = clip
        self.warp = warp

        if len(y_target) != len(time_target):
            raise ValueError('`y_target` and `time_target` must be equal in length.')  # noqa

        # Holds original values
        self.y_target = y_target
        self.time_target = time_target
        self.y_predict = None
        self.time_predict = None

        # Holds resampled values
        self.y_target_: Union[None, np.ndarray] = None
        self.y_predict_: Union[None, np.ndarray] = None
        self.time_: Union[None, np.ndarray] = None

        # Holds dynamic time-warped values
        self.y_target_warped_: Union[None, np.ndarray] = None
        self.y_predict_warped_: Union[None, np.ndarray] = None

        # Clip parameters
        self._clip_time = None
        self._clip_index = None

    def fit(self, y_predict: np.ndarray, time_predict: np.ndarray) -> None:
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

    def _fit(self, y_predict: Any, time_predict: Any) -> None:
        """Execute routine called with by the `fit` class method."""

        self.y_predict = y_predict
        self.time_predict = time_predict

        # Get the end time we should clip the signal at (in case our target
        # data is shorter than our simulation time)

        resampled_signals = align_signals_in_time(
            t_1=self.time_target,
            y_1=self.y_target,
            t_2=self.time_predict,
            y_2=self.y_predict
        )

        resampled_time: np.ndarray = resampled_signals[0]
        resampled_y_target: np.ndarray = resampled_signals[1]
        resampled_y_predict: np.ndarray = resampled_signals[2]

        # Find where to clip the resampled signals
        if self.clip:
            self._clip_time = np.min([self.time_target[-1], self.time_predict[-1]])  # type: ignore
            self._clip_index = np.argmin(np.abs(resampled_time - self._clip_time)) - 1  # type: ignore
        else:
            self._clip_time = resampled_time[0]
            self._clip_index = len(resampled_time) - 1

        # Store results
        self.y_target_ = resampled_y_target
        self.y_predict_ = resampled_y_predict
        self.time_ = resampled_time

    def _calc_dtw(self):
        """Perform dynamic time warping on prediction and targets."""

        # Exclude trailing (i.e. steady state) portion of the predicted waveform
        self.y_predict_warped_, self.y_target_warped_ = warp_signals(
            self.y_predict_[self._clip_index],
            self.y_target_[self._clip_index]
        )

    def score(self) -> Dict[str, Any]:
        """Evaluate the mechanical model using a selection of metrics.

        Returns
        -------
        Dict[str, Any]
            Dict that contains the results of the computed metrics. Keys are
            the same as `metrics` passed in to the constructor. The values are
            the outputs of the passed metric functions.

        """
        results = self._score(**self.metrics)
        return results

    def _score(self, **metrics) -> Dict[str, Any]:
        """Calculate the score of the predicted y values."""

        metric_results = apply_scalar_functions(
            self.y_predict_[:self._clip_index],
            self.y_target_[:self._clip_index],
            **metrics)

        return metric_results

    def poof(self, include_dtw=False, **kwargs):
        """Plot the aligned target and predicted values.

        Parameters
        ----------
        include_dtw : bool, optional
            Set to `True` to also plot the dynamic-time-warped signals.
            Default value is False.
        kwargs:
            Kwargs passed to matplotlib.pyplot.figure function.

        """
        plt.figure(**kwargs)
        plt.plot(self.time_, self.y_target_, label='Target')
        plt.plot(self.time_, self.y_predict_, label='Prediction')

        # Show the clip marks
        plt.plot([0, 0], [0, max(self.y_target_)], 'k--')  # start
        plt.plot([self.time_[self._clip_index], self.time_[self._clip_index]],
                 [0, max(self.y_target_)],
                 'k--')  # end

        plt.legend()

        if include_dtw:
            if self.y_predict_warped_ is None:
                self._calc_dtw()

            plt.plot(self.y_target_warped_, label='Target, time-warped')
            plt.plot(self.y_predict_warped_, label='Prediction, time-warped')
            plt.legend()

        plt.show()


# TODO: Update attributes
# TODO: Write tests
class ElectricalSystemEvaluator:
    """Evaluate the accuracy of an electrical system model's output.

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
        samples that don't include signal information.
        Default value is 0.05

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

    def __init__(self,
                 emf_target: np.ndarray,
                 time_target: np.ndarray,
                 metrics: Dict[str, Callable],
                 warp: bool = False,
                 clip_threshold: float = 0.125) -> None:
        """Constructor.

        """

        self.metrics = metrics
        self.warp = warp
        self.clip_threshold = clip_threshold

        # Holds original values
        self.emf_target = emf_target
        self.time_target = time_target
        self.emf_predict = None
        self.time_predict = None

        # Holds resampled values
        # Trailing underscores indicate resampled values were used.
        self.emf_target_: np.ndarray = None
        self.emf_predict_: np.ndarray = None
        self.time_: np.ndarray = None

        # Holds clipped resampled values (using spectrogram)
        self.time_clipped_ = None
        self.emf_target_clipped_ = None
        self.emf_predict_clipped_ = None

        # Holds dynamic time-warped values
        self.emf_target_warped_ = None
        self.emf_predict_warped_ = None

    def _make_clipped_signals(self,
                              target,
                              predict) -> None:

        emf_predict_start, emf_predict_end = find_signal_limits(predict)
        emf_target_start, emf_target_end = find_signal_limits(target, self.clip_threshold) # 0.075)

        start_index = np.min([emf_predict_start, emf_target_start])
        end_index = np.max([emf_predict_end, emf_target_end])
        emf_predict_clipped_ = self.emf_predict_[start_index:end_index]
        emf_target_clipped_ = self.emf_target_[start_index:end_index]
        time_clipped_ = self.time_[start_index:end_index]

        self.emf_predict_clipped_ = emf_predict_clipped_
        self.emf_target_clipped_ = emf_target_clipped_
        self.time_clipped_ = time_clipped_

        self._clip_indexes = {
            'predict': (emf_predict_start, emf_predict_end),
            'target': (emf_target_start, emf_target_end)
        }

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

        self.emf_predict = emf_predict
        self.time_predict = time_predict

        # Rectify
        emf_predict = np.abs(emf_predict)

        resampled_signals = align_signals_in_time(
            t_1=self.time_target,
            y_1=self.emf_target,
            t_2=self.time_predict,
            y_2=self.emf_predict,
            num_samples=len(self.time_target)
        )

        resampled_time = resampled_signals[0]
        resampled_emf_target = resampled_signals[1]
        resampled_emf_predict = resampled_signals[2]

        self.emf_target_ = resampled_emf_target
        self.emf_predict_ = resampled_emf_predict
        self.time_ = resampled_time

        self._make_clipped_signals(target=self.emf_target_,
                                   predict=self.emf_predict_)

        # emf_predict_start, emf_predict_end = find_signal_limits(resampled_emf_predict)
        # emf_target_start, emf_target_end = find_signal_limits(resampled_emf_target, 0.075)

        # start_index = np.min([emf_predict_start, emf_target_start])
        # end_index = np.max([emf_predict_end, emf_target_end])
        # emf_predict_clipped_ = self.emf_predict_[start_index:end_index]
        # emf_target_clipped_ = self.emf_target_[start_index:end_index]

        # emf_predict_clipped_ = self.emf_predict_[emf_predict_start:emf_predict_end]
        # emf_target_clipped_ = self.emf_target_[emf_target_start:emf_target_end]

        # length_difference = len(emf_predict_clipped_) - len(emf_target_clipped_)
        # if length_difference < 0:  # right-pad emf_predict_clipped_
        #     emf_predict_clipped_ = np.pad(emf_predict_clipped_, (0, abs(length_difference)), 'constant')
        # else:  # right-pad the target signal
        #     emf_target_clipped_ = np.pad(emf_target_clipped_, (0, length_difference), 'constant')

        # self.emf_predict_clipped_ = emf_predict_clipped_
        # self.emf_target_clipped_ = emf_target_clipped_

        # self._clip_indexes = {
        #     'predict': (emf_predict_start, emf_predict_end),
        #     'target': (emf_target_start, emf_target_end)
        # }

    def _calc_dtw(self):
        """Perform dynamic time warping on prediction and targets."""

        # Exclude trailing (i.e. steady state) portion of the predicted waveform
        self.emf_predict_warped_, self.emf_target_warped_ = warp_signals(
            self.emf_predict_clipped_,
            self.emf_target_clipped_
        )

    def score(self) -> Dict[str, Any]:
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
        Dict
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
        >>> es_evaluator.score(mean_difference=(lambda x, y: np.mean(x-y)), max_value=(lambda x,y: np.max([x, y])))  # noqa
        Score(mean_difference=0.21169084032224028, max_value=5.078793981160988)

        """

        metric_results = apply_scalar_functions(self.emf_predict_clipped_,
                                                self.emf_target_clipped_,
                                                **self.metrics)

        return metric_results

    def poof(self, include_dtw=False, **kwargs) -> None:
        """Plot the aligned target and predicted values.

        Parameters
        ----------
        include_dtw : bool, optional
            Set to `True` to also plot the dynamic-time-warped signals.
            Default value is False.
        kwargs:
            Kwargs passed to matplotlib.pyplot.plot function.

        """
        time = self.time_
        target = self.emf_target_
        predict = self.emf_predict_

        max_y = np.max([target, predict])

        plt.figure(**kwargs)
        plt.plot(time, target, 'k', label='Target')
        plt.plot(time, predict, 'r', label='Predictions')

        # Show the clip marks

        plt.vlines(time[self._clip_indexes['target'][0]], 0, max_y, color='k', linestyle='--')
        plt.vlines(time[self._clip_indexes['target'][1]], 0, max_y, color='k', linestyle='--')

        plt.vlines(time[self._clip_indexes['predict'][0]], 0, max_y, color='r', linestyle='--')
        plt.vlines(time[self._clip_indexes['predict'][1]], 0, max_y, color='r', linestyle='--')

        plt.legend()

        if include_dtw:
            plt.figure()
            if self.emf_target_warped_ is None:
                self._calc_dtw()
            plt.plot(self.emf_target_warped_, label='Target, time-warped')
            plt.plot(self.emf_predict_warped_, label='Predictions, time-warped')
            plt.legend()
        plt.show()


class LabeledVideoProcessor:

    def __init__(
            self,
            magnet_assembly: mechanical_components.MagnetAssembly,
            seconds_per_frame: float,
            pixel_scale: Optional[float]=None,
            impute_missing_values: bool=True
    ) -> None:
        """Post-processor for labeled magnet-assembly data.

        Parameters
        ----------
        magnet_assembly : MagnetAssembly
            The magnet assembly used to execute the simulation.
        seconds_per_frame : float
            Number of seconds per frame / datapoint.
            This is typically found in the `subsampled_seconds_per_frame` key
            of the generated .DONE file when using the OpenCV-based CLI
            helper script.
        pixel_scale : float
            The pixel scale to use (in mm per pixel). This value will override
            the recorded pixel values in the groundtruth_dataframe.
            Default value is None.
        impute_missing_values : bool
            Whether to impute missing values in the labeled data using preceding
            samples.
            Default is True.

        """

        self.magnet_assembly = magnet_assembly
        self.spf = seconds_per_frame
        self.pixel_scale = pixel_scale
        self.impute_missing_values = impute_missing_values

    def fit_transform(self, groundtruth_dataframe):
        """
        Process and transform the `groundtruth_dataframe` that is generated by
        the OpenCV-based CLI helper script and return the position of the moving
        magnet assembly in mm, relative to the top of the fixed magnet.

        Parameters
        ----------
        groundtruth_dataframe : dataframe
            Dataframe generated by the OpenCV-based CLI helper script.

        Returns
        -------
        mag_center_pos : array
            Position of the center of the bottom-most magnet in the assembly
            relative to the top of the fixed magnet. Unit is metres.
        timestamps : array
            Timestamp of each position. Unit is seconds.

        """
        df = groundtruth_dataframe
        df.columns = [col.lower() for col in df.columns]

        # Prevent a cryptic error from getting thrown later
        try:
            assert df is not None
        except AssertionError:
            raise AssertionError('Groundtruth dataframe is `None`.'
                                 + 'Was the groundtruth file parsed correctly? Does it exist?')  # noqa

        if self.pixel_scale is None:  # If we don't manually set pixel scale
            if np.any(df['y_pixel_scale'] == -1):  # ... and it isn't in the parsed file.  # noqa
                raise ValueError('Dataframe contains missing pixel scale values and the pixel scale has not been '  # noqa
                                 'manually specified.')
        else:
            df['y_pixel_scale'] = self.pixel_scale

        df['y'] = np.abs(df['end_y'] - df['start_y'])  # Calculate position
        df['y_mm'] = df['y'] * df['y_pixel_scale']  # Adjust with pixel scale
        df['y_prime_mm'] = df['y_mm']  # Get actual position
        # Adjust for top / bottom of magnet during labeling process
        df.loc[df['top_of_magnet'] == 1, 'y_prime_mm'] = df['y_prime_mm'] - self.magnet_assembly.get_length()  # noqa

        # Correct calculations made with missing values, if they exist.
        if self.impute_missing_values:
            missing_indexes = df.query('start_y == -1').index.values
            df = impute_missing(df, missing_indexes)

        timestamps = np.linspace(0, (len(df)-1)*self.spf, len(df))

        return (df['y_prime_mm'].values + self.magnet_assembly.l_m_mm / 2) / 1000, timestamps  # noqa


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
                raise ValueError('Too few many sequential missing values to be able to impute all missing values.')  # noqa
            if df_missing.loc[start_velocity_calc, 'start_y'] == -1:
                start_velocity_calc = start_velocity_calc - 1
                if df_missing.loc[start_velocity_calc, 'start_y'] == -1:
                    raise ValueError('Too many sequential missing values to be able to impute all missing values.')  # noqa
        except KeyError:
            raise IndexError('Too few points available to calculate velocity and impute missing values.')  # noqa

        velocity = (df_missing.loc[end_velocity_calc, 'y_prime_mm']
                    - df_missing.loc[start_velocity_calc, 'y_prime_mm'])

        df_missing.loc[index, 'y_prime_mm'] = df_missing.loc[index - 1, 'y_prime_mm'] + velocity  # noqa
    return df_missing
