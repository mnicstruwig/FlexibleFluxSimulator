from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import numpy as np
import pandas as pd

from collections import namedtuple
from glob import glob
import os
import warnings

from asteval import Interpreter
from scipy import signal


def fetch_key_from_dictionary(dictionary, key, error_message):
    """Fetch a value from a dictionary

    :param dictionary: The dictionary to search
    :param key: The key to look-up
    :param error_message: The error message to print when the key is not found and an exception is raised.
    :return: The value corresponding to the key
    """

    try:
        return dictionary[key]
    except KeyError:
        raise KeyError(error_message)


def calc_sample_delay(x, y):
    """Calculate the delay (in samples) between two signals using correlation.
    """
    corr_1 = signal.correlate(x, y)
    corr_2 = signal.correlate(y, x)
    sample_delay = int((np.abs(np.argmax(corr_1) -
                        np.argmax(corr_2))) / 2)
    return sample_delay


# TODO: Add test
def smooth_butterworth(values, critical_frequency, **kwargs):
    """Smooth `values` by applying a low-pass butterworth filter

    Parameters
    ----------
    values : array_like
        Values to smooth or filter.
    critical_frequency : float
        The critical frequency of the butterworth filter. This is the
        frequency at which the gain drops to -3 dB of the pass band.
        Normalized between 0 and 1, where 1 is the Nyquist Frequency.
    **kwargs
        Keyword arguments to pass to the `butter` class.

    Returns
    -------
    ndarray
        The filtered signal

    See Also
    --------
    scipy.signal.butter : module

    """
    if 'N' not in kwargs:
        N = 6
    # noinspection PyTupleAssignmentBalance
    b, a = signal.butter(N, Wn=critical_frequency, btype='low', output='ba')
    filtered_values = signal.lfilter(b, a, values)
    return filtered_values


# TODO: Add test
def smooth_savgol(values, **kwargs):
    """Smooth `values` using a Savitsky-Golay filter

    Parameters
    ----------
    values : array_like
        Values to smooth.
    **kwargs
        Keyword arguments to pass to the `savgol` class

    Returns
    -------
    ndarray
        Smoothed values.

    See Also
    --------
    scipy.signal.savgol_filter : module

    """
    try:
        if 'window_length' in kwargs and 'polyorder' in kwargs:
            return signal.savgol_filter(values, **kwargs)
        return signal.savgol_filter(values, 101, 2)
    except ValueError:
        warnings.warn('Filter window length exceeds signal length. No filtering is being applied.', RuntimeWarning)
        return values


# TODO: Write test
# TODO: Write documentation
def parse_output_expression(t, raw_output, **kwargs):
    """Parse and evaluate an expression of the raw output.

    `raw_output` is a (n, d) dimensional array where n is the number of
    timesteps in the simulation, and d is the number of outputs.

    """

    gradient_function = """
def g(x, y):
    delta_y = [i-j for i,j in zip(y[1:], y)]
    delta_x = [i-j for i,j in zip(x[1:], x)]

    # Fake last element so that length remains the same as inputs.
    return [y/x for x, y in zip(delta_x, delta_y)] + [delta_y[-1]/delta_x[-1]]"""

    df_out = pd.DataFrame()

    def _populate_asteval_symbol_table(ast_eval_interpretor):
        ast_eval_interpretor.symtable['t'] = t
        for i in range(raw_output.shape[0]):
            ast_eval_interpretor.symtable['x' + str(i + 1)] = raw_output[i, :]
        return ast_eval_interpretor

    aeval = Interpreter()
    aeval = _populate_asteval_symbol_table(aeval)
    aeval(gradient_function)  # Define gradient function

    for key, expr in kwargs.items():
        df_out[key] = aeval(expr)

    return df_out


def warp_signals(x1, x2):
    """Warp and align two signals using dynamic time warping.

    Parameters
    ----------
    x1 : array
        The first signal to be warped.
    x2 : array
        The second signal to be warped.

    Returns
    -------
    tuple, length=2 * len(arrays)
        Tuple containing the two warped signals.

    """

    distance, path = fastdtw(x1, x2, dist=euclidean)
    path = np.array(path)

    x1_indexes = path[:, 0]
    x2_indexes = path[:, 1]

    x1_warped = np.array([x1[i] for i in x1_indexes])
    x2_warped = np.array([x2[i] for i in x2_indexes])

    return x1_warped, x2_warped


def find_signal_limits(target, sampling_period, threshold=1e-4):
    """Find the beginning and end of a signal using its spectrogram."""
    freqs, times, spectrum_density = signal.spectrogram(target,
                                                        1/sampling_period,
                                                        nperseg=128)
    # rows --> time, columns --> frequencies
    spectrum_density = spectrum_density.T
    max_density = np.array([np.max(density) for density in spectrum_density])

    for i, val in enumerate(max_density):
        if val > threshold:
            start_index = i
            break
    for i, val in enumerate(max_density):
        if val > threshold:
            end_index = i

    return times[start_index], times[end_index]


def apply_scalar_functions(x1, x2, **func):
    """Apply a set of functions (that return a scalar result) to two arrays."""
    results = {name: function(x1, x2) for (name, function) in zip(func.keys(), func.values())}
    return results


# TODO: Add test (might need to be a bit creative)
def collect_samples(base_path, acc_pattern, adc_pattern, labeled_video_pattern):
    """Collect groundtruth samples from a directory with filename matching.

    Parameters
    ----------
    base_path : str
        Base path to the root directory containing the samples.
    acc_pattern : str
        Glob-compatible pattern to use to search for the recorded accelerometer
        measurements .csv files.
    adc_pattern : str
        Glob-compatible pattern to use to search for the recorded adc
        measurements .csv files.
    labeled_video_pattern : str
        Glob-compatible pattern to use to search for the labeled video
        .csv files.

    Returns
    -------
    list
        List of `Sample` objects, with attributes containing pandas dataframes
        of the labeled video and recorded accelerometer and adc data.

    """

    # Holds the final result
    Sample = namedtuple('Sample', ['acc_df', 'adc_df', 'video_labels_df', 'paths'])

    sample_collection = []

    acc_paths = glob(os.path.join(base_path, acc_pattern))
    adc_paths = glob(os.path.join(base_path, adc_pattern))
    labeled_video_paths = glob(os.path.join(base_path, labeled_video_pattern))

    acc_paths.sort()
    adc_paths.sort()
    labeled_video_paths.sort()

    # Sanity checks + warnings
    if len(acc_paths) != len(adc_paths):
        warnings.warn('Different number of acc and adc files. Things might break as a result.')

    if len(labeled_video_paths) != len(acc_paths) or len(labeled_video_paths) != len(adc_paths):
        warnings.warn('There are a different number of groundtruth files, or some of them could not be found.')

    if len(labeled_video_paths) == 0 and len(acc_paths) == 0 and len(adc_paths) == 0:
        warnings.warn('No groundtruth files were found.')

    for acc, adc, lvp in zip(acc_paths, adc_paths, labeled_video_paths):
        sample_collection.append(Sample(pd.read_csv(acc), pd.read_csv(adc), pd.read_csv(lvp), [acc, adc, lvp]))
    return sample_collection
