import copy
import os
import warnings
from collections import namedtuple
from glob import glob
from itertools import product, zip_longest
from typing import Tuple

import numpy as np
import pandas as pd
from asteval import Interpreter
from fastdtw import fastdtw
from scipy import signal
from scipy.spatial.distance import euclidean
from scipy.interpolate import UnivariateSpline


def rms(x):
    """Calculate the RMS of a signal."""
    return np.sqrt(np.mean(x**2))


def pretty_str(dict_):
    """Get a pretty string representation of a dictionary."""
    str_ = ""
    for key, val in dict_.items():
        str_ = str_ + f"{key} : {val}" + "\n"
    return str_


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


def get_sample_delay(x, y):
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
        N = 9
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


def grad(func, x, dx=1e-5):
    """Compute the gradient of function `func` at point `x` relative to `dx`"""
    dfunc_dx = (func(x + dx) - func(x - dx))/(2*dx)
    if np.isinf(dfunc_dx):
        return 0.0
    return dfunc_dx


# TODO: Write test
# TODO: Write documentation
def parse_output_expression(t, raw_output, **kwargs):
    """Parse and evaluate an expression of the raw output.

    `raw_output` is a (n, d) dimensional array where n is the number of
    timesteps in the simulation, and d is the number of outputs.

    """

# def g(x, y):
#     delta_y = [i-j for i,j in zip(y[1:], y)]
#     delta_x = [i-j for i,j in zip(x[1:], x)]

#     # Fake last element so that length remains the same as inputs.
#     return [y/x for x, y in zip(delta_x, delta_y)] + [delta_y[-1]/delta_x[-1]]

    df_out = pd.DataFrame()

    def gradient_function(x, y):
        return np.gradient(y)/np.gradient(x)

    def _populate_asteval_symbol_table(ast_eval_interpretor):
        ast_eval_interpretor.symtable['t'] = t
        for i in range(raw_output.shape[0]):
            ast_eval_interpretor.symtable['x' + str(i + 1)] = raw_output[i, :]
        return ast_eval_interpretor

    aeval = Interpreter()
    aeval = _populate_asteval_symbol_table(aeval)

    for key, expr in kwargs.items():
        if 'g(' in expr:
            split = expr.split(',')
            x = split[0].split('(')[1]
            y = split[1].split(')')[0]
            y = y.strip()
            x = aeval(x)
            y = aeval(y)

            df_out[key] = gradient_function(x, y)
        else:
            df_out[key] = aeval(expr)

    return df_out


def warp_signals(x1, x2, return_distance=False):
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

    if return_distance:
        return x1_warped, x2_warped, distance
    return x1_warped, x2_warped


def interpolate_and_resample(x, y, num_samples=10000, new_x_range=None):
    """Resample a signal using interpolation.

        This is useful for resampling two different signals with different
        sampling frequencies so that they have the same sampling frequency,
        which can be achieved by resampling both signals to have the same
        `num_samples` and `new_x_range`.

        Parameters
        ----------
        x : array_like
            The input values that correspond to output values `y`.
        y : array_like
            The output values to be interpolated.
        num_samples : int
            The number of sampling points that should be used in the resampled
            signal.
        new_x_range : tuple(int, int)
            The range of x values for which the resampled values should be
            returned.

        Returns
        -------
        new_x : array
            The new x values.
        interp: array
            The new resampled values of `y` corresponding to `new_x`.

        """
    interp = UnivariateSpline(x, y, s=0, ext='zeros')

    if new_x_range is not None:
        x_start = new_x_range[0]
        x_stop = new_x_range[1]
    else:
        x_start = 0
        x_stop = np.max(x)

    new_x = np.linspace(x_start, x_stop, num_samples)
    return new_x, interp(new_x)


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
    """Apply a set of functions (that return a scalar result) to two arrays.

    Parameters
    ----------
    x1 : array
        The first array.
    x2 : array
        The second array.
    **func
        The functions should be provided as a dictionary, `func` where the keys
        are the names of the sub-result that will be returned, and the values of
        `func` are the functions to be applied.

    Returns
    -------
    dict
        Results are returned as dictionary where the keys are the same as the
        keys provided by `func` and the values are the results of the applied
        functions.

    """
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

    acc_paths = glob(os.path.join(base_path, acc_pattern))
    adc_paths = glob(os.path.join(base_path, adc_pattern))
    labeled_video_paths = glob(os.path.join(base_path, labeled_video_pattern))

    acc_paths.sort()
    adc_paths.sort()
    labeled_video_paths.sort()

    # TODO: Move these sanity checks to /after/ reading in the files.
    if len(acc_paths) != len(adc_paths):
        warnings.warn('Different number of acc and adc files. Things might break as a result.')

    if len(labeled_video_paths) != len(acc_paths) or len(labeled_video_paths) != len(adc_paths):
        warnings.warn('There are a different number of groundtruth files, or some of them could not be found.')

    if len(labeled_video_paths) == 0 and len(acc_paths) == 0 and len(adc_paths) == 0:
        warnings.warn('No groundtruth files were found.')

    acc_dfs = [pd.read_csv(acc_path) for acc_path in acc_paths]
    adc_dfs = [pd.read_csv(adc_path) for adc_path in adc_paths]
    lvp_dfs = [pd.read_csv(lvp_path) for lvp_path in labeled_video_paths]

    paths = zip_longest(acc_paths,
                        adc_paths,
                        labeled_video_paths,
                        fillvalue=None)

    sample_collection = [Sample(acc, adc, lvp, path)
                         for acc, adc, lvp, path
                         in zip_longest(acc_dfs,
                                        adc_dfs,
                                        lvp_dfs,
                                        paths,
                                        fillvalue=None)]
    return sample_collection


# TODO: Add test
def build_paramater_grid(param_dict: dict, func_dict: dict = None) -> Tuple:
    """Build a grid of parameters for performing gridsearch.

    Functions can optionally be applied to the input parameters, which is
    useful when you wish to instantiate objects for use in the grid search.

    Parameters
    ----------
    param_dict : dict
        Dictionary where the keys are the names of the parameters, and the
        values are a list containing the values the parameter must have.
        If `func_dict` is provided, the keys of the parameters must match those
        of `func_dict`.
    func_dict : dict, optional
        Dictionary defining functions to be applied to values of `param_dict`.
        Keys must match the keys in `param_dict`.

    Returns
    -------
    parameter_grid : list
        A list containing all product permutations of the parameters, with
        functions in `funct_dict` applied (if `func_dict` was specified).
    value_grid : list
        A iterable containing the product permutation of the *values* of the
        parameters. This is useful when `func_dict` is specified, and the
        original values of the parameters would otherwise be lost (such as
        in the case where an object is returned). `value_grid` is only returned
        if `func_dict` is not None.

    """
    if func_dict:
        try:
            assert list(param_dict.keys()) == list(func_dict.keys())
        except AssertionError:
            raise AssertionError('The parameter keys and the function keys do\
            not match.')

        processed_param_dict = {}
        for key, values in param_dict.items():
            # Apply functions to parameter grid
            processed_param_dict[key] = [func_dict[key](value) for value in values]

        parameter_product = list(product(*processed_param_dict.values()))
        value_grid = list(product(*param_dict.values()))

        parameter_grid = []
        for param_set in parameter_product:
            dict_ = {key: param
                     for key, param
                     in zip(param_dict.keys(), param_set)}

            parameter_grid.append(dict_)

        return parameter_grid, value_grid
        return list(product(*parameter_grid.values())), list(product(*param_dict.values()))

    return list(product(*param_dict.values()))


def update_nested_attributes(primary, update_dict):
    """
    Update a number of parameters in a multi-tiered object and return a copy.

    Parameters
    ----------
    primary : object
        The object to update.
    update_dict : dict
        A dictionary whose keys should be a "dot" expression for the nested
        attributes to be updated, and whose values should be the new values
        for these attributes.
        Example key: 'primary_attribute.secondary_attribute.target_attribute'.

    Returns
    -------
    object
        The updated object.

    """
    new_primary = copy.deepcopy(primary)
    for key, val in update_dict.items():
        new_primary = update_attribute(new_primary,
                                       key,
                                       val)
    return new_primary


# TODO: Add tests
def update_attribute(primary, key_expr, value):
    """
    Update a key in a nested dictionary and return a copy.


    Parameters
    ----------
    primary : object
        The object (that may contain nested objects as attributes) to be
        updated.
    key_expr : str
        The dotted expression of the nested path of the key to update.
        Eg: `primary_key.secondary_key.target_key`. Keys must correspond
        to nested dictionary in `primary`.
    value:
        The new value for the key at `key_expr`.

    Returns
    -------
    dict
        The updated dictionary.

    Examples
    --------
    >>> dict_ = {'a': {'b': {'c': 'my_nested_value'}}}
    >>> new_dict = update_nested_dict(dict_, 'a.b.c', 'new_value')
    >>> new_dict
    {'a': {'b': {'c': 'new_value'}}}
    >>> dict
    {'a': {'b': {'c': 'my_nested_value'}}}

    """
    new_primary = copy.deepcopy(primary)
    new_primary_dict = new_primary.__dict__
    keys = key_expr.split('.')

    if len(keys) == 1:
        new_primary_dict[keys[0]] = value
    else:
        temp = new_primary_dict
        for key in keys[:-1]:  # Loop through keys except last one
            temp = temp[key]  # Enter the next dictionary
        temp.__dict__[keys[-1]] = value  # Set value at deepest key

    return new_primary
