import numpy as np
import pandas as pd
import warnings

from asteval import Interpreter
from scipy import signal


def fetch_key_from_dictionary(dictionary, key, error_message):
    """
    Fetches a value from a dictionary
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

    It is not recommended to use this helper function directly. For
    documentation and usage, see the `get_output` method.

    """
    df_out = pd.DataFrame()

    def _populate_asteval_symbol_table(ast_eval_interpretor):
        ast_eval_interpretor.symtable['t'] = t
        for i in range(raw_output.shape[0]):
            ast_eval_interpretor.symtable['x' + str(i + 1)] = raw_output[i, :]
        return ast_eval_interpretor

    aeval = Interpreter()
    aeval = _populate_asteval_symbol_table(aeval)

    for key, expr in kwargs.items():
        df_out[key] = aeval(expr)

    return df_out
