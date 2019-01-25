import warnings

from scipy.signal import savgol_filter


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


# TODO: Add test
def _smooth_savgol(values, **kwargs):
    """Smooth `values` using a Savitsky-Golay filter

    Parameters
    ----------
    values : array_like
        Values to smooth.
    **kwargs
        Keyword arguments to pass to the `savgol` class

    Returns
    -------
    array_like
        Smoothed values.


    See Also
    --------
    scipy.signal.savgol_filter : module

    """
    try:
        if 'window_length' in kwargs and 'polyorder' in kwargs:
            return savgol_filter(values, **kwargs)
        return savgol_filter(values, 101, 2)
    except ValueError:
        warnings.warn('Filter window length exceeds signal length. No filtering is being applied.', RuntimeWarning)
        return values
