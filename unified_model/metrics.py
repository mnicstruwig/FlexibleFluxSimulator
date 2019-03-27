import numpy as np


def corr_coeff(x1, x2):
    """Calculate the correlation coefficient."""
    return np.corrcoef(x1, x2)[0, 1]


def max_err(x1, x2):
    """Calculate the maximum error"""
    return np.max(np.abs(np.array(x1) - np.array(x2)))


def mean_absolute_percentage_err(x1, x2):
    """Calculate the mean absolute percentage error.

    Note, `x1` are the predicted values and `x2` are the truthful values.

    """
    return np.mean(np.abs((x2 - x1)/(x2+0.000001)))*100


def root_mean_square(x1, x2):
    """Calculate the RMS of two signals."""
    x1_rms = np.mean((np.sum(x1*x1)/len(x1)))
    x2_rms = np.mean((np.sum(x2*x2)/len(x2)))

    return x1_rms, x2_rms
