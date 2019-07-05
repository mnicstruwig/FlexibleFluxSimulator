import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.metrics import mutual_info_score


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
    x1_rms = np.sqrt((np.sum(x1*x1)/len(x1)))
    x2_rms = np.sqrt((np.sum(x2*x2)/len(x2)))

    return x1_rms, x2_rms


def root_mean_square_percentage_diff(x1, x2):
    """Calculate the percentage difference between the RMS of two signals.

    Calculation is done relative to `x2`. Therefore positive values indicate
    `x1` overestimates `x2`.
    """
    x1_rms, x2_rms = root_mean_square(x1, x2)
    return (x1_rms-x2_rms)/x2_rms*100


def dtw_euclid_distance(x1, x2):
    """Calculate the distance between two signals using dynamic time warping."""
    distance, path = fastdtw(x1, x2, 1)
    return distance


def mutual_information_score(x1, x2):
    joint = np.histogram2d(x1, x2, 100)[0]
    return mutual_info_score(None, None, joint)
