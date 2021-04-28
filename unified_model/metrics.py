"""
Metrics for calculating the accuracy of a model.
"""

import numpy as np
from dtw import dtw
from fastdtw import fastdtw
from scipy.stats import zscore


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


def similarity_measure(x1, x2) -> float:
    """Calculate a similarity measure between two signals using dynamic time warping.

    Source: https://cs.stackexchange.com/questions/53250/normalized-measure-from-dynamic-time-warping
    """
    D = dtw_euclid_distance(x1, x2)
    M = len(x1) * np.max(x1)  # The maximum possible DTW path distance
    S = (M - D) / M
    return S


def dtw_euclid_norm(x1, x2):
    return dtw_euclid_distance(x1/np.max(x1), x2/np.max(x2))


def dtw_euclid_z_norm(x1, x2):
    return dtw_euclid_distance(zscore(x1), zscore(x2))


def _joint_z_norm(x1, x2):
    # http://luscinia.sourceforge.net/page26/page14/page14.html
    joint_mean = np.sum([x1, x2])/(len(x1) + len(x2))
    x1_joint_std = np.sum((x1 - joint_mean)**2)
    x2_joint_std = np.sum((x2 - joint_mean)**2)
    joint_std = np.sqrt((x1_joint_std + x2_joint_std)/(len(x1) + len(x2) - 1))

    x1_norm = (x1-joint_mean)/joint_std
    x2_norm = (x2-joint_mean)/joint_std
    return x1_norm, x2_norm


def dtw_euclid_joint_z_norm(x1, x2):
    x1_norm, x2_norm = _joint_z_norm(x1, x2)
    return dtw_euclid_distance(x1_norm, x2_norm)


def _get_trend(X):
    X_prime = []
    for i in range(len(X) - 1):
        if X[i] == X[i + 1]:
            x_prime = 0
        else:
            x_prime = 2 * (X[i + 1] - X[i]) / (np.abs(X[i]) + np.abs(X[i + 1]))
        X_prime.append(x_prime)
    return np.array(X_prime)


def _calc_trend_warping_path(X_prime, Y_prime):
    alignment = dtw(X_prime, Y_prime, keep_internals=True)
    P_prime = list(zip(alignment.index1, alignment.index2))
    return P_prime


def _get_window_arr(P_prime, length):
    arr = np.zeros([length, length])
    for a, b in P_prime:
        arr[a, b] = True
        arr[a - 1, b] = True
        arr[a, b - 1] = True
        arr[a - 1, b - 1] = True
    return arr


def _ts_window_function(iw, jw, query_size, reference_size, arr):
    return arr[iw, jw]


def dtw_ts(x1, x2):
    X_prime = _get_trend(x1)
    Y_prime = _get_trend(x2)

    P_prime = _calc_trend_warping_path(X_prime, Y_prime)
    arr = _get_window_arr(P_prime, len(x1))

    alignment = dtw(x1,
                    x2,
                    window_type=_ts_window_function,
                    window_args={'arr': arr},
                    keep_internals=True)
    return alignment.distance
