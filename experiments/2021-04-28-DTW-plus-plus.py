import numpy as np
import matplotlib.pyplot as plt
from dtw import dtw
from fastdtw import fastdtw


# Let's calculate the trend sequence of a time series

t = np.linspace(0, 1, 1000)
X = np.sin(30*t) + np.cos(20*t)
Y = np.sin(32*t) + np.cos(20*t)


def get_trend(X):
    X_prime = []
    for i in range(len(X) - 1):
        if X[i] == X[i + 1]:
            x_prime = 0
        else:
            x_prime = 2 * (X[i + 1] - X[i]) / (np.abs(X[i]) + np.abs(X[i + 1]))
        X_prime.append(x_prime)
    return np.array(X_prime)


def calc_trend_warping_path(X_prime, Y_prime):
    alignment = dtw(X_prime, Y_prime, keep_internals=True)
    P_prime = list(zip(alignment.index1, alignment.index2))
    return P_prime


def get_window_arr(P_prime, length):
    arr = np.zeros([length, length])
    for a, b in P_prime:
        arr[a, b] = True
        arr[a - 1, b] = True
        arr[a, b - 1] = True
        arr[a - 1, b - 1] = True
    return arr


def ts_window_function(iw, jw, query_size, reference_size, arr):
    return arr[iw, jw]


def ts_dtw(X, Y):
    X_prime = get_trend(X)
    Y_prime = get_trend(Y)

    P_prime = calc_trend_warping_path(X_prime, Y_prime)
    arr = get_window_arr(P_prime, len(X))

    alignment = dtw(X,
                    Y,
                    window_type=ts_window_function,
                    window_args={'arr': arr},
                    keep_internals=False)
    return alignment.distance

# Run DTW again
alignment_ref = dtw(X, Y, keep_internals=True)
P_prime = ts_dtw(X, Y)
P_prime = np.array(P_prime)

X_noise = X + np.random.random(len(X))
