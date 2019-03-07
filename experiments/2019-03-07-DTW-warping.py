import numpy as np
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

import matplotlib.pyplot as plt


def make_square(timestamps, wave_start, wave_end, signal=None):
    magnitude = 1
    signal = [magnitude if (t >= wave_start and t <= wave_end) else 0 for t in timestamps]
    return np.array(signal)

def make_sine(timestamps, wave_start, wave_end, amplitude=1):
    period = 2*(wave_end - wave_start)
    f = 1/period

    timestep = timestamps[1]-timestamps[0]
    time = np.arange(0, period/2, timestep)
    y = amplitude*np.sin(2*np.pi*f*time)

    start_time_index = np.argmin(np.abs(timestamps - wave_start))
    y = np.append(np.zeros(start_time_index), y)
    y = np.append(y, np.zeros(len(timestamps) - len(y)))
    return y

def add_noise(signal, std):
    noise = np.random.normal(0, std, size=len(signal))
    return signal + noise



time = np.linspace(0, 6, 1000)

y1 = make_square(time, 1, 2)
y2 = make_square(time, 3, 4)


dist, curve = fastdtw(y1, y2, dist=euclidean)
curve = np.array(curve)

y1_indexes = curve[:, 0]
y2_indexes = curve[:, 1]

y1_warped = [y1[i] for i in y1_indexes]


# plt.plot(y1, label='y1')
# plt.plot(y1_warped, label='y1_warped')
# plt.plot(y2, label='y2')
# plt.legend()

# plt.show()

y1 = make_sine(time, 1, 2, 0.5)
y2 = make_sine(time, 3, 4, 1)

dist, curve = fastdtw(y1, y2, dist=euclidean)
curve = np.array(curve)

y1_indexes = curve[:, 0]
y2_indexes = curve[:, 1]

y1_warped = [y1[i] for i in y1_indexes]
y2_warped = [y2[i] for i in y2_indexes]

plt.plot(y1_warped, label='y1_warped')
plt.plot(y1, label='y1')
plt.plot(y2, label='y2')
plt.plot(y2_warped, label='y2_warped')
plt.legend()
plt.show()
