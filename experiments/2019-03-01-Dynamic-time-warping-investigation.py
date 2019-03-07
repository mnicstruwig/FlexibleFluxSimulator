import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from sklearn.metrics import mean_squared_error, mean_absolute_error

from scipy.spatial.distance import euclidean, chebyshev, minkowski

def make_square(timestamps, wave_start, wave_end, signal=None):
    magnitude = 1
    signal = [magnitude if (t >= wave_start and t <= wave_end) else 0 for t in timestamps]
    return np.array(signal)

def add_noise(signal, std):
    noise = np.random.normal(0, std, size=len(signal))
    return signal + noise

time = np.linspace(0, 6, 1000)

y1 = make_square(time, 1, 2)

def run_noise_experiment(signal_a, signal_b, noise_dict, **kwargs):
    noise_name_list = []
    metric_name = []
    score = []
    for noise_name, noise_signal in noise_dict.items():
        signal_b_noise = signal_b + noise_signal
        for name, metric in kwargs.items():
            score.append(metric(signal_a, signal_b_noise))
            noise_name_list.append(noise_name)
            metric_name.append(name)
    return pd.DataFrame(dict(noise=noise_name_list, metric=metric_name, score=score))

y1 = make_square(time, 1, 2)
y2 = make_square(time, 1, 2)

std_list = np.arange(0, 0.5, 0.05)

noise_dict = {}
for std in std_list:
    noise_dict[std] = np.random.normal(0, std, len(y2))

def dtw(x, y, dist='euclidean'):
    dict_dist = {'euclidean': euclidean,
                 'cheb': chebyshev,
                 'mink': minkowski}
    distance, _ = fastdtw(y1, y2, dist=dict_dist[dist])
    return distance

df_noise = run_noise_experiment(y1,
                                y2,
                                noise_dict,
                                mse=mean_squared_error,
                                mae=mean_absolute_error,
                                dtw=dtw)

sns.lmplot(x='noise', y='score', hue='metric', data=df_noise)
plt.title('Overlapping signals')
plt.tight_layout()

y2 = make_square(time, 3, 4)
df_noise = run_noise_experiment(y1,
                                y2,
                                noise_dict,
                                mse=mean_squared_error,
                                mae=mean_absolute_error,
                                dtw=dtw)

sns.lmplot(x='noise', y='score', hue='metric', data=df_noise)
plt.title('Same signal, time-shifted')
plt.tight_layout()


y2 = make_square(time, 3, 5)

df_noise = run_noise_experiment(y1,
                                y2,
                                noise_dict,
                                mse=mean_squared_error,
                                mae=mean_absolute_error,
                                dtw=dtw)

sns.lmplot(x='noise', y='score', hue='metric', data=df_noise)
plt.title('Same signal, time-stretched')
plt.tight_layout()

y2 = 2*make_square(time, 1, 2)
df_noise = run_noise_experiment(y1,
                                y2,
                                noise_dict,
                                mse=mean_squared_error,
                                mae=mean_absolute_error,
                                dtw=dtw)

sns.lmplot(x='noise', y='score', hue='metric', data=df_noise)
plt.title('Same signal, overlapping, different magnitude')
plt.tight_layout()

y2 = 2*make_square(time, 3, 4)
df_noise = run_noise_experiment(y1,
                                y2,
                                noise_dict,
                                mse=mean_squared_error,
                                mae=mean_absolute_error,
                                dtw=dtw)

sns.lmplot(x='noise', y='score', hue='metric', data=df_noise)
plt.title('Same signal, time-shifted, different magnitude')
plt.tight_layout()

y2 = make_square(time, 3, 4) + make_square(time, 2.5, 3.5)
df_noise = run_noise_experiment(y1,
                                y2,
                                noise_dict,
                                mse=mean_squared_error,
                                mae=mean_absolute_error,
                                dtw=dtw)

sns.lmplot(x='noise', y='score', hue='metric', data=df_noise)
plt.title('Different signals, time-shifted')
plt.tight_layout()

# Also test for signals of different length

time_a = np.linspace(0, 6, 1000)
time_b = np.linspace(0, 5, 1000)

y1 = make_square(time_a, 3, 4)
y2 = make_square(time_b, 3, 4)

df_noise = run_noise_experiment(y1,
                                y2,
                                noise_dict,
                                mse=mean_squared_error,
                                mae=mean_absolute_error,
                                dtw=dtw)

sns.lmplot(x='noise', y='score', hue='metric', data=df_noise)
plt.title('Same signals, different time lengths')
plt.tight_layout()

y2 = make_square(time_b, 1, 2)

df_noise = run_noise_experiment(y1,
                                y2,
                                noise_dict,
                                mse=mean_squared_error,
                                mae=mean_absolute_error,
                                dtw=dtw)

sns.lmplot(x='noise', y='score', hue='metric', data=df_noise)
plt.title('Same signals, time-shifted, different time lengths')
plt.tight_layout()
plt.show()



