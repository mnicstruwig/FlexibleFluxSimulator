import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import *

df = pd.read_csv('./experiments/emf_signal.csv')

time_ = df['time'].values
emf_predict_ = df['emf_predict_'].values
emf_target_ = df['emf_target_'].values

sampling_period = time_[1] - time_[0]
sampling_frequency = 1/sampling_period

# sample frequencies, segment times, spectrogram for each "bin"
f, t, s = signal.spectrogram(emf_target_, sampling_frequency, nperseg=64)

plt.pcolormesh(t, f, s)
plt.show()

s = s.T # row --> time, col --> frequency
thresh = 1e-4
maxs = np.array([np.max(slice_) for slice_ in s])

for i, val in enumerate(maxs):
    if val > thresh:
        start_index = i
        break

for i, val in enumerate(maxs):
    if val > thresh:
        end_index = i

