import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error

from unified_model.evaluate import ElectricalSystemEvaluator
from unified_model.utils.utils import find_signal_limits, warp_signals

df = pd.read_csv('./experiments/emf_signal.csv')

time_ = df['time'].values
emf_predict_ = df['emf_predict_'].values
emf_target_ = df['emf_target_'].values

wave = np.sin(2*np.pi*100*time_) + np.sin(2*np.pi*300*time_)
wave = emf_target_
wave = emf_predict_

test_time_arr = np.arange(0, 10, 0.001)
test_wave = np.sin(2*np.pi*3*test_time_arr)
padding = np.zeros(1000)

test_signal = np.concatenate([padding, test_wave, padding])
wave = test_signal


sampling_period = time_[1] - time_[0]
sampling_period = 0.001
sampling_frequency = 1/sampling_period

# sample frequencies, segment times, spectrogram for each "bin"
f, t, s = signal.spectrogram(wave, sampling_frequency, nperseg=128)

# plt.pcolormesh(t, f, s)
# plt.show()

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
start_time, end_time = find_signal_limits(emf_target_, time_[1]-time_[0])
print(start_time)
print(end_time)

ec_eval = ElectricalSystemEvaluator(emf_target_, time_)
ec_eval.fit(emf_predict_, time_)
score = ec_eval.score(mse=mean_squared_error)
print(score)
print(mean_squared_error(emf_predict_, emf_target_))

plt.plot(ec_eval.emf_predict_warped_)
plt.plot(ec_eval.emf_target_warped_)
plt.show()

