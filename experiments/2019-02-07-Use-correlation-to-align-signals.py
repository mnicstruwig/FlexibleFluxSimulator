import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import UnivariateSpline

from unified_model.electrical_system.evaluator import AdcProcessor, ElectricalSystemEvaluator
from unified_model.electrical_system.flux.model import flux_univariate_spline
from unified_model.electrical_system.flux.utils import FluxDatabase
from unified_model.electrical_system.load import SimpleLoad
from unified_model.mechanical_system.evaluator import LabeledVideoProcessor
from unified_model.utils.testing.testing_electrical_model import simulate_electrical_system

# Prerequisites
df_A_1_labeled = pd.read_csv('/home/michael/Dropbox/PhD/Python/Experiments/mechanical-model/2018-12-20/A/001_transcoded_subsampled_labels_2019-02-03-15:53:43.csv')

df_A_2_labeled = pd.read_csv('/home/michael/Dropbox/PhD/Python/Experiments/mechanical-model/2018-12-20/A/002_transcoded_subsampled_labels_2019-02-06-12:42:15.csv')

df_A_1_adc = pd.read_csv('/home/michael/Dropbox/PhD/Python/unified_model/notebooks/2018-12-20/A/log_17.csv')
df_A_2_adc = pd.read_csv('/home/michael/Dropbox/PhD/Python/unified_model/notebooks/2018-12-20/A/log_18.csv')

fdb = FluxDatabase(
    '/home/michael/Dropbox/PhD/Python/Research/fea-flux-curves-numr[5,15]-numz[17,33,66]-wdiam[0.15]-2018-12-07.csv',
    fixed_velocity=0.35)

phi_A = fdb.query(winding_num_z='17', winding_num_r='15')
flux_model_A = flux_univariate_spline(z_arr=fdb.z, phi_arr=phi_A, coil_center=58.5 / 1000, mm=10)

df_label = df_A_1_labeled
df_adc = df_A_1_adc

flux_model = flux_model_A
load_model = SimpleLoad(R=np.inf)

pixel_scale = 0.18745*1280/900
lp = LabeledVideoProcessor(L=125, mf=14, mm=10, seconds_per_frame=3 / 240)

y_relative_mm, timestamps = lp.fit_transform(df_label, impute_missing_values=True, pixel_scale=pixel_scale)

# Prepare groundtruth data
adc = AdcProcessor(voltage_division_ratio=1/0.342, smooth=True)
emf_truth, timestamps_truth = adc.fit_transform(df_adc)

# Simulate the system
emf_pred, timestamps_pred = simulate_electrical_system(y_relative_mm, timestamps, flux_model, load_model, True)
emf_pred = np.abs(emf_pred)

e_eval = ElectricalSystemEvaluator(emf_target=emf_truth, time_target=timestamps_truth)
e_eval.fit(emf_pred, timestamps_pred)

plt.plot(e_eval.resampled_timestamps, e_eval.resampled_emf_target)
plt.plot(e_eval.resampled_timestamps, e_eval.resampled_emf_pred)
plt.show()
