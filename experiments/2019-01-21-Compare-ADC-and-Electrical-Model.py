import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from unified_model.electrical_system.flux.model import flux_univariate_spline
from unified_model.electrical_system.flux.utils import FluxDatabase
from unified_model.electrical_system.load import SimpleLoad
from unified_model.evaluate import LabeledVideoProcessor
from unified_model.utils.testing.electrical_model import (_build_y_input_vector_at_timestamps,
                                                          apply_rectification,
                                                          simulate_electrical_system)

# Process the labeled data first

df_A_1 = pd.read_csv('/home/michael/Dropbox/PhD/Python/Experiments/mechanical-model/2018-12-20/A/001_transcoded_subsampled_labels_2019-02-03-15:53:43.csv')

df_A_2 = pd.read_csv('/home/michael/Dropbox/PhD/Python/Experiments/mechanical-model/2018-12-20/A/002_transcoded_subsampled_labels_2019-02-06-12:42:15.csv')

df_A_3 = pd.read_csv('/home/michael/Dropbox/PhD/Python/Experiments/mechanical-model/2018-12-20/A/003_transcoded_subsampled_labels_2019-02-07-10:46:33.csv')

df_A_4 = pd.read_csv('/home/michael/Dropbox/PhD/Python/Experiments/mechanical-model/2018-12-20/A/004_transcoded_subsampled_labels_2019-02-07-11:14:12.csv')

df_A_5 = pd.read_csv('/home/michael/Dropbox/PhD/Python/Experiments/mechanical-model/2018-12-20/A/005_transcoded_subsampled_labels_2019-02-07-13:18:00.csv')

df_B_1 = pd.read_csv('/home/michael/Dropbox/PhD/Python/Experiments/mechanical-model/2018-12-20/B/001_transcoded_subsampled_labels_2019-02-07-15:12:56.csv')

df_B_2 = pd.read_csv('/home/michael/Dropbox/PhD/Python/Experiments/mechanical-model/2018-12-20/B/002_transcoded_subsampled_labels_2019-02-07-15:42:41.csv')

# Very shaky video - might not be reliable
df_B_3 = pd.read_csv('/home/michael/Dropbox/PhD/Python/Experiments/mechanical-model/2018-12-20/B/003_transcoded_subsampled_labels_2019-02-07-16:01:07.csv')

df_B_4 = pd.read_csv('/home/michael/Dropbox/PhD/Python/Experiments/mechanical-model/2018-12-20/B/004_transcoded_subsampled_labels_2019-02-07-16:17:31.csv')

pixel_scale = 0.18745
lp = LabeledVideoProcessor(L=125, mm=10, seconds_per_frame=3 / 240)

fdb = FluxDatabase(
    '/home/michael/Dropbox/PhD/Python/Research/fea-flux-curves-numr[5,15]-numz[17,33,66]-wdiam[0.15]-2018-12-07.csv',
    fixed_velocity=0.35)
phi_A = fdb.query(winding_num_z='17', winding_num_r='15')
phi_B =  fdb.query(winding_num_z='33', winding_num_r='15')

load = SimpleLoad(R=np.inf)
flux_model_A = flux_univariate_spline(z_arr=fdb.z, phi_arr=phi_A, coil_center=58.5 / 1000, mm=10)
flux_model_B = flux_univariate_spline(z_arr=fdb.z, phi_arr=phi_B, coil_center=61 / 1000, mm=10)

#######################
##### CHOOSE HERE #####
df_imp = df_B_4
flux_model = flux_model_B
#########################

y_relative_mm, timestamps = lp.fit_transform(df_imp, impute_missing_values=True, pixel_scale=pixel_scale)
df_imp['y_prime_mm'].plot()

# Simulate the system
emf_values, emf_timestamps = simulate_electrical_system(y_relative_mm, timestamps, flux_model, load, True)
plt.figure()
plt.plot(emf_timestamps, np.abs(emf_values), 'g')
plt.show()
