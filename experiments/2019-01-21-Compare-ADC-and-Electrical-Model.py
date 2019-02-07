import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from unified_model.electrical_system.electrical_system import ElectricalSystem
from unified_model.electrical_system.load import SimpleLoad
from unified_model.electrical_system.flux.utils import FluxDatabase
from unified_model.electrical_system.flux.model import flux_interpolate, flux_univariate_spline

from unified_model.mechanical_system.evaluator import LabeledVideoProcessor, impute_missing

def _build_y_input_vector_at_timestamps(x3, timestamps):
    """
    Build the `y` input vector (at each timestamp!) that is fed to the
    electrical system in order to calculate the emf.

    y[0] --> x1 --> tube displacement
    y[1] --> x2 --> tube velocity (gradient of x1)
    y[2] --> x3 --> magnet displacement
    y[3] --> x4 --> magnet velocity (gradient of x3)
    y[4] --> x5 --> flux linkage

    We can "hack" around this by setting everything to zero, _except_ x3 and x4
    and then calculate the induced EMF by hand.

    Parameters
    ----------
    x3 : array_like
        The relative distance between the top of the fixed magnet and the bottom
    timestamps : array_like
        Timestamps corresponding to `x3`

    Returns
    -------
    (n, 5) ndarray
        y "input vector" where each row is a value of the input vector `y`.
    """

    x1 = np.zeros(len(x3))
    x2 = x1
    # x3 already defined
    x4 = np.gradient(x3) / np.gradient(timestamps)
    x5 = x1  # Not used, but must be present.

    return np.array([x1, x2, x3, x4, x5]).T

def apply_rectification(emf_values):
    """Do a "dumb" simulation of rectification of the EMF values."""

    for i, e in enumerate(emf_values):
        e = np.abs(e)

        if e > 0.2:
            e = e-0.2
        else:
            e = 0

        emf_values[i]= e
    return emf_values


def simulate_electrical_system(y_relative_mm, timestamps, flux_model, load, interpolate=True):

    x3 = y_relative_mm

    if interpolate:
        x3_interpolator = interp1d(timestamps, x3)
        timestamps = np.linspace(0, timestamps[-1], 10000)
        x3 = x3_interpolator(timestamps)

    ys = _build_y_input_vector_at_timestamps(x3, timestamps)

    electrical_system = ElectricalSystem(flux_model=flux_model, load_model=load, precompute_gradient=True)
    emf_values = np.array([electrical_system.get_emf(y) for y in ys])

    return emf_values, timestamps


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
lp = LabeledVideoProcessor(L=125, mf=14, mm=10, seconds_per_frame=3 / 240)

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

