from fastdtw import fastdtw
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.signal import savgol_filter, correlate, butter, lfilter, detrend
from scipy.spatial.distance import euclidean
from scipy.misc import derivative
import os

base_dir = os.getcwd()

from unified_model.mechanical_components.magnet_assembly import MagnetAssembly
from unified_model.evaluate import MechanicalSystemEvaluator, LabeledVideoProcessor


def rms(x):
    return np.sqrt(np.mean(x ** 2))


def magnetic_spring_factory(
    data_path="../unified_model/mechanical_components/spring/data/10x10alt.csv",
):
    df = pd.read_csv(os.path.join(base_dir, data_path))
    df.columns = ["z", "newtons"]
    df["z"] = df["z"] / 1000  # Convert to metres

    # filter signal
    filtered = savgol_filter(df["newtons"], window_length=7, polyorder=3)

    return interp1d(df["z"].values, filtered, fill_value=0, bounds_error=False)


def acc_factory(data_path="./data/2019-05-23/A/log_03_acc.csv"):
    df = pd.read_csv(data_path)
    filtered = savgol_filter(df["z_G"], window_length=7, polyorder=3)
    filtered = filtered * 9.81 - 9.81  # Convert to m/s^2 from g and remove gravity

    return interp1d(df["time(ms)"] / 1000, filtered, fill_value=0, bounds_error=False)


def coupling_factory(c=0.5, rl=30, rc=12.5):
    def _coupling(emf):
        i = emf / (rl + rc)
        return c * i

    return _coupling


def flux_factory(
    data_path="./data/fea-flux-curves/fea-flux-curves-numr[5,15]-numz[17,33,66]-wdiam[0.15]-cheight[8,12,14]-2019-04-11.csv",
    coil_center=59 / 1000,
    velocity=0.35,
):
    df_coil = pd.read_csv(data_path)
    df_A = pd.DataFrame()
    df_A["time(ms)"] = df_coil["Time [ms]"]
    df_A["phi"] = np.abs(df_coil.iloc[:, 1])
    df_A["z"] = df_A["time(ms)"] / 1000 * velocity

    phi_interp = interp1d(
        df_A["z"], df_A["phi"], bounds_error=False, fill_value=0, kind="cubic"
    )
    new_z = np.linspace(0, df_A["z"].max(), 2000)
    new_phi = phi_interp(new_z)

    max_idx = np.argmax(new_phi)
    z_max_pos = new_z[max_idx]

    # Reposition peak to center of coil
    new_z = new_z - (z_max_pos - coil_center - 5 / 1000)  # 5 --> mm / 2, magnet center

    # Re-interpolate
    phi_interp = interp1d(
        new_z, new_phi, bounds_error=False, fill_value=0, kind="quadratic"
    )

    # chop off first and last samples to prevent derivative errors
    dphi = [derivative(phi_interp, z, dx=np.diff(new_z)[0] / 2) for z in new_z[1:-1]]
    dphi_interp = interp1d(
        new_z[1:-1], dphi, bounds_error=False, fill_value=0, kind="quadratic"
    )

    return phi_interp, dphi_interp


def adc_factory(
    data_path="./data/2019-05-23/A/log_03_adc.csv",
    voltage_division_ratio=1 / 0.342,
    R_l=30,
    R_c=12.5,
):
    df = pd.read_csv(data_path)

    V_l = np.abs(df["V"].values)  # scaled load voltage
    time = df["time(ms)"].values / 1000
    V_l_f = V_l
    # V_l_f = savgol_filter(V_l, 13, 11, 0)
    V_l_f = np.abs(detrend(V_l_f))
    #    V_l_f = savgol_filter(V_l_f, 35, 7, 0)
    V_l_f = V_l_f * voltage_division_ratio  # Scale up to real load voltage
    V_oc = V_l_f * (R_l + R_c) / (R_l)

    return V_oc, time


def system(t, y, acc, spring, coupling, phi, dphi_dz, c, m):
    # y --> tube_x, tube_x_dot, mag_x, mag_x_dot, mag_flux
    x1, x2, x3, x4, x5 = y

    if x1 < 0 and x2 < 0:
        x1 = 0
        x2 = 0

    g = 9.81

    rel_mag_pos = x3 - x1
    rel_mag_vel = x4 - x2

    dx1 = x2
    dx2 = acc(t)
    dx3 = x4

    dx5 = dphi_dz(rel_mag_pos) * rel_mag_vel  # dphi_dz * dz_dt

    F_spr = spring(rel_mag_pos)
    F_cpl = coupling(dx5)

    dx4 = (F_spr - F_cpl - m * g - c * rel_mag_vel) / m

    return [dx1, dx2, dx3, dx4, dx5]


mag_ass = MagnetAssembly(n_magnet=1, l_m=10, l_mcd=0, dia_magnet=10, dia_spacer=0)

m = mag_ass.get_mass()

t_span = [0.0, 8.0]
y0 = [0.0, 0.0, 0.04, 0.0, 0.0]

acc = acc_factory()
spring = magnetic_spring_factory()
coupling = coupling_factory()
phi, dphi_dz = flux_factory()
c = 0.035

sol = solve_ivp(
    fun=lambda t, y: system(t, y, acc, spring, coupling, phi, dphi_dz, c, m),
    t_span=t_span,
    max_step=1e-3,
    first_step=1e-3,
    y0=y0,
)

y_predict = sol.y
x1, x2, x3, x4, x5 = y_predict[:]

mag_pos_rel = x3 - x1
mag_vel_rel = x4 - x2
flux = x5
time = sol.t
oc_emf = np.abs(np.gradient(flux) / np.gradient(time))

oc_emf_, adc_time = adc_factory()

plt.plot(time, np.abs(oc_emf))
plt.plot(adc_time, oc_emf_)
plt.show()

pixel_scale = 0.18451
lvp = LabeledVideoProcessor(
    L=125, mm=10, seconds_per_frame=2 / 120, pixel_scale=pixel_scale
)
groundtruth_df = pd.read_csv(
    "./data/2019-05-23/A/a003_transcoded_subsampled_labels_2019-06-20-15:14:29.csv"
)

mag_pos_rel_, time_ = lvp.fit_transform(groundtruth_df, True)
end_time = time_[-1]


# Resample to same sample frequency
mag_pos_rel_interp = interp1d(time, mag_pos_rel, fill_value=0, bounds_error=False)
mag_pos_rel_interp_ = interp1d(time_, mag_pos_rel_, fill_value=0, bounds_error=False)

time_max = np.max(np.concatenate([time_, time]))
num_samples = 5000
time_r = np.linspace(0, time_max, num_samples)
timestep_r = np.diff(time_r)[0]

mag_pos_rel_r = mag_pos_rel_interp(time_r)
mag_pos_rel_r_ = mag_pos_rel_interp_(time_r)

# Align signals in time
corr_ab = correlate(mag_pos_rel_r, mag_pos_rel_r_)
corr_ba = correlate(mag_pos_rel_r_, mag_pos_rel_r)

index_ab = np.argmax(corr_ab)
index_ba = np.argmax(corr_ba)

index_diff = np.abs(index_ab - index_ba)

# Shift target signal backwards
padding = np.zeros(int(index_diff / 2))
mag_pos_rel_r_ = np.concatenate([padding, mag_pos_rel_r_])

for i, val in enumerate(mag_pos_rel_r_):
    if val > 0:
        start_index = i
        break

for i, val in enumerate(mag_pos_rel_r_[::-1]):
    if val > 0:
        end_index = len(mag_pos_rel_r_) - (i)
        break

time_r = time_r[start_index:end_index]
mag_pos_rel_r_ = mag_pos_rel_r_[start_index:end_index]
mag_pos_rel_r = mag_pos_rel_r[start_index:end_index]

plt.plot(time_r, mag_pos_rel_r_, label="target")
plt.plot(time_r, mag_pos_rel_r, label="prediction")
plt.legend()
plt.show()

mech_dist, _ = fastdtw(mag_pos_rel_r_, mag_pos_rel_r, radius=2, dist=euclidean)
print("Mech dist --- {}".format(mech_dist))

R_l = 30
R_c = 12.5

cc_emf = oc_emf * R_l / (R_c + R_l)
cc_emf_ = oc_emf_ * R_l / (R_c + R_l)

rms_oc_emf = rms(oc_emf)
rms_oc_emf_ = rms(oc_emf_)

rms_cc_emf = rms(cc_emf)
rms_cc_emf_ = rms(cc_emf_)

print("OC RMS adc --- {}".format(rms_oc_emf_))
print("OC RMS est --- {}".format(rms_oc_emf))
print("OC RMS perc diff --- {}".format((-rms_oc_emf_ + rms_oc_emf) / rms_oc_emf_ * 100))
print("---")
print("CC RMS adc --- {}".format(rms_cc_emf_))
print("CC RMS est --- {}".format(rms_cc_emf))
print("CC RMS perc diff --- {}".format((-rms_cc_emf_ + rms_cc_emf) / rms_cc_emf_ * 100))
