import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import signal

from unified_model.electrical_components.load import SimpleLoad
from unified_model.evaluate import (
    LabeledVideoProcessor,
    ElectricalSystemEvaluator,
    AdcProcessor,
)
from unified_model.electrical_model import ElectricalModel
from unified_model.utils.utils import collect_samples, rms, find_signal_limits
from unified_model.utils.testing.testing_electrical_model import (
    simulate_electrical_system,
    apply_rectification,
)
from config import abc


base_groundtruth_path = "./data/2019-05-23/"
a_samples = collect_samples(
    base_path=base_groundtruth_path,
    acc_pattern="A/*acc*.csv",
    adc_pattern="A/*adc*.csv",
    labeled_video_pattern="A/*labels*.csv",
)

which = 1

pixel_scale = 0.18451  # Huawei P10 alternative
lvp = LabeledVideoProcessor(125, 10, 1 / 80, pixel_scale=pixel_scale)
adc = AdcProcessor(voltage_division_ratio=1 / 0.342, smooth=True)

# Processed values
adc_emf, adc_ts = adc.fit_transform(a_samples[which].adc_df)
adc_ts = adc_ts
mag_pos, ts = lvp.fit_transform(a_samples[which].video_labels_df, True)
mag_vel = np.gradient(mag_pos) / np.gradient(ts)

# Filtered values
mag_pos_ = signal.savgol_filter(mag_pos, 11, 5)
mag_vel_ = signal.savgol_filter(mag_vel, 11, 5)

mag_pos_interp = interpolate.interp1d(ts, mag_pos_, "quadratic")
mag_vel_interp = interpolate.interp1d(ts, mag_vel_, "quadratic")

# Interpolated values
ts__ = np.linspace(ts[0], ts[-1], 15000)
mag_pos__ = mag_pos_interp(ts__)
mag_vel__ = mag_vel_interp(ts__)

plt.plot(ts__, mag_pos__)
plt.show()

flux_model = abc.flux_models["A"]
dflux_model = abc.dflux_models["A"]
z = np.linspace(-0.0, 0.2, 10000)
phi = flux_model(z)
z_max = z[np.argmax(phi)]
squish_factor = 1.00
new_z = (z - z_max) * squish_factor

new_flux_model = interpolate.interp1d(
    new_z, phi / squish_factor, fill_value=0, bounds_error=False, kind="quadratic"
)
new_flux_model = interpolate.interp1d(
    new_z + z_max, new_flux_model(new_z), bounds_error=False, fill_value=0
)

dphi = np.gradient(new_flux_model(z)) / np.gradient(z)
new_dflux_model = interpolate.interp1d(z, dphi, bounds_error=False, fill_value=0)

plt.plot(z, phi)
plt.plot(z, new_flux_model(z), "--")
plt.show()

plt.plot(z, dflux_model(z))
plt.plot(z, new_dflux_model(z))
plt.show()

t_emf, emf = simulate_electrical_system(
    mag_pos__,
    mag_vel__,
    ts__,
    new_flux_model,
    new_dflux_model,
    SimpleLoad(30),
    coil_resistance=12.5,
)

emf = apply_rectification(emf, 0.12)


plt.plot(adc_ts, adc_emf)
plt.plot(t_emf, emf)
plt.show()

e_eval_m = ElectricalSystemEvaluator(adc_emf, adc_ts)
e_eval_m.fit(emf, t_emf, clip_threshold=1e-4)
e_eval_m.poof(False)

rms_target = rms(e_eval_m.emf_target_clipped_)
rms_predict = rms(e_eval_m.emf_predict_clipped_)

print()
print("Target RMS --- {}".format(rms_target))
print("Predict RMS --- {}".format(rms_predict))
print("Diff % --- {}".format(100 * (rms_predict - rms_target) / rms_predict))
