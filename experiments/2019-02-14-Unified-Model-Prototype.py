import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, r2_score, median_absolute_error

from collections import namedtuple
from glob import glob

from unified_model.coupling import CouplingModel
from unified_model.electrical_model import ElectricalModel
from unified_model.electrical_components.flux.utils import FluxDatabase
from unified_model.electrical_components.load import SimpleLoad
from unified_model.governing_equations import unified_ode
from unified_model.mechanical_model import MechanicalModel
from unified_model.mechanical_components.damper import Damper
from unified_model.mechanical_components.input_excitation.accelerometer import AccelerometerInput
from unified_model.mechanical_components.magnet_assembly import MagnetAssembly
from unified_model.mechanical_components.magnetic_spring import \
    MagneticSpring
from unified_model.unified import UnifiedModel
from unified_model.evaluate import ElectricalSystemEvaluator, MechanicalSystemEvaluator, LabeledVideoProcessor, AdcProcessor
from unified_model.pipeline import clip_x2
from unified_model.utils.utils import collect_samples
from unified_model.metrics import max_err, mean_absolute_percentage_err, corr_coeff, root_mean_square

# Path handling
fea_data_path = './unified_model/mechanical_components/spring/data/10x10alt.csv'

# Ground-truth files
base_groundtruth_path = './experiments/data/2018-12-20/'

a_samples = collect_samples(base_path=base_groundtruth_path,
                            acc_pattern='A/*acc*.csv',
                            adc_pattern='A/*adc*.csv',
                            labeled_video_pattern='A/*labels*.csv')

# SELECTION
which_sample = 2
acc_adc_data_path = a_samples[which_sample].acc_emf_df
video_labels_df = a_samples[which_sample].video_labels_df
which_device = 'A'

# MECHANICAL MODEL
spring = MagneticSpring(fea_data_file=fea_data_path,
                        model='savgol_smoothing',
                        model_type='interp')

magnet_assembly = MagnetAssembly(n_magnet=1,
                                 l_m=10,
                                 l_mcd=0,
                                 dia_magnet=10,
                                 dia_spacer=10,
                                 mat_magnet='NdFeB',
                                 mat_spacer='iron')

damper = Damper(model='constant', model_kwargs={'damping_coefficient': 0.05})
accelerometer = AccelerometerInput(raw_accelerometer_input=acc_adc_data_path,
                                   accel_column='z_G',
                                   time_column='time(ms)',
                                   time_unit='ms',
                                   smooth=True,
                                   interpolate=True)

mechanical_model = MechanicalModel(name='mech_system')
mechanical_model.set_magnetic_spring(spring)
mechanical_model.set_magnet_assembly(magnet_assembly)
mechanical_model.set_damper(damper)
mechanical_model.set_input(accelerometer)

# ELECTRICAL MODEL
flux_database = FluxDatabase(csv_database_path='/home/michael/Dropbox/PhD/Python/Research/fea-flux-curves-numr[5,15]-numz[17,33,66]-wdiam[0.15]-2018-12-07.csv', fixed_velocity=0.35)

coil_center = {'A': 58.5/1000,
               'B': 61/1000}

winding_num_z = {'A': '17',
                 'B': '33'}

winding_num_r = {'A': '15',
                 'B': '15'}

flux_model = flux_database.query_to_model(flux_model_type='unispline',
                                          coil_center=coil_center[which_device],
                                          mm=10,
                                          winding_num_z=winding_num_z[which_device],
                                          winding_num_r=winding_num_r[which_device])

load_model = SimpleLoad(np.inf)  # open-circuit

electrical_model = ElectricalModel(name='A')
electrical_model.set_flux_model(flux_model, precompute_gradient=True)
electrical_model.set_load_model(load_model)

# COUPLING MODEL
coupling_model = CouplingModel().set_coupling_constant(c=0)

# SYSTEM MODEL
governing_equations = unified_ode

# UNIFIED MODEL
unified_model = UnifiedModel(name='Unified')
unified_model.set_mechanical_model(mechanical_model)
unified_model.set_electrical_model(electrical_model)
unified_model.set_coupling_model(coupling_model)
unified_model.set_governing_equations(governing_equations)

unified_model.set_post_processing_pipeline(clip_x2, name='clip tube velocity')

y0 = [0, 0, 0.04, 0, 0]
unified_model.solve(t_start=0, t_end=15,
                    y0=y0,
                    t_max_step=1e-3)

# POST-PROCESSING
df_result = unified_model.get_result(time='t',
                                     x1='x1',
                                     x2='x2',
                                     x3='x3',
                                     x4='x4',
                                     x5='x5',
                                     rel_disp='x3-x1',
                                     rel_vel='x4-x2')

df_result['emf'] = np.gradient(df_result['x5'].values)/np.gradient(df_result['time'].values)

# COMPARISON
pixel_scale = 0.18745
voltage_div_ratio = 1/0.342

lp = LabeledVideoProcessor(L=125, mm=10, seconds_per_frame=3 / 240)
adc = AdcProcessor(voltage_div_ratio, smooth=True)

# Target values
y_target, time_target = lp.fit_transform(video_labels_df , True, pixel_scale)
emf_target, emf_time_target = adc.fit_transform(acc_adc_data_path)

# Predicted values
y_predicted = df_result['rel_disp'].values
emf_predicted = df_result['emf']
time_predicted = df_result['time'].values

# BEGIN EVALUATION
mech_eval = MechanicalSystemEvaluator(y_target,
                                      time_target)
mech_eval.fit(y_predicted, time_predicted)

mech_scores = mech_eval.score(mae=mean_absolute_error,
                              mse=mean_squared_error,
                              mde=median_absolute_error,
                              corr=corr_coeff,
                              explained_var=explained_variance_score,
                              r2=r2_score,
                              max_err=max_err,
                              mape=mean_absolute_percentage_err)

mech_eval.poof(include_dtw=True)

# Evaluate electrical system
ec_eval = ElectricalSystemEvaluator(emf_target, emf_time_target)
ec_eval.fit(emf_predicted, time_predicted)

elec_scores = ec_eval.score(rms=root_mean_square)
ec_eval.poof(include_dtw=True)

print('Mechanical System:')
print(mech_scores)
print('Electrical System:')
print(elec_scores)

