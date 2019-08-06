from config import abc
from itertools import product
import os
import numpy as np
import pandas as pd
from plotnine import *
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, r2_score, median_absolute_error
from tqdm import tqdm

from unified_model.metrics import *
from unified_model.evaluate import AdcProcessor, LabeledVideoProcessor, MechanicalSystemEvaluator
from unified_model.unified import UnifiedModel
from unified_model.coupling import ConstantCoupling
from unified_model.electrical_model import ElectricalModel
from unified_model.electrical_system.load import SimpleLoad
from unified_model.mechanical_model import MechanicalModel
from unified_model.mechanical_system.damper import DamperConstant
from unified_model.mechanical_system.spring.mechanical_spring import MechanicalSpring
from unified_model.mechanical_system.input_excitation.accelerometer import AccelerometerInput
from unified_model.utils.utils import collect_samples, build_paramater_grid, update_nested_attributes
from unified_model.governing_equations import unified_ode
from unified_model.pipeline import clip_x2


base_unified_model = UnifiedModel.load_from_disk('../my_saved_model/')

base_groundtruth_path = './data/2019-05-23/'

pixel_scales = {'A': 0.18451, 'B': 0.148148}
seconds_per_frame = {'A': 1/80, 'B': 1/60}
samples = {}
samples['A'] = collect_samples(base_path=base_groundtruth_path,
                            acc_pattern='A/*acc*.csv',
                            adc_pattern='A/*adc*.csv',
                            labeled_video_pattern='A/*labels*.csv')
samples['B'] = collect_samples(base_path=base_groundtruth_path,
                            acc_pattern='B/*acc*.csv',
                            adc_pattern='B/*adc*.csv',
                            labeled_video_pattern='B/*labels*.csv')
accelerometer_inputs = {}
accelerometer_inputs['A'] = [AccelerometerInput(raw_accelerometer_input=sample.acc_df,
                                           accel_column='z_G',
                                           time_column='time(ms)',
                                           accel_unit='g',
                                           time_unit='ms',
                                           smooth=True,
                                           interpolate=True)
                        for sample
                        in samples['A']]

accelerometer_inputs['B'] = [AccelerometerInput(raw_accelerometer_input=sample.acc_df,
                                           accel_column='z_G',
                                           time_column='time(ms)',
                                           accel_unit='g',
                                           time_unit='ms',
                                           smooth=True,
                                           interpolate=True)
                        for sample
                        in  samples['B']]


def make_mechanical_spring(damper_constant):
    return MechanicalSpring(push_direction='down',
                            position=110/1000,
                            strength=1000,
                            pure=False,
                            damper_constant=damper_constant)
####################
which_device = 'B'
####################

pixel_scale = pixel_scales[which_device]
accelerometer_inputs = accelerometer_inputs[which_device]
seconds_per_frame = seconds_per_frame[which_device]

damping_coefficients = np.linspace(0.03, 0.04, 10)
mech_spring_coefficients = np.linspace(0, 0.25, 5) #[0.0125]  # Found from investigation
constant_coupling_values = np.linspace(0.3, 2, 10)

# Single-values (set parameters for test)
flux_models = which_device
dflux_models = which_device
coil_resistance = which_device
damping_coefficients = [0.035]
mech_spring_coefficients = [0.0000]
constant_coupling_values = [0.5]
rectification_drop = [0.10]

param_dict = {'mechanical_model.damper': damping_coefficients,
              'mechanical_model.mechanical_spring': mech_spring_coefficients,
              'coupling_model': constant_coupling_values,
              'electrical_model.rectification_drop': rectification_drop,
              'electrical_model.flux_model': flux_models,
              'electrical_model.dflux_model': dflux_models,
              'electrical_model.load': coil_resistance}

func_dict = {'mechanical_model.damper': DamperConstant,
             'mechanical_model.mechanical_spring': make_mechanical_spring,
             'coupling_model': ConstantCoupling,
             'electrical_model.rectification_drop': lambda x: rectification_drop[0],
             'electrical_model.flux_model': lambda x: abc.flux_models[x],
             'electrical_model.dflux_model': lambda x: abc.dflux_models[x],
             'electrical_model.load': lambda x: SimpleLoad(R=abc.coil_resistance[x])}

param_grid, val_grid = build_paramater_grid(param_dict, func_dict)

labeled_video_processor = LabeledVideoProcessor(L=125,
                                                mm=10,
                                                seconds_per_frame=seconds_per_frame,
                                                pixel_scale=pixel_scale)

voltage_division_ratio = 1/0.342
adc_processor = AdcProcessor(voltage_division_ratio=voltage_division_ratio,
                             smooth=True)


mechanical_metrics = {'dtw_euclid_m': dtw_euclid_distance}
mechanical_v_metrics = {'dtw_euclid_mv': dtw_euclid_distance}
electrical_metrics = {'rms_perc_diff': root_mean_square_percentage_diff,
                      'dtw_euclid_e': dtw_euclid_distance}


which_sample = 1
mech_scores = []
mech_v_scores = []
elec_scores = []
y_target, y_time_target = labeled_video_processor.fit_transform(samples[which_device][which_sample].video_labels_df,
                                                              impute_missing_values=True)

emf_target, emf_time_target = adc_processor.fit_transform(samples[which_device][which_sample].adc_df)


yv_target = signal.savgol_filter(y_target, 9, 4)
yv_target = np.gradient(yv_target)/np.gradient(y_time_target)


for param_set in tqdm(param_grid):
    new_unified_model = update_nested_attributes(base_unified_model,
                                                 update_dict=param_set)

    new_unified_model.solve(t_start=0,
                            t_end=8,
                            t_max_step=1e-3,
                            y0=[0., 0., 0.04, 0., 0.])


    m_score, m_eval = new_unified_model.score_mechanical_model(metrics_dict=mechanical_metrics,
                                                               y_target=y_target,
                                                               time_target=y_time_target,
                                                               prediction_expr='x3-x1',
                                                               warp=False,
                                                               return_evaluator=True)

    mv_score, mv_eval = new_unified_model.score_mechanical_model(metrics_dict=mechanical_v_metrics,
                                                                 y_target=yv_target,
                                                                 time_target=y_time_target,
                                                                 prediction_expr='x4-x2',
                                                                 warp=False,
                                                                 return_evaluator=True)

    e_score, e_eval = new_unified_model.score_electrical_model(time_target=emf_time_target,
                                                               emf_target=emf_target,
                                                               metrics_dict=electrical_metrics,
                                                               prediction_expr='g(t, x5)',
                                                               warp=False,
                                                               return_evaluator=True,
                                                               closed_circuit=True)
    mech_v_scores.append(mv_score)
    mech_scores.append(m_score)
    elec_scores.append(e_score)


def scores_to_dataframe(scores, param_values_grid, param_names):
    metrics = list(scores[0]._asdict().keys())
    accumulated_metrics = {m:[] for m in metrics}

    # Get score metrics
    for s in scores:
        for m in metrics:
            accumulated_metrics[m].append(s._asdict()[m])

    # Get parameter values that led to scores
    for i, name in enumerate(param_names):
        accumulated_metrics[name] = [param_values_tuple[i]
                                     for param_values_tuple
                                     in param_values_grid]

    return pd.DataFrame(accumulated_metrics)


df = scores_to_dataframe(mech_scores, val_grid, param_names=['friction_damping', 'spring_damping', 'em_coupling'])
df_elec = scores_to_dataframe(elec_scores, val_grid, param_names=['friction_damping', 'spring_damping', 'em_coupling'])
df_mv = scores_to_dataframe(mech_v_scores, val_grid, param_names=['friction_damping', 'spring_damping', 'em_coupling'])

df['dtw_euclid_mv'] = df_mv['dtw_euclid_mv']
df['abs_rms_perc_diff'] = np.abs(df_elec['rms_perc_diff'])
df['dtw_euclid_e'] = df_elec['dtw_euclid_e']

df.to_csv('result.csv')

df['dtw_euclid_m_'] = df['dtw_euclid_m']/np.max(df['dtw_euclid_m'])
df['dtw_euclid_mv_'] = df['dtw_euclid_mv']/np.max(df['dtw_euclid_mv'])
df['mms'] = (df['dtw_euclid_m_'] + df['dtw_euclid_mv_'])/2  # Mean Mech Score --> MMS

df.to_csv('result_{}.csv'.format(which_sample))
