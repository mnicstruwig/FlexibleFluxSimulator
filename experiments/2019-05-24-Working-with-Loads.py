from experiments.config import abc

import numpy as np
import matplotlib.pyplot as plt

from unified_model.metrics import *
from unified_model.evaluate import AdcProcessor
from unified_model.unified import UnifiedModel
from unified_model.coupling import ConstantCoupling
from unified_model.electrical_model import ElectricalModel
from unified_model.electrical_system.load import SimpleLoad
from unified_model.mechanical_model import MechanicalModel
from unified_model.mechanical_system.damper import DamperConstant
from unified_model.mechanical_system.input_excitation.accelerometer import AccelerometerInput
from unified_model.utils.utils import collect_samples
from unified_model.governing_equations import unified_ode
from unified_model.pipeline import clip_x2

import warnings
warnings.simplefilter('ignore')

base_groundtruth_path = './experiments/data/2019-05-23/'
a_samples = collect_samples(base_path=base_groundtruth_path,
                            acc_pattern='A/*acc*.csv',
                            adc_pattern='A/*adc*.csv',
                            labeled_video_pattern='A/*labels*.csv')

mechanical_model = MechanicalModel(name='Mechanical Model')
mechanical_model.set_spring(abc.spring)
mechanical_model.set_magnet_assembly(abc.magnet_assembly)
mechanical_model.set_damper(DamperConstant(damping_coefficient=0.055))  # Tweaking will need to happen


accelerometer_inputs = [AccelerometerInput(raw_accelerometer_input=sample.acc_df,
                                           accel_column='z_G',
                                           time_column='time(ms)',
                                           accel_unit='g',
                                           time_unit='ms',
                                           smooth=True,
                                           interpolate=True)
                        for sample
                        in a_samples]

which_sample = 1
mechanical_model.set_input(accelerometer_inputs[which_sample])  # Choose which input to system

electrical_model = ElectricalModel(name=None)
electrical_model.set_coil_resistance(abc.coil_resistance['A'])  # Guessing this value for the time being
electrical_model.set_load_model(SimpleLoad(R=np.inf))  # Make sure this is correct!
electrical_model.set_flux_model(abc.flux_models['A'], precompute_gradient=True)

coupling_model = ConstantCoupling(c=0)  # This will need to be found.

unified_model = UnifiedModel(name='Unified Model')
unified_model.add_mechanical_model(mechanical_model)
unified_model.add_electrical_model(electrical_model)
unified_model.add_coupling_model(coupling_model)
unified_model.add_governing_equations(unified_ode)
unified_model.add_post_processing_pipeline(clip_x2, name='clip tube velocity')

# Execute and collect results
unified_model.solve(t_start=0,
                    t_end=10,
                    t_max_step=1e-3,
                    y0=[0., 0., 0.04, 0., 0.])

result = unified_model.get_result(time='t',
                                  x1='x1',
                                  x2='x2',
                                  acc='g(t, x2)',
                                  rel_pos='x3-x1',
                                  rel_vel='x4-x2',
                                  flux='x5',
                                  emf='g(t, x5)')

# Mechanical Scoring
# Will only be handled once video has been labeled.

# EMF Scoring

electrical_metrics = {'rms': root_mean_square,
                      'rms_err_perc': root_mean_square_percentage_diff}

voltage_division_ratio = 1/0.342

adc_processor = AdcProcessor(voltage_division_ratio=voltage_division_ratio,
                             smooth=False,
                             critical_frequency=1 / 8)

emf_scores, e_eval = unified_model.score_electrical_model(metrics_dict=electrical_metrics,
                                                          adc_df=a_samples[which_sample].adc_df,
                                                          adc_processor=adc_processor,
                                                          prediction_expr='g(t, x5)',
                                                          return_evaluator=True)

# result.plot(x='time', y='rel_pos') result.plot(x='time', y='flux')
# result.plot(x='time', y='emf')
# plt.figure()
e_eval.poof(True)

print(emf_scores)
