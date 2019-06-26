from experiments.config import abc
from itertools import product
import numpy as np
import pandas as pd
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
from unified_model.utils.utils import collect_samples, build_paramater_grid
from unified_model.governing_equations import unified_ode
from unified_model.pipeline import clip_x2

unified_model = UnifiedModel.load_from_disk('./my_saved_model/')

base_groundtruth_path = './experiments/data/2019-05-23/'
a_samples = collect_samples(base_path=base_groundtruth_path,
                            acc_pattern='A/*acc*.csv',
                            adc_pattern='A/*adc*.csv',
                            labeled_video_pattern='A/*labels*.csv')


accelerometer_inputs = [AccelerometerInput(raw_accelerometer_input=sample.acc_df,
                                           accel_column='z_G',
                                           time_column='time(ms)',
                                           accel_unit='g',
                                           time_unit='ms',
                                           smooth=True,
                                           interpolate=True)
                        for sample
                        in a_samples]

def make_mechanical_spring(damper_constant):
    return MechanicalSpring(push_direction='down',
                            position=110/1000,
                            strength=1000,
                            pure=False,
                            damper_constant=damper_constant)

damping_coefficients = np.linspace(0.01, 0.5, 2)
mech_spring_coefficients = [0.0125]  # Found from investigation
constant_coupling_values = np.linspace(0.5, 2, 2)

param_dict = {'damper': damping_coefficients,
              'mechanical_spring': mech_spring_coefficients,
              'coupling': constant_coupling_values}

func_dict = {'damper': DamperConstant,
             'mechanical_spring': make_mechanical_spring,
             'coupling': ConstantCoupling}

param_grid, val_grid = build_paramater_grid(param_dict, func_dict)

pixel_scale = 0.18451  # Huawei P10 alternative
labeled_video_processor = LabeledVideoProcessor(L=125,
                                                mm=10,
                                                seconds_per_frame=2/118,
                                                pixel_scale=pixel_scale)

mechanical_metrics = {'dtw_euclid': dtw_euclid_distance}


def search_grid(sample_collection, base_unified_model, param_grid):
    pass


# TODO: Make a utility to do this
scores = []
mech_evals = []
which_sample = 1
for damper, mech_spring, coupling in tqdm(list(param_grid)):
    # Update
    unified_model.mechanical_model.set_input(accelerometer_inputs[which_sample])
    unified_model.mechanical_model.set_damper(damper)
    unified_model.mechanical_model.set_mechanical_spring(mech_spring)
    unified_model.add_coupling_model(coupling)

    # Solve
    unified_model.solve(t_start=0,
                        t_end=10,
                        t_max_step=1e-3,
                        y0=[0., 0., 0.04, 0., 0.])

    mech_scores, m_eval = unified_model.score_mechanical_model(metrics_dict=mechanical_metrics,
                                                               video_labels_df=a_samples[which_sample].video_labels_df,
                                                               labeled_video_processor=labeled_video_processor,
                                                               prediction_expr='x3-x1',
                                                               return_evaluator=True,
                                                               use_processed_signals=False)
    scores.append(mech_scores)
    mech_evals.append(m_eval)


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


df = scores_to_dataframe(scores, val_grid, param_names=['friction_damping', 'spring_damping', 'em_coupling'])

# from plotnine import *
# p = ggplot(aes(x='friction_damping', y='em_coupling', size='dtw_euclid'), df)
# p = p + geom_point()
# p.__repr__()

