import matplotlib.pyplot as plt
import ray
import time
import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm

from config import abc
from unified_model.coupling import ConstantCoupling
from unified_model.evaluate import AdcProcessor, LabeledVideoProcessor
from unified_model.mechanical_system.damper import DamperConstant
from unified_model.mechanical_system.input_excitation.accelerometer import \
    AccelerometerInput
from unified_model.mechanical_system.spring.mechanical_spring import \
    MechanicalSpring
from unified_model.metrics import (dtw_euclid_distance,
                                   root_mean_square_percentage_diff)
from unified_model.unified import UnifiedModel
from unified_model.utils.utils import (build_paramater_grid, collect_samples,
                                       update_nested_attributes)

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
accelerometer_inputs['A'] = [
    AccelerometerInput(raw_accelerometer_input=sample.acc_df,
                       accel_column='z_G',
                       time_column='time(ms)',
                       accel_unit='g',
                       time_unit='ms',
                       smooth=True,
                       interpolate=True) for sample in samples['A']
]

accelerometer_inputs['B'] = [
    AccelerometerInput(raw_accelerometer_input=sample.acc_df,
                       accel_column='z_G',
                       time_column='time(ms)',
                       accel_unit='g',
                       time_unit='ms',
                       smooth=True,
                       interpolate=True) for sample in samples['B']
]


def make_mechanical_spring(damper_constant):
    return MechanicalSpring(push_direction='down',
                            position=110/1000,
                            strength=1000,
                            pure=False,
                            damper_constant=damper_constant)


####################
which_device = 'A'
which_sample = 0

pixel_scale = pixel_scales[which_device]
seconds_per_frame = seconds_per_frame[which_device]

accelerometer_inputs = accelerometer_inputs[which_device]
input_ = [which_sample]
####################


# Gridsearch parameters
damping_coefficients = np.linspace(0.03, 0.04, 10)
mech_spring_coefficients = np.linspace(0, 0.25, 5) #[0.0125]  # Found from investigation
constant_coupling_values = np.linspace(0.3, 2, 10)
rectification_drop = [0.10]
flux_models = [which_device]
dflux_models = [which_device]
coil_resistance = [which_device]

# Overrides (set parameters for test)
# damping_coefficients = [0.035]
# mech_spring_coefficients = [0.0000]
# constant_coupling_values = [0.5]

param_dict = {
    'mechanical_model.input_': input_,
    'mechanical_model.damper': damping_coefficients,
    'mechanical_model.mechanical_spring': mech_spring_coefficients,
    'coupling_model': constant_coupling_values,
    'electrical_model.rectification_drop': rectification_drop,
    'electrical_model.flux_model': flux_models,
    'electrical_model.dflux_model': dflux_models,
    'electrical_model.coil_resistance': coil_resistance
}

func_dict = {
    'mechanical_model.input_': lambda x: accelerometer_inputs[x],
    'mechanical_model.damper': DamperConstant,
    'mechanical_model.mechanical_spring': make_mechanical_spring,
    'coupling_model': ConstantCoupling,
    'electrical_model.rectification_drop': lambda x: x,
    'electrical_model.flux_model': lambda x: abc.flux_models[x],
    'electrical_model.dflux_model': lambda x: abc.dflux_models[x],
    'electrical_model.coil_resistance': lambda x: abc.coil_resistance[x]
}

translation_dict = {
    'mechanical_model.damper': 'friction_damping',
    'mechanical_model.mechanical_spring': 'spring_damping',
    'coupling_model': 'coupling_factor'
}

# Build the grid to search
param_grid, val_grid = build_paramater_grid(param_dict, func_dict)

# Prepare scoring objects + groundtruth
labeled_video_processor = LabeledVideoProcessor(
    L=125,
    mm=10,
    seconds_per_frame=seconds_per_frame,
    pixel_scale=pixel_scale)

voltage_division_ratio = 1/0.342
adc_processor = AdcProcessor(
    voltage_division_ratio=voltage_division_ratio,
    smooth=True)

# Metrics
mechanical_metrics = {'dtw': dtw_euclid_distance}
mechanical_v_metrics = {'dtw': dtw_euclid_distance}
electrical_metrics = {'rms_perc_diff': root_mean_square_percentage_diff,
                      'dtw': dtw_euclid_distance}

metrics = {
    'mechanical': mechanical_metrics,
    'electrical': electrical_metrics
}

mech_scores = []
mech_v_scores = []
elec_scores = []

# Target values
y_target, y_time_target = labeled_video_processor.fit_transform(
    samples[which_device][which_sample].video_labels_df,
    impute_missing_values=True
)
emf_target, emf_time_target = adc_processor.fit_transform(
    samples[which_device][which_sample].adc_df
)
yv_target = signal.savgol_filter(y_target, 9, 4)
yv_target = np.gradient(yv_target)/np.gradient(y_time_target)

# Multiprocessing stuff
ray.init(ignore_reinit_error=True)

@ray.remote
def run_cell(base_unified_model,
             parameter_set,
             y_target,
             y_time_target,
             emf_target,
             metrics):
    """Run a single cell of a gridsearch."""

    mechanical_metrics = metrics['mechanical']
    electrical_metrics = metrics['electrical']

    new_unified_model = update_nested_attributes(base_unified_model,
                                                 update_dict=parameter_set)
    new_unified_model.solve(
        t_start=0,
        t_end=8,
        t_max_step=1e-3,
        y0=[0., 0., 0.04, 0., 0.]
    )

    m_score, m_eval = new_unified_model.score_mechanical_model(
        metrics_dict=mechanical_metrics,
        y_target=y_target,
        time_target=y_time_target,
        prediction_expr='x3-x1',
        warp=False,
        return_evaluator=True
    )

    e_score, e_eval = new_unified_model.score_electrical_model(
        metrics_dict=electrical_metrics,
        emf_target=emf_target,
        time_target=emf_time_target,
        prediction_expr='g(t, x5)',
        warp=False,
        closed_circuit=True,
        clip_threshold=1e-1,
        return_evaluator=True
    )

    scores = {
        'mechanical': m_score,
        'electrical': e_score
    }

    curves = {
        'time': m_eval.time_,
        'y': m_eval.y_predict_,
        'emf': e_eval.emf_predict_
    }

    return scores, curves


jobs = []
param_log = {}

# RUN
for param_set in param_grid:
    # Send to Ray
    cell_id = run_cell.remote(
        base_unified_model,
        param_set,
        y_target,
        y_time_target,
        emf_target,
        metrics
    )
    # Contain list of values
    jobs.append(cell_id)


def get_ray_results(jobs):
    """Get Ray results in a nice way."""
    for job_id in jobs:
        yield ray.get(job_id)

# Track progress
done = False
start_time = time.time()
while(not done):
    ready, remaining = ray.wait(jobs, num_returns=len(jobs), timeout=5.)
    num_ready = len(ready)

    current_time = time.time()
    minutes = int((current_time - start_time)/60)
    seconds = int((current_time - start_time)%60)
    print(f'Progress --- {num_ready} out of {len(jobs)} --- {minutes}m{seconds}s')

    if num_ready == len(jobs):
        done = True


def curves_to_dataframe(curves, sampling_rate=3):
    """Convert a list of dictionaries, `curves`, into a pandas dataframe.

    Each key of each element in `curves` will become a column.
    """
    df_list = []
    for i, c in enumerate(curves):
        # Subsample
        subsampled_curve = {}
        for k, v in c.items():
            subsampled_curve[k] = v[::sampling_rate]

        single_curve = pd.DataFrame(subsampled_curve)
        single_curve['idx'] = i
        df_list.append(single_curve)

    return pd.concat(df_list)


def scores_to_dataframe(scores_dict,
                        param_values_grid,
                        translation_dict):
    """
    Transform scores, with model parameters that produced those scores, into
    a dataframe.
    """

    accumulated_scores = {}
    accumulated_parameters = {}
    for i, score_dict in enumerate(scores):  # Each score collection
        # Accumulate metrics
        for category, score_obj in score_dict.items():  # Each category of score
            for metric, value in score_obj._asdict().items():  # For each attribute
                metric_name = category[:4] + '_' + metric
                # Accumulate the scores
                try:
                    accumulated_scores[metric_name].append(value)
                except KeyError:  # When encountering metric for the first time
                    accumulated_scores[metric_name] = [value]
        # Accumulate parameters in outer loop (one set produces score set)
        for param, param_alt_name in translation_dict.items():
            try:
                accumulated_parameters[param_alt_name].append(param_values_grid[i][param])
            except KeyError:
                accumulated_parameters[param_alt_name] = [param_values_grid[i][param]]

    accumulated_scores.update(accumulated_parameters)  # Join
    return pd.DataFrame(accumulated_scores)


# Collect results
scores = [result[0] for result in get_ray_results(jobs)]
curves = [result[1] for result in get_ray_results(jobs)]

df_scores = scores_to_dataframe(scores, val_grid, translation_dict)
df_curves = curves_to_dataframe(curves)

df_scores.to_csv(f'{which_device}_{which_sample}_score.csv')
df_curves.to_csv(f'{which_device}_{which_sample}_curves.xz')

metric='mech_dtw_euclid_m'
lowest_mech_error = df_scores.sort_values(by=metric).index[0]
best_curve = df_curves.query(f'idx == {lowest_mech_error}')
time_groundtruth = 
