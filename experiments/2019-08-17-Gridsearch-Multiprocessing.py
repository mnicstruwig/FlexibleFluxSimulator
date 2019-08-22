import ray
from plotnine import *
import time
import numpy as np
import pandas as pd
from scipy import signal
import warnings

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

warnings.filterwarnings('ignore', category=FutureWarning)

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
which_sample = 2

pixel_scale = pixel_scales[which_device]
seconds_per_frame = seconds_per_frame[which_device]

accelerometer_inputs = accelerometer_inputs[which_device]
input_ = [which_sample]
####################


# Gridsearch parameters
damping_coefficients = np.linspace(0.02, 0.05, 20)
mech_spring_coefficients = np.linspace(0, 0.5, 5) #[0.0125]  # Found from investigation
constant_coupling_values = np.linspace(0.1, 3, 20)
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
### SMOOTH TARGET VALUES  (OPTIONAL?)
y_target = signal.savgol_filter(y_target, 9, 3)

emf_target, emf_time_target = adc_processor.fit_transform(
    samples[which_device][which_sample].adc_df
)
yv_target = signal.savgol_filter(y_target, 9, 4)
yv_target = np.gradient(yv_target)/np.gradient(y_time_target)


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


def indexify(array_like, step):
    """Get the indexes to step through an array-like"""
    total_size = len(array_like)
    indexes = np.arange(0, total_size, step)

    if indexes[-1] < total_size:
        indexes = np.append(indexes, total_size -1)
    return zip(indexes, indexes[1:])


def execute_in_batches(param_grid, batch_size=48):
    """Execute the gridsearch in batches using Ray."""

    start_time = None
    current_number = 0
    scores = np.array([])
    curves = np.array([])

    # For each batch
    for start, stop in indexify(param_grid, batch_size):
        jobs = []  # Hold each batch of jobs
        for param_set in param_grid[start:stop]:
            # Send to Ray
            id = run_cell.remote(
                base_unified_model,
                param_set,
                y_target,
                y_time_target,
                emf_target,
                metrics
            )

            jobs.append(id)

        # Wait for completion of batch...
        start_time, current_number = wait_for_completion(
            jobs,
            start_time,
            current_number,
            total=len(param_grid)
        )
        # Then retrieve results
        results = list(get_ray_results(jobs))  # Stop from getting consumed 
        scores = np.append(scores, np.array([x[0] for x in results]))
        curves = np.append(curves, np.array([x[1] for x in results]))

    return scores, curves


def get_ray_results(jobs):
    """Get Ray results in a nice way."""
    for job_id in jobs:
        yield ray.get(job_id)


def wait_for_completion(jobs, start_time=None, current_number=0, total=None):
    if start_time is None:
        start_time = time.time()

    done = False
    while(not done):
        ready, remaining = ray.wait(jobs, num_returns=len(jobs), timeout=5.)
        num_ready = len(ready)
        num_completed = len(ready) + current_number

        current_time = time.time()
        minutes = int((current_time - start_time)/60)
        seconds = int((current_time - start_time)%60)
        print(f'Progress --- {num_completed} out of {total} --- {minutes}m{seconds}s')

        if num_ready == len(jobs):
            done = True
    return start_time, num_completed


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


# Execute
ray.init()
scores, curves = execute_in_batches(param_grid)
ray.shutdown()

df_scores = scores_to_dataframe(scores, val_grid, translation_dict)
df_curves = curves_to_dataframe(curves, sampling_rate=4)

df_scores.to_csv(f'{which_device}_{which_sample}_score.csv')
df_curves.to_csv(f'{which_device}_{which_sample}_curves.xz')


def plot_comparison(df_scores,
                    df_curves,
                    time_target,
                    signal_target,
                    signal_column,
                    metric_column,
                    n=1):

    df_groundtruth = pd.DataFrame()
    df_groundtruth['time'] = time_target
    df_groundtruth[signal_column] = signal_target
    df_groundtruth['label'] = 'Groundtruth'
    df_groundtruth['idx'] = 'None'


    best_idx = df_scores.sort_values(metric_column).index[:n]
    best_curves = df_curves[df_curves['idx'].isin(best_idx)]
    best_curves['label'] = 'best candidates'  # Improve in future

    df_curves['label'] = 'experiments'

    p = ggplot(aes(x='time', y=signal_column, group='idx', color='label'), df_curves)
    p = (
        p
        + geom_line(alpha=0.1)  # Experiments
        + geom_line(data=best_curves, size=1)  # Best `n` curves
        + geom_line(data=df_groundtruth, size=1)  # Groundtruth
        + scale_color_manual(['black', '#fb3640', '#007fce'])
    )

    p.__repr__()


plot_comparison(
    df_scores,
    df_curves,
    y_time_target,
    y_target,
    'y',
    'mech_dtw',
    n=1
)


# df_scores = df_scores.reset_index()
# df_scores = df_scores.rename({'index':'idx'}, axis=1)
# df_join = df_curves.join(df_scores, on='idx', lsuffix='_c', rsuffix='_s', how='inner')

