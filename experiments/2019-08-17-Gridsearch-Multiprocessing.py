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
                                       update_nested_attributes, interpolate_and_resample)

#warnings.filterwarnings('ignore', category=FutureWarning)


def make_mechanical_spring(damper_constant):
    return MechanicalSpring(push_direction='down',
                            position=110/1000,
                            strength=1000,
                            pure=False,
                            damper_constant=damper_constant)


def make_groundtruth_dict(target_time, target_values, label):
    return {
        'time': target_time,
        'values': target_values,
        'label': label
    }


def export_groundtruth(which_device, which_sample, *args):
    """Example element of *args:

    a_groundtruth_dict = {
        'time': y_time_target,
        'value': y_target,
        'signal': 'y'
    }

    Exported in "tidy" format.

    """
    df = pd.concat([pd.DataFrame(dict_) for dict_ in args])
    df.to_parquet(f'{which_device}_{which_sample}_groundtruth.parq',
                  engine='pyarrow',
                  compression='brotli')


def get_mechanical_groundtruth(which_device, which_sample):
    labeled_video_processor = LabeledVideoProcessor(
        L=125,
        mm=10,
        seconds_per_frame=seconds_per_frame,
        pixel_scale=pixel_scale)

    y_target, y_time_target = labeled_video_processor.fit_transform(
        samples[which_device][which_sample].video_labels_df,
        impute_missing_values=True
    )
    # SMOOTH TARGET VALUES  (OPTIONAL?)
    y_target = signal.savgol_filter(y_target, 9, 3)

    return make_groundtruth_dict(y_time_target, y_target, 'y')


def get_electrical_groundtruth(which_device, which_sample):
    voltage_division_ratio = 1/0.342
    adc_processor = AdcProcessor(
        voltage_division_ratio=voltage_division_ratio,
        smooth=True)

    emf_target, emf_time_target = adc_processor.fit_transform(
        samples[which_device][which_sample].adc_df
    )
    return make_groundtruth_dict(emf_time_target, emf_target, 'emf')


def build_groundtruth_data(which_device, which_sample, save_to_disk=True):
    """Prepare and return the groundtruth data for an experiment.

    Also has the side effect of writing the groundtruth to file.
    """

    mechanical_groundtruth_dict = get_mechanical_groundtruth(
        which_device,
        which_sample
    )
    electrical_groundtruth_dict = get_electrical_groundtruth(
        which_device,
        which_sample
    )

    if save_to_disk:
        export_groundtruth(which_device,
                           which_sample,
                           mechanical_groundtruth_dict,
                           electrical_groundtruth_dict)

    return mechanical_groundtruth_dict, electrical_groundtruth_dict


@ray.remote
def run_cell(base_unified_model,
             parameter_set,
             y_time_target,
             y_target,
             emf_time_target,
             emf_target,
             metrics):
    """Run a single cell of a gridsearch.

    Parameters
    ----------
    base_unified_model : UnifiedModel
        The base unified model that will be updated with the parameters in
        `parameter_set`.
    parameter_set : dict
        A dictionary where keys are the attributes of `base_unified_model`
        and the values are the new values that these attributes that must
        hold. See the `build_parameter_grid` function.
    y_time_target : array
        The time values of the mechanical target.
    y_target: array
        The actual values of the mechanical target.
    emf_time_target : array
        The time values of the emf target.
    emf_target : array
        The actual values of the emf target
    metrics : dict
        A dictionary with two elements at keys 'mechanical' and 'electrical'.
        The value of `metrics['mechanical']` must be a dict whose keys are a
        user-selected name for the metric, and the value must be a metric
        function (see the `metrics` module for examples) that is used to score
        the mechanical system's performance. Likewise, the value of
        `metrics['electrical']` must be a dict whose keys are a user-selected
        name for the metric, and the value must be a metric function is used to
        score the electrical system's performance.

    Returns
    -------
    scores : dict
        Dictionary with keys 'mechanical' and 'electrical' that contain the
        calculated metrics specified by the `metrics` argument.
    curves : dict
        Dictionary containing the predicted mechanical and electrical signals,
        as well as the corresponding timestamps, as calculated by the model.

    """

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


def indexify(array_like, step):
    """Get the indexes to step through an array-like"""
    total_size = len(array_like)
    indexes = np.arange(0, total_size, step)

    if indexes[-1] < total_size:
        indexes = np.append(indexes, total_size)
    return zip(indexes, indexes[1:])


def execute_in_batches(base_unified_model,
                       param_grid,
                       y_time_target,
                       y_target,
                       emf_time_target,
                       emf_target,
                       metrics,
                       batch_size=48):
    """Execute the grid search in batches."""

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
                y_time_target,
                y_target,
                emf_time_target,
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
        # To list to stop from getting consumed
        results = list(get_ray_results(jobs))
        # By copying values, we may allow for Ray to evict these results from
        # the Object Store.
        scores = np.append(scores, np.array([x[0] for x in results], copy=True))
        curves = np.append(curves, np.array([x[1] for x in results], copy=True))

        # Delete reference to allow cleaning of object store (stops memory from filling up)
        for result in results:
            del result

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
        minutes = int((current_time - start_time) / 60)
        seconds = int((current_time - start_time) % 60)
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


# ═════════════════════════════════
# Data preparation
# ═════════════════════════════════
base_unified_model = UnifiedModel.load_from_disk('../my_saved_model/')

base_groundtruth_path = './data/2019-05-23_B/'

pixel_scales = {'A': 0.154508, 'B': 0.154508, 'C': 0.154508} # 'B': 0.148148} A: 0.18451
seconds_per_frames = {'A': 1/60, 'B': 1/60, 'C': 1/60}
samples = {}
samples['A'] = collect_samples(base_path=base_groundtruth_path,
                               acc_pattern='A/*acc*.csv',
                               adc_pattern='A/*adc*.csv',
                               labeled_video_pattern='A/*labels*.csv')
samples['B'] = collect_samples(base_path=base_groundtruth_path,
                               acc_pattern='B/*acc*.csv',
                               adc_pattern='B/*adc*.csv',
                               labeled_video_pattern='B/*labels*.csv')
samples['C'] = collect_samples(base_path=base_groundtruth_path,
                               acc_pattern='C/*acc*.csv',
                               adc_pattern='C/*adc*.csv',
                               labeled_video_pattern='B/*labels*.csv')
accelerometer_inputs = {}
accelerometer_inputs['A'] = [
    AccelerometerInput(
        raw_accelerometer_input=sample.acc_df,
        accel_column='z_G',
        time_column='time(ms)',
        accel_unit='g',
        time_unit='ms',
        smooth=True,
        interpolate=True)
    for sample
    in samples['A']
]

accelerometer_inputs['B'] = [
    AccelerometerInput(
        raw_accelerometer_input=sample.acc_df,
        accel_column='z_G',
        time_column='time(ms)',
        accel_unit='g',
        time_unit='ms',
        smooth=True,
        interpolate=True)
    for sample
    in samples['B']
]

accelerometer_inputs['C'] = [
    AccelerometerInput(
        raw_accelerometer_input=sample.acc_df,
        accel_column='z_G',
        time_column='time(ms)',
        accel_unit='g',
        time_unit='ms',
        smooth=True,
        interpolate=True)
    for sample
    in samples['C']
]

# ═════════════════════════════════
# Experiment Details
# ═════════════════════════════════
which_device = 'C'
which_samples = [3, 4]

pixel_scale = pixel_scales[which_device]
seconds_per_frame = seconds_per_frames[which_device]

accelerometer_input = accelerometer_inputs[which_device]

# ═════════════════════════════════
# Metrics
# ═════════════════════════════════
mechanical_metrics = {'dtw': dtw_euclid_distance}
mechanical_v_metrics = {'dtw': dtw_euclid_distance}
electrical_metrics = {'rms_perc_diff': root_mean_square_percentage_diff,
                      'dtw': dtw_euclid_distance}

metrics = {
    'mechanical': mechanical_metrics,
    'electrical': electrical_metrics
}


ray.init(
    memory= 8 * 1024 * 1024 * 2014,
    object_store_memory=4 * 1024 * 1024 * 1024,
    ignore_reinit_error=True
)
for which_sample in which_samples:
    # ═════════════════════════════════
    # Prepare Groundtruth
    # ═════════════════════════════════
    y_groundtruth, emf_groundtruth = build_groundtruth_data(
        which_device,
        which_sample
    )
    y_time_target = y_groundtruth['time']
    y_target = y_groundtruth['values']
    emf_time_target = emf_groundtruth['time']
    emf_target = emf_groundtruth['values']

    # ═════════════════════════════════
    # Gridsearch parameters
    # ═════════════════════════════════
    damping_coefficients = np.linspace(0.02, 0.05, 20)
    mech_spring_coefficients = np.linspace(0, 0.5, 5) #[0.0125]  # Found from investigation
    constant_coupling_values = np.linspace(0.1, 3, 10)
    rectification_drop = [0.10]

    # Overrides (set parameters for test)
    # damping_coefficients = [0.035]
    # mech_spring_coefficients = [0.0000]
    # constant_coupling_values = [0.5]

    param_dict = {  # Holds the values we want to map
        'mechanical_model.input_': [which_sample],
        'mechanical_model.damper': damping_coefficients,
        'mechanical_model.mechanical_spring': mech_spring_coefficients,
        'coupling_model': constant_coupling_values,
        'electrical_model.rectification_drop': rectification_drop,
        'electrical_model.flux_model': [which_device],
        'electrical_model.dflux_model': [which_device],
        'electrical_model.coil_resistance': [which_device]
    }

    func_dict = {  # Maps the parameter values to objects used in simulation
        'mechanical_model.input_': lambda x: accelerometer_input[x],
        'mechanical_model.damper': DamperConstant,
        'mechanical_model.mechanical_spring': make_mechanical_spring,
        'coupling_model': ConstantCoupling,
        'electrical_model.rectification_drop': lambda x: x,
        'electrical_model.flux_model': lambda x: abc.flux_models[x],
        'electrical_model.dflux_model': lambda x: abc.dflux_models[x],
        'electrical_model.coil_resistance': lambda x: abc.coil_resistance[x]
    }

    # Allows us to save parameters with nicer names in the dataframe
    translation_dict = {
        'mechanical_model.damper': 'friction_damping',
        'mechanical_model.mechanical_spring': 'spring_damping',
        'coupling_model': 'coupling_factor'
    }

    # Build the grid to search
    param_grid, val_grid = build_paramater_grid(param_dict, func_dict)

    # ═════════════════════════════════
    # Execute
    # ═════════════════════════════════
    scores, curves = execute_in_batches(
        base_unified_model=base_unified_model,
        param_grid=param_grid,
        y_time_target=y_time_target,
        y_target=y_target,
        emf_time_target=emf_time_target,
        emf_target=emf_target,
        metrics=metrics,
        batch_size=16
    )

    df_scores = scores_to_dataframe(scores, val_grid, translation_dict)
    df_curves = curves_to_dataframe(curves, sampling_rate=3)

    df_scores.to_parquet(f'{which_device}_{which_sample}_score.parq',
                         engine='pyarrow',
                         compression='brotli')
    df_curves.to_parquet(f'{which_device}_{which_sample}_curves.parq',
                         engine='pyarrow',
                         compression='brotli')

    # Try ease memory a bit
    del scores
    del curves
