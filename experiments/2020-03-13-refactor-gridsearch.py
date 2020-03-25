import numpy as np
import ray
from scipy.signal import savgol_filter

from itertools import product
from collections import namedtuple
import logging
from copy import copy

from unified_model import UnifiedModel
from unified_model import MechanicalModel
from unified_model import ElectricalModel
from unified_model import CouplingModel
from unified_model import mechanical_components
from unified_model import electrical_components
from unified_model import governing_equations
from unified_model import pipeline
from unified_model import evaluate
from unified_model import metrics

from unified_model.utils.utils import collect_samples
from config import abc_config

logging.basicConfig(format='%(asctime)s :: %(levelname)s :: %(message)s', level=logging.INFO)

class AbstractUnifiedModelFactory:
    def __init__(self,
                 mechanical_components,
                 electrical_components,
                 coupling_models,
                 governing_equations,
                 ):
        self.mechanical_components = mechanical_components
        self.electrical_components = electrical_components
        self.coupling_models = coupling_models
        self.governing_equations = governing_equations

        self.combined = {}
        self.combined.update(mechanical_components)
        self.combined.update(electrical_components)
        self.combined['coupling_model'] = coupling_models
        self.combined['governing_equations' ] = self.governing_equations
        self.passed_kwargs = []

    def generate(self):
        """Generate a UnifiedModelFactory for a combination of all components

        Each yielded UnifiedModelFactory can produce a UnifiedModel that
        consists of one of all the possible combination of `mechanical`-
        `electrical`-, and `coupling`-models and governing equations specified
        in the constructor.

        Yields
        ------
        UnifiedModelFactory
            A unified model factory containing one combination of parameters.

        """
        param_product = product(*[v for v in self.combined.values()])
        for pp in param_product:
            new_kwargs = {k:v for k, v in zip(self.combined.keys(), pp)}
            self.passed_kwargs.append(new_kwargs)
            yield UnifiedModelFactory(**new_kwargs)


class UnifiedModelFactory:
    """A factory that produces UnifiedModel object.

    Not designed to be used directly -- use the `AbstractUnifiedModelFactory`
    instead.
    """

    def __init__(self,
                 damper=None,
                 magnet_assembly=None,
                 magnetic_spring=None,
                 mechanical_spring=None,
                 input_excitation=None,
                 coil_resistance=None,
                 rectification_drop=None,
                 load_model=None,
                 flux_model=None,
                 dflux_model=None,
                 coupling_model=None,
                 governing_equations=None):
        self.damper=damper
        self.magnet_assembly=magnet_assembly
        self.magnetic_spring = magnetic_spring
        self.mechanical_spring = mechanical_spring
        self.input_excitation=input_excitation
        self.coil_resistance=coil_resistance
        self.rectification_drop=rectification_drop
        self.load_model=load_model
        self.flux_model=flux_model
        self.dflux_model=dflux_model
        self.coupling_model=coupling_model
        self.governing_equations=governing_equations

    def make(self):
        mechanical_model = (
            MechanicalModel()
            .set_input(self.input_excitation)
            .set_damper(self.damper)
            .set_magnet_assembly(self.magnet_assembly)
            .set_mechanical_spring(self.mechanical_spring)
            .set_magnetic_spring(self.magnetic_spring)
        )
        electrical_model = (
            ElectricalModel()
            .set_coil_resistance(self.coil_resistance)
            .set_rectification_drop(self.rectification_drop)
            .set_load_model(self.load_model)
            .set_flux_model(self.flux_model, self.dflux_model)
        )
        unified_model = (
            UnifiedModel()
            .set_mechanical_model(mechanical_model)
            .set_electrical_model(electrical_model)
            .set_coupling_model(self.coupling_model)
            .set_governing_equations(self.governing_equations)
            .set_post_processing_pipeline(pipeline.clip_x2, name='clip x2')
        )
        return unified_model

    def get_parameters_as_str(self, parameters_to_get):

        def _walk_attrs(obj, search_str):
            split_ = search_str.split('.')
            value = obj
            for s in split_:
                value = value.__dict__[s]
            return value

        parameter_str = ''
        for param in parameters_to_get:
            attr_value = _walk_attrs(self, param)
            parameter_str += f'{param}: {attr_value}\n'
        return parameter_str


class ConstantDamperFactory:
    def __init__(self, c_list):
        self.c_list = c_list

    def make(self):
        return [mechanical_components.ConstantDamper(c) for c in self.c_list]


class MechanicalSpringFactory:
    def __init__(self, position, damper_constant_list):
        self.position = position
        self.damper_constant_list = damper_constant_list

    def make(self):
        return [mechanical_components.MechanicalSpring(self.position,
                                                       damper_constant=dc)
                for dc
                in self.damper_constant_list]


class CouplingModelFactory:
    def __init__(self, coupling_factor_list):
        self.coupling_factor_list = coupling_factor_list

    def make(self):
        return [CouplingModel().set_coupling_constant(c) for c in self.coupling_factor_list]


class AccelerometerInputsFactory:
    def __init__(self, sample_list, acc_input_kwargs=None):
        self.sample_list = sample_list
        self.acc_input_kwargs = {} if acc_input_kwargs is None else acc_input_kwargs
        self._set_defaults()

    def _set_defaults(self):
        self.acc_input_kwargs.setdefault('accel_column', 'z_G'),
        self.acc_input_kwargs.setdefault('time_column', 'time(ms)'),
        self.acc_input_kwargs.setdefault('accel_unit', 'g'),
        self.acc_input_kwargs.setdefault('time_unit', 'ms'),
        self.acc_input_kwargs.setdefault('smooth', True),
        self.acc_input_kwargs.setdefault('interpolate', True)

    def make(self):
        accelerometer_inputs = []
        for sample in self.sample_list:
            acc_input = mechanical_components.AccelerometerInput(
                raw_accelerometer_input=sample.acc_df,
                accel_column=self.acc_input_kwargs.setdefault('accel_column', 'z_G'),
                time_column=self.acc_input_kwargs.setdefault('time_column', 'time(ms)'),
                accel_unit=self.acc_input_kwargs.setdefault('accel_unit', 'g'),
                time_unit=self.acc_input_kwargs.setdefault('time_unit', 'ms'),
                smooth=self.acc_input_kwargs.setdefault('smooth', True),
                interpolate=self.acc_input_kwargs.setdefault('interpolate', True)
            )
            accelerometer_inputs.append(acc_input)
        return accelerometer_inputs


class GroundTruthFactory:
    def __init__(self, samples_list, lvp_kwargs, adc_kwargs):
        self.samples_list = samples_list
        self.lvp_kwargs = lvp_kwargs
        self.adc_kwargs = adc_kwargs

        self.lvp = evaluate.LabeledVideoProcessor(**lvp_kwargs)
        self.adc = evaluate.AdcProcessor(**adc_kwargs)
        self.MechGroundtruth = namedtuple('MechanicalGroundtruth', ['y_diff', 'time'])
        self.ElecGroundtruth = namedtuple('ElectricalGroundtruth', ['emf', 'time'])
        self.Groundtruth = namedtuple('Groundtruth', ['mech', 'elec'])

    def _make_mechanical_groundtruth(self, sample):
        y_target, y_time_target = self.lvp.fit_transform(
            sample.video_labels_df,
            impute_missing_values=True
        )
        y_target = savgol_filter(y_target, 9, 3)

        return self.MechGroundtruth(y_target,
                                   y_time_target)

    def _make_electrical_groundtruth(self, sample):
        emf_target, emf_time_target = self.adc.fit_transform(sample.adc_df)
        return self.ElecGroundtruth(emf_target,
                                    emf_time_target)

    def make(self):
        groundtruths = []
        for sample in self.samples_list:
            mech_groundtruth = self._make_mechanical_groundtruth(sample)
            elec_groundtruth = self._make_electrical_groundtruth(sample)

            groundtruths.append(
                self.Groundtruth(mech_groundtruth, elec_groundtruth)
            )

        return groundtruths


def chunk(array_like, chunk_size):
    """Chunk up an array-like. Generator"""
    total_size = len(array_like)
    indexes = list(range(0, total_size, chunk_size))

    # Make sure we get the final chunk:
    if indexes[-1] < total_size:
        indexes.append(total_size)

    for start, stop in zip(indexes, indexes[1:]):
        yield array_like[start:stop]

@ray.remote
def run_cell(unified_model_factory, groundtruth, metrics):
    model = unified_model_factory.make()
    model.solve(t_start=0,
                t_end=8,
                t_max_step=1e-3,
                y0=[0.0, 0.0, 0.04, 0.0, 0.0])

    mech_score, mech_eval = model.score_mechanical_model(
        time_target=groundtruth.mech.time,
        y_target=groundtruth.mech.y_diff,
        metrics_dict=metrics['mechanical'],
        prediction_expr='x3-x1',
        return_evaluator=True
    )

    elec_score, elec_eval = model.score_electrical_model(
        time_target=groundtruth.elec.time,
        emf_target=groundtruth.elec.emf,
        metrics_dict=metrics['electrical'],
        prediction_expr='g(t, x5)',
        return_evaluator=True,
        clip_threshold=1e-1
    )

    scores = {'mechanical': mech_score,
              'electrical': elec_score}

    curves = {'time': mech_eval.time_,
              'y_diff': mech_eval.y_predict_,
              'emf': elec_eval.emf_predict_}

    return scores, curves


def execute_gridsearch_for_sample(abstract_unified_model_factory,
                                  groundtruth,
                                  metrics,
                                  batch_size=8):

    # This doesn't make me happy, but I'm out of clean ideas for now If I want
    # this to work and _also_ preserve order.
    model_factories = list(abstract_unified_model_factory.generate())
    total_completed = 0
    total_tasks = len(model_factories)

    grid_scores = []
    grid_curves = []
    for model_factory_batch in chunk(model_factories, batch_size):
        task_queue = []
        for model_factory in model_factory_batch:
            task_id = run_cell.remote(model_factory, groundtruth, metrics)
            task_queue.append(task_id)

        ready = []
        while len(ready) < len(task_queue):
            ready, remaining = ray.wait(task_queue, num_returns=len(task_queue), timeout=5.)
            logging.info(f'Progress: {len(ready)+total_completed}/{total_tasks}')

        # Once all tasks are completed...
        total_completed += len(ready)# ... increment the total completed counter ...
        results = [ray.get(task_id) for task_id in task_queue]  # ... and fetch results

        # Parse the results
        for result in results:
            grid_scores.append(copy(result[0]))
            grid_curves.append(copy(result[1]))

        del results  # Remove reference so Ray can free memory as needed
    return grid_scores, grid_curves

base_groundtruth_path = './data/2019-05-23_C/'
samples = {}
samples['A'] = collect_samples(base_path=base_groundtruth_path,
                               acc_pattern='A/*acc*.csv',
                               adc_pattern='A/*adc*.csv',
                               video_label_pattern='A/*labels*.csv')
samples['B'] = collect_samples(base_path=base_groundtruth_path,
                               acc_pattern='B/*acc*.csv',
                               adc_pattern='B/*adc*.csv',
                               video_label_pattern='B/*labels*.csv')
samples['C'] = collect_samples(base_path=base_groundtruth_path,
                               acc_pattern='C/*acc*.csv',
                               adc_pattern='C/*adc*.csv',
                               video_label_pattern='B/*labels*.csv')

input_excitation_factories = {k: AccelerometerInputsFactory(samples[k])
                              for k in ['A', 'B', 'C']}

magnetic_spring = mechanical_components.MagneticSpringInterp(
    fea_data_file='./data/magnetic-spring/10x10alt.csv',
    filter_obj=lambda x: savgol_filter(x, 27, 5)
)

magnet_assembly = mechanical_components.MagnetAssembly(
    n_magnet=1,
    l_m=10,
    l_mcd=0,
    dia_magnet=10,
    dia_spacer=10
)

mech_components = {
    'input_excitation': input_excitation_factories['A'].make()[:1],
    'magnetic_spring': [magnetic_spring],
    'magnet_assembly': [magnet_assembly],
    'damper': ConstantDamperFactory(np.linspace(0.01, 0.07, 10)).make(),  # np.linspace(0.01, 0.07, 5)
    'mechanical_spring':  MechanicalSpringFactory(110/1000,
                                                  [0]).make(),
}

elec_components = {
    'coil_resistance': [abc_config.coil_resistance['A']],
    'rectification_drop': [0.1],
    'load_model': [electrical_components.SimpleLoad(30)],
    'flux_model': [abc_config.flux_models['A']],
    'dflux_model': [abc_config.dflux_models['A']],

}

governing_equations = [governing_equations.unified_ode]

coupling_models = CouplingModelFactory([0]).make()

abstract_model_factory = AbstractUnifiedModelFactory(
    mech_components,
    elec_components,
    coupling_models,
    governing_equations
)

groundtruth_factory = GroundTruthFactory(
    samples['A'][:5],
    lvp_kwargs=dict(L=125,
                    mm=10,
                    seconds_per_frame=1/60,
                    pixel_scale=0.154508),
    adc_kwargs=dict(voltage_division_ratio=1 / 0.342)
)


# Metrics
mechanical_metrics = {
    'dtw_distance': metrics.dtw_euclid_distance,
}
electrical_metrics = {
    'rms_perc_diff': metrics.root_mean_square_percentage_diff,
    'dtw_distance': metrics.dtw_euclid_distance
}

metrics = {'mechanical': mechanical_metrics,
           'electrical': electrical_metrics}

# Get groundtruth and score
groundtruth = groundtruth_factory.make()[0]

# Initialize Ray
ray.init(
    memory=8 * 1024 * 1024 * 1024,
    object_store_memory=4 * 1024 * 1024 * 1024,
    ignore_reinit_error=True
)

grid_scores, grid_curves = execute_gridsearch_for_sample(
    abstract_model_factory,
    groundtruth,
    metrics
)

params = abstract_model_factory.passed_kwargs[0]

poi = 'mechanical_spring.k'

def get_nested_param(obj, path):
    split_ = path.split('.')
    temp = obj[split_[0]]
    for s in split_[1:]:
        if isinstance(temp, dict):  # If we have a dict...
            temp = temp[s]
        else:  # If we have an object
            temp = temp.__dict__[s]
    return temp


def scores_to_dataframe(grid_scores, param_dict_list, params_of_interest):
    pass
