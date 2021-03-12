from itertools import product
from dataclasses import dataclass
from copy import copy

from typing import List, Any, Dict, Callable
import numpy as np
from scipy.signal import savgol_filter

from unified_model import CouplingModel
from unified_model import mechanical_components
from unified_model import electrical_components
from unified_model import governing_equations
from unified_model import evaluate
from unified_model import metrics
from unified_model import gridsearch

from unified_model.utils.utils import collect_samples
from local_config import ABC_CONFIG

class QuasiKarnoppDamperFactory:
    """A factory that returns `QuasiKarnopDamper` objects."""
    def __init__(self,
                 b_m1_list,
                 b_m2_list,
                 magnet_assembly,
                 tube_inner_radius_mm):
        self.b_m1_list = b_m1_list
        self.b_m2_list = b_m2_list
        self.magnet_assembly = copy(magnet_assembly)
        self.tube_inner_radius_mm = tube_inner_radius_mm

        self.param_tuples = product(self.b_m1_list, self.b_m2_list)


    def make(self):
        return [mechanical_components.damper.QuasiKarnoppDamper(b_m1,
                                                                b_m2,
                                                                self.magnet_assembly,
                                                                self.tube_inner_radius_mm)
                for b_m1, b_m2 in self.param_tuples]

class ConstantDamperFactory:
    """A factory that returns `ConstantDamper` objects."""
    def __init__(self, c_list):
        self.c_list = c_list

    def make(self):
        """Make the ConstantDamper objects."""
        return [mechanical_components.ConstantDamper(c) for c in self.c_list]


class MechanicalSpringFactory:
    def __init__(self, magnet_assembly, position, damping_coefficient_list):
        self.position = position
        self.magnet_assembly = copy(magnet_assembly)
        self.damping_coefficient_list = damping_coefficient_list

    def make(self):
        return [mechanical_components.MechanicalSpring(self.magnet_assembly,
                                                       self.position,
                                                       damping_coefficient=dc)

                for dc
                in self.damping_coefficient_list]


class CouplingModelFactory:
    def __init__(self, coupling_factor_list):
        self.coupling_factor_list = coupling_factor_list

    def make(self):
        return [CouplingModel().set_coupling_constant(c)
                for c in self.coupling_factor_list]


class AccelerometerInputsFactory:
    def __init__(self, sample_list, acc_input_kwargs=None):
        self.sample_list = sample_list
        self.acc_input_kwargs = {} if acc_input_kwargs is None else acc_input_kwargs  # noqa
        self._set_defaults()

    def _set_defaults(self):
        self.acc_input_kwargs.setdefault('accel_column', 'z_G'),
        self.acc_input_kwargs.setdefault('time_column', 'time(ms)'),
        self.acc_input_kwargs.setdefault('accel_unit', 'g'),
        self.acc_input_kwargs.setdefault('time_unit', 'ms'),
        self.acc_input_kwargs.setdefault('smooth', True),
        self.acc_input_kwargs.setdefault('interpolate', True)

    def make(self) -> np.ndarray:
        accelerometer_inputs = []
        for sample in self.sample_list:
            acc_input = mechanical_components.AccelerometerInput(
                raw_accelerometer_input=sample.acc_df,
                accel_column=self.acc_input_kwargs.setdefault('accel_column', 'z_G'),  # noqa
                time_column=self.acc_input_kwargs.setdefault('time_column', 'time(ms)'),  # noqa
                accel_unit=self.acc_input_kwargs.setdefault('accel_unit', 'g'),
                time_unit=self.acc_input_kwargs.setdefault('time_unit', 'ms'),
                smooth=self.acc_input_kwargs.setdefault('smooth', True),
                interpolate=self.acc_input_kwargs.setdefault('interpolate', True)  # noqa
            )
            accelerometer_inputs.append(acc_input)
        return np.array(accelerometer_inputs)

@dataclass
class MechanicalGroundtruth:
    y_diff: Any
    time: Any

@dataclass
class ElectricalGroundtruth:
    emf: Any
    time: Any

@dataclass
class Groundtruth:
    mech: MechanicalGroundtruth
    elec: ElectricalGroundtruth

class GroundTruthFactory:
    def __init__(self,
                 samples_list,
                 lvp_kwargs,
                 adc_kwargs):
        """Helper Factory to get groundtruth data in a batch."""

        self.samples_list = samples_list
        self.lvp_kwargs = lvp_kwargs
        self.adc_kwargs = adc_kwargs

        self.lvp = evaluate.LabeledVideoProcessor(**lvp_kwargs)
        self.adc = evaluate.AdcProcessor(**adc_kwargs)

    def _make_mechanical_groundtruth(self, sample):
        y_target, y_time_target = self.lvp.fit_transform(
            sample.video_labels_df,
            impute_missing_values=True
        )
        y_target = savgol_filter(y_target, 9, 3)

        return MechanicalGroundtruth(y_target,
                                     y_time_target)

    def _make_electrical_groundtruth(self, sample):
        emf_target, emf_time_target = self.adc.fit_transform(sample.adc_df)
        return ElectricalGroundtruth(emf_target,
                                     emf_time_target)

    def make(self):
        groundtruths = []
        for sample in self.samples_list:
            try:
                mech_groundtruth = self._make_mechanical_groundtruth(sample)
                elec_groundtruth = self._make_electrical_groundtruth(sample)

                groundtruths.append(
                    Groundtruth(mech_groundtruth, elec_groundtruth)
                )
            except AttributeError:
                pass

        return groundtruths


# Prepare data
BASE_GROUNDTRUTH_PATH = '../data/2019-05-23_D/'
samples = {}
samples['A'] = collect_samples(base_path=BASE_GROUNDTRUTH_PATH,
                               acc_pattern='A/*acc*.csv',
                               adc_pattern='A/*adc*.csv',
                               video_label_pattern='A/*labels*.csv')
samples['B'] = collect_samples(base_path=BASE_GROUNDTRUTH_PATH,
                               acc_pattern='B/*acc*.csv',
                               adc_pattern='B/*adc*.csv',
                               video_label_pattern='B/*labels*.csv')
samples['C'] = collect_samples(base_path=BASE_GROUNDTRUTH_PATH,
                               acc_pattern='C/*acc*.csv',
                               adc_pattern='C/*adc*.csv',
                               video_label_pattern='C/*labels*.csv')

which_device = 'A'
which_input = np.array(range(len(samples[which_device])))  # type:ignore
which_input = [0]

magnet_assembly = ABC_CONFIG.magnet_assembly

# Groundtruth
groundtruth_factory = GroundTruthFactory(samples_list=samples[which_device],  # noqa
                                         lvp_kwargs=dict(magnet_assembly=magnet_assembly,
                                                         seconds_per_frame=1 / 60,  # noqa
                                                         pixel_scale=0.154508),
                                         adc_kwargs=dict(voltage_division_ratio=1 / 0.342))  # noqa


# TODO: Consider changing the factory to make it more user-friendly
groundtruth = groundtruth_factory.make()
mech_y_targets = [gt.mech.y_diff for gt in groundtruth]
mech_time_targets = [gt.mech.time for gt in groundtruth]
elec_emf_targets = [gt.elec.emf for gt in groundtruth]
elec_time_targets = [gt.elec.time for gt in groundtruth]


# Components
input_excitation_factories = {device: AccelerometerInputsFactory(samples[device])  # noqa
                              for device in ['A', 'B', 'C']}
magnetic_spring = mechanical_components.MagneticSpringInterp(
    fea_data_file='../data/magnetic-spring/10x10alt.csv',
    magnet_length=10 / 1000,
    filter_callable=lambda x: savgol_filter(x, 27, 5)
)

mech_components = {
    'magnetic_spring': [magnetic_spring],
    'magnet_assembly': [magnet_assembly],
    'damper': QuasiKarnoppDamperFactory(
        b_m1_list=np.linspace(0, 0.5, 10),
        b_m2_list=np.linspace(-0.2, 0.2, 10),
        magnet_assembly=magnet_assembly,
        tube_inner_radius_mm=5.5).make(),
    'mechanical_spring': MechanicalSpringFactory(
        magnet_assembly=magnet_assembly,
        position=110 / 1000,
        damping_coefficient_list=np.linspace(0, 10, 10)).make()
}

elec_components = {
    'rectification_drop': [0.1],
    'load_model': [electrical_components.SimpleLoad(R=30)],
    'coil_configuration': [ABC_CONFIG.coil_configs[which_device]],
    'flux_model': [ABC_CONFIG.flux_models[which_device]],
    'dflux_model': [ABC_CONFIG.dflux_models[which_device]],

}
coupling_models = CouplingModelFactory(np.linspace(0, 10, 15)).make()
governing_equations = [governing_equations.unified_ode]


# Models we want to simulate
abstract_model_factory = gridsearch.AbstractUnifiedModelFactory(
    mech_components,
    elec_components,
    coupling_models,
    governing_equations
)


# Inputs we want to excite the system with
input_excitations = input_excitation_factories[which_device].make()[which_input]

# Curves we want to capture
curve_expressions = {
    't': 'time',
    'x3-x1': 'y_diff',
    'g(t, x5)': 'emf'
}

# Expressions we want to score
score_metrics = {
    'x3-x1': gridsearch.EvaluatorFactory(evaluator_cls=evaluate.MechanicalSystemEvaluator,
                                         expr_targets=mech_y_targets,
                                         time_targets=mech_time_targets,
                                         metrics={'y_diff_dtw_distance': metrics.dtw_euclid_distance}).make()[which_input],  # noqa

    'g(t, x5)': gridsearch.EvaluatorFactory(evaluator_cls=evaluate.ElectricalSystemEvaluator,  # noqa
                                            expr_targets=elec_emf_targets,
                                            time_targets=elec_time_targets,
                                            metrics={'rms_perc_diff': metrics.root_mean_square_percentage_diff,  # noqa
                                                     'emf_dtw_distance': metrics.dtw_euclid_distance}).make()[which_input]  # noqa
}

# Metrics we want to calculate
calc_metrics = None

# Parameters we want to track
parameters_to_track = [
    'damper.cdc',
    'damper.mdc',
    'coupling_model.coupling_constant',
    'mechanical_spring.damping_coefficient',
    'coil_config.coil_resistance',
    'load_model.R'
]


# Run the gridsearch
grid_executor = gridsearch.GridsearchBatchExecutor(abstract_model_factory,
                                                   input_excitations,
                                                   curve_expressions,
                                                   score_metrics,
                                                   calc_metrics=calc_metrics,  # noqa <-- use this for optimization, not scoring
                                                   parameters_to_track=parameters_to_track)  # noqa

grid_executor.preview()
grid_executor.run(f'./{which_device}_single_input.parquet', batch_size=24)  # Execute

# DEBUG
# import pyarrow.parquet as pq
# df = pq.read_table('A.parquet').to_pandas()
# print(df.groupby('model_id').mean().sort_values(by='y_diff_dtw_distance').reset_index()[['model_id', 'y_diff_dtw_distance', 'damper.damping_coefficient']])
