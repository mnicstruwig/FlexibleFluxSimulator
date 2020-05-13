from collections import namedtuple
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
from local_config import abc_config


class ConstantDamperFactory:
    def __init__(self, c_list):
        self.c_list = c_list

    def make(self):
        return [mechanical_components.ConstantDamper(c) for c in self.c_list]


class MechanicalSpringFactory:
    def __init__(self, position, damping_coefficient_list):
        self.position = position
        self.damping_coefficient_list = damping_coefficient_list

    def make(self):
        return [mechanical_components.MechanicalSpring(self.position,
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

    def make(self) -> List[mechanical_components.AccelerometerInput]:
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
        return accelerometer_inputs


class GroundTruthFactory:
    def __init__(self, samples_list, lvp_kwargs, adc_kwargs):
        self.samples_list = samples_list
        self.lvp_kwargs = lvp_kwargs
        self.adc_kwargs = adc_kwargs

        self.lvp = evaluate.LabeledVideoProcessor(**lvp_kwargs)
        self.adc = evaluate.AdcProcessor(**adc_kwargs)
        self.MechGroundtruth = gridsearch.MechanicalGroundtruth
        self.ElecGroundtruth = gridsearch.ElectricalGroundtruth
        self.Groundtruth = gridsearch.Groundtruth

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


class EvaluatorFactory:
    def __init__(self,
                 evaluator_cls: Any,
                 expr_targets: List,
                 time_targets: List,
                 metrics: Dict[str, Callable],
                 **kwargs) -> None:

        # Do some validation
        assert len(expr_targets) == len(time_targets)

        self.evaluator_cls = evaluator_cls
        self.expr_targets = expr_targets
        self.time_targets = time_targets
        self.metrics = metrics
        self.evaluator_kwargs = kwargs

    def make(self) -> List[Any]:
        evaluator_list = []
        for expr_target, time_target in zip(self.expr_targets, self.time_targets):  # noqa
            evaluator = self.evaluator_cls(expr_target,
                                           time_target,
                                           self.metrics,
                                           **self.evaluator_kwargs)
            evaluator_list.append(evaluator)
        return evaluator_list


# Prepare data
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

# Groundtruth
groundtruth_factory = GroundTruthFactory(samples_list=samples['A'][:5],  # noqa <-- take the first five groundtruth samples
                                         lvp_kwargs=dict(L=125,
                                                         mm=10,
                                                         seconds_per_frame=1/60,
                                                         pixel_scale=0.154508),
                                         adc_kwargs=dict(voltage_division_ratio=1 / 0.342)  # noqa
)
groundtruth = groundtruth_factory.make()  # TODO: Consider changing the factory to make it more user-friendly
mech_y_targets = [gt.mech.y_diff for gt in groundtruth]
mech_time_targets = [gt.mech.time for gt in groundtruth]
elec_emf_targets = [gt.elec.emf for gt in groundtruth]
elec_time_targets = [gt.elec.time for gt in groundtruth]


# Components
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
    'magnetic_spring': [magnetic_spring],
    'magnet_assembly': [magnet_assembly],
    'damper': ConstantDamperFactory(np.linspace(0.01, 0.07, 10)).make(),
    'mechanical_spring':  MechanicalSpringFactory(110/1000, np.linspace(0, 3, 5)).make()
}
elec_components = {
    'coil_resistance': [abc_config.coil_resistance['A']],
    'rectification_drop': [0.1],
    'load_model': [electrical_components.SimpleLoad(30)],
    'flux_model': [abc_config.flux_models['A']],
    'dflux_model': [abc_config.dflux_models['A']],

}
coupling_models = CouplingModelFactory(np.linspace(0, 3, 2)).make()
governing_equations = [governing_equations.unified_ode]


# Models we want to simulate
abstract_model_factory = gridsearch.AbstractUnifiedModelFactory(
    mech_components,
    elec_components,
    coupling_models,
    governing_equations
)


# Inputs we want to excite the system with
input_excitations = input_excitation_factories['A'].make()[:3]

# Curves we want to capture
curve_expressions = {
    't': 'time',
    'x3-x1': 'y_diff',
    'g(t, x5)': 'emf'
}

# Expressions we want to score
score_metrics = {
    'x3-x1': EvaluatorFactory(evaluator_cls=evaluate.MechanicalSystemEvaluator,
                              expr_targets=mech_y_targets,
                              time_targets=mech_time_targets,
                              metrics={'y_diff_dtw_distance': metrics.dtw_euclid_distance}).make()[:len(input_excitations)],  # noqa

    'g(t, x5)': EvaluatorFactory(evaluator_cls=evaluate.ElectricalSystemEvaluator,  # noqa
                                 expr_targets=elec_emf_targets,
                                 time_targets=elec_time_targets,
                                 metrics={'rms_perc_diff': metrics.root_mean_square_percentage_diff,  # noqa
                                          'emf_dtw_distance': metrics.dtw_euclid_distance}).make()[:len(input_excitations)]  # noqa
}

# Metrics we want to calculate
calc_metrics = None

# Parameters we want to track
parameters_to_track = [
    'damper.damping_coefficient',
    'coupling_model.coupling_constant',
    'mechanical_spring.damping_coefficient'
]


# Run the gridsearch
grid_executor = gridsearch.GridsearchBatchExecutor(abstract_model_factory,
                                                   input_excitations,
                                                   curve_expressions,
                                                   score_metrics,
                                                   calc_metrics=calc_metrics,  # noqa <-- use this for optimization, not scoring
                                                   parameters_to_track=parameters_to_track)  # noqa

grid_executor.preview()
results = grid_executor.run()  # Execute
