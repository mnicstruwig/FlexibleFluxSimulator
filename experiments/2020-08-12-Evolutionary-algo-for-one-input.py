"""
Try use `nevergrad` to find the parameters for the model for a single device /
input excitation
"""
from concurrent import futures
from dataclasses import dataclass
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import nevergrad as ng
import numpy as np
from scipy.signal import savgol_filter
# Local imports
from unified_model import (CouplingModel, ElectricalModel, MechanicalModel,
                           UnifiedModel, electrical_components, evaluate,
                           governing_equations, gridsearch,
                           mechanical_components, metrics, pipeline)
from unified_model.utils.utils import collect_samples

from local_config import ABC_CONFIG

# Parameters
c = 1  # number of coils
m = 1  # number of magnets
l_m = 10  # length of the magnets in mm
dia_m = 10  # diameter of the magnets in mm
l_ccd = 0  # distance between centers of coils in mm
l_mcd = 0  # distance between centers of magnets in mm

# Components
def smoothing_filter(x_arr):
    return savgol_filter(x_arr, window_length=27, polyorder=5)


magnetic_spring = mechanical_components.MagneticSpringInterp(
    fea_data_file='./data/magnetic-spring/10x10alt.csv',
    magnet_length=l_m/1000,
    filter_callable=smoothing_filter
)

magnet_assembly = mechanical_components.MagnetAssembly(
    n_magnet=m,
    l_m=l_m,
    l_mcd=0,
    dia_magnet=dia_m,
    dia_spacer=dia_m,
)

load = electrical_components.SimpleLoad(R=20)
v_rect_drop = 0.1


# Function to generate our parameters that can vary
def make_damper(c):
    return mechanical_components.ConstantDamper(c)


def make_mech_spring(c):
    return mechanical_components.MechanicalSpring(
        position=110/1000,
        magnet_length=l_m/1000,
        damping_coefficient=c
    )


def make_coupling(c):
    return CouplingModel().set_coupling_constant(c)


@dataclass
class ModelComponents:
    """A class that holds unified model components"""
    flux_model: Any
    dflux_model: Any
    r_coil: float

@dataclass
class ModelPackage:
    """A model package consists of an input, evaluators and model components"""
    input_excitation: List[Any]
    score_metric_dict: Dict[str, Any]
    model_components: ModelComponents


def build_model(model_package: ModelPackage,
                fric_damp: float,
                mech_spring_damp: float,
                coupling_damp: float) -> UnifiedModel:
    mechanical_model = (
        MechanicalModel()
        .set_damper(make_damper(fric_damp))
        .set_magnet_assembly(magnet_assembly)
        .set_magnetic_spring(magnetic_spring)
        .set_mechanical_spring(make_mech_spring(mech_spring_damp))
        .set_input(model_package.input_excitation)
    )

    electrical_model = (
        ElectricalModel()
        .set_coil_resistance(model_package.model_components.r_coil)
        .set_rectification_drop(v_rect_drop)
        .set_load_model(load)
        .set_flux_model(model_package.model_components.flux_model,
                        model_package.model_components.dflux_model)
    )

    unified_model = (
        UnifiedModel()
        .set_mechanical_model(mechanical_model)
        .set_electrical_model(electrical_model)
        .set_coupling_model(make_coupling(coupling_damp))
        .set_post_processing_pipeline(pipeline.clip_x2, name='')
        .set_governing_equations(governing_equations.unified_ode)
    )

    return unified_model


def run_simulation(unified_model: UnifiedModel) -> UnifiedModel:
    y0 = [
        0.0,   # x1 at t=0 -> tube displacement in m
        0.0,   # x2 at t=0 -> tube velocity in m/s
        0.05,  # x3 at t=0 -> magnet displacement in m
        0.0,   # x4 at t=0 -> magnet velocity in m/s
        0.0    # x5 at t=0 -> flux linkage
    ]

    unified_model.solve(  # This can take a little while...
        t_start=0,
        t_end=7,
        y0=y0,  # Initial conditions we defined above
        t_eval=np.linspace(0, 7, 1000),
        t_max_step=1e-3
    )

    return unified_model

@dataclass
class Groundtruth:
    y_target: Any
    y_time: Any
    emf_target: Any
    emf_time: Any



class InputEvaluatorFactory:
    def __init__(self,
                 samples_list,  # acc_df, adc_df, video_labels_df
                 lvp_kwargs,
                 adc_kwargs):
        self.samples_list = samples_list
        self.lvp_kwargs = lvp_kwargs
        self.adc_kwargs = adc_kwargs

        self.lvp = evaluate.LabeledVideoProcessor(**lvp_kwargs)
        self.adc = evaluate.AdcProcessor(**adc_kwargs)

    def _make_input_excitation(self, sample):
        input_excitation = mechanical_components.AccelerometerInput(
            raw_accelerometer_input=sample.acc_df,
            accel_column='z_G',
            time_column='time(ms)',
            accel_unit='g',
            time_unit='ms',
            smooth=True,
            interpolate=True
        )

        return input_excitation

    def _make_evaluator_dict(self, groundtruth):
        return {
            'x3-x1': evaluate.MechanicalSystemEvaluator(groundtruth.y_target,
                                                        groundtruth.y_time,
                                                        {'y_diff_dtw_distance': metrics.dtw_euclid_distance}),
            'g(t, x5)': evaluate.ElectricalSystemEvaluator(groundtruth.emf_target,
                                                           groundtruth.emf_time,
                                                           {'emf_dtw_distance': metrics.dtw_euclid_distance,
                                                            'rms_perc_diff': metrics.root_mean_square_percentage_diff})
        }

    def _get_mech_groundtruth(self, sample):
        y_target, y_time_target = self.lvp.fit_transform(
            sample.video_labels_df,
            impute_missing_values=True
        )
        y_target = savgol_filter(y_target, 9, 3)

        return y_target, y_time_target

    def _get_elec_groundtruth(self, sample):
        emf_target, emf_time_target = self.adc.fit_transform(sample.adc_df)
        return emf_target, emf_time_target

    def _get_groundtruth(self, sample):
        y, y_time = self._get_mech_groundtruth(sample)
        emf, emf_time = self._get_elec_groundtruth(sample)

        return Groundtruth(
            y_target=y,
            y_time=y_time,
            emf_target=emf,
            emf_time=emf_time,
        )

    def make(self):
        input_evaluator_list = []
        for sample in self.samples_list:
            input_excitation = self._make_input_excitation(sample)
            groundtruth = self._get_groundtruth(sample)
            evaluator_dict = self._make_evaluator_dict(groundtruth)

            input_evaluator_list.append((input_excitation, evaluator_dict))

        return input_evaluator_list


def calc_loss(model_package: ModelPackage,
              fric_damp: float,
              mech_spring_damp: float,
              coupling_damp: float) -> float:
    unified_model = build_model(
        model_package=model_package,
        fric_damp=fric_damp,
        mech_spring_damp=mech_spring_damp,
        coupling_damp=coupling_damp,
    )

    unified_model = run_simulation(unified_model)

    scores = {}
    for expr, evaluator in model_package.score_metric_dict.items():
        results = unified_model.get_result(time='t', prediction=expr)
        evaluator.fit(results['prediction'].values, results['time'].values)
        score = evaluator.score()
        scores.update(score)

    best_mech_score = 9.5753
    best_elec_score = 84.888
    best_abs_rms_score = 1
    return (
        + 0*scores['y_diff_dtw_distance']/best_mech_score
        + 0*scores['emf_dtw_distance']/best_elec_score
        + 1*np.abs(scores['rms_perc_diff'])/best_abs_rms_score
    )



def calc_loss_for_all(param_dict, model_package_list: List[ModelPackage]):
    losses = []
    for i, model_package in enumerate(model_package_list):  # noqa
        loss = calc_loss(model_package,
                         **param_dict)
        losses.append(loss)
    return np.mean(losses)


base_groundtruth_path = './data/2019-05-23_D/'
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
                               video_label_pattern='C/*labels*.csv')


def build_model_packages(samples_list, model_components):
    input_evaluator_factory = InputEvaluatorFactory(
        samples_list,
        lvp_kwargs=dict(mm=10,
                        seconds_per_frame=1/60,
                        pixel_scale=0.154508),
        adc_kwargs=dict(voltage_division_ratio=1 / 0.342)  # noqa
    )
    input_evaluator_tuples = input_evaluator_factory.make()

    model_packages = []
    for input_excitation, score_metric_dict in input_evaluator_tuples:
        model_packages.append(ModelPackage(input_excitation,
                                           score_metric_dict,
                                           model_components))

    return model_packages


def build_model_components(which_device):
    return ModelComponents(
        flux_model=ABC_CONFIG.flux_models[which_device],
        dflux_model=ABC_CONFIG.dflux_models[which_device],
        r_coil=ABC_CONFIG.coil_resistance[which_device]
    )


model_package_list_A = build_model_packages(samples['A'], build_model_components('A'))
model_package_list_B = build_model_packages(samples['B'], build_model_components('B'))
model_package_list_C = build_model_packages(samples['C'], build_model_components('C'))

model_package_list = model_package_list_A + model_package_list_B + model_package_list_C  # noqa

instrumentation = ng.p.Instrumentation(
    fric_damp=ng.p.Scalar(0.0406837, lower=0, upper=0.1),
    mech_spring_damp=ng.p.Scalar(3.9272, lower=0, upper=10),
    coupling_damp=ng.p.Scalar(3.49686, lower=0, upper=5)
)


# Optimization step
def callback(optimizer, candidate, value):
    values_dict = candidate.value[1]
    output_values = {k: np.round(v, 4) for k, v in values_dict.items()}
    print(f'Values: {output_values} :: Loss: {np.round(value, 5)}')


optimizer = ng.optimizers.TwoPointsDE(parametrization=instrumentation, budget=200, num_workers=4)
optimizer.register_callback('tell', callback)


def wrapper(fric_damp, mech_spring_damp, coupling_damp):
    param_dict = {
        'fric_damp': fric_damp,
        'mech_spring_damp': mech_spring_damp,
        'coupling_damp': coupling_damp
    }

    return calc_loss_for_all(
        param_dict=param_dict,
        model_package_list=model_package_list  # Global variable
    )


with futures.ProcessPoolExecutor(max_workers=optimizer.num_workers) as executor:
    recommendation = optimizer.minimize(wrapper, executor=executor, batch_mode=False)

best_values = recommendation.value[1]
loss = recommendation.loss
# > best values
# {'fric_damp': 0.03747113788977376,
#  'mech_spring_damp': 6.820248958164491,
#  'coupling_damp': 2.1510305735759934}
