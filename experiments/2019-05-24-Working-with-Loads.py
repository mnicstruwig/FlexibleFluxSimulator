from experiments.config import abc

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    explained_variance_score,
    r2_score,
    median_absolute_error,
)

from unified_model.metrics import *
from unified_model.evaluate import AdcProcessor, LabeledVideoProcessor
from unified_model.unified import UnifiedModel
from unified_model.coupling import CouplingModel
from unified_model.electrical_model import ElectricalModel
from unified_model.electrical_components.load import SimpleLoad
from unified_model.mechanical_model import MechanicalModel
from unified_model.mechanical_components.damper import ConstantDamper
from unified_model.mechanical_components.mechanical_spring import MechanicalSpring
from unified_model.mechanical_components.input_excitation.accelerometer import (
    AccelerometerInput,
)
from unified_model.utils.utils import collect_samples
from unified_model.governing_equations import unified_ode
from unified_model.pipeline import clip_x2

import warnings

warnings.simplefilter("ignore")

base_groundtruth_path = "./data/2019-05-23_C/"
a_samples = collect_samples(
    base_path=base_groundtruth_path,
    acc_pattern="A/*acc*.csv",
    adc_pattern="A/*adc*.csv",
    labeled_video_pattern="A/*labels*.csv",
)

max_height = 110 / 1000

mechanical_spring = MechanicalSpring(
    push_direction="down",
    position=max_height,
    pure=False,
    strength=1000,
    damper_constant=0.06,
)

mechanical_model = MechanicalModel(name="Mechanical Model")
mechanical_model.set_max_height(max_height)
mechanical_model.set_magnetic_spring(abc.spring)
mechanical_model.set_mechanical_spring(
    mechanical_spring
)  # <-- do more investigating with this guy
mechanical_model.set_magnet_assembly(abc.magnet_assembly)
mechanical_model.set_damper(
    ConstantDamper(damping_coefficient=0.0350)
)  # Tweaking will need to happen


accelerometer_inputs = [
    AccelerometerInput(
        raw_accelerometer_input=sample.acc_df,
        accel_column="z_G",
        time_column="time(ms)",
        accel_unit="g",
        time_unit="ms",
        smooth=True,
        interpolate=True,
    )
    for sample in a_samples
]

which_sample = 2
mechanical_model.set_input(
    accelerometer_inputs[which_sample]
)  # Choose which input to system

electrical_model = ElectricalModel(name="Electrical Model")
electrical_model.set_coil_resistance(
    abc.coil_resistance["A"]
)  # Guessing this value for the time being
electrical_model.set_load_model(SimpleLoad(R=30))  # Make sure this is correct!
electrical_model.set_flux_model(abc.flux_models["A"], abc.dflux_models["A"])

coupling_model = CouplingModel().set_coupling_constant(
    1.0
)  # This will need to be found.

unified_model = UnifiedModel(name="Unified Model")
unified_model.set_mechanical_model(mechanical_model)
unified_model.set_electrical_model(electrical_model)
unified_model.set_coupling_model(coupling_model)
unified_model.set_governing_equations(unified_ode)
unified_model.set_post_processing_pipeline(clip_x2, name="clip tube velocity")

# Execute and collect results
unified_model.solve(t_start=0, t_end=8, t_max_step=1e-3, y0=[0.0, 0.0, 0.04, 0.0, 0.0])

result = unified_model.get_result(
    time="t",
    x1="x1",
    x2="x2",
    x3="x3",
    x4="x4",
    acc="g(t, x2)",
    rel_pos="x3-x1",
    rel_vel="x4-x2",
    flux="x5",
    emf="g(t, x5)",
)

# Mechanical Scoring
# pixel_scale = 0.1785  # Huawei P10
pixel_scale = 0.18451  # Huawei P10 alternative
# pixel_scale = 0.18745 # Samsung S7

labeled_video_processor = LabeledVideoProcessor(
    L=125, mm=10, seconds_per_frame=2 / 120, pixel_scale=pixel_scale
)

mechanical_metrics = {"dtw_dist": dtw_euclid_distance}


mech_scores, m_eval = unified_model._score_mechanical_model(
    metrics_dict=mechanical_metrics,
    video_labels_df=a_samples[which_sample].video_labels_df,
    labeled_video_processor=labeled_video_processor,
    prediction_expr="x3-x1",
    return_evaluator=True,
    use_processed_signals=False,
)

m_eval.poof(True)

# EMF Scoring

electrical_metrics = {
    "rms": root_mean_square,
    "rms_err_perc": root_mean_square_percentage_diff,
}

voltage_division_ratio = 1 / 0.342

adc_processor = AdcProcessor(voltage_division_ratio=voltage_division_ratio, smooth=True)

emf_scores, e_eval = unified_model._score_electrical_model(
    metrics_dict=electrical_metrics,
    adc_df=a_samples[which_sample].adc_df,
    adc_processor=adc_processor,
    prediction_expr="g(t, x5)",
    return_evaluator=True,
    closed_circuit=True,
    use_processed_signals=False,
)

# result.plot(x='time', y='rel_pos') result.plot(x='time', y='flux')
# result.plot(x='time', y='emf')
# plt.figure()
e_eval.poof(True)

print(mech_scores)
print(emf_scores)

# try:
#     unified_model.save_to_disk('./my_saved_model/')
# except FileExistsError:
#     print('Model not saved, since path already exists.')
