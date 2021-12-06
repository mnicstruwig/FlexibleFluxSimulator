import argparse
import json

import nevergrad as ng
from ffs.evaluate import Measurement
from ffs.unified import UnifiedModel
from ffs.utils.utils import collect_samples


device_A_config = {
    "height": 110 / 1000,
    "magnetic_spring": {
        "fea_data_file": "/Users/michael/Nextcloud/nextcloud/PhD/Python/ffs/data/magnetic-spring/10x10alt.csv",
        "filter_callable": "auto",
        "magnet_assembly": "dep:magnet_assembly",
    },
    "magnet_assembly": {
        "m": 1,
        "l_m_mm": 10,
        "l_mcd_mm": 0,
        "dia_magnet_mm": 10,
        "dia_spacer_mm": 10,
    },
    "mechanical_spring": {
        "magnet_assembly": "dep:magnet_assembly",
        "strength": 10000000.0,
        "damping_coefficient": None,
    },
    "mechanical_damper": {
        "damping_coefficient": None,
        "magnet_assembly": "dep:magnet_assembly",
    },
    "input_excitation": None,
    "coil_configuration": {
        "c": 1,
        "n_z": 17,
        "n_w": 15,
        "l_ccd_mm": 0,
        "ohm_per_mm": 0.0010789999999999999,
        "tube_wall_thickness_mm": 2,
        "coil_wire_radius_mm": 0.0715,
        "coil_center_mm": 59,
        "inner_tube_radius_mm": 5.5,
        "custom_coil_resistance": 12.5,
    },
    "flux_model": {
        "coil_configuration": "dep:coil_configuration",
        "magnet_assembly": "dep:magnet_assembly",
        "curve_model_path": "../data/flux_curve_model/flux_curve_model_2021_05_11.model",
    },
    "rectification_drop": 0.05,
    "load_model": {"R": 30},
    "coupling_model": {"coupling_constant": None},
    "extra_components": None,
    "governing_equations": {
        "module_path": "ffs.governing_equations",
        "func_name": "unified_ode",
    },
}

device_B_config = {
    "height": 110 / 1000,
    "magnetic_spring": {
        "fea_data_file": "/Users/michael/Nextcloud/nextcloud/PhD/Python/ffs/data/magnetic-spring/10x10alt.csv",
        "filter_callable": "auto",
        "magnet_assembly": "dep:magnet_assembly",
    },
    "magnet_assembly": {
        "m": 1,
        "l_m_mm": 10,
        "l_mcd_mm": 0,
        "dia_magnet_mm": 10,
        "dia_spacer_mm": 10,
    },
    "mechanical_spring": {
        "magnet_assembly": "dep:magnet_assembly",
        "strength": 10000000.0,
        "damping_coefficient": None,
    },
    "mechanical_damper": {
        "damping_coefficient": None,
        "magnet_assembly": "dep:magnet_assembly",
    },
    "input_excitation": None,
    "coil_configuration": {
        "c": 1,
        "n_z": 15,
        "n_w": 33,
        "l_ccd_mm": 0,
        "ohm_per_mm": 0.0010789999999999999,
        "tube_wall_thickness_mm": 2,
        "coil_wire_radius_mm": 0.0715,
        "coil_center_mm": 61,
        "inner_tube_radius_mm": 5.5,
        "custom_coil_resistance": 23.5,
    },
    "flux_model": {
        "coil_configuration": "dep:coil_configuration",
        "magnet_assembly": "dep:magnet_assembly",
        "curve_model_path": "../data/flux_curve_model/flux_curve_model_2021_05_11.model",
    },
    "rectification_drop": 0.05,
    "load_model": {"R": 30},
    "coupling_model": {"coupling_constant": None},
    "extra_components": None,
    "governing_equations": {
        "module_path": "ffs.governing_equations",
        "func_name": "unified_ode",
    },
}


device_C_config = {
    "height": 110 / 1000,
    "magnetic_spring": {
        "fea_data_file": "/Users/michael/Nextcloud/nextcloud/PhD/Python/ffs/data/magnetic-spring/10x10alt.csv",
        "filter_callable": "auto",
        "magnet_assembly": "dep:magnet_assembly",
    },
    "magnet_assembly": {
        "m": 1,
        "l_m_mm": 10,
        "l_mcd_mm": 0,
        "dia_magnet_mm": 10,
        "dia_spacer_mm": 10,
    },
    "mechanical_spring": {
        "magnet_assembly": "dep:magnet_assembly",
        "strength": 10000000.0,
        "damping_coefficient": None,
    },
    "mechanical_damper": {
        "damping_coefficient": None,
        "magnet_assembly": "dep:magnet_assembly",
    },
    "input_excitation": None,
    "coil_configuration": {
        "c": 1,
        "n_z": 15,
        "n_w": 33,
        "l_ccd_mm": 0,
        "ohm_per_mm": 0.0010789999999999999,
        "tube_wall_thickness_mm": 2,
        "coil_wire_radius_mm": 0.0715,
        "coil_center_mm": 61,
        "inner_tube_radius_mm": 5.5,
        "custom_coil_resistance": 17.8,
    },
    "flux_model": {
        "coil_configuration": "dep:coil_configuration",
        "magnet_assembly": "dep:magnet_assembly",
        "curve_model_path": "../data/flux_curve_model/flux_curve_model_2021_05_11.model",
    },
    "rectification_drop": 0.05,
    "load_model": {"R": 30},
    "coupling_model": {"coupling_constant": None},
    "extra_components": None,
    "governing_equations": {
        "module_path": "ffs.governing_equations",
        "func_name": "unified_ode",
    },
}

device_D_config = {
    "height": 140 / 1000,
    "magnetic_spring": {
        "fea_data_file": "/Users/michael/Nextcloud/nextcloud/PhD/Python/ffs/data/magnetic-spring/10x10alt.csv",
        "filter_callable": "auto",
        "magnet_assembly": "dep:magnet_assembly",
    },
    "magnet_assembly": {
        "m": 2,
        "l_m_mm": 10,
        "l_mcd_mm": 24,
        "dia_magnet_mm": 10,
        "dia_spacer_mm": 10,
    },
    "mechanical_spring": {
        "magnet_assembly": "dep:magnet_assembly",
        "strength": 10000000.0,
        "damping_coefficient": None,
    },
    "mechanical_damper": {
        "damping_coefficient": None,
        "magnet_assembly": "dep:magnet_assembly",
    },
    "input_excitation": None,
    "coil_configuration": {
        "c": 1,
        "n_z": 88,
        "n_w": 20,
        "l_ccd_mm": 0,
        "ohm_per_mm": 0.0010789999999999999,
        "tube_wall_thickness_mm": 2,
        "coil_wire_radius_mm": 0.0715,
        "coil_center_mm": 78,
        "inner_tube_radius_mm": 5.5,
        "custom_coil_resistance": None,
    },
    "flux_model": {
        "coil_configuration": "dep:coil_configuration",
        "magnet_assembly": "dep:magnet_assembly",
        "curve_model_path": "../data/flux_curve_model/flux_curve_model_2021_05_11.model",
    },
    "rectification_drop": 0.05,
    "load_model": {"R": 30},
    "coupling_model": {"coupling_constant": None},
    "extra_components": None,
    "governing_equations": {
        "module_path": "ffs.governing_equations",
        "func_name": "unified_ode",
    },
}

# Define our devices

device_A = UnifiedModel.from_config(device_A_config)
device_B = UnifiedModel.from_config(device_B_config)
device_C = UnifiedModel.from_config(device_C_config)
device_D = UnifiedModel.from_config(device_D_config)


# Write config out to Disk
with open('device_a.json', 'w') as f:
    config = device_A.get_config(kind='json')
    f.write(config)

with open('device_b.json', 'w') as f:
    config = device_B.get_config(kind='json')
    f.write(config)

with open('device_c.json', 'w') as f:
    config = device_C.get_config(kind='json')
    f.write(config)

with open('device_d.json', 'w') as f:
    config = device_D.get_config(kind='json')
    f.write(config)

# Fetch our samples / measurements
BASE_GROUNDTRUTH_PATH = "../data/2019-05-23/"
samples = {}

samples["A"] = collect_samples(
    base_path=BASE_GROUNDTRUTH_PATH + 'A/',
    acc_pattern="*acc*.csv",
    adc_pattern="*adc*.csv",
    video_label_pattern="*labels*.csv",
)
samples["B"] = collect_samples(
    base_path=BASE_GROUNDTRUTH_PATH + 'B/',
    acc_pattern="*acc*.csv",
    adc_pattern="*adc*.csv",
    video_label_pattern="*labels*.csv",
)
samples["C"] = collect_samples(
    base_path=BASE_GROUNDTRUTH_PATH + 'C/',
    acc_pattern="*acc*.csv",
    adc_pattern="*adc*.csv",
    video_label_pattern="*labels*.csv",
)
samples["D"] = collect_samples(
    base_path='../data/2021-03-05/D',
    acc_pattern="*acc*.csv",
    adc_pattern="*adc*.csv",
    video_label_pattern="*labels*.csv",
)

measurements = {
    'A': [Measurement(s, device_A) for s in samples['A']],
    'B': [Measurement(s, device_B) for s in samples['B']],
    'C': [Measurement(s, device_C) for s in samples['C']],
    'D': [Measurement(s, device_D) for s in samples['D']]
}

# Time for the parameter search
models_and_measurements = [
    (device_A, measurements['A']),
    (device_B, measurements['B']),
    (device_C, measurements['C']),
    (device_D, measurements['D']),
]

instruments = {
    'mech_damping_coefficient': ng.p.Scalar(init=2, lower=0, upper=10),
    'coupling_constant': ng.p.Scalar(init=0, lower=0, upper=10),
    'mech_spring_damping_coefficient': ng.p.Scalar(init=0, lower=0, upper=10),
}

# results = parameter_search.mean_of_scores(
#     models_and_measurements=models_and_measurements,
#     instruments=instruments,
#     cost_metric='power',
#     budget=3
# )
