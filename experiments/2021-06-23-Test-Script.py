# Let's define our model that we need to simulate
from scipy.signal import savgol_filter
from unified_model.unified import UnifiedModel
from unified_model import mechanical_components
from unified_model import electrical_components
from unified_model import CouplingModel
from unified_model import metrics
from unified_model import evaluate
from unified_model.utils.utils import collect_samples
from flux_modeller.model import CurveModel
from unified_model.evaluate import Measurement
from unified_model import parameter_search

import nevergrad as ng
import pandas as pd
import cloudpickle
import numpy as np
import copy

from typing import Tuple, Union, Any, Dict, List
import warnings

warnings.simplefilter("ignore")

with open("../scripts/ABC.config", "rb") as f:
    ABC_CONFIG = cloudpickle.load(f)


def _create_model_D_components():
    coil_config = electrical_components.CoilConfiguration(
        c=1,
        n_z=88,
        n_w=20,
        l_ccd_mm=0,
        ohm_per_mm=1079 / 1000 / 1000,
        tube_wall_thickness_mm=2,
        coil_wire_radius_mm=0.143 / 2,
        coil_center_mm=78,
        inner_tube_radius_mm=5.5,
    )

    magnet_assembly = mechanical_components.MagnetAssembly(
        m=2, l_m_mm=10, l_mcd_mm=24, dia_magnet_mm=10, dia_spacer_mm=10
    )

    curve_model = CurveModel.load(
        "../data/flux_curve_model/flux_curve_model_2021_05_11.model"
    )
    z, phi = curve_model.predict_curves(np.array([[coil_config.n_z, coil_config.n_w]]))
    phi = phi.flatten()
    flux_model_interp = electrical_components.FluxModelInterp(
        coil_config=coil_config, magnet_assembly=magnet_assembly
    )
    flux_model_interp.fit(z.values, phi)

    mech_spring = mechanical_components.MechanicalSpring(
        magnet_assembly=magnet_assembly, position=140 / 1000, damping_coefficient=None
    )

    return (
        coil_config,
        magnet_assembly,
        flux_model_interp.flux_model,
        flux_model_interp.dflux_model,
        mech_spring,
    )


def _create_model_O_components():
    coil_config = electrical_components.CoilConfiguration(
        c=1,
        n_z=68,
        n_w=20,
        l_ccd_mm=0,
        ohm_per_mm=1079 / 1000 / 1000,
        tube_wall_thickness_mm=2,
        coil_wire_radius_mm=0.143 / 2,
        coil_center_mm=72,
        inner_tube_radius_mm=5.5,
    )

    magnet_assembly = mechanical_components.MagnetAssembly(
        m=2, l_m_mm=10, l_mcd_mm=17, dia_magnet_mm=10, dia_spacer_mm=10
    )

    curve_model = CurveModel.load(
        "../data/flux_curve_model/flux_curve_model_2021_05_11.model"
    )
    z, phi = curve_model.predict_curves(np.array([[coil_config.n_z, coil_config.n_w]]))
    phi = phi.flatten()
    flux_model_interp = electrical_components.FluxModelInterp(
        coil_config=coil_config, magnet_assembly=magnet_assembly
    )
    flux_model_interp.fit(z.values, phi)

    mech_spring = mechanical_components.MechanicalSpring(
        magnet_assembly=magnet_assembly, position=125 / 1000, damping_coefficient=None
    )

    return (
        coil_config,
        magnet_assembly,
        flux_model_interp.flux_model,
        flux_model_interp.dflux_model,
        mech_spring,
    )


def _create_model_A_B_C_components(which_device):
    coil_config = ABC_CONFIG.coil_configs[which_device]
    magnet_assembly = ABC_CONFIG.magnet_assembly
    flux_model = ABC_CONFIG.flux_models[which_device]
    dflux_model = ABC_CONFIG.dflux_models[which_device]

    # We only define this so the spring position is fixed (gets overridden with parameters later)
    mech_spring = mechanical_components.MechanicalSpring(
        magnet_assembly=magnet_assembly,
        position=110 / 1000,
        damping_coefficient=None,  # Make sure things break if we forget
    )

    return (coil_config, magnet_assembly, flux_model, dflux_model, mech_spring)


def _prepare_prototype_model(which_device, path="um_prototype.model/") -> UnifiedModel:
    model_prototype = UnifiedModel.load_from_disk(path)
    model_prototype.mechanical_model.set_damper(None)
    model_prototype.set_coupling_model(None)

    # Get and assign components
    if which_device in ["A", "B", "C"]:
        (
            coil_config,
            magnet_assembly,
            flux_model,
            dflux_model,
            mech_spring,
        ) = _create_model_A_B_C_components(which_device)
    elif which_device == "D":
        (
            coil_config,
            magnet_assembly,
            flux_model,
            dflux_model,
            mech_spring,
        ) = _create_model_D_components()
    elif which_device == "O":
        (
            coil_config,
            magnet_assembly,
            flux_model,
            dflux_model,
            mech_spring,
        ) = _create_model_O_components()
    else:
        raise ValueError('`which_device` must be "A", "B", "C", "D" or "O".')

    model_prototype.electrical_model.set_coil_configuration(coil_config)
    model_prototype.electrical_model.set_flux_model(flux_model, dflux_model)
    model_prototype.mechanical_model.set_magnet_assembly(magnet_assembly)
    model_prototype.mechanical_model.set_mechanical_spring(mech_spring)

    return model_prototype


def _get_measurements(which_device, model_prototype):
    # Prepare data
    if which_device == "O":
        samples_list = collect_samples(
            base_path="../data/2021-06-11/E/",
            acc_pattern="*acc*.csv",
            adc_pattern="*adc*.csv",
            video_label_pattern="*labels*.csv",
        )
    elif which_device == "D":
        samples_list = collect_samples(
            base_path="../data/2021-03-05/D/",
            acc_pattern="*acc*.csv",
            adc_pattern="*adc*.csv",
            video_label_pattern="*labels*.csv",
        )
    elif which_device in ["A", "B", "C"]:
        samples_list = collect_samples(
            base_path="../data/2019-05-23/",
            acc_pattern=f"{which_device}/*acc*.csv",
            adc_pattern=f"{which_device}/*adc*.csv",
            video_label_pattern=f"{which_device}/*labels*.csv",
        )
    else:
        raise ValueError("Samples for device not registered.")

    measurements = [evaluate.Measurement(s, model_prototype) for s in samples_list]
    return measurements


def get_prototype_and_measurements(
    which_device: UnifiedModel, path: str = "um_prototype.model/"
) -> Tuple[UnifiedModel, List[Measurement]]:

    model_prototype = _prepare_prototype_model(which_device, path)
    measurements = _get_measurements(which_device, model_prototype)

    return model_prototype, measurements


def make_unified_model_from_params(
    model_prototype, damper_cdc, coupling_constant, mech_spring_constant
) -> UnifiedModel:
    model = copy.deepcopy(model_prototype)

    damper = mechanical_components.MassProportionalDamper(
        damper_cdc, model.mechanical_model.magnet_assembly
    )

    coupling_model = CouplingModel().set_coupling_constant(coupling_constant)

    mech_spring = mechanical_components.MechanicalSpring(
        magnet_assembly=model.mechanical_model.magnet_assembly,
        position=model.mechanical_model.mechanical_spring.position,
        damping_coefficient=mech_spring_constant,
    )

    model.mechanical_model.set_damper(damper)
    model.set_coupling_model(coupling_model)
    model.mechanical_model.set_mechanical_spring(mech_spring)

    return model


def make_unified_model_from_path(model_prototype, param_path) -> UnifiedModel:

    with open(param_path, "rb") as f:
        params = cloudpickle.load(f)

    return make_unified_model_from_params(
        model_prototype=model_prototype,
        damper_cdc=params["damper_cdc"],
        coupling_constant=params["coupling_constant"],
        mech_spring_constant=params["mech_spring_constant"],
    )


instruments = {
    "damping_coefficient": ng.p.Scalar(lower=0, upper=10),
    "coupling_constant": ng.p.Scalar(lower=0, upper=10),
    "mech_spring_constant": ng.p.Scalar(lower=0, upper=10),
}

for dev in ["A", "B", "C", "D"]:
    print(f"🏃 :: {dev}")
    model_prototype, measurements = get_prototype_and_measurements(dev)
    candidate_params = parameter_search.mean_of_votes(
        model_prototype=model_prototype,
        measurements=measurements,
        instruments=instruments,
        budget=20,
    )

    # Write to disk
    with open(f"candidate_params_m1_{dev}_debug.params", "wb") as f:
        cloudpickle.dump(candidate_params, f)

    # print('MEAN LOSS:: ', np.mean(candidate_params['loss']))
    # print('------')
    # print(candidate_params)
    # print('------')
