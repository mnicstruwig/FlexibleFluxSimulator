"""
A module for finding the optimal energy harvester
"""
import copy
import warnings
from itertools import product
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import ray
from flux_modeller.model import CurveModel
from tqdm import tqdm

from unified_model.electrical_components.coil import CoilConfiguration
from unified_model.electrical_components.flux.model import FluxModelInterp
from unified_model.gridsearch import UnifiedModelFactory
from unified_model.local_exceptions import ModelError
from unified_model.mechanical_components.damper import MassProportionalDamper
from unified_model.mechanical_components.magnet_assembly import MagnetAssembly
from unified_model.mechanical_components.mechanical_spring import MechanicalSpring
from unified_model.unified import UnifiedModel


def _get_new_flux_curve(
    curve_model: CurveModel, coil_configuration: CoilConfiguration
) -> Tuple[np.ndarray, np.ndarray]:
    """Get new z and phi values  from coil parameters and a `CurveModel`."""
    n_z = coil_configuration.n_z
    n_w = coil_configuration.n_w

    coil_params = np.array([[n_z, n_w]], dtype="int")  # type: ignore
    X = coil_params.reshape(1, -1)  # type: ignore
    return curve_model.predict_curves(X)


def _get_new_flux_and_dflux_model(
    curve_model: CurveModel,
    coil_configuration: CoilConfiguration,
    magnet_assembly: MagnetAssembly,
) -> Tuple[Any, Any]:
    """Predict and return a new flux and dflux model from a CurveModel"""
    flux_interp_model = FluxModelInterp(coil_configuration, magnet_assembly)

    z_arr, phi = _get_new_flux_curve(
        curve_model=curve_model, coil_configuration=coil_configuration
    )

    flux_interp_model.fit(z_arr, phi.flatten())
    return flux_interp_model.flux_model, flux_interp_model.dflux_model


# TODO: This should be available somewhere else
def calc_rms(x):
    """Calculate the RMS of `x`."""
    return np.sqrt(np.sum(x ** 2) / len(x))


@ray.remote
def _calc_constant_velocity_rms(
    curve_model: CurveModel,
    coil_configuration: CoilConfiguration,
    magnet_assembly: MagnetAssembly,
) -> float:
    """Calculate the open-circuit RMS for a simple emf curve."""

    flux_interp_model = FluxModelInterp(
        coil_configuration, magnet_assembly, curve_model
    )
    z_arr, phi = _get_new_flux_curve(
        curve_model=curve_model, coil_configuration=coil_configuration
    )
    flux_interp_model.fit(z_arr, phi.flatten())

    # Use constant velocity case
    dflux_curve = flux_interp_model.dflux_model
    velocity = 0.35  # doesn't matter
    z = np.linspace(0, 0.3, 1000)
    emf = np.array([dflux_curve.get(z) * velocity for z in z])  # type: ignore
    return calc_rms(emf)


def find_optimal_spacing(
    curve_model: CurveModel,
    coil_config: CoilConfiguration,
    magnet_assembly: MagnetAssembly,
) -> float:
    """Find spacing between each coil / magnet that produces the largest RMS.

    This is calculuated by finding the RMS of the produced EMF assuming a
    constant velocity. This function requires a running `ray` instance.

    Parameters
    ----------
    curve_model : CurveModel
        The trained CurveModel to use to predict the flux curve.
    coil_config: CoilConfiguration
        The coil configuration that must be simulated.
    magnet_assembly : MagnetAssembly
        The magnet assembly that must be simulated.

    Returns
    -------
    float
        The magnet spacing and/or coil spacing that produces the maximum RMS.

    Notes
    -----
    The optimal magnet spacing and/or coil spacing are equal to one another, and
    so can be used interchangeably depending on the context in which the optimal
    spacing is required.

    """
    coil_config = copy.copy(coil_config)
    coil_config.c = 2
    l_ccd_list = np.arange(1e-6, 0.05, 0.001)  # type: ignore
    task_ids = []
    for l_ccd in l_ccd_list:
        coil_config.l_ccd_mm = l_ccd * 1000  # Convert to mm
        task_ids.append(
            _calc_constant_velocity_rms.remote(
                curve_model, coil_config, magnet_assembly
            )
        )
    rms = ray.get(task_ids)
    return l_ccd_list[np.argmax(rms)]


def precompute_best_spacing(
    n_z_arr: np.ndarray,
    n_w_arr: np.ndarray,
    curve_model: CurveModel,
    coil_config: CoilConfiguration,
    magnet_assembly: MagnetAssembly,
    output_path: str,
) -> None:
    """Precompute the best spacing for coil parameters and save to disk.

    Parameters
    ----------
    n_z_arr : np.ndarray
        An array of the number of windings in the z (axial) direction to be
        considered.
    n_w_arr : np.ndarray
        An array of the number of windings in the w (radial) direction to be
        considered.
    curve_model : CurveModel
        The trained curve model to use to predict the flux waveform.
    magnet_assembly : MagnetAssembly
        The magnet assembly to use to excite the coil.
    output_path : str
        The path to store the output of the precomputation, which is a .csv
        file.

    """
    nz_nw_product: Union[List, np.ndarray] = list(product(n_z_arr, n_w_arr))
    results = []
    for n_z, n_w in tqdm(nz_nw_product):
        coil_config_copy = copy.copy(coil_config)
        coil_config_copy.n_w = n_w
        coil_config_copy.n_z = n_z
        results.append(
            find_optimal_spacing(curve_model, coil_config_copy, magnet_assembly)
        )

    nz_nw_product = np.array(nz_nw_product)
    df = pd.DataFrame(
        {
            "n_z": nz_nw_product[:, 0],
            "n_w": nz_nw_product[:, 1],
            "optimal_spacing_mm": results,
        }
    )
    # TODO: Consider a better format. Don't want to use pickle
    # due to version incompatibilities that can arise
    df.to_csv(output_path)


def lookup_best_spacing(path: str, n_z: int, n_w: int) -> float:
    """Return the optimal spacing between coils / magnet centers in mm.

    Parameters
    ----------
    path : str
        Path to a .csv file that contains optimal spacing for each value of
        `n_z` and `n_c`. Must have an `optimal_spacing_mm`, `n_z` and `n_w`
        column.
    n_z : int
        The value of `n_z` (number of coil windings in the axial direction) to
        lookup the optimal spacing for.
    n_w : int
        The value of `n_w` (number of coil windings in the radial direction) to
        lookup the optimal spacing for.

    Returns
    -------
    float
        The optimal coil or magnet center distance, in mm.

    """
    df = pd.read_csv(path)
    # TODO: Fix underlying data file so that optimal distances values are in mm.
    # For now, we use a loose heuristic to make sure we return in mm.
    if (
        df["optimal_spacing_mm"].max() < 1
    ):  # Hack to check if we should return in mm.  # noqa
        df["optimal_spacing_mm"] = df["optimal_spacing_mm"] * 1000
    result = df.query(f"n_z == {n_z} and n_w == {n_w}")[
        "optimal_spacing_mm"
    ].values  # noqa

    if not result:
        raise ValueError(f"Coil parameters not found in {path}")

    return result[0]


def calc_p_load_avg(x, r_load):
    """Calculate the average power over the load."""
    v_rms = calc_rms(x)
    return v_rms * v_rms / r_load


# TODO: Docstring
def evolve_simulation_set(
    unified_model_factory: UnifiedModelFactory,
    input_excitations: List[Any],
    curve_model: CurveModel,
    coil_config_params: Dict,
    magnet_assembly_params: Dict,
    mech_spring_params: Dict,
    damper_model_params: Dict,
    height_mm: float,
) -> List[UnifiedModel]:
    """Update the simulation set with new subsidiary models."""

    coil_configuration = CoilConfiguration(**coil_config_params)
    magnet_assembly = MagnetAssembly(**magnet_assembly_params)

    damper_model_params["magnet_assembly"] = magnet_assembly
    damper = MassProportionalDamper(**damper_model_params)

    mech_spring_params["magnet_assembly"] = magnet_assembly
    new_mech_spring = MechanicalSpring(**mech_spring_params)
    new_flux_model, new_dflux_model = _get_new_flux_and_dflux_model(
        curve_model=curve_model,
        coil_configuration=coil_configuration,
        magnet_assembly=magnet_assembly,
    )

    new_factory = UnifiedModelFactory(
        damper=damper,  # New
        magnet_assembly=magnet_assembly,  # New
        magnetic_spring=unified_model_factory.magnetic_spring,
        mechanical_spring=new_mech_spring,  # New
        height_mm=height_mm,
        rectification_drop=unified_model_factory.rectification_drop,
        load_model=unified_model_factory.load_model,
        coil_configuration=coil_configuration,  # New
        flux_model=new_flux_model,  # New
        dflux_model=new_dflux_model,  # New
        coupling_model=unified_model_factory.coupling_model,
        governing_equations=unified_model_factory.governing_equations,
        model_id=unified_model_factory.model_id,
    )

    unified_models = [new_factory.make(input_) for input_ in input_excitations]
    return unified_models


@ray.remote
def simulate_unified_model_for_power(model: UnifiedModel, **solve_kwargs) -> Dict:
    """Simulate a unified model and return the average load power.

    Parameters
    ----------
    unified_model : UnifiedModel
        The unified model to simulate.
    solve_kwargs : dict
        Keyword arguments passed to the `.solve` methof the `unified_model`.

    Returns
    -------
    dict
        Output of the `.calulate_metrics` method of `unified_model`.

    """
    model.reset()  # Make sure we're starting from a clean slate

    if not solve_kwargs:
        solve_kwargs = {}

    solve_kwargs.setdefault("t_start", 0)
    solve_kwargs.setdefault("t_end", 8)
    solve_kwargs.setdefault("y0", [0.0, 0.0, 0.04, 0.0, 0.0])
    solve_kwargs.setdefault("t_max_step", 1e-3)
    solve_kwargs.setdefault("t_eval", np.arange(0, 8, 1e-3))  # type: ignore

    # If our device is not valid, we want to return `None` as our result.
    try:
        model.solve(**solve_kwargs)

        if model.electrical_model is None:
            raise ValueError("ElectricalModel is not specified.")

        results = model.calculate_metrics(
            "g(t, x5)",
            {
                "p_load_avg": lambda x: calc_p_load_avg(
                    x, model.electrical_model.load_model.R
                )  # noqa
            },
        )
    except ModelError:
        warnings.warn("Device configuration not valid, skipping.")

        results = {"p_load_avg": None}

    return results
