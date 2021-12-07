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

from .electrical_components.coil import CoilConfiguration
from .electrical_components.flux import FluxModelInterp
from .local_exceptions import ModelError
from .mechanical_components.damper import MassProportionalDamper
from .mechanical_components.magnet_assembly import MagnetAssembly
from .mechanical_components.mechanical_spring import MechanicalSpring
from .unified import UnifiedModel


# TODO: This should be available somewhere else
def calc_rms(x):
    """Calculate the RMS of `x`."""
    return np.sqrt(np.sum(x ** 2) / len(x))


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


@ray.remote
def simulate_unified_model_for_power(model: UnifiedModel, **solve_kwargs) -> Dict:
    """Simulate a unified model and return the average load power.

      Parameters
      ----------
    m  : UnifiedModel
          The unified model to simulate.
      solve_kwargs : dict
          Keyword arguments passed to the `.solve` method them .

      Returns
      -------
      dict
          Output of the `.calulate_metrics` method ofm .

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

        results = model.calculate_metrics(
            "g(t, x5)",
            {
                "p_load_avg": lambda x: calc_p_load_avg(
                    x, model.electrical_model.load_model.R  # type: ignore
                )
            },
        )
    except ModelError:
        warnings.warn("Device configuration not valid, skipping.")
        results = {"p_load_avg": None}

    return results
