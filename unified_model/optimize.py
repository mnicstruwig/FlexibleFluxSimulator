"""
A module for finding the optimal energy harvester
"""
from copy import copy
from itertools import product
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import ray
from tqdm import tqdm

from flux_modeller.model import CurveModel
from unified_model.electrical_components.flux.model import FluxModelInterp
from unified_model.mechanical_components.magnet_assembly import MagnetAssembly
from unified_model.gridsearch import UnifiedModelFactory
from unified_model.unified import UnifiedModel


def _get_new_flux_curve(
        curve_model: CurveModel,
        coil_model_params: Dict
) -> Tuple[np.ndarray, np.ndarray]:
    """Get new z and phi values  from coil parameters and a `CurveModel`."""
    coil_model_params = copy(coil_model_params)
    n_z = coil_model_params['n_z']
    n_w = coil_model_params['n_w']

    coil_params = np.array([[n_z, n_w]])  # type: ignore
    X = coil_params.reshape(1, -1)  # type: ignore
    return curve_model.predict_curve(X)


def _get_coil_resistance(beta: float,
                         n_z: int,
                         n_w: int,
                         l_th: float,
                         r_c: float,
                         r_t: float,
                         c: int,
                         **kwargs):
    """Get the resistance of the coil based on its parameters

    Parameters:
    ----------
    beta : float
        Wire resistance per unit length in Ohm per mm.
    n_z : int
        The number of windings in the z-direction per coil.
    n_w : int
        The number of windings in the radial direction per coil.
    l_th : float
        The thickness of the tube wall in mm.
    r_c : float
        The radius of the wire in mm.
    r_t : float
        The inner radius of the tube in mm.
    num_coils : int
        The number of coils that make up the winding configuration.

    """
    return 2*np.pi*beta*n_w*n_z*(l_th+r_t+r_c+2*r_c*(n_w+1)/2)*c


def get_new_flux_and_dflux_model(curve_model, coil_model_params):
    flux_interp_model = FluxModelInterp(**coil_model_params)

    z_arr, phi = _get_new_flux_curve(curve_model=curve_model,
                                     coil_model_params=coil_model_params)

    flux_interp_model.fit(z_arr, phi.flatten())
    return flux_interp_model.flux_model, flux_interp_model.dflux_model


def evolve_simulation_set(unified_model_factory: UnifiedModelFactory,
                          input_excitations: List[Any],
                          curve_model: Any,
                          coil_model_params: Dict,
                          magnet_assembly_params: Dict) -> List[UnifiedModel]:
    """Update the simulation set with new flux and coil resistance models."""

    new_flux_model, new_dflux_model = get_new_flux_and_dflux_model(
        curve_model=curve_model,
        coil_model_params=coil_model_params
    )

    new_coil_resistance = _get_coil_resistance(**coil_model_params)

    new_magnet_assembly = MagnetAssembly(**magnet_assembly_params)

    new_factory = UnifiedModelFactory(
        damper=unified_model_factory.damper,
        magnet_assembly=new_magnet_assembly,  # New
        magnetic_spring=unified_model_factory.magnetic_spring,
        mechanical_spring=unified_model_factory.mechanical_spring,
        coil_resistance=new_coil_resistance,  # New
        rectification_drop=unified_model_factory.rectification_drop,
        load_model=unified_model_factory.load_model,
        flux_model=new_flux_model,  # New
        dflux_model=new_dflux_model,  # New
        coupling_model=unified_model_factory.coupling_model,
        governing_equations=unified_model_factory.governing_equations,
        model_id=unified_model_factory.model_id
    )

    unified_models = [new_factory.make(input_) for input_ in input_excitations]
    return unified_models


# TODO: This should be available somewhere else
def calc_rms(x):
    return np.sqrt(np.sum(x**2)/len(x))

def calc_p_load_avg(x, r_load):
    v_rms = calc_rms(x)
    return v_rms*v_rms/r_load

@ray.remote
def _calc_constant_velocity_rms(curve_model, coil_model_params):
    """Calculate the open-circuit RMS for a simple emf curve."""
    flux_interp_model = FluxModelInterp(**coil_model_params)
    z_arr, phi = _get_new_flux_curve(curve_model=curve_model,
                                     coil_model_params=coil_model_params)
    flux_interp_model.fit(z_arr, phi.flatten())


    # Use constant velocity case
    dflux_curve = flux_interp_model.dflux_model
    velocity = 0.35  # doesn't matter
    z = np.linspace(0, 0.3, 1000)
    emf = np.array([dflux_curve.get(z)*velocity for z in z])
    return calc_rms(emf)


def find_optimal_spacing(curve_model, coil_model_params):
    """Find spacing between each coil / magnet that produces the largest RMS"""
    cmp = copy(coil_model_params)  # Dicts are mutable
    cmp['c'] = 2
    l_ccd_list = np.arange(1e-6, 0.05, 0.001)  # type: ignore
    task_ids = []
    for l_ccd in l_ccd_list:
        cmp['l_ccd'] = l_ccd
        task_ids.append(_calc_constant_velocity_rms.remote(curve_model, cmp))
    rms = ray.get(task_ids)
    return l_ccd_list[np.argmax(rms)]


def precompute_best_spacing(n_z_arr: np.ndarray,
                            n_w_arr: np.ndarray,
                            curve_model: CurveModel,
                            coil_model_params: Dict,
                            output_path: str) -> None:
    """Precompute the best spacing for coil parameters and save to disk"""
    nz_nw_product = np.array(list(product(n_z_arr, n_w_arr)))  # type: ignore
    results = []
    for n_z, n_w in tqdm(nz_nw_product):
        coil_model_params['n_z'] = n_z
        coil_model_params['n_w'] = n_w
        results.append(find_optimal_spacing(curve_model, coil_model_params))

    df = pd.DataFrame({
        'n_z': nz_nw_product[:, 0],
        'n_w': nz_nw_product[:, 1],
        'optimal_spacing_mm': results
    })
    # TODO: Consider a better format. Don't want to use pickle
    # due to version incompatibilities that can arise
    df.to_csv(output_path)


def lookup_best_spacing(path, n_z, n_w):
    df = pd.read_csv(path)
    result = df.query(f'n_z == {n_z} and n_w == {n_w}')['optimal_spacing_mm'].values

    if not result:
        raise ValueError(f'Coil parameters not found in {path}')

    return result[0]


@ray.remote
def simulate_unified_model(unified_model: UnifiedModel, **solve_kwargs):
    unified_model.reset()  # Make sure we're starting from a clean slate

    if not solve_kwargs:
        solve_kwargs = {}

    solve_kwargs.setdefault('t_start', 0)
    solve_kwargs.setdefault('t_end', 8)
    solve_kwargs.setdefault('y0', [0., 0., 0.04, 0., 0.])
    solve_kwargs.setdefault('t_max_step', 1e-3)
    solve_kwargs.setdefault('t_eval', np.arange(0, 8, 1e-3))  # type: ignore

    unified_model.solve(**solve_kwargs)

    results = unified_model.calculate_metrics('g(t, x5)', {
        'p_load_avg': lambda x: calc_p_load_avg(x, unified_model.electrical_model.load_model.R)
    })

    return results
