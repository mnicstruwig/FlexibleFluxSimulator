"""
A module for finding the optimal energy harvester
"""
from typing import Any, Tuple, List, Dict
from unified_model.gridsearch import UnifiedModelFactory
from unified_model.electrical_components.flux.model import FluxModelInterp
from unified_model.unified import UnifiedModel

import numpy as np
from flux_curve_modelling.model import CurveModel

def _get_new_flux_curve(curve_model: CurveModel,
                        n_z: int,
                        n_w: int) -> Tuple[np.ndarray, np.ndarray]:
    """Get new z and phi values  from coil parameters and a trained `CurveModel`."""

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
                                     n_z=coil_model_params['n_z'],
                                     n_w=coil_model_params['n_w'])


    flux_interp_model.fit(z_arr, phi.flatten())
    return flux_interp_model.flux_model, flux_interp_model.dflux_model


def evolve_simulation_set(unified_model_factory: UnifiedModelFactory,
                          input_excitations: List[Any],
                          curve_model: Any,
                          coil_model_params: Dict) -> List[UnifiedModel]:
    """Update the simulation set with new flux and coil resistance models."""

    new_flux_model, new_dflux_model = get_new_flux_and_dflux_model(
        curve_model=curve_model,
        coil_model_params=coil_model_params
    )

    new_coil_resistance = _get_coil_resistance(**coil_model_params)

    new_factory = UnifiedModelFactory(
        damper=unified_model_factory.damper,
        magnet_assembly=unified_model_factory.magnet_assembly,
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
