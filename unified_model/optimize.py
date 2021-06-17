"""
A module for finding the optimal energy harvester
"""
import copy
from itertools import product
from typing import Any, Dict, List, Tuple, Union

import cloudpickle
import numpy as np
import pandas as pd
import ray
import nevergrad as ng
from flux_modeller.model import CurveModel
from tqdm import tqdm

from unified_model.coupling import CouplingModel
from unified_model.electrical_components.coil import CoilConfiguration
from unified_model.electrical_components.flux.model import FluxModelInterp
from unified_model.gridsearch import UnifiedModelFactory
from unified_model.mechanical_components.damper import (MassProportionalDamper,
                                                        QuasiKarnoppDamper)
from unified_model.mechanical_components.magnet_assembly import MagnetAssembly
from unified_model.mechanical_components.mechanical_spring import \
    MechanicalSpring  # noqa
from unified_model.unified import UnifiedModel
from unified_model.evaluate import Measurement
from unified_model import metrics


def _solve_and_score_single_device_and_measurement(
        model_prototype: UnifiedModel,
        measurement: Measurement,
        damping_coefficient: float,
        coupling_constant: float,
        mech_spring_constant: float
) -> float:
    """Solve and score a unified model for a single ground truth measurement."""

    model = copy.deepcopy(model_prototype)

    # Build a new model using parameters
    if model.mechanical_model is not None:

        # Damper
        model.mechanical_model.set_damper(
            MassProportionalDamper(
                damping_coefficient=damping_coefficient,
                magnet_assembly=model.mechanical_model.magnet_assembly
            )
        )

        # Mechanical spring
        model.mechanical_model.set_mechanical_spring(
            MechanicalSpring(
                magnet_assembly=model.mechanical_model.magnet_assembly,
                position=model.mechanical_model.mechanical_spring.position
            )
        )

        # Input excitation
        model.mechanical_model.set_input(measurement.input_)
    else:
        raise ValueError('MechanicalModel can not be None.')

    # Coupling model
    model.set_coupling_model(
        CouplingModel().set_coupling_constant(coupling_constant)
    )

    model.solve(
        t_start=0,
        t_end=8.,
        y0=[0., 0., 0.04, 0., 0.],
        t_eval=np.linspace(0, 8, 1000),
        t_max_step=1e-2
    )

    # Score the solved model against the ground truth.
    mech_result, mech_eval = model.score_mechanical_model(
        y_target=measurement.groundtruth.mech['y_diff'],
        time_target=measurement.groundtruth.mech['time'],
        metrics_dict={'dtw_distance': metrics.dtw_euclid_norm_by_length},
        prediction_expr='x3-x1',
        return_evaluator=True
    )
    elec_result, elec_eval = model.score_electrical_model(
        emf_target=measurement.groundtruth.elec['emf'],
        time_target=measurement.groundtruth.elec['time'],
        metrics_dict={'rms_perc_diff': metrics.root_mean_square_percentage_diff,
                       'dtw_distance': metrics.dtw_euclid_norm_by_length},
        prediction_expr='g(t, x5)',
        return_evaluator=True
    )

    return mech_result['dtw_distance'] + elec_result['dtw_distance']


def meta_minimize_for_mean_of_votes(
        model_prototype: UnifiedModel,
        measurements: List[Measurement],
        instruments: Dict[str, ng.p.Scalar],
        budget: int = 500
) -> Dict[str, List[float]]:
    """Perform the `mean of votes` evolutionary optimization.

    For each measurement, a set of model parameters is found that minimizes the
    cost function. The recommended parameters for each measurement, alongside
    the corresponding loss is returned.

    Currently, only the friction damping coefficient, coupling coefficient and
    mechanical spring damping coefficient are considered for optimization. These must be
    specified as `nevergrad` scalars using the `instruments` argument.

    The cost function is the sum of the DTW distance between the simulated and
    measured devices. There are two dtw distances. The first is the DTW distance
    between the simulated and measured position of the magnet assembly. The
    second is the DTW distance between the simulated and measured load voltage.

    Examples
    --------

    >>> # instrumentation example
    >>> instruments = {
    ...     'damping_coefficient': ng.p.Scalar(init=5),
    ...     'coupling_constant': ng.p.Scalar(init=5),
    ...     'mech_spring_constant': ng.p.Scalar(init=0)
    ...     }

    TODO: Rest of docstring
    """

    # Some quick validation.
    if 'damping_coefficient' not in instruments:
        raise KeyError('damping_coefficient must be present in `instruments`')
    if 'coupling_constant' not in instruments:
        raise KeyError('coupling_constant must be present in `instruments`')
    if 'mech_spring_constant' not in instruments:
        raise KeyError('mech_spring_constant must be present in `instruments`')

    if not measurements:
        raise ValueError('No measurements were passed.')

    recommended_params: Dict[str, List[float]] = {
        'damping_coefficient': [],
        'coupling_constant': [],
        'mech_spring_constant': [],
        'loss': []
    }

    for i, m in enumerate(measurements):
        print(f'-> {i+1}/{len(measurements)}')

        instrum = ng.p.Instrumentation(
            model_prototype=model_prototype,
            measurement=m,
            damping_coefficient=instruments['damping_coefficient'],
            coupling_constant=instruments['coupling_constant'],
            mech_spring_constant=instruments['mech_spring_constant']
        )

        optimizer = ng.optimizers.OnePlusOne(
            parametrization=instrum,
            budget=budget
        )

        recommendation = optimizer.minimize(
            _solve_and_score_single_device_and_measurement
        )

        recommended_params['damping_coefficient'].append(recommendation.value[1]['damping_coefficient'])  # noqa
        recommended_params['coupling_constant'].append(recommendation.value[1]['coupling_constant'])  # noqa
        recommended_params['mech_spring_constant'].append(recommendation.value[1]['mech_spring_constant'])  # noqa

        if recommendation.loss:
            recommended_params['loss'].append(recommendation.loss)
        else:
            raise ValueError('A loss could not be computed.')

    return recommended_params


def _get_new_flux_curve(
        curve_model: CurveModel,
        coil_configuration: CoilConfiguration
) -> Tuple[np.ndarray, np.ndarray]:
    """Get new z and phi values  from coil parameters and a `CurveModel`."""
    n_z = coil_configuration.n_z
    n_w = coil_configuration.n_w

    coil_params = np.array([[n_z, n_w]], dtype='int')  # type: ignore
    X = coil_params.reshape(1, -1)  # type: ignore
    return curve_model.predict_curves(X)


def get_new_flux_and_dflux_model(
        curve_model: CurveModel,
        coil_configuration: CoilConfiguration,
        magnet_assembly: MagnetAssembly) -> Tuple[Any, Any]:
    """Predict and return a new flux and dflux model from a CurveModel"""
    flux_interp_model = FluxModelInterp(coil_configuration, magnet_assembly)

    z_arr, phi = _get_new_flux_curve(curve_model=curve_model,
                                     coil_configuration=coil_configuration)

    flux_interp_model.fit(z_arr, phi.flatten())
    return flux_interp_model.flux_model, flux_interp_model.dflux_model


def evolve_simulation_set(unified_model_factory: UnifiedModelFactory,
                          input_excitations: List[Any],
                          curve_model: CurveModel,
                          coil_config_params: Dict,
                          magnet_assembly_params: Dict,
                          mech_spring_params: Dict,
                          damper_model_params: Dict) -> List[UnifiedModel]:
    """Update the simulation set with new flux and coil resistance models."""

    coil_configuration = CoilConfiguration(**coil_config_params)
    magnet_assembly = MagnetAssembly(**magnet_assembly_params)

    damper_model_params['magnet_assembly'] = magnet_assembly
    damper = QuasiKarnoppDamper(**damper_model_params)

    mech_spring_params['magnet_assembly'] = magnet_assembly
    new_mech_spring = MechanicalSpring(**mech_spring_params)
    new_flux_model, new_dflux_model = get_new_flux_and_dflux_model(
        curve_model=curve_model,
        coil_configuration=coil_configuration,
        magnet_assembly=magnet_assembly
    )

    new_factory = UnifiedModelFactory(
        damper=damper,  # New
        magnet_assembly=magnet_assembly,  # New
        magnetic_spring=unified_model_factory.magnetic_spring,
        mechanical_spring=new_mech_spring,  # New
        rectification_drop=unified_model_factory.rectification_drop,
        load_model=unified_model_factory.load_model,
        coil_configuration=coil_configuration,  # New
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
    """Calculate the RMS of `x`."""
    return np.sqrt(np.sum(x**2) / len(x))


@ray.remote
def _calc_constant_velocity_rms(curve_model: CurveModel,
                                coil_configuration: CoilConfiguration,
                                magnet_assembly: MagnetAssembly) -> float:
    """Calculate the open-circuit RMS for a simple emf curve."""

    flux_interp_model = FluxModelInterp(coil_configuration,
                                        magnet_assembly,
                                        curve_model)
    z_arr, phi = _get_new_flux_curve(curve_model=curve_model,
                                     coil_configuration=coil_configuration)
    flux_interp_model.fit(z_arr, phi.flatten())

    # Use constant velocity case
    dflux_curve = flux_interp_model.dflux_model
    velocity = 0.35  # doesn't matter
    z = np.linspace(0, 0.3, 1000)
    emf = np.array([dflux_curve.get(z) * velocity for z in z])  # type: ignore
    return calc_rms(emf)


def find_optimal_spacing(curve_model: CurveModel,
                         coil_config: CoilConfiguration,
                         magnet_assembly: MagnetAssembly) -> float:
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
            _calc_constant_velocity_rms.remote(curve_model,
                                               coil_config,
                                               magnet_assembly)
        )
    rms = ray.get(task_ids)
    return l_ccd_list[np.argmax(rms)]


def precompute_best_spacing(n_z_arr: np.ndarray,
                            n_w_arr: np.ndarray,
                            curve_model: CurveModel,
                            coil_config: CoilConfiguration,
                            magnet_assembly: MagnetAssembly,
                            output_path: str) -> None:
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
            find_optimal_spacing(curve_model,
                                 coil_config_copy,
                                 magnet_assembly)
        )

    nz_nw_product = np.array(nz_nw_product)
    df = pd.DataFrame({
        'n_z': nz_nw_product[:, 0],
        'n_w': nz_nw_product[:, 1],
        'optimal_spacing_mm': results
    })
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
    if df['optimal_spacing_mm'].max() < 1:  # Hack to check if we should return in mm.  # noqa
        df['optimal_spacing_mm'] = df['optimal_spacing_mm'] * 1000
    result = df.query(f'n_z == {n_z} and n_w == {n_w}')['optimal_spacing_mm'].values  # noqa

    if not result:
        raise ValueError(f'Coil parameters not found in {path}')

    return result[0]


def calc_p_load_avg(x, r_load):
    """Calculate the average power over the load."""
    v_rms = calc_rms(x)
    return v_rms * v_rms / r_load


@ray.remote
def simulate_unified_model_for_power(
        unified_model: UnifiedModel,
        **solve_kwargs
) -> Dict:
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
    unified_model.reset()  # Make sure we're starting from a clean slate

    if not solve_kwargs:
        solve_kwargs = {}

    solve_kwargs.setdefault('t_start', 0)
    solve_kwargs.setdefault('t_end', 8)
    solve_kwargs.setdefault('y0', [0., 0., 0.04, 0., 0.])
    solve_kwargs.setdefault('t_max_step', 1e-3)
    solve_kwargs.setdefault('t_eval', np.arange(0, 8, 1e-3))  # type: ignore

    unified_model.solve(**solve_kwargs)

    if unified_model.electrical_model is None:
        raise ValueError('ElectricalModel is not specified.')

    results = unified_model.calculate_metrics('g(t, x5)', {
        'p_load_avg': lambda x: calc_p_load_avg(x, unified_model.electrical_model.load_model.R)  # noqa
    })

    return results
