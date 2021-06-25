"""A module for finding the parameter of a unified model."""

import copy
from typing import Dict, List, Tuple, Any

import nevergrad as ng
import numpy as np
import ray  # type: ignore

from .coupling import CouplingModel
from .evaluate import Measurement
from .mechanical_components import MassProportionalDamper, MechanicalSpring
from .metrics import (dtw_euclid_norm_by_length, power_difference_perc,
                      root_mean_square_percentage_diff)
from .unified import UnifiedModel


def _import_spinner() -> Any:
    # https://stackoverflow.com/a/22424821
    import sys
    try:
        get_ipython = sys.modules['IPython'].get_ipython  # type: ignore
        if 'IPKernelApp' in get_ipython().config:
            from halo import HaloNotebook as Halo
    except KeyError:
        from halo import Halo  # type: ignore
    return Halo  # type: ignore


def _assert_valid_cost_metric(cost_metric: str) -> None:
    if cost_metric not in ['dtw', 'power', 'combined']:
        raise ValueError("`cost_metric` must be one of 'dtw', 'power' or 'combined'.")  # noqa


def mean_of_votes(
        model_prototype: UnifiedModel,
        measurements: List[Measurement],
        instruments: Dict[str, ng.p.Scalar],
        cost_metric: str,
        budget: int = 500
) -> Dict[str, List[float]]:
    """Perform the `mean of votes` evolutionary parameter search optimization.

    For each measurement, a set of model parameters is found that minimizes the
    cost function. The recommended parameters for each measurement, alongside
    the corresponding loss is returned.

    Currently, only the friction damping coefficient, coupling coefficient and
    mechanical spring damping coefficient are considered for the parameter
    search. These must be specified as `nevergrad` scalars using the
    `instruments` argument.

    The cost function is the sum of the DTW distance between the simulated and
    measured devices. There are two dtw distances. The first is the DTW distance
    between the simulated and measured position of the magnet assembly. The
    second is the DTW distance between the simulated and measured load voltage.

    Parameters
    ----------
    model_prototype : UnifiedModel
        A fully-instantiated UnifiedModel object that will be used as the basis
        for the parameter search. The damper, coupling model, mechanical spring
        and input excitation will be replaced during the parameter search. Every
        other component must be specified.
    measurements : List[Measurement]
        A list of Measurement objects that will be used to both drive the
        unified model (by using the input excitation present in each Measurement
        object) and also to compare the simulation results with the ground truth
        present in each Measurement.
    instruments : Dict[str, ng.p.Scalar]
        The `nevergrad` parametrization instruments that will be evolved in
        order to find the most accurate set of parameters. Keys must correspond
        to all three of {'damping_coefficient', 'coupling_constant',
        'mech_spring_constant'} and values must be a nevergrad `Scalar`. See the
        relevant `Scalar` documentation for how initial values and limits can be
        set.
    cost_metric : str
        The cost metric to use. Must be 'dtw,'power' or 'combined'.
    budget : int
        The budget (maximum number of allowed iterations) that the evolutionary
        algorithm will run for, for *each* measurement in `measurements`.
        Optional.

    Returns
    -------
    Dict[str, List[float]]
        The optimization results. Each key is one of {'damping_coefficient',
        'coupling_constant', 'mech_spring_constant'}, and each value corresponds
        to a list of optimal parameter values found for each measurement in
        `measurements`.

    Examples
    --------

    >>> # instrumentation example
    >>> instruments = {
    ...     'damping_coefficient': ng.p.Scalar(init=5),
    ...     'coupling_constant': ng.p.Scalar(init=5),
    ...     'mech_spring_constant': ng.p.Scalar(init=0)
    ...     }

    See Also
    --------
    evaluate.Measurement : Class
        The Measurement class that contains both input excitation and measured
        ground truth information.
    nevergrad.p.Scalar : Class
        The nevergrad Scalar parametrization instrument.

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

    _assert_valid_cost_metric(cost_metric)

    model = copy.deepcopy(model_prototype)

    recommended_params: Dict[str, List[float]] = {
        'damping_coefficient': [],
        'coupling_constant': [],
        'mech_spring_constant': [],
        'loss': []
    }
    ray.init()
    tasks = []
    for m in measurements:

        @ray.remote
        def wrapper():
            instrum = ng.p.Instrumentation(
                model_prototype=model,
                measurement=m,
                cost_metric=cost_metric,
                damping_coefficient=instruments['damping_coefficient'],
                coupling_constant=instruments['coupling_constant'],
                mech_spring_constant=instruments['mech_spring_constant']
            )

            optimizer = ng.optimizers.PSO(
                parametrization=instrum,
                budget=budget
            )
            recommendation = optimizer.minimize(
                _calculate_cost_for_single_measurement
            )
            return recommendation

        ref = wrapper.remote()
        tasks.append(ref)

    Halo = _import_spinner()
    spinner = Halo(text=f'Running :: 0 / {len(tasks)}', spinner='dots', color='red')  # noqa
    spinner.start()

    ready: List[Any] = []
    while len(ready) < len(tasks):
        ready, _ = ray.wait(tasks, num_returns=len(tasks), timeout=30)
        spinner.text = f'Running :: {len(ready)} / {len(tasks)}'

    spinner.succeed('Simulation complete!')
    results = ray.get(tasks)

    for recommendation in results:
        recommended_params['damping_coefficient'].append(recommendation.value[1]['damping_coefficient'])  # noqa
        recommended_params['coupling_constant'].append(recommendation.value[1]['coupling_constant'])  # noqa
        recommended_params['mech_spring_constant'].append(recommendation.value[1]['mech_spring_constant'])  # noqa

        if recommendation.loss:
            recommended_params['loss'].append(recommendation.loss)
        else:
            raise ValueError('A loss could not be computed.')

    ray.shutdown()
    return recommended_params


def mean_of_scores(
        models_and_measurements: List[Tuple[UnifiedModel, List[Measurement]]],
        instruments: Dict[str, ng.p.Scalar],
        cost_metric: str,
        budget: int = 500,
        verbose: bool = True
) -> Dict[str, float]:
    """Perform the `mean of scores` evolutionary parameter search optimization.

    A set of model parameters is found that minimizes the average cost functions
    across all measurements. The recommended parameters, alongside the
    corresponding loss, is returned.

    Currently, only the friction damping coefficient, coupling coefficient and
    mechanical spring damping coefficient are considered for the parameter
    search. These must be specified as `nevergrad` scalars using the
    `instruments` argument.

    The cost function is the mean of the sum of the DTW distance between the
    simulated and measured devices. There are two dtw distances. The first is
    the DTW distance between the simulated and measured position of the magnet
    assembly. The second is the DTW distance between the simulated and measured
    load voltage.

    Parameters
    ----------
    models_and_measurements : List[Tuple[UnifiedModel, List[Measurement]]]
        A list of tuples. The first element of each tuple is a
        fully-instantiated UnifiedModel object that will be used as the basis
        for the parameter search. The second element of each tuple is a list of
        corresponding `Measurement` objects to the unified model. For the
        unified model, the damper, coupling model, mechanical spring and input
        excitation will be replaced during the parameter search. Every other
        component must be specified.
    instruments : Dict[str, ng.p.Scalar]
        The `nevergrad` parametrization instruments that will be evolved in
        order to find the most accurate set of parameters. Keys must correspond
        to all three of {'damping_coefficient', 'coupling_constant',
        'mech_spring_constant'} and values must be a nevergrad `Scalar`. See the
        relevant `Scalar` documentation for how initial values and limits can be
        set.
    cost_metric : str
        The cost metric to use. Must be 'dtw,'power' or 'combined'.
    budget : int
        The budget (maximum number of allowed iterations) that the evolutionary
        algorithm will run for, for *each* measurement in `measurements`.
        Optional.
    verbose : bool
        Set to `True` to print out the Loss value of the cost function during
        optimization. Optional.

    Returns
    -------
    Dict[str, float]
        The optimization results. Each key is one of {'damping_coefficient',
        'coupling_constant', 'mech_spring_constant'}, and each value corresponds
        to the optimal parameter values found by minimizing the cost function.

    Examples
    --------

    >>> # instrumentation example
    >>> instruments = {
    ...     'damping_coefficient': ng.p.Scalar(init=5),
    ...     'coupling_constant': ng.p.Scalar(init=5),
    ...     'mech_spring_constant': ng.p.Scalar(init=0)
    ...     }

    See Also
    --------
    evaluate.Measurement : Class
        The Measurement class that contains both input excitation and measured
        ground truth information.
    nevergrad.p.Scalar : Class
        The nevergrad Scalar parametrization instrument.

    """
    _assert_valid_cost_metric(cost_metric)

    instrum = ng.p.Instrumentation(
        models_and_measurements=models_and_measurements,
        damping_coefficient=instruments['damping_coefficient'],
        coupling_constant=instruments['coupling_constant'],
        mech_spring_constant=instruments['mech_spring_constant']
    )

    optimizer = ng.optimizers.PSO(
        parametrization=instrum,
        budget=budget
    )

    Halo = _import_spinner()
    spinner = Halo(text='Starting...', spinner='dots')
    spinner.start()

    def callback(optmizer, candidate, value):
        try:
            best_score = np.round(optimizer.provide_recommendation().loss, 5)
        except TypeError:
            best_score = None
        latest_score = np.round(value, 5)
        spinner.text = f'{optimizer.num_ask} / {budget} - latest: {latest_score} - best: {best_score}'  # noqa

    if verbose:
        optimizer.register_callback('tell', callback)

    ray.init()
    recommendation = optimizer.minimize(
        _calculate_cost_for_multiple_devices_multiple_measurements
    )
    spinner.succeed('Done!')

    recommended_params = {
        'damping_coefficient': recommendation.value[1]['damping_coefficient'],
        'coupling_constant': recommendation.value[1]['coupling_constant'],
        'mech_spring_constant': recommendation.value[1]['mech_spring_constant'],
        'loss': recommendation.loss
    }

    ray.shutdown()
    return recommended_params


def _calculate_cost_for_single_measurement(
        model_prototype: UnifiedModel,
        measurement: Measurement,
        cost_metric: str,
        damping_coefficient: float,
        coupling_constant: float,
        mech_spring_constant: float
) -> float:
    """Return the cost of a unified model for a single ground truth measurement.

    This function is intended to be passed to a `nevergrad` optimizer.

    """

    model = copy.deepcopy(model_prototype)

    # Quick verification
    assert model.mechanical_model is not None
    assert model.mechanical_model.magnet_assembly is not None
    assert model.mechanical_model.mechanical_spring is not None

    # Build a new model using parameters

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
            position=model.mechanical_model.mechanical_spring.position,
            damping_coefficient=mech_spring_constant
        )
    )

    # Input excitation
    model.mechanical_model.set_input(measurement.input_)

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
    mech_result = _score_mechanical_model(model, measurement)
    elec_result = _score_electrical_model(model, measurement)

    if cost_metric == 'dtw':
        return mech_result['dtw_distance'] + elec_result['dtw_distance']
    elif cost_metric == 'power':
        return np.abs(mech_result['watt_perc_diff'])
    elif cost_metric == 'combined':
        return mech_result['dtw_distance'] + elec_result['dtw_distance'] + np.abs(elec_result['watts_perc_diff'])  # noqa
    else:
        raise ValueError("`cost_metric` must be one of 'dtw', 'power' or 'combined'.")  # noqa


@ray.remote
def _calculate_cost_for_single_measurement_remote(
        model_prototype: UnifiedModel,
        measurement: Measurement,
        cost_metric: str,
        damping_coefficient: float,
        coupling_constant: float,
        mech_spring_constant: float
) -> float:
    """A ray-wrapped version to calculate the cost if a single measurement."""
    return _calculate_cost_for_single_measurement(
        model_prototype=model_prototype,
        measurement=measurement,
        cost_metric=cost_metric,
        damping_coefficient=damping_coefficient,
        coupling_constant=coupling_constant,
        mech_spring_constant=mech_spring_constant
    )


def _calculate_cost_for_multiple_devices_multiple_measurements(
        models_and_measurements: List[Tuple[UnifiedModel, List[Measurement]]],
        damping_coefficient: float,
        coupling_constant: float,
        mech_spring_constant: float
) -> float:
    """Return cost of multiple unified models with multiple measurements.

    The cost is the average cost across all devices and their respective ground
    truth measurements.  This function is intended to be passed to a `nevergrad`
    optimizer.

    """
    costs = []
    tasks = []
    for model, measurements in models_and_measurements:
        for measurement in measurements:

            task_id = _calculate_cost_for_single_measurement_remote.remote(
                model_prototype=model,
                measurement=measurement,
                damping_coefficient=damping_coefficient,
                coupling_constant=coupling_constant,
                mech_spring_constant=mech_spring_constant
            )
            tasks.append(task_id)

    ready: List[Any] = []
    while len(ready) < len(tasks):
        ready, _ = ray.wait(tasks, num_returns=len(tasks), timeout=30)

    costs = ray.get(tasks)
    mean_cost = np.mean(costs)
    return mean_cost  # type: ignore


def _score_mechanical_model(
        model: UnifiedModel,
        measurement: Measurement
) -> Dict[str, float]:
    """Score the unified model against a ground truth measurement.

    This function scores the mechanical component of the model.

    """
    mech_result, _ = model.score_mechanical_model(
        y_target=measurement.groundtruth.mech['y_diff'],
        time_target=measurement.groundtruth.mech['time'],
        metrics_dict={'dtw_distance': dtw_euclid_norm_by_length},
        prediction_expr='x3-x1',
        return_evaluator=False
    )
    return mech_result


def _score_electrical_model(
        model: UnifiedModel,
        measurement: Measurement
) -> Dict[str, float]:
    """Score the unified model against a ground truth measurement.

    This function scores the electrical component of the model.

    """
    elec_result, _ = model.score_electrical_model(
        emf_target=measurement.groundtruth.elec['emf'],
        time_target=measurement.groundtruth.elec['time'],
        metrics_dict={'rms_perc_diff': root_mean_square_percentage_diff,
                      'dtw_distance': dtw_euclid_norm_by_length,
                      'watts_perc_diff': power_difference_perc},
        prediction_expr='g(t, x5)',
        return_evaluator=False
    )
    return elec_result
