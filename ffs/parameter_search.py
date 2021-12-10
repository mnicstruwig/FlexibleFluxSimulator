"""A module for finding the parameter of a unified model."""

import copy
from datetime import datetime
from typing import Any, Dict, List, Tuple

import nevergrad as ng
import numpy as np
import ray  # type: ignore

from .evaluate import Measurement, Sample
from .metrics import (
    dtw_euclid_norm_by_length,
    power_difference_perc,
    root_mean_square_percentage_diff,
)
from .unified import UnifiedModel


def _assert_valid_cost_metric(cost_metric: str) -> None:
    if cost_metric not in ["dtw", "power", "combined"]:
        raise ValueError(
            "`cost_metric` must be one of 'dtw', 'power' or 'combined'."
        )  # noqa


def mean_of_scores(
    models_and_samples: List[Tuple[UnifiedModel, List[Sample]]],
    instruments: Dict[str, ng.p.Scalar],
    cost_metric: str,
    budget: int = 500,
    verbose: bool = True,
    log_to_disk: bool = False,
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
    models_and_samples : List[Tuple[UnifiedModel, List[Sample]]]
        A list of tuples. The first element of each tuple is a
        fully-instantiated UnifiedModel object that will be used as the basis
        for the parameter search. The second element of each tuple is a list of
        corresponding `Sample` objects to the unified model. For the
        unified model, the damper, coupling model, mechanical spring and input
        excitation will be replaced during the parameter search. Every other
        component must be specified.
    instruments : Dict[str, ng.p.Scalar]
        The `nevergrad` parametrization instruments that will be evolved in
        order to find the most accurate set of parameters. Keys must correspond
        to all three of {'mech_damping_coefficient', 'coupling_constant',
        'mech_spring_damping_coefficient'} and values must be a nevergrad
        `Scalar`. See the relevant `Scalar` documentation for how initial values
        and limits can be set.
    cost_metric : str
        The cost metric to use. Must be 'dtw', 'power' or 'combined'.
    budget : int
        The budget (maximum number of allowed iterations) that the evolutionary
        algorithm will run for, for *each* measurement in `measurements`.
        Optional.
    verbose : bool
        Set to `True` to print out the Loss value of the cost function during
        optimization. Optional. Default value is `True`.
    log_to_disk : bool
        Set to `True` to log the cost function scores to disk during
        optimization. Optional. Default value is `False`.

    Returns
    -------
    Dict[str, float]
        The optimization results. Keys are {'mech_damping_coefficient',
        'coupling_constant', 'mech_spring_damping_coefficient'}, and each value
        corresponds to the optimal parameter values found by minimizing the cost
        function.

    Examples
    --------

    >>> # instrumentation example
    >>> instruments = {
    ...     'mech_damping_coefficient': ng.p.Scalar(init=5),
    ...     'coupling_constant': ng.p.Scalar(init=5),
    ...     'mech_spring_damping_coefficient': ng.p.Scalar(init=0)
    ...     }

    See Also
    --------
    nevergrad.p.Scalar : Class
        The nevergrad Scalar parametrization instrument.

    """
    _assert_valid_cost_metric(cost_metric)

    instrum = ng.p.Instrumentation(
        models_and_samples=models_and_samples,
        cost_metric=cost_metric,
        mech_damping_coefficient=instruments["mech_damping_coefficient"],
        coupling_constant=instruments["coupling_constant"],
        mech_spring_damping_coefficient=instruments["mech_spring_damping_coefficient"],
    )

    optimizer = ng.optimizers.OnePlusOne(parametrization=instrum, budget=budget)

    def callback(optimizer, candidate, value):
        try:
            best_score = np.round(optimizer.provide_recommendation().loss, 5)
        except TypeError as e:
            best_score = None
        latest_score = np.round(value, 5)
        text = f"{optimizer.num_ask} / {budget} - latest: {latest_score} - best: {best_score}"  # noqa
        print(text, end="\r")  # Rewrite our latest outputline each time

    if verbose:
        optimizer.register_callback("tell", callback)

    if log_to_disk:
        log_file = f"evo_log_{datetime.now()}.json"
        logger = ng.callbacks.ParametersLogger(log_file)
        optimizer.register_callback("tell", logger)

    ray.init()
    recommendation = optimizer.minimize(
        _calculate_cost_for_multiple_devices_multiple_samples
    )

    recommended_params = {
        "mech_damping_coefficient": recommendation.value[1]["mech_damping_coefficient"],
        "coupling_constant": recommendation.value[1]["coupling_constant"],
        "mech_spring_damping_coefficient": recommendation.value[1][
            "mech_spring_damping_coefficient"
        ],
        "loss": recommendation.loss,
    }

    ray.shutdown()
    return recommended_params


def _calculate_cost_for_single_sample(
    model_prototype: UnifiedModel,
    sample: Sample,
    cost_metric: str,
    mech_damping_coefficient: float,
    coupling_constant: float,
    mech_spring_damping_coefficient: float,
) -> float:
    """Return the cost of a unified model for a single ground truth sample.

    This function is intended to be passed to a `nevergrad` optimizer.

    """
    model = copy.deepcopy(model_prototype)

    # Quick verification
    assert model.magnet_assembly is not None
    assert model.mechanical_spring is not None

    # Update the model using the suggested parameters
    model = model.update_params(
        [
            ("mechanical_damper.damping_coefficient", mech_damping_coefficient),
            ("coupling_model.coupling_constant", coupling_constant),
            ("mechanical_spring.damping_coefficient", mech_spring_damping_coefficient),
        ]
    )

    measurement = Measurement(sample, model)
    # Set the new input excitation
    model.with_input_excitation(measurement.input_)

    model.solve(
        t_start=0,
        t_end=8.0,
        y0=[0.0, 0.0, 0.04, 0.0, 0.0],
        t_eval=np.linspace(0, 8, 1000),
        t_max_step=1e-3,
    )

    # Score the solved model against the ground truth.
    mech_result = _score_mechanical_model(model, measurement)
    elec_result = _score_electrical_model(model, measurement)

    if cost_metric == "dtw":
        return mech_result["dtw_distance"] + elec_result["dtw_distance"]
    elif cost_metric == "power":
        return np.abs(elec_result["watt_perc_diff"])
    elif cost_metric == "combined":
        return (
            mech_result["dtw_distance"]
            + elec_result["dtw_distance"]
            + np.abs(elec_result["watt_perc_diff"])
        )  # noqa
    else:
        raise ValueError(
            "`cost_metric` must be one of 'dtw', 'power' or 'combined'."
        )  # noqa


# TODO: Docs
@ray.remote
def _calculate_cost_for_single_sample_distributed(
    model_prototype: UnifiedModel,
    sample: Sample,
    cost_metric: str,
    mech_damping_coefficient: float,
    coupling_constant: float,
    mech_spring_damping_coefficient: float,
) -> float:
    """A ray-wrapped version to calculate the cost if a single sample."""
    return _calculate_cost_for_single_sample(
        model_prototype=model_prototype,
        sample=sample,
        cost_metric=cost_metric,
        mech_damping_coefficient=mech_damping_coefficient,
        coupling_constant=coupling_constant,
        mech_spring_damping_coefficient=mech_spring_damping_coefficient,
    )


def _calculate_cost_for_multiple_devices_multiple_samples(
    models_and_samples: List[Tuple[UnifiedModel, List[Sample]]],
    cost_metric: str,
    mech_damping_coefficient: float,
    coupling_constant: float,
    mech_spring_damping_coefficient: float,
) -> float:
    """Return cost of multiple unified models with multiple samples.

    The cost is the average cost across all devices and their respective ground
    truth measurements.  This function is intended to be passed to a `nevergrad`
    optimizer.

    """
    costs = []
    tasks = []
    for model, samples in models_and_samples:
        for sample in samples:
            task_id = _calculate_cost_for_single_sample_distributed.remote(
                model_prototype=model,
                sample=sample,
                cost_metric=cost_metric,
                mech_damping_coefficient=mech_damping_coefficient,
                coupling_constant=coupling_constant,
                mech_spring_damping_coefficient=mech_spring_damping_coefficient,
            )
            tasks.append(task_id)

    ready: List[Any] = []
    while len(ready) < len(tasks):
        ready, _ = ray.wait(tasks, num_returns=len(tasks), timeout=30)

    costs = ray.get(tasks)
    mean_cost = np.mean(costs)
    return mean_cost  # type: ignore


def _score_mechanical_model(
    model: UnifiedModel, measurement: Measurement
) -> Dict[str, float]:
    """Score the unified model against a ground truth measurement.

    This function scores the mechanical component of the model.

    """
    mech_result, _ = model._score_mechanical_model(
        y_target=measurement.groundtruth.mech["y_diff"],
        time_target=measurement.groundtruth.mech["time"],
        metrics_dict={"dtw_distance": dtw_euclid_norm_by_length},
        prediction_expr="x3-x1",
        return_evaluator=False,
    )
    return mech_result


def _score_electrical_model(
    model: UnifiedModel, measurement: Measurement
) -> Dict[str, float]:
    """Score the unified model against a ground truth measurement.

    This function scores the electrical component of the model.

    """
    elec_result, _ = model._score_electrical_model(
        emf_target=measurement.groundtruth.elec["emf"],
        time_target=measurement.groundtruth.elec["time"],
        metrics_dict={
            "rms_perc_diff": root_mean_square_percentage_diff,
            "dtw_distance": dtw_euclid_norm_by_length,
            "watt_perc_diff": power_difference_perc,
        },
        prediction_expr="g(t, x5)",
        return_evaluator=False,
    )
    return elec_result
