"""
Perform batch simulations, in parallel.
"""
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import ray

from .evaluate import Measurement, Sample
from .unified import UnifiedModel
from .utils.utils import batchify


@ray.remote
def _score_measurement(
    model: UnifiedModel,
    solve_kwargs: Dict,
    measurement: Measurement,
    mech_pred_expr: str,
    mech_metrics: Dict,
    elec_pred_expr: str,
    elec_metrics: Dict,
    prediction_expr: str,
    prediction_metrics: Dict
):
    result, _ = model.score_measurement(
        measurement=measurement,
        solve_kwargs=solve_kwargs,
        mech_pred_expr=mech_pred_expr,
        mech_metrics_dict=mech_metrics,
        elec_pred_expr=elec_pred_expr,
        elec_metrics_dict=elec_metrics,
    )

    if prediction_expr is not None and prediction_metrics is not None:
        result_metric = model.calculate_metrics(
            prediction_expr=prediction_expr,
            metric_dict=prediction_metrics
        )
        result.update(result_metric)

    return result


# TODO: Update docs
def solve_for_batch(
    base_model_config: Dict,
    params: List[List[Tuple[str, Any]]],
    samples: List[Sample],
    mech_pred_expr: str = None,
    mech_metrics: Dict = None,
    elec_pred_expr: str = None,
    elec_metrics: Dict = None,
    prediction_expr: str = None,
    prediction_metrics: Dict = None,
    solve_kwargs: Dict = None,
    output_root_dir: str = '.',
) -> None:
    """
    Solve, score and store solutions for a batch of interpolated device designs.

    Solves, scores and writes-out results for a base model configuration that is
    updated with every parameter set specified in `params`. Each parameter set
    is solved and scored against every ground truth Sample in `samples` for the
    metrics and expressions specified in the `*_pred_exr` and `*_metrics`
    arguments.

    Parameters
    ----------
    base_model_config : Dict
        Dict representing the base `UnifiedModel` configuration that will be
        updated using parameters in `params`. The configuration can be obtained pf
        from the `UnifiedModel` class methods.
    params : List[List[Tuple[str, Any]]]
        List of parameters to calculate solutions for. Each parameter in
        `params` will be used to update the `base_model_config`. See the
        `Examples` section for the shape of `params`. Each parameter in `params`
        must be compatible with the `UnifiedModel.update_params` method.
    samples : List[Sample]
        List of instantiated `Sample`s that contain both the input excitation
        and groundtruth data. Typically collected using the
        `utils.utils.collect_samples` function.
    mech_pred_expr : str
         Expression that is evaluated and used as the predictions for the
         mechanical system. Any reasonable expression is possible. You
         can refer to each of the differential equations referenced by the
         `governing_equations` using the letter `x` with the number appended.
         For example, `x1` refers to the first differential equation, and
         `x2` refers to the second differential equation. Optional.
    mech_metrics : Dict[str, Any]
        Metrics to compute on the predicted and target mechanical data. Keys is
        a user-chosen name given to the metric returned in the result. Values
        must be a function, that accepts two numpy arrays (arr_predict,
        arr_target), and computes a scalar value. See the `metrics` module for
        some built-in metrics. Optional.
    elec_pred_expr : str
        Expression that is evaluated and used as the predictions for the
        electrical system. Identical in functionality to `mech_pred_expr`.
        Optional.
    elec_metrics: Dict[str, Any]
        Metrics to compute on the predicted and target electrical data.
        Identical in functionality to `mech_metrics`. Optional.
    prediction_expr : str
        Expression that is evaluated and used as input for `prediction_metrics`.
        Identical in functionality to `mech_pred_expr`.  Optional.
    prediction_metrics: Dict[str, Any]
        Metrics to compute on `prediction_expr`. Identical in functionality to
        `mech_metrics`. Optional.
    solve_kwargs : Dict[str, Any]
        Additional keyword arguments passed to the `UnifiedModel.solve` method for each
        model that is solved. Optional.

    Examples
    --------
    >>> # Example `params`
    >>> params = [[('mechanical_model.damper.damping_coefficient', 5.3)]]
    >>> param_set_1 = [
    ...     ('mechanical_model.damper.damping_coefficient', 1.5)
    ...     ('coupling_model.coupling_constant', 0.1)
    ... ]
    >>> param_set_2 = [
    ...     ('mechanical_model.damper.damping_coefficient', 2.0)
    ...     ('coupling_model.coupling_constant', 0.2)
    ... ]
    >>> param_set_3 = [
    ...     ('mechanical_model.damper.damping_coefficient', 3.0)
    ...     ('coupling_model.coupling_constant', 0.3)
    ... ]
    >>> params = [param_set_1, param_set_2, param_set_3]

    >>> # Example mech_metrics
    >>> mech_metrics = {
    ...    'dtw_mech_norm' : metrics.dtw_euclid_norm_by_length
    ...    'dtw_mech' : metrics.dtw_euclid_distance
    ... }

    """
    start_time = datetime.now()
    output_path = os.path.abspath(output_root_dir + f"/batch_run_{start_time}.parquet")

    if ray.is_initialized():  # Make sure we start fresh
        ray.shutdown()
    ray.init()

    base_model = UnifiedModel.from_config(base_model_config)

    if not solve_kwargs:
        solve_kwargs = {
            "t_start": 0.0,
            "t_end": 8,
            "y0": [0.0, 0.0, 0.04, 0.0, 0.0],
            "t_max_step": 1e-3,
            "t_eval": np.arange(0, 8, 1e-3),
            "method": "RK45",
        }

    batches = batchify(params, batch_size=256)

    for batch_number, current_batch in enumerate(batches):  # For each batch
        print(f"✨ Running batch {batch_number+1} out of {len(batches)}... ")

        results: List[Any] = []
        tasks: List[Any] = []

        for param_set in current_batch:  # For each parameter set
            model = base_model.update_params(param_set)

            for i, s in enumerate(samples):  # For each sample

                task_id = _score_measurement.remote(  # type: ignore
                    model=model,
                    solve_kwargs=solve_kwargs,
                    measurement=Measurement(s, model),
                    mech_pred_expr=mech_pred_expr,
                    mech_metrics=mech_metrics,
                    elec_pred_expr=elec_pred_expr,
                    elec_metrics=elec_metrics,
                    prediction_expr=prediction_expr,
                    prediction_metrics=prediction_metrics
                )
                tasks.append(task_id)

                info = {}

                # Add parameter information
                for param_name, param_value in param_set:
                    info[param_name] = param_value

                # Add input excitation number
                info["input"] = i

                # Add the full config
                info["config"] = model.get_config(kind="json")

                results.append(info)

        ready: List = []
        while len(ready) < len(tasks):  # Wait for every task to finish
            ready, waiting = ray.wait(tasks, num_returns=len(tasks), timeout=60)
            print(f"🕙 Still waiting for {len(waiting)} jobs...")

        # Update our results array with scored measurement results
        for i, r in enumerate(ready):
            result = ray.get(r)
            results[i].update(result)

        df = pd.DataFrame(results)
        table = pa.Table.from_pandas(df)
        pq.write_to_dataset(table, output_path)

    ray.shutdown()
