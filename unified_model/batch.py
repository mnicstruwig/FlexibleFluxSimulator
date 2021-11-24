"""
Perform batch simulations, in parallel.
"""
from typing import Dict, Tuple, List, Any
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import ray
import pyarrow as pa
import pyarrow.parquet as pq

from unified_model.unified import UnifiedModel
from unified_model.evaluate import Measurement, Sample
from unified_model.utils.utils import batchify


@ray.remote
def _score_measurement(
        model: UnifiedModel,
        solve_kwargs: Dict,
        measurement: Measurement,
        mech_pred_expr: str,
        mech_metrics: Dict,
        elec_pred_expr: str,
        elec_metrics: Dict
):
    result, _ = model.score_measurement(
        measurement=measurement,
        solve_kwargs=solve_kwargs,
        mech_pred_expr=mech_pred_expr,
        mech_metrics_dict=mech_metrics,
        elec_pred_expr=elec_pred_expr,
        elec_metrics_dict=elec_metrics
    )

    return result


# TODO: Docstring, batchify
def solve_for_batch(
        base_model_config: Dict,
        params: List[Tuple[str, Any]],
        samples: List[Sample],
        mech_pred_expr: str = None,
        mech_metrics: Dict = None,
        elec_pred_expr: str = None,
        elec_metrics: Dict = None,
        solve_kwargs: Dict = None
):
    start_time = datetime.now()

    if ray.is_initialized():  # Make sure we start fresh
        ray.shutdown()
    ray.init()

    base_model = UnifiedModel.from_config(base_model_config)

    if not solve_kwargs:
        solve_kwargs = {
            't_start': 0.,
            't_end': 8,
            'y0': [0., 0., 0.04, 0., 0.],
            't_max_step': 1e-3,
            't_eval': np.arange(0, 8, 1e-3),
            'method': "RK23"
        }

    batches = batchify(params, batch_size=256)

    for batch_number, current_batch in enumerate(batches):  # For each batch
        print(f'✨ Running batch {batch_number+1} out of {len(batches)}... ')

        results: List[Any] = []
        tasks: List[Any] = []

        for param_set in current_batch:  # For each parameter set
            model = base_model.update_params(param_set)

            for i, s in enumerate(samples):  # For each sample

                task_id = _score_measurement.remote(
                    model=model,
                    solve_kwargs=solve_kwargs,
                    measurement=Measurement(s, model),
                    mech_pred_expr=mech_pred_expr,
                    mech_metrics=mech_metrics,
                    elec_pred_expr=elec_pred_expr,
                    elec_metrics=elec_metrics
                )
                tasks.append(task_id)

                info = {}

                # Add parameter information
                for param_name, param_value in param_set:
                    info[param_name] = param_value

                # Add input excitation number
                info['input'] = i

                # Add the full config
                info['config'] = model.get_config(as_type='json')

                results.append(info)

        ready: List = []
        while len(ready) < len(tasks):  # Wait for every task to finish
            ready, waiting = ray.wait(tasks, num_returns=len(tasks), timeout=60)
            print(f'🕙 Still waiting for {len(waiting)} jobs...')

        # Update our results array with scored measurement results
        for i, r in enumerate(ready):
            result = ray.get(r)
            results[i].update(result)

        df = pd.DataFrame(results)
        table = pa.Table.from_pandas(df)
        pq.write_to_dataset(table, f"./batch_run_{start_time}.parquet")

    ray.shutdown()
