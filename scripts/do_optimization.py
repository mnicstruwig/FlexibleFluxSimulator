from glob import glob
from itertools import product
from typing import Any, Dict, List
import warnings

import ray
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.signal import savgol_filter
from tqdm import tqdm
from unified_model import (CouplingModel, ElectricalModel, MechanicalModel,
                           UnifiedModel, electrical_components,
                           mechanical_components, optimize)
from unified_model.governing_equations import unified_ode
from unified_model.utils.utils import batchify
from unified_model.local_exceptions import ModelError


def calc_rms(x):
    """Calculate the RMS of `x`."""
    return np.sqrt(np.sum(x ** 2) / len(x))


def calc_p_load_avg(x, r_load):
    """Calculate the average power over the load."""
    v_rms = calc_rms(x)
    return v_rms * v_rms / r_load


@ray.remote
def simulate_and_calculate_power(model: UnifiedModel):
    try:
        model.solve(
            t_start=0.,
            t_end=8,
            y0=[0., 0., 0.04, 0., 0.],
            t_max_step=1e-4,
            t_eval=np.arange(0, 8, 1e-3),
            method="RK23"
        )

        results = model.calculate_metrics(
            prediction_expr='g(t, x5)',
            metric_dict={
                'p_load_avg': lambda x: calc_p_load_avg(x, model.electrical_model.load_model.R)
            }
        )
    except ModelError as e:
        warnings.warn(e)
        warnings.warn("Model didn't pass validation. Skipping.")

        results = {'p_load_avg': None}

    return results


# PARAMETERS
BATCH_SIZE = 128
c = 1
m = 1
n_z_arr = np.arange(5, 101, 5)
n_w_arr = np.arange(5, 101, 5)
c_c_arr = np.arange(20, 82, 10)

# Collect samples
acc_inputs = []
log_files = glob("./data/2019-05-23/A/log*_acc.csv")
log_files.sort()  # Don't forget!
for log_file in log_files:
    acc_input = mechanical_components.AccelerometerInput(
        raw_accelerometer_data_path=log_file,
        accel_column="z_G",
        time_column="time(ms)",
        accel_unit="g",
        time_unit="ms",
        smooth=True,
        interpolate=True,
    )
    acc_inputs.append(acc_input)

acc_inputs = acc_inputs[1:2]
assert len(acc_inputs) != 0

batches = batchify(list(product(n_z_arr, n_w_arr, c_c_arr)), BATCH_SIZE)

print(f'Batches: {len(batches)}')
print(f'Input Excitations per model: {len(acc_inputs)}')

ray.init(num_cpus=12)
for batch_number, batch in enumerate(batches):

    task_ids: List[Any] = []
    output: Dict[str, Any] = {
        'c': [],
        'm': [],
        'n_z': [],
        'n_w': [],
        'c_c': [],
        'input': [],
        'p_load_avg': [],
        'config': []
    }

    print(f'✨ Running batch {batch_number} out of {len(batches)}... ')
    for n_z, n_w, c_c in batch:
        for i, input_ in enumerate(acc_inputs):
            # Build our model
            optimal_spacing = optimize.lookup_best_spacing(
                path="./data/flux_curve_model/optimal_l_ccd_0_200_2_v2.csv",
                n_z=n_z,
                n_w=n_w
            ) if (c > 1 or m > 1) else 0

            magnet_assembly = mechanical_components.MagnetAssembly(
                m=m,
                l_m_mm=10,
                l_mcd_mm=optimal_spacing if m > 1 else 0,
                dia_magnet_mm=10,
                dia_spacer_mm=10
            )

            coil_config = electrical_components.CoilConfiguration(
                c=c,
                n_z=n_z,
                n_w=n_w,
                l_ccd_mm=optimal_spacing if c > 1 else 0,
                ohm_per_mm=1079/ 1000 / 1000,
                tube_wall_thickness_mm=2,
                coil_wire_radius_mm=0.143 / 2,
                coil_center_mm=c_c,
                inner_tube_radius_mm=5.5
            )

            magnetic_spring = mechanical_components.MagneticSpringInterp(
                fea_data_file="./data/magnetic-spring/10x10alt.csv",
                magnet_length=10 / 1000,
                filter_callable=lambda x: savgol_filter(x, window_length=27, polyorder=5)
            )

            mechanical_spring = mechanical_components.MechanicalSpring(
                magnet_assembly=magnet_assembly,
                damping_coefficient=3.108
            )

            mechanical_damper = mechanical_components.MassProportionalDamper(
                damping_coefficient=4.272,
                magnet_assembly=magnet_assembly
            )

            flux_model = electrical_components.FluxModelPretrained(
                coil_config=coil_config,
                magnet_assembly=magnet_assembly,
                curve_model_path='./data/flux_curve_model/flux_curve_model_2021_05_11.model'
            )

            load_model = electrical_components.SimpleLoad(R=5)

            # Compose
            mechanical_model = (
                MechanicalModel()
                + magnet_assembly
                + magnetic_spring
                + mechanical_spring
                + mechanical_damper
            )

            electrical_model = (
                ElectricalModel()
                + coil_config
                + flux_model
                + load_model
                + 0.01  # Rectification drop.  TODO: Make this a class.
            )

            coupling_model = CouplingModel().set_coupling_constant(5.096)

            model = (
                UnifiedModel()
                .set_mechanical_model(mechanical_model)
                .set_electrical_model(electrical_model)
                .set_coupling_model(coupling_model)
                .set_governing_equations(unified_ode)
                .set_height(105)
            )

            model.mechanical_model.set_input(input_)

            task_ids.append(simulate_and_calculate_power.remote(model))

            output['c'].append(c)
            output['m'].append(m)
            output['n_z'].append(n_z)
            output['n_w'].append(n_w)
            output['c_c'].append(c_c)
            output['input'].append(i)
            output['config'].append(model.get_config('json'))

    # Wait for results after submitting jobs for the batch
    ready: List[Any] = []
    while (len(ready) < len(task_ids)):
        ready, waiting = ray.wait(task_ids, num_returns=len(task_ids), timeout=30)
        print(f'🕙 Still waiting for {len(waiting)} jobs...')

    results = [ray.get(id_) for id_ in task_ids]
    p_load_avg = [r['p_load_avg'] for r in results]
    output['p_load_avg'] = p_load_avg

    df = pd.DataFrame(output)
    table = pa.Table.from_pandas(df)
    pq.write_to_dataset(table, f"./output/{c}c{m}m_i1_DOP853_small_ts.parquet")

ray.shutdown()
