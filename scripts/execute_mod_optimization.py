"""Execute a gridsearch that creates an optimization dataset.

In addition to the normal optimization process, this gridsearch also explores
the coil center position as an optimizable variable.
"""

from copy import copy
from glob import glob
from itertools import product
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import ray
from scipy.signal import savgol_filter

from flux_modeller.model import CurveModel
from unified_model import (
    CouplingModel,
    electrical_components,
    governing_equations,
    gridsearch,
    mechanical_components,
    optimize,
)

# PARAMETERS
c = 1
m = 1
n_z_arr = np.arange(6, 101, 4)
n_w_arr = np.arange(6, 101, 4)
c_c_arr = np.arange(20, 82, 5)

# Mechanical components
magnetic_spring = mechanical_components.MagneticSpringInterp(
    fea_data_file="./data/magnetic-spring/10x10alt.csv",
    magnet_length=10 / 1000,
    filter_callable=lambda x: savgol_filter(x, window_length=27, polyorder=5),
)

damper_model_params: Dict[str, Any] = {
    "damping_coefficient": 4.272,
    "magnet_assembly": None,
}

# Electrical Components
load = electrical_components.SimpleLoad(R=5)
v_rect_drop = 0.1
coupling_model = CouplingModel().set_coupling_constant(5.096)

coil_model_params: Dict[str, Any] = {
    "c": None,
    "n_z": None,
    "n_w": None,
    "l_ccd_mm": None,
    "ohm_per_mm": 1079 / 1000 / 1000,
    "tube_wall_thickness_mm": 2,
    "coil_wire_radius_mm": 0.143 / 2,
    "coil_center_mm": None,  # An additional optimization parameter
    "inner_tube_radius_mm": 5.5,
}

magnet_assembly_params: Dict[str, Any] = {
    "m": None,
    "l_m_mm": 10,
    "l_mcd_mm": 0,
    "dia_magnet_mm": 10,
    "dia_spacer_mm": 10,
}

mech_spring_params: Dict[str, Any] = {
    "magnet_assembly": None,
    "strength": 1e7,
    "damping_coefficient": 3.108,
}

curve_model = CurveModel.load(
    "./data/flux_curve_model/flux_curve_model_2021_05_11.model"
)

height_mm = 105

# Build our first "template" factory
unified_model_factory = gridsearch.UnifiedModelFactory(
    damper=None,
    magnet_assembly=None,
    mechanical_spring=None,
    magnetic_spring=magnetic_spring,
    coil_configuration=None,
    height_mm=height_mm,
    rectification_drop=v_rect_drop,
    load_model=load,
    flux_model=None,
    dflux_model=None,
    coupling_model=coupling_model,
    governing_equations=governing_equations.unified_ode,
    model_id=0,
)

# Choose our input excitations

acc_inputs = []
for log_file in glob("./data/2019-05-23/A/log*_acc.csv"):
    acc_input = mechanical_components.AccelerometerInput(
        raw_accelerometer_input=log_file,
        accel_column="z_G",
        time_column="time(ms)",
        accel_unit="g",
        time_unit="ms",
        smooth=True,
        interpolate=True,
    )
    acc_inputs.append(acc_input)

acc_inputs = acc_inputs[1:3]
assert len(acc_inputs) != 0  # Safety check


def batchify(x, batch_size):
    """Batch a list `x` into batches of size `batch_size`."""
    total_size = len(x)
    indexes = np.arange(0, total_size, batch_size)

    if indexes[-1] < total_size:
        indexes = np.append(indexes, [total_size])  # type: ignore

    return [x[indexes[i] : indexes[i + 1]] for i in range(len(indexes) - 1)]


# Actual experiment
ray.init(ignore_reinit_error=True, num_cpus=12)


BATCH_SIZE = 256
nz_nw_cc_product = np.array(list(product(n_z_arr, n_w_arr, c_c_arr)))  # type: ignore
n_z_list: List[float] = []
n_w_list: List[float] = []
c_c_list: List[float] = []
input_ids = []
submitted = []

print(f"Executing {len(nz_nw_cc_product)} device simulations.")
print(f"There are {len(acc_inputs)} inputs per simulation.")
batches = batchify(nz_nw_cc_product, BATCH_SIZE)
for batch_num, batch in enumerate(batches):
    print(f"Executing batch {batch_num+1} out of {len(batches)}...")
    for n_z, n_w, c_c in batch:
        coil_model_params_copy = copy(coil_model_params)
        coil_model_params_copy["n_z"] = n_z
        coil_model_params_copy["n_w"] = n_w
        coil_model_params_copy["c"] = c
        coil_model_params_copy["coil_center_mm"] = c_c

        magnet_assembly_params_copy = copy(magnet_assembly_params)
        magnet_assembly_params_copy["m"] = m

        damper_model_params = copy(damper_model_params)

        # To make sure our arrays are the same length at the end
        n_z_values = [n_z] * len(acc_inputs)
        n_w_values = [n_w] * len(acc_inputs)
        c_c_values = [c_c] * len(acc_inputs)
        n_z_list = n_z_list + n_z_values
        n_w_list = n_w_list + n_w_values
        c_c_list = c_c_list + c_c_values

        # Start with default values. Lookup optimal spacing if necessary.
        coil_model_params_copy["l_ccd_mm"] = 0
        magnet_assembly_params_copy["l_mcd_mm"] = 0

        optimal_spacing = None
        if coil_model_params_copy["c"] > 1:
            optimal_spacing = optimize.lookup_best_spacing(
                path="./data/flux_curve_model/optimal_l_ccd_0_200_2_v2.csv",
                n_z=n_z,
                n_w=n_w,
            )
            coil_model_params_copy["l_ccd_mm"] = optimal_spacing

        if magnet_assembly_params_copy["m"] > 1:
            optimal_spacing = optimize.lookup_best_spacing(
                path="./data/flux_curve_model/optimal_l_ccd_0_200_2_v2.csv",
                n_z=n_z,
                n_w=n_w,
            )
            magnet_assembly_params_copy["l_mcd_mm"] = optimal_spacing

        simulation_models = optimize.evolve_simulation_set(
            unified_model_factory=unified_model_factory,
            input_excitations=acc_inputs,
            curve_model=curve_model,
            coil_config_params=coil_model_params_copy,
            magnet_assembly_params=magnet_assembly_params_copy,
            mech_spring_params=mech_spring_params,
            damper_model_params=damper_model_params,
            height_mm=height_mm,
        )
        for i, um in enumerate(simulation_models):
            submitted.append(optimize.simulate_unified_model_for_power.remote(um))
            input_ids.append(i)

    print(f"Submitted {len(submitted)} tasks...")
    ready: List[Any] = []
    while len(ready) < len(submitted):  # Wait for batch to complete
        ready, waiting = ray.wait(submitted, num_returns=len(submitted), timeout=30)
        print(f"Completed {len(ready)} out of {len(submitted)} tasks...")
    print("Finished batch!")

    # Export results
    results = ray.get(ready)
    df = pd.DataFrame(
        {
            "n_z": n_z_list,
            "n_w": n_w_list,
            "c_c": c_c_list,
            "input": input_ids,
            "p_load_avg": [r["p_load_avg"] for r in results],
        }
    )
    table = pa.Table.from_pandas(df)
    pq.write_to_dataset(table, f"./output/{c}c{m}m_amd_radau.parquet")

    # Clear
    del results
    del df
    del table

    submitted = []
    n_z_list = []
    n_w_list = []
    c_c_list = []
    input_ids = []

ray.shutdown()
