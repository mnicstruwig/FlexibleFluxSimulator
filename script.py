from copy import copy
from itertools import product

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import ray
from flux_modeller.model import CurveModel
from scipy.signal import savgol_filter

from unified_model import \
    gridsearch  # <-- The new set of tools we'll be using exist in the `gridsearch` module
from unified_model import (CouplingModel, electrical_components,
                           governing_equations, mechanical_components,
                           optimize)

# PARAMETERS
n_z_arr = np.arange(6, 201, 2)
n_w_arr = np.arange(6, 201, 2)
c = 1
m = 2

# Mechanical components
magnetic_spring = mechanical_components.MagneticSpringInterp(
    fea_data_file='./data/magnetic-spring/10x10alt.csv',
    magnet_length=10/1000,
    filter_callable=lambda x: savgol_filter(x, window_length=27, polyorder=5)
)

mech_spring = mechanical_components.MechanicalSpring(
    position=110/1000,
    damping_coefficient=7.778,
    magnet_length=10/1000
)
damper = mechanical_components.ConstantDamper(0.0433)

# Electrical Components
load = electrical_components.SimpleLoad(R=30)
v_rect_drop = 0.1
coupling_model = CouplingModel().set_coupling_constant(4.444)

coil_model_params = {
    'c': None,
    'n_z': None,
    'n_w': None,
    'l_ccd_mm': None,
    'ohm_per_mm': 1361/1000/1000,
    'tube_wall_thickness_mm': 2,
    'coil_wire_radius_mm': 0.143/2,
    'coil_center_mm': 59,
    'outer_tube_radius_mm': 5.5
}

magnet_assembly_params = {
    'm': None,
    'l_m_mm': 10,
    'l_mcd_mm': None,
    'dia_magnet_mm': 10,
    'dia_spacer_mm': 10
}

curve_model = CurveModel.load('./data/flux_curve_model.model')

# Build our first "template" factory
unified_model_factory = gridsearch.UnifiedModelFactory(
    damper=damper,
    magnet_assembly=None,
    mechanical_spring=mech_spring,
    magnetic_spring=magnetic_spring,
    coil_model=None,
    rectification_drop=v_rect_drop,
    load_model=load,
    flux_model=None,
    dflux_model=None,
    coupling_model=coupling_model,
    governing_equations=governing_equations.unified_ode,
    model_id=0,
)

# Choose our input excitations
from glob import glob

acc_inputs = []
for log_file in glob('./data/2019-05-23_D/A/log*_acc.csv'):
    acc_input = mechanical_components.AccelerometerInput(
        raw_accelerometer_input=log_file,
        accel_column='z_G',
        time_column='time(ms)',
        accel_unit='g',
        time_unit='ms',
        smooth=True,
        interpolate=True
    )
    acc_inputs.append(acc_input)


def batchify(x, batch_size):
    total_size = len(x)
    indexes = np.arange(0, total_size, batch_size)

    if indexes[-1] < total_size:
        indexes = np.append(indexes, [total_size])  # type: ignore

    return [x[indexes[i]:indexes[i+1]] for i in range(len(indexes)-1)]


# Actual experiment
ray.init(ignore_reinit_error=True)


batch_size = 256
nz_nw_product = np.array(list(product(n_z_arr, n_w_arr)))  # type: ignore
n_z_list = []
n_w_list = []
input_ids = []
submitted = []

print(f'Executing {len(nz_nw_product)} device simulations.')
print(f'There are {len(acc_inputs)} inputs per simulation.')
batches = batchify(nz_nw_product, batch_size)
for batch_num, batch in enumerate(batches):
    print(f'Executing batch {batch_num+1} out of {len(batches)}...')
    for n_z, n_w in batch:
        coil_model_params_copy = copy(coil_model_params)
        coil_model_params_copy['n_z'] = n_z
        coil_model_params_copy['n_w'] = n_w
        coil_model_params_copy['c'] = c

        magnet_assembly_params_copy = copy(magnet_assembly_params)
        magnet_assembly_params_copy['m'] = m

        # To make sure our arrays are the same length at the end
        n_z_values = [n_z]*len(acc_inputs)
        n_w_values = [n_w]*len(acc_inputs)
        n_z_list = n_z_list + n_z_values
        n_w_list = n_w_list + n_w_values

        # Start with default values. Lookup optimal spacing if necessary.
        coil_model_params_copy['l_ccd_mm'] = 0
        magnet_assembly_params_copy['l_mcd_mm'] = 0

        optimal_spacing = None
        if coil_model_params_copy['c'] > 1:
            optimal_spacing = optimize.lookup_best_spacing(
                path='./data/optimal_l_ccd_0_200_2.csv',
                n_z=n_z,
                n_w=n_w
            )
            coil_model_params_copy['l_ccd_mm'] = optimal_spacing

        if magnet_assembly_params_copy['m'] > 1:
            optimal_spacing = optimize.lookup_best_spacing(
                path='./data/optimal_l_ccd_0_200_2.csv',
                n_z=n_z,
                n_w=n_w
            )
            magnet_assembly_params_copy['l_mcd_mm'] = optimal_spacing

        simulation_models = optimize.evolve_simulation_set(
            unified_model_factory=unified_model_factory,
            input_excitations=acc_inputs,
            curve_model=curve_model,
            coil_model_params=coil_model_params_copy,
            magnet_assembly_params=magnet_assembly_params_copy
        )
        for i, um in enumerate(simulation_models):
            submitted.append(optimize.simulate_unified_model.remote(um))
            input_ids.append(i)

    print(f'Submitted {len(submitted)} tasks...')
    ready = []
    while len(ready) < len(submitted):  # Wait for batch to complete
        ready, waiting = ray.wait(submitted, num_returns=len(submitted), timeout=30)
        print(f'Completed {len(ready)} out of {len(submitted)} tasks...')
    print(f'Finished batch!')

    # Export results
    results = ray.get(ready)
    df = pd.DataFrame({
        'n_z': n_z_list,
        'n_w': n_w_list,
        'input': input_ids,
        'p_load_avg': [r['p_load_avg'] for r in results]
    })
    table = pa.Table.from_pandas(df)
    pq.write_to_dataset(table, f'./{c}c{m}m.parquet')

    # Clear
    del results
    del df
    del table

    submitted = []
    n_z_list = []
    n_w_list = []
    input_ids = []

ray.shutdown()
