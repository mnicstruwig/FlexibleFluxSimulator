import pandas as pd
import numpy as np
from tqdm import tqdm
import ray
import pyarrow.parquet as pq
import pyarrow as pa
from itertools import product
from copy import copy
from scipy.signal import savgol_filter

from unified_model import MechanicalModel
from unified_model import ElectricalModel
from unified_model import mechanical_components
from unified_model import electrical_components
from unified_model import CouplingModel
from unified_model import governing_equations
from unified_model import gridsearch  # <-- The new set of tools we'll be using exist in the `gridsearch` module
from unified_model import optimize

from flux_curve_modelling.model import CurveModel
from unified_model.electrical_components.flux.model import FluxModelInterp


# Mechanical components
magnetic_spring = mechanical_components.MagneticSpringInterp(
    fea_data_file='./experiments/data/magnetic-spring/10x10alt.csv',
    magnet_length=10/1000,
    filter_callable=lambda x: savgol_filter(x, window_length=27, polyorder=5)
)

# NB: This will need to be update for the multi-magnet case!
magnet_assembly = mechanical_components.MagnetAssembly(
    n_magnet=1,
    l_m=10,
    l_mcd=0,
    dia_magnet=10,
    dia_spacer=10
)
mech_spring = mechanical_components.MechanicalSpring(
    position=110/1000,
    damping_coefficient=7.778,
    magnet_length=10/1000
)
damper = mechanical_components.ConstantDamper(0.0433)

# Electrical Components
R_coil = None  # Need to get from `optimize` module
load = electrical_components.SimpleLoad(R=30)
v_rect_drop = 0.1
coupling_model = CouplingModel().set_coupling_constant(4.444)

# Initial flux model
coil_params = {
   'beta': 1361/1000/1000,
    'n_z': 20,
    'n_w': 20,
    'l_th': 2,
    'r_c': 0.143/2,
    'c': 1,
    'm': 1,
    'c_c': 0.059,
    'r_t': 5.5,
}


curve_model = CurveModel.load('./experiments/flux_curve_model.model')

# Build our first "template" factory
unified_model_factory = gridsearch.UnifiedModelFactory(
    damper=damper,
    magnet_assembly=magnet_assembly,
    mechanical_spring=mech_spring,
    magnetic_spring=magnetic_spring,
    coil_resistance=None,
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
for log_file in glob('./experiments/data/2019-05-23_D/A/log*_acc.csv'):
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

print(len(acc_inputs))

def batchify(x, batch_size):
    total_size = len(x)
    indexes = np.arange(0, total_size, batch_size)

    if indexes[-1] < total_size:
        indexes = np.append(indexes, [total_size])

    return [x[indexes[i]:indexes[i+1]] for i in range(len(indexes)-1)]


# Actual experiment

ray.init(ignore_reinit_error=True)

n_z_arr = np.arange(2, 201, 40)
n_w_arr = np.arange(2, 201, 40)

batch_size = 256
nz_nw_product = np.array(list(product(n_z_arr, n_w_arr)))  # type: ignore
n_z_list = []
n_w_list = []
input_ids = []
submitted = []

print(f'Pending simulations: {len(nz_nw_product)}')
batches = batchify(nz_nw_product, batch_size)
for batch_num, batch in enumerate(batches):
    print(f'Executing batch {batch_num+1} out of {len(batches)}')
    for n_z, n_w in batch:
        coil_params_copy = copy(coil_params)
        coil_params_copy['n_z'] = n_z
        coil_params_copy['n_w'] = n_w
        coil_params_copy['c'] = 2

        # To make sure our arrays are the same length at the end
        n_z_values = [n_z]*len(acc_inputs)
        n_w_values = [n_w]*len(acc_inputs)
        n_z_list = n_z_list + n_z_values
        n_w_list = n_w_list + n_w_values

        if coil_params_copy['c'] > 1:
            coil_params_copy['l_ccd'] = optimize.lookup_best_spacing(
                path='./experiments/optimal_l_ccd_0_200_5.csv',
                n_z=n_z,
                n_w=n_w
            )
        else:
            coil_params_copy['l_ccd'] = 0

        simulation_models = optimize.evolve_simulation_set(unified_model_factory=unified_model_factory,
                                                          input_excitations=acc_inputs,
                                                          curve_model=curve_model,
                                                          coil_model_params=coil_params_copy)
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
    pq.write_to_dataset(table, 'test_run.parquet')

    # Clear
    del results
    del df
    del table

    submitted = []
    n_z_list = []
    n_w_list = []

ray.shutdown()
