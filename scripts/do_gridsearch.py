from itertools import product

from scipy.signal import savgol_filter
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from tqdm import tqdm

from unified_model import mechanical_components
from unified_model import electrical_components
from unified_model import CouplingModel
from unified_model import governing_equations
from unified_model import optimize
from unified_model import MechanicalModel, ElectricalModel, CouplingModel, UnifiedModel
from unified_model import pipeline

from unified_model.evaluate import Measurement
from unified_model.utils.utils import collect_samples


def calc_rms(x):
    """Calculate the RMS of `x`."""
    return np.sqrt(np.sum(x ** 2) / len(x))

def calc_p_load_avg(x, r_load):
    """Calculate the average power over the load."""
    v_rms = calc_rms(x)
    return v_rms * v_rms / r_load


# PARAMETERS
c = 1
m = 1
n_z_arr = np.arange(6, 101, 4)
n_w_arr = np.arange(6, 101, 4)
c_c_arr = np.arange(20, 82, 4)
param_combinations = np.array(list(product(n_z_arr, n_w_arr, c_c_arr)))

samples = collect_samples(
    base_path='./data/2019-05-23/',
    acc_pattern='A/*acc*.csv',
    adc_pattern='A/*adc*.csv',
    video_label_pattern='A/*labels*.csv'
)

samples = samples[1:3]
assert len(samples) != 0

for n_z, n_w, c_c in tqdm(param_combinations):
    # Magnetic spring
    mag_spring = mechanical_components.MagneticSpringInterp(
        fea_data_file="./data/magnetic-spring/10x10alt.csv",
        magnet_length=10 / 1000,
        filter_callable=lambda x: savgol_filter(x, window_length=27, polyorder=5),
    )

    # Magnet assembly
    optimal_spacing = optimize.lookup_best_spacing(
        path="./data/flux_curve_model/optimal_l_ccd_0_200_2_v2.csv",
        n_z=n_z,
        n_w=n_w,
    )

    mag_assembly = mechanical_components.MagnetAssembly(
        m=m,
        l_m_mm=10.,
        l_mcd_mm=optimal_spacing if m > 1 else 0.,
        dia_magnet_mm=10.,
        dia_spacer_mm=10.
    )

    # Damper
    mech_damper = mechanical_components.MassProportionalDamper(
        damping_coefficient=4.272,
        magnet_assembly=mag_assembly
    )

    mech_spring = mechanical_components.MechanicalSpring(
        magnet_assembly=mag_assembly,
        damping_coefficient=3.108
    )

    # Coil configuration
    coil_config = electrical_components.CoilConfiguration(
        c=c,
        n_z=n_z,
        n_w=n_w,
        l_ccd_mm=optimal_spacing if c > 1 else 0.,
        ohm_per_mm=1079/1000/1000,
        tube_wall_thickness_mm=2,
        coil_wire_radius_mm=0.143 / 2,
        coil_center_mm=c_c,
        inner_tube_radius_mm=5.5
    )

    # Flux model
    flux_model = electrical_components.FluxModelPretrained(
        coil_config=coil_config,
        magnet_assembly=mag_assembly,
        curve_model_path='./data/flux_curve_model/flux_curve_model_2021_05_11.model'
    )

    mech_model = (
        MechanicalModel()
        .set_damper(mech_damper)
        .set_magnet_assembly(mag_assembly)
        .set_magnetic_spring(mag_spring)
        .set_mechanical_spring(mech_spring)
    )

    elec_model = (
        ElectricalModel()
        .set_coil_configuration(coil_config)
        .set_flux_model(flux_model.flux_model, flux_model.dflux_model)
        .set_load_model(electrical_components.SimpleLoad(R=5))
        .set_rectification_drop(0.01)
    )

    coupling_model = CouplingModel().set_coupling_constant(5.096)


    results = []
    for s in tqdm(samples):
        um = (
            UnifiedModel()
            .set_mechanical_model(mech_model)
            .set_electrical_model(elec_model)
            .set_coupling_model(coupling_model)
            .set_governing_equations(governing_equations.unified_ode)
            .set_post_processing_pipeline(pipeline.clip_x2, name="clip x2")  # Remove in future
            .set_height(105)
        )

        measurement = Measurement(s, um)
        um.mechanical_model.set_input(measurement.input_)  # type: ignore

        um.solve(
            t_start=0,
            t_end=8,
            y0=[0., 0., 0.04, 0., 0.],
            t_max_step=1e-3,
            t_eval=np.arange(0, 8, 1e-3)
        )

        result = um.calculate_metrics(
            prediction_expr='g(t,x5)',
            metric_dict={
                'p_load_avg': lambda x: calc_p_load_avg(x, um.electrical_model.load_model.R)  # type: ignore
            }
        )

        results.append(result)

    # Build out results
    output = {
        'n_z': [],
        'n_w': [],
        'c_c': [],
        'input': [],
        'p_load_avg': []
    }

    print('Writing out results...')
    for i, r in enumerate(results):
        output['n_z'].append(n_z)
        output['n_w'].append(n_w)
        output['c_c'].append(c_c)
        output['input'].append(i)
        output['p_load_avg'].append(r['p_load_avg'])

    df = pd.DataFrame(output)
    table = pa.Table.from_pandas(df)
    pq.write_to_dataset(table, f"./output/{c}c{m}m_alt.parquet")
