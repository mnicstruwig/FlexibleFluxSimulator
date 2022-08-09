from itertools import product
from typing import Any

import numpy as np
from ffs import batch
from ffs.coupling import CouplingModel
from ffs.electrical_components.coil import CoilConfiguration
from ffs.electrical_components.flux import FluxModelPretrained
from ffs.electrical_components.load import SimpleLoad
from ffs.governing_equations import unified_ode
from ffs.mechanical_components.damper import MassProportionalDamper
from ffs.mechanical_components.magnet_assembly import MagnetAssembly
from ffs.mechanical_components.magnetic_spring import \
    MagneticSpringInterp
from ffs.mechanical_components.mechanical_spring import \
    MechanicalSpring
from ffs.unified import UnifiedModel
from ffs.utils.utils import collect_samples
from ffs import optimize

# Parameters
c = 2
m = 2
n_z_arr = np.arange(6, 101, 10)
n_w_arr = np.arange(6, 101, 10)
c_c_arr = np.arange(20, 82, 10)

# Convert this into our parameter sets
gridsearch_params: Any = []
for n_z, n_w, c_c in product(n_z_arr, n_w_arr, c_c_arr):
    param = [
        ('magnet_assembly.m', m),
        ('coil_configuration.c', c),
        ('coil_configuration.n_z', n_z),
        ('coil_configuration.n_w', n_w),
        ('coil_configuration.c_c', c_c),
    ]

    # We calculate the optimal coil / magnet spacing using experimental data
    # directly (in th case of c > 1 and/or m > 1)
    spacing = None
    if c > 1 or m > 1:
        spacing = optimize.lookup_best_spacing(
            '../data/flux_curve_model/optimal_l_ccd_0_200_2_v2.csv',
            n_z=n_z,
            n_w=n_w
        )
        if c > 1:
            param.append(('coil_configuration.l_ccd_mm', spacing))
        if m > 1:
            param.append(('magnet_assembly.l_mcd_mm', spacing))

    gridsearch_params.append(param)

print(len(gridsearch_params))

# Let's define our model
magnet_assembly = MagnetAssembly(
    m=1,  # Mutable
    l_m_mm=10,  # Mutable
    l_mcd_mm=0,
    dia_magnet_mm=10,
    dia_spacer_mm=10
)

mech_damper = MassProportionalDamper(
    damping_coefficient=4.272,
    magnet_assembly=magnet_assembly
)

coupling_model = CouplingModel(
    coupling_constant=5.096
)

mech_spring = MechanicalSpring(
    magnet_assembly=magnet_assembly,
    damping_coefficient=3.108
)

magnetic_spring = MagneticSpringInterp(
    fea_data_file='../data/magnetic-spring/10x10alt.csv',
    magnet_assembly=magnet_assembly
)

coil_config = CoilConfiguration(
    c=1,  # Mutable
    n_z=1,  # Mutable
    n_w=1,  # Mutable
    l_ccd_mm=0,  # Mutable
    ohm_per_mm=1079 / 1000 / 1000,
    tube_wall_thickness_mm=2,
    coil_wire_radius_mm=0.0715,
    coil_center_mm=1,  # Mutable
    inner_tube_radius_mm=5.5,
)

flux_model = FluxModelPretrained(
    coil_configuration=coil_config,
    magnet_assembly=magnet_assembly,
    curve_model_path='../data/flux_curve_model/flux_curve_model_2021_05_11.model'
)

load = SimpleLoad(R=5)


base_model = (
    UnifiedModel()
    .with_height(105 / 1000)
    .with_magnet_assembly(magnet_assembly)
    .with_coil_configuration(coil_config)
    .with_magnetic_spring(magnetic_spring)
    .with_mechanical_damper(mech_damper)
    .with_coupling_model(coupling_model)
    .with_mechanical_spring(mech_spring)
    .with_rectification_drop(0.01)
    .with_flux_model(flux_model)
    .with_load_model(load)
    .with_governing_equations(unified_ode)
)

samples = collect_samples(
    base_path='../data/2019-05-23/A/',
    acc_pattern='*acc*.csv',
    adc_pattern='*adc*.csv',
    video_label_pattern='*labels*.csv'
)

def calc_rms(x):
    """Calculate the RMS of `x`."""
    return np.sqrt(np.sum(x ** 2) / len(x))


def calc_p_load_avg(x, r_load):
    """Calculate the average power over the load."""
    v_rms = calc_rms(x)
    return v_rms * v_rms / r_load


# Run the gridsearch
batch.solve_for_batch(
    base_model_config=base_model.get_config(),
    params=gridsearch_params,
    samples=samples[:1],
    prediction_expr='g(t, x5)',
    prediction_metrics={'p_load_avg': lambda x: calc_p_load_avg(x, load.R)},
    output_root_dir='.'
)
