"""
Try use `nevergrad` to find the parameters for the model for a single device /
input excitation
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# Local imports
from unified_model import UnifiedModel
from unified_model import MechanicalModel
from unified_model import ElectricalModel
from unified_model import CouplingModel
from unified_model import evaluate
from unified_model import mechanical_components
from unified_model import electrical_components
from unified_model import governing_equations
from unified_model import pipeline

from local_config import ABC_CONFIG

# Parameters
c = 1  # number of coils
m = 1  # number of magnets
l_m = 10  # length of the magnets in mm
dia_m = 10  # diameter of the magnets in mm
l_ccd = 0  # distance between centers of coils in mm
l_mcd = 0  # distance between centers of magnets in mm
c_c = 0.059  # location of center of first coil in m

# Components
input_excitation = mechanical_components.AccelerometerInput(
    raw_accelerometer_input='./data/2019-05-23_C/A/log_03_acc.csv',  # Choose the accelerometer file we want to use
    accel_column='z_G',  
    time_column='time(ms)',
    accel_unit='g',
    time_unit='ms',
    smooth=True,
    interpolate=True
)


def smoothing_filter(x_arr):
    return savgol_filter(x_arr, window_length=27, polyorder=5)

magnetic_spring = mechanical_components.MagneticSpringInterp(
    fea_data_file='./data/magnetic-spring/10x10alt.csv',
    magnet_length=l_m/1000,
    filter_callable=smoothing_filter
)

mag_assembly = mechanical_components.MagnetAssembly(
    n_magnet=m,
    l_m=l_m,
    l_mcd=0,
    dia_magnet=dia_m,
    dia_spacer=dia_m,
)

R_coil = 12.5
load = electrical_components.SimpleLoad(R=20)
v_rect_drop = 0.1

flux_model = electrical_components.FluxModelInterp(
    c=c,
    m=m,
    c_c=c_c,
    l_ccd=l_ccd,
    l_mcd=l_mcd
)

# Function to generate our parameters that can vary
def make_damper(c):
    return mechanical_components.ConstantDamper(c)


def make_mech_spring(c):
    return mechanical_components.MechanicalSpring(
        position=110/1000,
        magnet_length=l_m/1000,
        damping_coefficient=c
    )


def make_coupling(c):
    return CouplingModel().set_coupling_constant(1.)


def build_model(fric_damp, mech_spring_damp, coupling_damp):
    mechanical_model = (
        MechanicalModel()
        .set_damper(make_damper(fric_damp))
        .set_magnet_assembly(mag_assembly)
        .set_magnetic_spring(magnetic_spring)
        .set_input(input_excitation)
    )

    electrical_model = (
        ElectricalModel()
        .set_coil_resistance(R_coil)
        .set_rectification_drop(v_rect_drop)
        .set_load_model(load)
        .set_flux_model(ABC_CONFIG.flux_models['A'], ABC_CONFIG.dflux_models['A'])
    )

    unified_model = (
        UnifiedModel()
        .set_mechanical_model(mechanical_model)
        .set_electrical_model(electrical_model)
        .set_coupling_model(make_coupling(coupling_damp))
        .set_post_processing_pipeline(pipeline.clip_x2, name='')
        .set_governing_equations(governing_equations.unified_ode)
    )

    return unified_model


def run_simulation(fric_damp, mech_spring_damp, coupling_damp):
    print('Building model...')
    um = build_model(fric_damp, mech_spring_damp, coupling_damp)

    y0 = [
        0.0,  # x1 at t=0 -> tube displacement in m
        0.0,  # x2 at t=0 -> tube velocity in m/s
        0.05, # x3 at t=0 -> magnet displacement in m
        0.0,  # x4 at t=0 -> magnet velocity in m/s
        0.0   # x5 at t=0 -> flux linkage in 
    ]

    print('Running simulation...')
    um.solve(  # This can take a little while...
        t_start=0,
        t_end=7,
        y0=y0,  # Initial conditions we defined above
        t_eval=np.linspace(0, 7, 1000),  # Don't make this too big (low accuracy) or small (long run time)
    )

    print('Generating results...')
    result = um.get_result(
        time='t',  # time,
        rel_pos_mag='x3-x1',  # Relative position of magnet in m
        rel_vel_mag='x4-x2',  # Relative velocity of magnet in m
        flux='x5',  # Flux linkage
        emf='g(t, x5)'  # EMF, which is the gradient, with respect to time, of the flux. In volts.
    )

    return result
