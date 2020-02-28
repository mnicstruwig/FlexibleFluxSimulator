import numpy as np
import pandas as pd

from config import abc_config

from unified_model.unified import UnifiedModel
from unified_model.mechanical_model import MechanicalModel
from unified_model.electrical_model import ElectricalModel
from unified_model.coupling import ConstantCoupling
from unified_model.mechanical_system.damper import ConstantDamper
from unified_model.mechanical_system.spring.mechanical_spring import MechanicalSpring
from unified_model.mechanical_system.spring.magnetic_spring import MagneticSpring
from unified_model.mechanical_system.magnet_assembly import MagnetAssembly
from unified_model.mechanical_system.input_excitation import accelerometer
from unified_model.governing_equations import unified_ode
from unified_model.pipeline import clip_x2

from unified_model.electrical_system.load import SimpleLoad

from unified_model.utils.utils import collect_samples

samples = collect_samples(
    base_path='./data/2019-05-23_B/A/',
    acc_pattern='*acc*.csv',
    adc_pattern='*adc*.csv',
    video_label_pattern='*labels*.csv'
)

accelerometer_inputs = [
    accelerometer.AccelerometerInput(
        raw_accelerometer_input=sample.acc_df,
        accel_column='z_G',
        time_column='time(ms)',
        accel_unit='g',
        time_unit='ms',
        smooth=True,
        interpolate=True
    )
    for sample in samples
]

mechanical_model = MechanicalModel(name='MechanicalModel')
mechanical_model.set_mechanical_spring(
    MechanicalSpring(
        position=110/1000,
        damper_constant=0)
)
mechanical_model.set_magnetic_spring(abc_config.spring)
mechanical_model.set_magnet_assembly(
    MagnetAssembly(
        n_magnet=1,
        h_magnet=10,
        h_spacer=0,
        dia_magnet=10,
        dia_spacer=10
    )
)
mechanical_model.set_damper(
    ConstantDamper(damping_coefficient=0.035)
)
mechanical_model.set_input(accelerometer_inputs[0])

# Electrical model
electrical_model = ElectricalModel(name='electrical model')
electrical_model.set_coil_resistance(abc_config.coil_resistance['A'])
electrical_model.set_load_model(SimpleLoad(R=30))
electrical_model.set_rectification_drop(0.10)

electrical_model.set_flux_model(
    flux_model=abc_config.flux_models['A'],
    dflux_model=abc_config.dflux_models['A']
)

# Build the unified model
unified_model = UnifiedModel(name='unified model')
unified_model.add_mechanical_model(mechanical_model)
unified_model.add_electrical_model(electrical_model)
unified_model.add_coupling_model(ConstantCoupling(c=1.0))
unified_model.add_governing_equations(unified_ode)
unified_model.add_post_processing_pipeline(clip_x2, name='clip tube velocity')

# Solve
unified_model.solve(
    t_start=0,
    t_end=8,
    y0=[0., 0., 0.04, 0., 0.],  #tube, tube_dot, mag, mag_dot, flux
    t_max_step=1e-3,
)

# Get result
result = unified_model.get_result(
    time='t',
    rel_pos='x3-x1',
    rel_vel='x4-x2',
    tube_acc='g(t, x2)',
    flux='x5',
    emf='g(t, x5)'
)
