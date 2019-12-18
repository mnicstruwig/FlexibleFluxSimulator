import numpy as np
import pandas as pd

from config import abc

from unified_model.unified import UnifiedModel
from unified_model.mechanical_model import MechanicalModel
from unified_model.electrical_model import ElectricalModel

from unified_model.coupling import ConstantCoupling

from unified_model.mechanical_system.damper import ConstantDamper
from unified_model.mechanical_system.spring.mechanical_spring import MechanicalSpring
from unified_model.mechanical_system.spring.magnetic_spring import MagneticSpring
from unified_model.mechanical_system.input_excitation.accelerometer import AccelerometerInput
from unified_model.governing_equations import unified_ode

from unified_model.utils.utils import collect_samples


mechanical_model = MechanicalModel(name='Mechanical Model')
mechanical_model.set_mechanical_spring(
    MechanicalSpring(
        position=110/1000,
        damper_constant=0)
)
mechanical_model.set_magnetic_spring(
    MagneticSpring(abc.spring)
)

samples = collect_samples(
    base_path='./data/2019-05-23_B/A/',
    acc_pattern='*acc*.csv',
    adc_pattern='*adc*.csv',
    video_label_pattern='*labels*.csv'
)

