from experiments.config import abc

from unified_model.unified import UnifiedModel
from unified_model.mechanical_model import MechanicalModel
from unified_model.mechanical_system.damper import DamperConstant
from unified_model.mechanical_system.input_excitation.accelerometer import AccelerometerInput
from unified_model.utils.utils import collect_samples


base_groundtruth_path = './experiments/data/2019-05-23/'
a_samples = collect_samples(base_path=base_groundtruth_path,
                            acc_pattern='A/*acc*.csv',
                            adc_pattern='A/*adc*.csv',
                            labeled_video_pattern='A/*labels*.csv')

mechanical_model = MechanicalModel(name=None)
mechanical_model.set_spring(abc.spring)
mechanical_model.set_magnet_assembly(abc.magnet_assembly)
mechanical_model.set_damper(DamperConstant(damping_coefficient=0.05))  # Tweaking will need to happen


accelerometer_inputs = [AccelerometerInput(raw_accelerometer_input=sample.acc_df,
                                           accel_column='z_G',
                                           time_column='time(ms)',
                                           accel_unit='g',
                                           time_unit='ms',
                                           smooth=True,
                                           interpolate=True)
                        for sample
                        in a_samples]

mechanical_model.set_input(accelerometer_inputs[0])  # Choose which input to system
