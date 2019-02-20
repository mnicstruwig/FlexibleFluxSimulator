import os
import numpy as np
import matplotlib.pyplot as plt

from unified_model.mechanical_system.spring.magnetic_spring import MagneticSpring
from unified_model.mechanical_system.magnet_assembly import MagnetAssembly
from unified_model.mechanical_system.damper import Damper
from unified_model.mechanical_system.input_excitation.accelerometer import AccelerometerInput
from unified_model.mechanical_model import MechanicalModel
from unified_model.electrical_model import ElectricalModel
from unified_model.electrical_system.flux.utils import FluxDatabase
from unified_model.electrical_system.load import SimpleLoad
from unified_model.coupling import ConstantCoupling
from unified_model.governing_equations import unified_ode, unified_ode_mechanical_only
from unified_model.unified import UnifiedModel

from unified_model.mechanical_system.evaluator import MechanicalSystemEvaluator
from unified_model.utils.testing.testing_electrical_model import simulate_electrical_system, _build_y_input_vector_at_timestamps


# Path handling
#file_path = os.path.abspath(__file__)
#dir_ = os.path.dirname(file_path)
#fea_data_path = os.path.abspath(os.path.join(dir_, '../unified_model/mechanical_system/spring/data/10x10alt.csv'))

fea_data_path = './unified_model/mechanical_system/spring/data/10x10alt.csv'
acc_data_path = './experiments/data/2018-12-20/A/log_17.csv'
#acc_data_path = '/home/michael/Dropbox/PhD/Python/unified_model/experiments/data/2018-10-04/log_02.csv'

spring = MagneticSpring(fea_data_file=fea_data_path,
                        model='savgol_smoothing',
                        model_type='interp')

magnet_assembly = MagnetAssembly(n_magnet=1,
                                 h_magnet=10,
                                 h_spacer=0,
                                 dia_magnet=10,
                                 dia_spacer=10,
                                 mat_magnet='NdFeB',
                                 mat_spacer='iron')

damper = Damper(model='constant', model_kwargs={'damping_coefficient': 0.03})
accelerometer = AccelerometerInput(raw_accelerometer_input=acc_data_path,
                                   accel_column='z_G',
                                   time_column='time(ms)',
                                   time_unit='ms',
                                   smooth=True,
                                   interpolate=True)

mechanical_model = MechanicalModel(name='mech_system')
mechanical_model.set_spring(spring)
mechanical_model.set_magnet_assembly(magnet_assembly)
mechanical_model.set_damper(damper)
mechanical_model.set_input(accelerometer)

flux_database = FluxDatabase(csv_database_path='/home/michael/Dropbox/PhD/Python/Research/fea-flux-curves-numr[5,15]-numz[17,33,66]-wdiam[0.15]-2018-12-07.csv',
                             fixed_velocity=0.35)

coil_center = {'A': 58.5/1000,
                'B': 61/1000}

winding_num_z = {'A': '17',
                 'B': '33'}

winding_num_r = {'A': '15',
                 'B': '15'}


which_device = 'A'

flux_model = flux_database.query_to_model(flux_model_type='unispline',
                                          coil_center=coil_center[which_device],
                                          mm=10,
                                          winding_num_z=winding_num_z[which_device],
                                          winding_num_r=winding_num_r[which_device])
load_model = SimpleLoad(np.inf)

electrical_model = ElectricalModel(name='A')
electrical_model.set_flux_model(flux_model, precompute_gradient=True)
electrical_model.set_load_model(load_model)

coupling_model = ConstantCoupling(c=0)

governing_equations = unified_ode

unified_model = UnifiedModel(name='Unified')
unified_model.add_mechanical_model(mechanical_model)
unified_model.add_electrical_model(electrical_model)
unified_model.add_coupling_model(coupling_model)
unified_model.add_governing_equations(governing_equations)

y0 = [0, 0, 0.04, 0, 0]
unified_model.solve(t_start=0, t_end=8,
                    y0=y0,
                    t_max_step=1e-3,
                    )

df = unified_model.get_result(time='t',
                              x1='x1',
                              x2='x2',
                              x3='x3',
                              x4='x4',
                              x5='x5',
                              rel_disp = 'x3-x1',
                              rel_vel = 'x4-x2')

df.plot(x='time', y='rel_disp')
plt.show()

df.plot(x='time', y='x1')
df.plot(x='time', y='x2')
df.plot(x='time', y='x3')
df.plot(x='time', y='x4')
df.plot(x='time', y='x5')
df.plot(x='time', y='rel_vel')

plt.figure()
plt.plot(np.abs(np.gradient(df['x5'].values))/np.gradient(df['time'].values), label='pred emf')
plt.legend()
plt.show()

# Debug
emf_values, timestamps = simulate_electrical_system(df['rel_disp'].values,
                                                    unified_model.t,
                                                    flux_model=unified_model.electrical_model.flux_model,
                                                    load_model=unified_model.electrical_model.load_model,
                                                    interpolate=False)


ys = _build_y_input_vector_at_timestamps(df['rel_disp'].values, unified_model.t)
emf_values = [unified_model.electrical_model.get_emf(y) for y in ys]

plt.plot(np.abs(emf_values))
plt.show()

# #emf_pred = np.gradient(df['x5'].values)/np.gradient(df['time'].values)


