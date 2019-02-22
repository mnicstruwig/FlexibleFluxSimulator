import numpy as np
import pandas as pd

from unified_model.coupling import ConstantCoupling
from unified_model.electrical_model import ElectricalModel
from unified_model.electrical_system.flux.utils import FluxDatabase
from unified_model.electrical_system.load import SimpleLoad
from unified_model.evaluate import (AdcProcessor, ElectricalSystemEvaluator,
                                    LabeledVideoProcessor,
                                    MechanicalSystemEvaluator)
from unified_model.governing_equations import unified_ode
from unified_model.mechanical_model import MechanicalModel
from unified_model.mechanical_system.damper import Damper
from unified_model.mechanical_system.input_excitation.accelerometer import \
    AccelerometerInput
from unified_model.mechanical_system.magnet_assembly import MagnetAssembly
from unified_model.mechanical_system.spring.magnetic_spring import \
    MagneticSpring
from unified_model.unified import UnifiedModel

# Path handling
#file_path = os.path.abspath(__file__)
#dir_ = os.path.dirname(file_path)
#fea_data_path = os.path.abspath(os.path.join(dir_, '../unified_model/mechanical_system/spring/data/10x10alt.csv'))

fea_data_path = './unified_model/mechanical_system/spring/data/10x10alt.csv'

# Accelerometer and EMF measurements
acc_emf_A_1 = './experiments/data/2018-12-20/A/log_17.csv'
acc_emf_A_2 = './experiments/data/2018-12-20/A/log_18.csv'
acc_emf_A_3 = './experiments/data/2018-12-20/A/log_19.csv'

acc_emf_B_1 = './experiments/data/2018-12-20/B/log_23.csv'


# Ground-truth mechanical observations
df_A_1 = pd.read_csv('/home/michael/Dropbox/PhD/Python/Experiments/mechanical-model/2018-12-20/A/001_transcoded_subsampled_labels_2019-02-03-15:53:43.csv')
df_A_2 = pd.read_csv('/home/michael/Dropbox/PhD/Python/Experiments/mechanical-model/2018-12-20/A/002_transcoded_subsampled_labels_2019-02-06-12:42:15.csv')
df_A_3 = pd.read_csv('/home/michael/Dropbox/PhD/Python/Experiments/mechanical-model/2018-12-20/A/003_transcoded_subsampled_labels_2019-02-07-10:46:33.csv')

df_B_1 = pd.read_csv('/home/michael/Dropbox/PhD/Python/Experiments/mechanical-model/2018-12-20/B/001_transcoded_subsampled_labels_2019-02-07-15:12:56.csv')

# SELECTION
acc_adc_data_path = acc_emf_B_1
df = df_B_1
which_device = 'B'

### MECHANICAL MODEL ###
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

damper = Damper(model='constant', model_kwargs={'damping_coefficient': 0.05})
accelerometer = AccelerometerInput(raw_accelerometer_input=acc_adc_data_path,
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

### ELECTRICAL MODEL ###
flux_database = FluxDatabase(csv_database_path='/home/michael/Dropbox/PhD/Python/Research/fea-flux-curves-numr[5,15]-numz[17,33,66]-wdiam[0.15]-2018-12-07.csv',
                             fixed_velocity=0.35)

coil_center = {'A': 58.5/1000,
                'B': 61/1000}

winding_num_z = {'A': '17',
                 'B': '33'}

winding_num_r = {'A': '15',
                 'B': '15'}



flux_model = flux_database.query_to_model(flux_model_type='unispline',
                                          coil_center=coil_center[which_device],
                                          mm=10,
                                          winding_num_z=winding_num_z[which_device],
                                          winding_num_r=winding_num_r[which_device])
load_model = SimpleLoad(np.inf)

electrical_model = ElectricalModel(name='A')
electrical_model.set_flux_model(flux_model, precompute_gradient=True)
electrical_model.set_load_model(load_model)

### COUPLING MODEL ###
coupling_model = ConstantCoupling(c=0)

### SYSTEM MODEL ###
governing_equations = unified_ode

### UNIFIED MODEL ###
unified_model = UnifiedModel(name='Unified')
unified_model.add_mechanical_model(mechanical_model)
unified_model.add_electrical_model(electrical_model)
unified_model.add_coupling_model(coupling_model)
unified_model.add_governing_equations(governing_equations)

y0 = [0, 0, 0.04, 0, 0]
unified_model.solve(t_start=7, t_end=15,
                    y0=y0,
                    t_max_step=1e-3,
                    )

### POST-PROCESSING ###
df_result = unified_model.get_result(time='t',
                                     x1='x1',
                                     x2='x2',
                                     x3='x3',
                                     x4='x4',
                                     x5='x5',
                                     rel_disp = 'x3-x1',
                                     rel_vel = 'x4-x2',
                                     my_result='x1+x2+x3'
)

df_result['emf'] = np.gradient(df_result['x5'].values)/np.gradient(df_result['time'].values)


### COMPARISON ###

pixel_scale = 0.2666
voltage_div_ratio = 1/0.342

lp = LabeledVideoProcessor(L=125, mm=10, seconds_per_frame=3 / 240)
adc = AdcProcessor(voltage_div_ratio, smooth=True)

# Target values
y_target, time_target = lp.fit_transform(df, True, pixel_scale)
emf_target, emf_time_target = adc.fit_transform(acc_adc_data_path)

# Predicted values
y_predicted = df_result['rel_disp'].values
emf_predicted = df_result['emf']
time_predicted = df_result['time'].values

# Evaluate mechanical system
mech_eval = MechanicalSystemEvaluator(y_target,
                                      time_target
)
mech_eval.fit(y_predicted, time_predicted)
mech_eval.poof()

# Evaluate electrical system 

ec_eval = ElectricalSystemEvaluator(emf_target, emf_time_target)
ec_eval.fit(emf_predicted, time_predicted)
ec_eval.poof()
