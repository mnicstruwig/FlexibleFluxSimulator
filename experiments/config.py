"""
Constants for each microgenerator device / collection of devices.
"""
from collections import namedtuple
from unified_model.electrical_system.flux.utils import FluxDatabase
from unified_model.mechanical_system.magnet_assembly import MagnetAssembly
from unified_model.mechanical_system.spring.magnetic_spring import MagneticSpring

config_parameters = ['coil_center',
                     'coil_resistance',
                     'winding_num_z',
                     'winding_num_r',
                     'coil_height',
                     'spring',
                     'flux_database',
                     'flux_models',
                     'magnet_assembly']

Config = namedtuple('Config', config_parameters, defaults=[None]*len(config_parameters))

# ABC DEVICES
abc_coil_center = {'A': 59/1000,
                   'B': 61/1000,
                   'C': 63/1000}

abc_coil_resistance = {'A': 20,
                       'B': 40,
                       'C': 60}

abc_winding_num_z = {'A': '17',
                     'B': '33',
                     'C': '66'}

abc_winding_num_r = {'A': '15',
                     'B': '15',
                     'C': '5'}

abc_coil_height = {'A': '0.008meter',
                   'B': '0.012meter',
                   'C': '0.014meter'}

abc_magnet_assembly = MagnetAssembly(n_magnet=1,
                                     h_magnet=10,
                                     h_spacer=0,
                                     dia_magnet=10,
                                     dia_spacer=10,
                                     mat_magnet='NdFeB',
                                     mat_spacer='iron')

abc_spring_fea_data_path = './unified_model/mechanical_system/spring/data/10x10alt.csv'
abc_spring = MagneticSpring(fea_data_file=abc_spring_fea_data_path,
                            model='savgol_smoothing',
                            model_type='interp')

abc_flux_database = FluxDatabase(csv_database_path='/home/michael/Dropbox/PhD/Python/Research/fea-flux-curves-numr[5,15]-numz[17,33,66]-wdiam[0.15]-cheight[8,12,14]-2019-04-11.csv', fixed_velocity=0.35)

abc_flux_models = {}
for device in ['A', 'B', 'C']:
    abc_flux_models[device] = abc_flux_database.query_to_model(flux_model_type='unispline',
                                                               coil_center=abc_coil_center[device],
                                                               mm=10,
                                                               winding_num_z=abc_winding_num_z[device],
                                                               winding_num_r=abc_winding_num_r[device],
                                                               coil_height=abc_coil_height[device])

abc = Config(coil_center=abc_coil_center,
             coil_resistance=abc_coil_resistance,
             winding_num_z=abc_winding_num_z,
             winding_num_r=abc_winding_num_r,
             coil_height=abc_coil_height,
             spring=abc_spring,
             flux_database=abc_flux_database,
             flux_models=abc_flux_models,
             magnet_assembly=abc_magnet_assembly)
