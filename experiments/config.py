"""
Constants for each microgenerator device / collection of devices.
"""
import os

from scipy.signal import savgol_filter

from collections import namedtuple
from unified_model.electrical_components.flux.utils import FluxDatabase
from unified_model.mechanical_components.magnet_assembly import MagnetAssembly
from unified_model.mechanical_components.magnetic_spring import MagneticSpringInterp
from unified_model.electrical_components.flux.model import FluxModelInterp

base_dir = os.getcwd()

config_parameters = ['coil_center',
                     'coil_resistance',
                     'winding_num_z',
                     'winding_num_r',
                     'coil_height',
                     'spring',
                     'flux_database',
                     'flux_models',
                     'dflux_models',
                     'magnet_assembly']

Config = namedtuple('Config', config_parameters, defaults=[None]*len(config_parameters))

# ABC DEVICES
abc_coil_center = {'A': 59/1000,
                   'B': 61/1000,
                   'C': 63/1000}

abc_coil_resistance = {'A': 12.5,
                       'B': 23.5,
                       'C': 17.8}

abc_winding_num_r = {'A': '15',
                     'B': '15',
                     'C': '5'}

abc_winding_num_z = {'A': '17',
                     'B': '33',
                     'C': '66'}

abc_coil_height = {'A': '0.008meter',
                   'B': '0.012meter',
                   'C': '0.014meter'}

abc_magnet_assembly = MagnetAssembly(n_magnet=1,
                                     l_m=10,
                                     l_mcd=0,
                                     dia_magnet=10,
                                     dia_spacer=10,
                                     mat_magnet='NdFeB',
                                     mat_spacer='iron')

abc_spring_fea_data_path = os.path.join(base_dir, './data/magnetic-spring/10x10alt.csv')
abc_spring = MagneticSpringInterp(fea_data_file=abc_spring_fea_data_path,
                                  filter_obj=lambda x: savgol_filter(x, 27, 5))

abc_flux_database = FluxDatabase(csv_database_path='./data/fea-flux-curves/fea-flux-curves-numr[5,15]-numz[17,33,66]-wdiam[0.15]-cheight[8,12,14]-2019-04-11.csv', fixed_velocity=0.35)

abc_flux_models = {}
abc_dflux_models = {}
for device in ['A', 'B', 'C']:
    flux_model, dflux_model = abc_flux_database.query_to_model(
        FluxModelInterp,
        {
            'c': 1,
            'm': 1,
            'c_c': abc_coil_center[device],
            'l_ccd': 0,
            'l_mcd': 0
        },
        coil_height=abc_coil_height[device],
        winding_num_z=abc_winding_num_z[device],
        winding_num_r=abc_winding_num_r[device]
    )

    abc_flux_models[device] = flux_model
    abc_dflux_models[device] = dflux_model

abc_config = Config(coil_center=abc_coil_center,
                    coil_resistance=abc_coil_resistance,
                    winding_num_z=abc_winding_num_z,
                    winding_num_r=abc_winding_num_r,
                    coil_height=abc_coil_height,
                    spring=abc_spring,
                    flux_database=abc_flux_database,
                    flux_models=abc_flux_models,
                    dflux_models=abc_dflux_models,
                    magnet_assembly=abc_magnet_assembly)
