"""
Constants for each microgenerator device / collection of devices.
"""
import os
from dataclasses import dataclass
from typing import Any, Dict

from scipy.signal import savgol_filter

from collections import namedtuple
from unified_model.electrical_components.flux.utils import FluxDatabase
from unified_model.mechanical_components.magnet_assembly import MagnetAssembly
from unified_model.mechanical_components.magnetic_spring import MagneticSpringInterp
from unified_model.electrical_components.flux.model import FluxModelInterp
from unified_model.electrical_components.coil import CoilModel

base_dir = os.getcwd()


# ABC DEVICES
abc_coil_center = {'A': 59,
                   'B': 61,
                   'C': 63}

# Is now calculated by the coil model, but we override it to make real-world measurements
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

abc_magnet_assembly = MagnetAssembly(m=1,
                                     l_m_mm=10,
                                     l_mcd_mm=0,
                                     dia_magnet_mm=10,
                                     dia_spacer_mm=10,
                                     mat_magnet='NdFeB',
                                     mat_spacer='iron')

abc_spring_fea_data_path = os.path.join(base_dir, '../data/magnetic-spring/10x10alt.csv')
abc_spring = MagneticSpringInterp(fea_data_file=abc_spring_fea_data_path,
                                  magnet_length=10/1000,
                                  filter_callable=lambda x: savgol_filter(x, 27, 5))  # noqa

abc_flux_database = FluxDatabase(csv_database_path='../flux_modeller/data/fea-flux-curves-numr[5,15]-numz[17,33,66]-wdiam[0.15]-cheight[8,12,14]-2019-04-11.csv', fixed_velocity=0.35)

abc_flux_models = {}
abc_dflux_models = {}
abc_coil_models = {}
for device in ['A', 'B', 'C']:
    coil_model = CoilModel(
        c=1,
        n_z=int(abc_winding_num_z[device]),
        n_w=int(abc_winding_num_r[device]),
        l_ccd_mm=0,
        ohm_per_mm=1361/1000/1000,
        tube_wall_thickness_mm=2,
        coil_wire_radius_mm=0.143/2,
        coil_center_mm=abc_coil_center[device],
        outer_tube_radius_mm=5.5,
        coil_resistance=abc_coil_resistance[device]
    )
    flux_model, dflux_model = abc_flux_database.query_to_model(
        FluxModelInterp,
        {
            'coil_model': coil_model,
            'magnet_assembly': abc_magnet_assembly
        },
        coil_height=abc_coil_height[device],
        winding_num_z=abc_winding_num_z[device],
        winding_num_r=abc_winding_num_r[device]
    )

    abc_flux_models[device] = flux_model
    abc_dflux_models[device] = dflux_model
    abc_coil_models[device] = coil_model


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

@dataclass
class Config:
    """A configuration class"""
    coil_models: Dict
    winding_num_z: Dict
    winding_num_r: Dict
    coil_height: Dict
    spring: Any
    flux_database: Any
    flux_models: Dict
    dflux_models: Dict
    magnet_assembly: Any


ABC_CONFIG = Config(coil_models=abc_coil_models,
                    winding_num_z=abc_winding_num_z,
                    winding_num_r=abc_winding_num_r,
                    coil_height=abc_coil_height,
                    spring=abc_spring,
                    flux_database=abc_flux_database,
                    flux_models=abc_flux_models,
                    dflux_models=abc_dflux_models,
                    magnet_assembly=abc_magnet_assembly)
