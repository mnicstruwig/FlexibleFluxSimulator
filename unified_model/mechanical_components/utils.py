import numpy as np
import pandas as pd
from unified_model.utils.utils import fetch_key_from_dictionary


def calc_contact_surface_area_cylinder(diameter, height):
    """
    Calculate the outer surface area of a cylinder, excluding the top caps
    :param diameter: Diameter of the cylinder
    :param height: Height of the cylinder
    :return: The outer surface area of the cylinder
    """

    return np.pi * diameter * height


def calc_volume_cylinder(diameter, height):
    """
    Calculate the volume of a cylinder
    :param diameter: The cylinder diameter
    :param height: The cylinder height
    :return: The volume of the cylinder
    """
    radius = diameter / 2
    return np.pi * radius * radius * height


def read_raw_file(file_name, z_unit='mm'):
    """
    Reads in the raw FEA data from CSV where the first column is the distance between
    the two magnets and the second coloumn is the force.
    :return: The dataframe containing the FEA data
    """
    df = pd.read_csv(file_name, header=None)
    df.columns = ['z', 'force']

    if z_unit is 'mm':
        df['z'] = df['z']/1000

    return df


def get_model_function(model_dict, model):
    """
    Fetches the model for the magnetic spring
    :model_dict: Dictionary containing model definitions
    :model: The key for the model to load from `model_dict`
    :return: The magnetic spring model
    """
    return fetch_key_from_dictionary(model_dict, model, 'The model is not defined!')
