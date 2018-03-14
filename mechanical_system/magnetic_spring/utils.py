import pandas as pd
from utils.utils import fetch_key_from_dictionary


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
