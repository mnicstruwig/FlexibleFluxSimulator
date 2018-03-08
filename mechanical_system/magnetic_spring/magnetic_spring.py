import pandas as pd
from scipy import optimize

# Local imports
from mechanical_system.magnetic_spring.model import coulombs_law_model, coulombs_law_modified_model

MODEL_DICT = {
    'coulombs_unmodified': coulombs_law_model,
    'coulombs_modified': coulombs_law_modified_model
}


class MagneticSpring(object):
    """
    This is the magnetic spring object
    """

    def __init__(self, fea_data_file, model='coulombs_modified'):
        self.fea_dataframe = self._read_raw_file(fea_data_file)
        self.model = self._set_model(model)
        self.model_parameters = self._fit_model()

    def _read_raw_file(self, file_name):
        """
        Reads in the raw FEA data from CSV where the first column is the distance between
        the two magnets and the second coloumn is the force.
        :return: The dataframe containing the FEA data
        """
        df = pd.read_csv(file_name, header=None)
        df.columns = ['z', 'force']
        return df

    def _set_model(self, model):
        """
        Sets the model for the magnetic spring
        :return: The magnetic spring model
        """
        try:
            return MODEL_DICT[model]
        except KeyError:
            print('That model is not defined')

    def _fit_model(self):
        """
        Fits the model using a curve fit and returns the model's parameters
        :return: The model's parameters as an array
        """
        popt, _ = optimize.curve_fit(self.model, self.fea_dataframe.z.values, self.fea_dataframe.force.values)
        return popt

    def get_force(self, z_array):
        """
        Calculate and return the force between the two magnets
        :param z_array: An array of z values
        :return: The corresponding force between two magnets at each z value
        """
        return [self.model(z, *self.model_parameters) for z in z_array]
