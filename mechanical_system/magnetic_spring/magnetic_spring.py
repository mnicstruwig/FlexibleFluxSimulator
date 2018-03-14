from scipy import optimize

# Local imports
from mechanical_system.magnetic_spring.model import coulombs_law, coulombs_law_modified
from mechanical_system.magnetic_spring.utils import read_raw_file, get_model_function

MODEL_DICT = {
    'coulombs_unmodified': coulombs_law,
    'coulombs_modified': coulombs_law_modified
}


class MagneticSpring(object):
    """
    The magnetic spring object
    """

    def __init__(self, fea_data_file, model='coulombs_modified'):
        self.fea_dataframe = read_raw_file(fea_data_file)
        self.model_parameters = None
        self._set_model(model)
        self._fit_model_parameters()

    def _fit_model_parameters(self):
        """
        Fits the model using a curve fit and sets the model's parameters
        """
        popt, _ = optimize.curve_fit(self.model, self.fea_dataframe.z.values, self.fea_dataframe.force.values)
        self.model_parameters = popt

    def _set_model(self, model):
        """
        Sets the model function from a pre-defined dictionary.
        """
        self.model = get_model_function(MODEL_DICT, model)

    def get_force(self, z):
        """
        Calculate and return the force between two magnets for a single position
        :param z: Distance between two magnets
        :return: Force between two magnets in Newtons
        """
        return self.model(z, *self.model_parameters)

    def get_force_array(self, z_array):
        """
        Calculate and return the force between the two magnets
        :param z_array: An array of z values
        :return: The corresponding force between two magnets at each z value
        """
        return [self.model(z, *self.model_parameters) for z in z_array]
