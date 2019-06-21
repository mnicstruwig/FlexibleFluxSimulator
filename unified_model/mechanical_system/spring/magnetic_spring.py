from scipy import optimize

# Local imports
from unified_model.mechanical_system.spring.model import coulombs_law, \
    coulombs_law_modified, power_series_2, \
    power_series_3, savgol_smoothing
from unified_model.mechanical_system.spring.utils import read_raw_file, \
    get_model_function

MODEL_DICT = {
    'coulombs_unmodified': coulombs_law,
    'coulombs_modified': coulombs_law_modified,
    'power_series_2': power_series_2,
    'power_series_3': power_series_3,
    'savgol_smoothing': savgol_smoothing
}


# TODO: Complete docstrings
class MagneticSpring(object):
    """
    A magnetic spring model object

    Attributes
    ----------
    fea_dataframe : datafrane
        Dataframe containing processed FEA magnet force readings.
    model : fun
        The model to fit to the FEA magnet force readings.
    model_parameters : array_like
        The magnetic spring model parameters that are found by fitting the model
        to the FEA data file.
    model_type : string
        The type of model function that `model` is. Either "iterative" or
        "interp". If the model is "iterative", its parameters are found by
        performing a curve fit optimization. If the model is "interp" or an
        interpolation function (from the `scipy.interpolate` module),
        the model function is fit directly to the data.

    """

    def __init__(self,
                 fea_data_file,
                 model='coulombs_modified',
                 model_type='iterative'):
        """
        Constructor.

        Parameters
        ----------
        fea_data_file : str
            Path to the .CSV FEA data file to fit the model to. This first
            column of this file must contain the `z` values or displacement in
            mm between the top of the fixed magnet and the bottom of the moving
            magnet assembly. The second column must contain the force between
            the two magnets in newtons.
        model : str
            The model to use to fit the FEA data.
        model_type str
            The type of model specified in `model`.
            Either "iterative" or "interp". If the model is "iterative", its
            parameters are found by performing a curve fit optimization. If the
            model is "interp" or an interpolation function (from the
            `scipy.interpolate` module), the model function is fit directly to
            the data.
            Default is "iterative".
        """
        self.fea_dataframe = read_raw_file(fea_data_file)
        self.model = None
        self.model_parameters = None
        self.model_type = model_type
        self._set_model(model)
        self._fit_model_parameters()

    def _fit_model_parameters(self):
        """
        Fit the model using an iterative curve fit or interpolation and set the
        model's parameters.
        """
        if self.model_type is 'iterative':
            popt, _ = optimize.curve_fit(self.model,
                                         self.fea_dataframe.z.values,
                                         self.fea_dataframe.force.values)
            self.model_parameters = popt
        elif self.model_type is 'interp':
            self.model = self.model(self.fea_dataframe.z.values,
                                    self.fea_dataframe.force.values)

    def _set_model(self, model):
        """
        Set the model.
        """
        self.model = get_model_function(MODEL_DICT, model)

    def get_force(self, z):
        """
        Calculate and return the force between two magnets for a single position
        :param z: Distance between two magnets
        :return: Force between two magnets in Newtons
        """
        if self.model_type == 'interp':
            return self.model(z)
        return self.model(z, *self.model_parameters)

    def get_force_array(self, z_array):
        """
        Calculate and return the force between the two magnets
        :param z_array: An array of z values
        :return: The corresponding force between two magnets at each z value
        """
        return [self.get_force(z) for z in z_array]
