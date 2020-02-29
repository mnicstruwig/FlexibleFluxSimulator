import numpy as np
import pandas as pd
from scipy import optimize
from scipy import interpolate
from scipy.signal import savgol_filter

# Local imports
from unified_model.mechanical_components.spring.utils import read_raw_file, \
    get_model_function


def _model_savgol_smoothing(z_arr, force_arr):
    """
    Apply a savgol filter and interpolate the result
    """
    filtered_force_arr = savgol_filter(force_arr, 27, 5)
    interp = interpolate.interp1d(z_arr, filtered_force_arr, fill_value=0,
                                  bounds_error=False)
    return interp


def _model_coulombs_law(z, m):
    """A magnetic spring model that implements a variant of Coulomb's Law.
    """
    u0 = 4 * np.pi * 10 ** (-7)
    return u0 * m * m / (4 * np.pi * z * z)


def _model_coulombs_law_modified(z, A, X, G):
    """A modified magnet spring model modeled after Coulomb's Law."""
    return X / (A * z * z + G)


def _model_power_series_2(z, a0, a1, a2):
    return a0 + a1 * z + a2 * z * z


def _model_power_series_3(z, a0, a1, a2, a3):
    part0 = a0
    part1 = a1 * z
    part2 = a2 * z ** 2
    part3 = a3 * z ** 3
    return part0 + part1 + part2 + part3


def _preprocess(dataframe, filter_obj):
    """Filter the `force` values in `dataframe` using `filter_obj`."""
    if filter_obj:
        dataframe['force'] = filter_obj(dataframe['force'].values)
        return dataframe
    return dataframe


class MagneticSpringInterp:
    """
    A magnetic spring model that uses interpolation.

    This means that the model is not an explicit mathematical model -- it only
    receives datapoints, and interpolates between those points.

    Parameters
    ----------
    fea_data_file : str
        Path to the FEA magnet force readings file. Position values must be in
        a column with name 'z' (with unit metres) and force values must be in a
        column with name 'force' (with unit Newtons).
    filter : obj
        A filter to smooth the data in the data file. Optional.
    **model_kwargs :
        Keyword arguments passed to `scipy.interpolate.interp1d`.

    Attributes
    ----------
    fea_dataframe : dataframe
        Pandas dataframe containing the processed FEA magnet force readings.

    """

    def __init__(self, fea_data_file, filter_obj=None, **model_kwargs):
        """Constructor."""
        self.fea_data_file = fea_data_file
        self.fea_dataframe = _preprocess(pd.read_csv(fea_data_file),
                                         filter_obj)
        self._model = self._fit_model(**model_kwargs)

    def _fit_model(self, **model_kwargs):
        """Fit the 1d interpolation model."""
        # Set a few defaults
        model_kwargs.setdefault('fill_value', 0)
        model_kwargs.setdefault('bounds_error', False)

        return interpolate.interp1d(self.fea_dataframe.z.values,
                                    self.fea_dataframe.force.values,
                                    **model_kwargs)

    def get_force(self, z):
        """Calculate the force between two magnets at a distance `z` apart.

        Parameters
        ----------
        z : float
            The distance between the two magnets (in metres).

        Returns
        -------
        float
            The force in Newtons.

        """
        return self._model(z)


class MagnetSpringAnalytic:
    """
    A magnet spring model that uses an analytic model and a fitting process.

    Parameters
    ----------
    fea_data_file : str
        Path to the FEA magnet force readings file. Position values must be in
        a column with name 'z' (with unit metres) and force values must be in a
        column with name 'force' (with unit Newtons).
    filter : obj
        A filter to smooth the data in the data file. Optional.
    model : func
        A function representing the magnetic spring model. Must be compatible
        with `scipy.optimize.curve_fit`.

    Attributes
    ----------
    fea_dataframe : dataframe
        Pandas dataframe containing the processed FEA magnet force readings.

    """
    def __init__(self, fea_data_file, model, filter_obj=None):
        """Constructor"""
        self.fea_dataframe = _preprocess(pd.read_csv(fea_data_file),
                                         filter_obj)
        self._model_params = self._fit_model(model)

    def _fit_model(self, model):
        """Find the best-fit parameters for `model`"""
        popt, _ = optimize.curve_fit(model,
                                     self.fea_dataframe.z.values,
                                     self.fea_dataframe.force.values)
        return popt

    def get_force(self, z):
        """Calculate the force between two magnets at a distance `z` apart.

        Parameters
        ----------
        z : float
            The distance between the two magnets (in metres).

        Returns
        -------
        float
            The force in Newtons.

        """
        return self._model(z, *self._model_params)

