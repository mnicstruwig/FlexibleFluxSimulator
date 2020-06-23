from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy import optimize
from scipy import interpolate
from scipy.signal import savgol_filter


def _model_savgol_smoothing(z_arr, force_arr):
    """
    Apply a savgol filter_callable and interpolate the result
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


def _preprocess(dataframe: pd.DataFrame,
                filter_callable: Optional[Callable]) -> pd.DataFrame:
    """Filter the `force` values in `dataframe` using `filter_callable`."""
    if filter_callable:
        dataframe['force'] = filter_callable(dataframe['force'].values)
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
    magnet_height : float
        The height of the magnet in *metres*. This is used to offset the model
        so that the distance between the magnets is relative to the center of
        the moving magnet.
    filter_callable : Callable
        A filter_callable to smooth the data in the data file. Optional.
    **model_kwargs :
        Keyword arguments passed to `scipy.interpolate.interp1d`.

    Attributes
    ----------
    fea_dataframe : dataframe
        Pandas dataframe containing the processed FEA magnet force readings.

    """

    def __init__(self,
                 fea_data_file: str,
                 magnet_height: float,
                 filter_callable: Callable = None,
                 **model_kwargs) -> None:
        """Constructor."""
        self.fea_data_file = fea_data_file
        self.filter_callable = filter_callable
        self.magnet_height = magnet_height

        self.fea_dataframe = _preprocess(pd.read_csv(fea_data_file),
                                         filter_callable)
        self._model = self._fit_model(self.fea_dataframe,
                                      self.magnet_height,
                                      **model_kwargs)

    def __repr__(self):
        return f'MagneticSpringInterp({self.fea_data_file}, {self.filter_callable})'  # noqa

    @staticmethod
    def _fit_model(fea_dataframe: pd.DataFrame,
                   magnet_height: float,
                   **model_kwargs) -> Callable:
        """Fit the 1d interpolation model."""
        # Set a few defaults
        model_kwargs.setdefault('fill_value', 0)
        model_kwargs.setdefault('bounds_error', False)

        return interpolate.interp1d(fea_dataframe.z.values + magnet_height,
                                    fea_dataframe.force.values,
                                    **model_kwargs)

    def get_force(self, z: float) -> float:
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
    model : func
        A function representing the magnetic spring model. Must be compatible
        with `scipy.optimize.curve_fit`.
    filter_callable : Callable
        A filter_callable to smooth the data in the data file. Optional.
    Attributes
    ----------
    fea_dataframe : dataframe
        Pandas dataframe containing the processed FEA magnet force readings.

    """
    def __init__(self,
                 fea_data_file: str,
                 model: Callable,
                 filter_callable: Callable = None) -> None:
        """Constructor"""
        self.fea_dataframe = _preprocess(pd.read_csv(fea_data_file),
                                         filter_callable)
        self._model_params = self._fit_model(model)

    def _fit_model(self, model: Callable) -> np.ndarray:
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
