import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from flux_modeller.model import CurveModel
from scipy.interpolate import UnivariateSpline, interp1d
from ..mechanical_components.magnet_assembly import MagnetAssembly
from ..utils.utils import FastInterpolator, grad
from ..electrical_components.coil import CoilConfiguration


class FluxModelPretrained:
    """A flux model that uses a trained CurveModel."""

    def __init__(
        self,
        coil_configuration: CoilConfiguration,
        magnet_assembly: MagnetAssembly,
        curve_model_path: str,
    ) -> None:
        """Constructor."""

        self._set_up(
            coil_configuration=coil_configuration,
            magnet_assembly=magnet_assembly,
            curve_model_path=curve_model_path,
        )

    def __repr__(self):
        to_print = ", ".join([f"{k}={v}" for k, v in self.__dict__.items()])
        return f"FluxModelPretrained({to_print})"

    def update(self, um):
        """Update and the internal state when notified."""
        self._set_up(
            coil_configuration=um.coil_configuration,
            magnet_assembly=um.magnet_assembly,
            curve_model_path=self.curve_model_path,
        )

    def _set_up(self, coil_configuration, magnet_assembly, curve_model_path):

        self.curve_model_path = curve_model_path

        # Load the CurveModel and predict the flux values
        self.curve_model = CurveModel.load(curve_model_path)
        z, phi = self.curve_model.predict_curves(
            np.array([[coil_configuration.n_z, coil_configuration.n_w]])
        )
        phi = phi.flatten()

        # Create the actual flux model
        self.flux_model_interp = FluxModelInterp(
            coil_configuration=coil_configuration, magnet_assembly=magnet_assembly
        )

        self.flux_model_interp.fit(z, phi)

        self.flux_model = self.flux_model_interp.flux_model
        self.dflux_model = self.flux_model_interp.dflux_model

    def get_flux(self, z):
        """Get the flux at relative magnet position `z` (in metres)."""
        return self.flux_model.get(z)

    def get_dflux(self, z):
        """Get the flux derivative at relative magnet position `z` (in metres)."""
        return self.dflux_model.get(z)

    def to_json(self):
        return {
            "coil_configuration": "dep:coil_configuration",
            "magnet_assembly": "dep:magnet_assembly",
            "curve_model_path": self.curve_model_path,
        }


class FluxModelInterp:
    """A flux model that uses interpolation."""

    def __init__(
        self,
        coil_configuration: CoilConfiguration,
        magnet_assembly: MagnetAssembly,
    ) -> None:
        """A flux model that relies on interpolation.

        Parameters
        ----------
        coil_configuration: CoilConfiguration
            The coil configuration to use when creating the interpolated flux
            model.
        magnet_assembly: MagnetAssembly
            The magnet assembly model to use when creating the interpolated flux
            model.

        """
        self.c = None
        self.c_c = None
        self.l_ccd = None
        self.m = None
        self.l_mcd = None
        self.flux_model = None
        self.dflux_model = None

        # Do actual setting up
        self._set_up(
            coil_configuration=coil_configuration, magnet_assembly=magnet_assembly
        )

        self._validate()

    # keep as separate func to make `.update` method to be DRY
    def _set_up(
        self, coil_configuration: CoilConfiguration, magnet_assembly: MagnetAssembly
    ) -> None:

        self.c = coil_configuration.c

        # `flux_inteprolate` requires measurements in SI units.
        self.c_c = coil_configuration.coil_center_mm / 1000

        # `flux_inteprolate` requires measurements in SI units.
        self.l_ccd = coil_configuration.l_ccd_mm / 1000

        self.m = magnet_assembly.m
        # `flux_inteprolate` requires measurements in SI units.
        self.l_mcd = magnet_assembly.l_mcd_mm / 1000

    def update(self, um):
        """Update the internal state when notified."""
        self._set_up(
            coil_configuration=um.coil_configuration, magnet_assembly=um.magnet_assembly
        )

        self._validate()

    def _validate(self):
        """Do some internal validation of the parameters"""
        # TODO: Figure out a better way of warning the user that doesn't spam
        # gridsearch.

        _validate_coil_params(c=self.c, m=self.m, l_ccd=self.l_ccd, l_mcd=self.l_mcd)

    def __repr__(self):
        to_print = ", ".join([f"{k}={v}" for k, v in self.__dict__.items()])
        return f"FluxModelInterp({to_print})"

    def fit(self, z_arr, phi_arr):
        """Fit the interpolated model to data.

        Note that `z_arr` and `phi_arr` must correspond to a single magnet
        moving through a single coil. The superposition flux curve will be
        constructed using this primitive for the case when c > 1 and/or m > 1.

        Also, note that the values of `z_arr` should correspond to the relative
        position between the *centers* of the magnet and coil.

        Parameters
        ----------
        z_arr : array-like
            The z-values of the magnet position's center relative to the coil
            position center.  In metres.
        phi_arr : array-like
            The corresponding flux values at `z_arr`.

        """

        self.flux_model, self.dflux_model = _make_superposition_curve(
            z_arr=z_arr,
            phi_arr=phi_arr,
            c=self.c,
            m=self.m,
            l_ccd=self.l_ccd,
            l_mcd=self.l_mcd,
            c_c=self.c_c,
        )

    def get_flux(self, z):
        """Get the flux at relative magnet position `z` (in metres)."""
        return self.flux_model.get(z)

    def get_dflux(self, z):
        """Get the flux derivative at relative magnet pos `z` (in metres)."""
        return self.dflux_model.get(z)

    def to_json(self):
        raise NotImplementedError(
            "FluxModelInterp is not currently supported for converting to a configuration. Use `FluxModelPretrained` instead."
        )


# TODO: Add docs
def _make_superposition_curve(
    z_arr: np.ndarray,
    phi_arr: np.ndarray,
    c: int,
    m: int,
    l_ccd: float,
    l_mcd: float,
    c_c: float,
) -> Tuple[FastInterpolator, FastInterpolator]:
    """Make the superposition flux curve."""
    if c == 1 and m == 1:  # Simplest case
        return interpolate_flux(z_arr, phi_arr, coil_center=c_c)

    flux_interp_list = []
    dflux_interp_list = []
    for i in range(c):  # For each coil
        for j in range(m):  # For each magnet
            # Generate a interpolator for each individual flux curve
            flux_interp, dflux_interp = interpolate_flux(
                z_arr=z_arr,
                phi_arr=(-1) ** (i + j)
                * phi_arr,  # noqa.  Remembering to alternate the polarity...
                coil_center=c_c
                - j * l_mcd
                + i * l_ccd,  # noqa ... and shift the center (peak)
            )
            flux_interp_list.append(flux_interp)
            dflux_interp_list.append(dflux_interp)

    # Scale the z range to compensate for the number of coils and magnets
    # TODO: Add a resolution argument for finer sampling?
    z_arr_width = max(z_arr) - min(z_arr)
    new_z_start = c_c - z_arr_width / 2
    new_z_end = c_c + c * l_ccd + z_arr_width / 2

    new_z_arr = np.linspace(new_z_start, new_z_end, len(z_arr) * (c + m))

    # Sum across each interpolator to build the superposition flux curve
    phi_super = []
    dphi_super = []
    for z in new_z_arr:
        phi_separate = [flux_interp.get(z) for flux_interp in flux_interp_list]
        dphi_separate = [dflux_interp.get(z) for dflux_interp in dflux_interp_list]
        phi = sum(phi_separate)
        dphi = sum(dphi_separate)
        phi_super.append(phi)
        dphi_super.append(dphi)

    # Now, generate a new interpolator with the superposition curve
    phi_super_interpolator = FastInterpolator(new_z_arr, phi_super)
    dphi_super_interpolator = FastInterpolator(new_z_arr, dphi_super)

    return phi_super_interpolator, dphi_super_interpolator


def _validate_coil_params(c, m, l_ccd, l_mcd):
    """Validate the coil parameters for correctness."""
    if l_ccd < 0:
        raise ValueError("l_ccd must be > 0")
    if l_mcd < 0:
        raise ValueError("l_mcd must be > 0")
    if c > 1 and m > 1:
        if l_ccd != l_mcd:
            warnings.warn("l_ccd != l_mcd, this is unusual.", RuntimeWarning)  # noqa

    if l_ccd == 0 and c > 1:
        raise ValueError("l_ccd = 0, but c > 1")

    if l_mcd == 0 and m > 1:
        raise ValueError("l_mcd = 0, but m > 1")


def _find_min_max_arg_gradient(arr):
    """Find the arguments that give the min/max gradient of `arr`."""
    grad_arr = np.gradient(arr)  # type:ignore
    return np.argmin(grad_arr), np.argmax(grad_arr)


def interpolate_flux(z_arr, phi_arr, coil_center):
    """Model the flux by interpolating between values

    Parameters
    ----------
    z_arr : array-like
        The z position of the *center* of the uppermost magnet in the magnet
        assembly. Must be the same length as `phi_arr`. In metres.
    phi_arr : array-like
        The flux captured by the coil, corresponding with the magnet assembly
        position given by `z_arr`. Must be the same length as `z_arr`.
    coil_center : float
        The position of the center of the coil of the microgenerator, relative
        to the *top* of the fixed magnet. In metres.
    custom_interpolator : Class
        A custom interpolator class

    Returns
    -------
    interpolator object
        The interpolator that can be called with `z` values to return the flux
        linkage.

    """
    z_arr = z_arr
    phi_arr = phi_arr
    phi_interpolator = interp1d(
        z_arr, phi_arr, "cubic", bounds_error=False, fill_value=0
    )

    z_arr_fine = np.linspace(z_arr.min(), z_arr.max(), 10 * len(z_arr))
    # Get smooth fit before shifting
    new_phi_arr = np.array([phi_interpolator(z) for z in z_arr_fine])  # type: ignore

    # Find the value of z when the flux linkage is maximum (i.e. center of
    # magnet in center of coil)
    peak_idx = np.argmax(np.abs(new_phi_arr))
    z_when_phi_peak = z_arr_fine[peak_idx]
    # Shift the array so that the maximum flux linkage occurs at the specified
    # coil center
    z_arr_fine = z_arr_fine - (z_when_phi_peak - coil_center)

    # Reinterpolate with new z values to update our flux linkage model
    phi_interpolator = interp1d(
        z_arr_fine, new_phi_arr, bounds_error=False, fill_value=0
    )
    fast_phi_interpolator = FastInterpolator(z_arr_fine, new_phi_arr)

    # Get an interpolator for the gradient
    # We ignore start/end values of z to prevent gradient anomalies
    dphi_dz = np.array(
        [grad(phi_interpolator, z) for z in z_arr_fine[1:-1]], dtype=np.float64
    )
    fast_dphi_interpolator = FastInterpolator(z_arr_fine[1:-1], dphi_dz)

    return fast_phi_interpolator, fast_dphi_interpolator


def flux_univariate_spline(z_arr, phi_arr, coil_center, mm):
    """Model the flux curve by interpolating between values using univariate spline.

    This flux model supports pre-computation of the gradient.

    Parameters
    ----------
    z_arr : array
        The z position of the *bottom* of the magnet assembly that corresponds
        with `phi_arr`. Must be the same length as `phi_arr`. In metres.
    phi_arr : array
        The flux captured by the coil, corresponding with the magnet assembly
        position given by `z_arr`. Must be the same length as `z_arr`.
    coil_center : float
        The position (in metres) of the center of the coil of the microgenerator,
        relative to the *top* of the fixed magnet.
        TODO: Potentially change this to something more natural (eg. relative to the
        bottom of the actual device.)
    mm : float
        The total height of the magnet assembly (in m).
        TODO: Consider a better unit

    Returns
    -------
    `UnivariateSpline` object
        The interpolator that can be called with `z` values to return the flux linkage.

    """
    warnings.warn("Univariate spline as flux model is deprecated for the time being!")
    magnet_assembly_center = mm / 2
    z_arr = (
        z_arr - z_arr[np.abs(phi_arr).argmax()] + coil_center - magnet_assembly_center
    )
    interpolator = UnivariateSpline(z_arr, np.abs(phi_arr), k=3, s=0, ext="zeros")
    return interpolator




def _parse_raw_flux_input(raw_flux_input):
    """Parse a raw flux input.

    Supports either a pandas dataframe, or a .csv file.

    Parameters
    ----------
    raw_flux_input : string or pandas dataframe
        A string referencing the .csv file to load or a pandas dataframe that
        will be loaded directly.

    Returns
    -------
    pandas dataframe
        Pandas dataframe containing the raw flux input.

    """
    if isinstance(raw_flux_input, str):
        return pd.read_csv(raw_flux_input)
    if isinstance(raw_flux_input, pd.DataFrame):
        return raw_flux_input


class FluxDatabase:
    """Convert .csv produced by Maxwell parametric simulation into a flux database.

    Attributes
    ----------
    raw_database : pandas dataframe
        Pandas dataframe containing the raw loaded .csv as exported by ANSYS
        Maxwell
    velocity : float
        Velocity in m/s of the moving magnet as specified in the ANSYS Maxwell
        simulation.
    lut : dict
        Lookup table used internally to map parameters to keys. Should not be
        accessed or modified by the user.
    database : dict
        Internal database that, in conjunction with `lut` is used to store the
        flux data.

    """

    def __init__(self, csv_database_path: str, fixed_velocity: float) -> None:
        """Initialize the FluxDatabase

        Parameters
        ----------
        csv_database_path : str
            Path to the raw .csv file exported from ANSYS Maxwell.
        fixed_velocity : float
            Velocity in m/s of the moving magnet as specified in the ANSYS
            Maxwell simulation.

        """

        self.raw_database = pd.read_csv(csv_database_path)
        self.velocity = fixed_velocity
        self.lut = None
        self.database = {}
        self._produce_database()

    @staticmethod
    def _extract_parameter_from_str(str_):
        """Extract parameters from a string generated by ANSYS Maxwell

        Parameters
        ----------
        str_ : str
            String generated by ANSYS Maxwell when exporting a parametric
            simulation. Typically found as a field heading in the
            exported .csv file.

        Returns
        -------
        dict
            Dictionary containing the parameter name and parameter values as
            key-value pairs.

        """
        split_ = str_.split()
        unprocessed_params = [s for s in split_ if "=" in s]
        param_names = [param.split("=")[0] for param in unprocessed_params]
        param_values = [
            param.split("=")[1].replace("'", "") for param in unprocessed_params
        ]

        param_dict = {}
        for name, value in zip(param_names, param_values):
            param_dict[name] = value
        return param_dict

    def _produce_database(self):
        """Build the flux database."""

        # TODO: Put in separate preprocessing function
        self.time = self.raw_database.iloc[:, 0].values / 1000
        self.z = self.time * self.velocity

        self._create_index(
            self._extract_parameter_from_str(self.raw_database.columns[1]).keys()
        )

        # Add flux curves to database
        for col in self.raw_database.columns[1:]:  # First column is time info
            key_dict = self._extract_parameter_from_str(col)
            self.add(key_dict, value=self.raw_database[col].values)

    def _make_db_key(self, **kwargs):
        """Build a database key using the internal look-up table."""
        db_key = [None] * len(self.lut)
        for key in kwargs:
            db_key[self.lut[key]] = kwargs[key]
        if None in db_key:
            raise KeyError("Not all keys specified")
        return tuple(db_key)

    def add(self, key_dict, value):
        """Add an entry to the database.

        This method should preferably not be used to build the original
        database, and should instead be used to add once-off or the odd
        additional sample to a database that has been built from a .csv file
        exported by ANSYS Maxwell.

        Parameters
        ----------
        key_dict : dict
            Key-value pairs to be used as the lookup for `value`. All keys
            must be used when performing lookup, i.e. if multiple keys are
            specified in `key_dict`, multiple keys must be used for the lookup
            of `value`.
        value :
            Any data structure to store in the database. Accessible using
            the `query` method.

        Returns
        -------
        None

        Example
        -------
        >>> key_dict = {'param_1' : param_1_value, 'param_2': param_2_value)
        >>> value = np.ones(3)
        >>> my_flux_database.add(key_dict, value)
        >>> my_flux_database.query(param_1=param_1_value, param_2=param_2_value)
        array([1, 1, 1])

        """
        db_key = self._make_db_key(**key_dict)
        self.database[db_key] = value

    # TODO: Update docs
    def query_to_model(self, model_cls, model_kwargs, **kwargs):
        """Query the database and return a flux model.

        This is intended to be a convenience function. It works identically
        to the `query` method, but returns a flux model instead of the actual
        flux curve.

        Parameters
        ----------
        model_cls : cls
            The model class.
        model_kwargs: dict
            The kwargs to pass to the constructor of the model class.
        **kwargs
            Keyword argument passed to the `query` method.

        Returns
        -------
        flux_model: obj
            A flux model object that can be called with `z` values to
            return the flux linkage at that position z.

        See Also
        --------
        self.query : underlying method.

        """
        phi = self.query(**kwargs)
        model = model_cls(**model_kwargs)
        model.fit(self.z, phi)
        return model

    def query(self, **kwargs):
        """Query the database

        Parameters
        ----------
        **kwargs
            kwargs to use in order to query the database. The key-value pairs
            must correspond to the `key_dict` used when adding the item to the
            database

        Returns
        -------
        value
            The data structure stored under the key-value pairs.

        Example
        -------
        >>> key_dict = {'param_1' : param_1_value, 'param_2': param_2_value)
        >>> value = np.ones(3)  # The value we want to lookup
        >>> my_flux_database.add(key_dict, value)
        >>> my_flux_database.query(param_1=param_1_value, param_2=param_2_value)
        array([1, 1, 1])  # returns `value`

        """
        db_key = self._make_db_key(**kwargs)
        return self.database[db_key]

    def _create_index(self, key_list):
        """
        Create the key look-up table using a list of keys.

        Parameters
        ----------
        key_list : (str, ) array_like
            Keys to use to build the index.

        """
        if self.lut is None:
            self.lut = {}
            for i, k in enumerate(key_list):
                self.lut[k] = i
        else:
            raise ValueError("Index cannot be created more than once.")

    def itervalues(self):
        """Iterate through the values in the database."""
        for key, value in self.database.items():
            yield key, value
