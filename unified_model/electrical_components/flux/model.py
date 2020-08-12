from numba import jitclass, int32, float64
import numpy as np
from scipy.interpolate import interp1d, UnivariateSpline
import warnings
from functools import reduce
from unified_model.utils.utils import grad


# TODO: Document this class
class FluxModelInterp:
    def __init__(self, c, m, c_c, l_ccd=0, l_mcd=0):
        self.c = c
        self.m = m
        self.c_c = c_c
        self.l_ccd = l_ccd
        self.l_mcd = l_mcd
        self.flux_model = None
        self.dflux_model = None

        self._validate()

    def _validate(self):
        """Do some internal validation of the parameters"""
        if self.l_ccd < 0:
            raise ValueError('l_ccd must be > 0')
        if self.l_mcd < 0:
            raise ValueError('l_mcd must be > 0')

        if self.l_ccd != self.l_mcd and (self.l_ccd != 0 or self.l_mcd != 0) and m != 1:
            warnings.warn('l_ccd != l_mcd, this is unusual.', RuntimeWarning)

        if self.l_ccd == 0 and self.c > 1:
            raise ValueError('l_ccd = 0, but c > 1')

        if self.l_mcd == 0 and self.m > 1:
            raise ValueError('l_mcd = 0, but m > 1')

    def __repr__(self):
        to_print = ', '.join([f'{k}={v}' for k, v in self.__dict__.items()])
        return f'FluxModelInterp({to_print})'

    def fit(self, z_arr, phi_arr):
        self.flux_model, self.dflux_model = self._make_superposition_curve(
            z_arr,
            phi_arr
        )

    def _make_superposition_curve(self, z_arr, phi_arr):
        """Make the superposition flux curve"""

        if self.c == 1 and self.m == 1:  # Simplest case
            return flux_interpolate(z_arr, phi_arr, coil_center=self.c_c)

        flux_interp_list = []
        dflux_interp_list = []
        for i in range(self.c):  # For each coil
            for j in range(self.m):  # For each magnet
                # Generate the interpolator for the individualized curve
                flux_interp, dflux_interp = flux_interpolate(
                    z_arr,
                    (-1) ** (i+j) * phi_arr,  # Remember to alternate the polarity
                    coil_center=self.c_c + j * self.l_mcd + i * self.l_ccd  # Shift the center
                )
                flux_interp_list.append(flux_interp)
                dflux_interp_list.append(dflux_interp)

        # Scale the z range to compensate for the number of coils and magnets
        # TODO: Add a resolution arg for finer sampling?
        z_arr_width = max(z_arr) - min(z_arr)
        new_z_start = self.c_c - z_arr_width/2
        new_z_end = (self.c_c
                     + self.c * self.l_ccd
                     + self.m * self.l_mcd
                     + z_arr_width/2)

        new_z_arr = np.linspace(new_z_start,
                                new_z_end,
                                len(z_arr)*(self.c + self.m))
        phi_super = []
        for z in new_z_arr:
            phi = sum(flux_interp(z) for flux_interp in flux_interp_list)
            phi_super.append(phi)

        # Now, generate a new interpolator with the superposition curve
        # TODO: Consider turning this into a helper
        phi_super_interpolator = interp1d(new_z_arr,
                                          phi_super,
                                          kind='cubic',
                                          bounds_error=False,
                                          fill_value=0)

        # Do the same for the gradient
        dphi_dz_super = [grad(phi_super_interpolator, z) for z in new_z_arr]
        dphi_super_interpolator = interp1d(new_z_arr,
                                           dphi_dz_super,
                                           kind='cubic',
                                           bounds_error=False,
                                           fill_value=0)

        return phi_super_interpolator, dphi_super_interpolator

    def flux(self, z):
        return self.flux_model.get(z)

    def dflux(self, z):
        return self.dflux_model.get(z)


def _find_min_max_arg_gradient(arr):
    """Find the arguments that give the min/max gradient of `arr`."""
    grad_arr = np.gradient(arr)
    return np.argmin(grad_arr), np.argmax(grad_arr)


# TODO: Docs
@jitclass([('x', float64[:]), ('y', float64[:]), ('length', int32)])  # noaq
class FastFluxInterpolator:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.length = len(x)

    def get(self, x):
        return np.interp(x, self.x, self.y)


def flux_interpolate(z_arr, phi_arr, coil_center):
    """Model the flux by interpolating between values

    Parameters
    ----------
    z_arr : array
        The z position of the *center* of the uppermost magnet in the magnet
        assembly. Must be the same length as `phi_arr`. In metres.
    phi_arr : array
        The flux captured by the coil, corresponding with the magnet assembly
        position given by `z_arr`. Must be the same length as `z_arr`.
    coil_center : float
        The position (in metres) of the center of the coil of the
        microgenerator, relative to the *top* of the fixed magnet.
    custom_interpolator : class
        A custom interpolator class

    Returns
    -------
    `interp1d` object
        The interpolator that can be called with `z` values to return the flux
        linkage.

    """
    z_arr = z_arr
    phi_arr = phi_arr
    phi_interpolator = interp1d(z_arr, phi_arr, 'cubic', bounds_error=False, fill_value=0)

    z_arr_fine = np.linspace(z_arr.min(), z_arr.max(), 10*len(z_arr))
    new_phi_arr = np.array([phi_interpolator(z) for z in z_arr_fine])  # Get smooth fit before shifting

    # Find the value of z when the flux linkage is maximum (i.e. center of
    # magnet in center of coil)
    operator = np.argmax if max(phi_arr) > 0 else np.argmin
    z_when_phi_peak = z_arr_fine[operator(new_phi_arr)]
    # Shift the array so that the maximum flux linkage occurs at the specified
    # coil center
    z_arr_fine = z_arr_fine - (z_when_phi_peak - coil_center)
    # Reinterpolate with new z values to update our flux linkage model
    phi_interpolator = interp1d(z_arr_fine, new_phi_arr, bounds_error=False, fill_value=0)
    fast_phi_interpolator = FastFluxInterpolator(z_arr_fine, new_phi_arr)
    # Get an interpolator for the gradient
    # We ignore start/end values of z to prevent gradient errors
    dphi_dz = np.array([grad(phi_interpolator, z) for z in z_arr_fine[1:-1]], dtype=np.float64)
    fast_dphi_interpolator = FastFluxInterpolator(z_arr_fine[1:-1], dphi_dz)

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
    warnings.warn('Univariate spline as flux model is deprecated for the time being!')
    magnet_assembly_center = mm/2
    z_arr = z_arr - z_arr[np.abs(phi_arr).argmax()] + coil_center - magnet_assembly_center
    interpolator = UnivariateSpline(z_arr, np.abs(phi_arr), k=3, s=0, ext='zeros')
    return interpolator
