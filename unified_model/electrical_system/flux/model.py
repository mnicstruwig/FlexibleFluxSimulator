import numpy as np
from scipy.interpolate import interp1d, UnivariateSpline
import warnings
from unified_model.utils.utils import grad


def _find_min_max_arg_gradient(arr):
    """Find the arguments that give the min/max gradient of `arr`."""
    grad_arr = np.gradient(arr)
    return np.argmin(grad_arr), np.argmax(grad_arr)


def flux_interpolate(z_arr, phi_arr, coil_center, mm):
    """Model the flux by interpolating between values

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
    mm : float
        The total height of the magnet assembly (in m).

    Returns
    -------
    `interp1d` object
        The interpolator that can be called with `z` values to return the flux linkage.

    """
    phi_interpolator = interp1d(z_arr,
                                np.abs(phi_arr),
                                kind='cubic',
                                bounds_error=False,
                                fill_value=0)

    z_arr_fine = np.linspace(z_arr.min(), z_arr.max(), 5*len(z_arr))
    new_phi_arr = phi_interpolator(z_arr_fine)  # Get smooth fit before shifting

    z_max = z_arr_fine[np.argmax(new_phi_arr)]

    # dphi/dz = 0 happens when center of magnet passes through center of coil
    z_arr_fine = z_arr_fine - (z_max - coil_center) - mm/2
    # Reinterpolate with new z values
    phi_interpolator = interp1d(z_arr_fine,
                                new_phi_arr,
                                kind='cubic',
                                bounds_error=False,
                                fill_value=0)

    # Get an interpolator for the gradient
    # We ignore start/end values of z to prevent gradient errors
    dphi_dz = [grad(phi_interpolator, z) for z in z_arr_fine[1:-1]]
    dphi_interpolator = interp1d(z_arr_fine[1:-1],
                                 dphi_dz,
                                 kind='cubic',
                                 bounds_error=False,
                                 fill_value=0)

    return phi_interpolator, dphi_interpolator


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
