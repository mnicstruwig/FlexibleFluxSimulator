import numpy as np
from scipy.interpolate import interp1d

def flux_interpolate(z_arr, phi_arr, coil_center, mf):
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
        TODO: Potentially change this to something more natural (eg. relative to the
        bottom of the actual device.)
    mf : float
        The total height of the magnet assembly (in mm).
        TODO: Consider a better unit

    """
    magnet_assembly_center = mf/2
    z_arr = z_arr - z_arr[np.abs(phi_arr).argmax()] + coil_center - magnet_assembly_center/1000
    interpolator = interp1d(z_arr, np.abs(phi_arr), fill_value=0, bounds_error=False)
    return interpolator
