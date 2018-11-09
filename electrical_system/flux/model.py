import numpy as np
from scipy.interpolate import interp1d

def flux_interpolate(z_arr, phi_arr, coil_center):
    """Model the flux by interpolating between values"""
    z_arr = z_arr - z_arr[np.abs(phi_arr).argmax()] + coil_center
    interpolator = interp1d(z_arr, phi_arr, fill_value=0, bounds_error=False)
    return interpolator
