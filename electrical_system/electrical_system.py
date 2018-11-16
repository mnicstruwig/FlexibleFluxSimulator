import numpy as np
from unified_model.electrical_system.flux.model import flux_interpolate
from unified_model.utils.utils import fetch_key_from_dictionary

# TODO: Come up with something better (eg. "basic")
FLUX_MODEL_DICT = {
    'flux_interpolate': flux_interpolate
}

def _gradient(f, x, delta_x=1e-3):
    """Compute the gradient of function `f` at point `y` relative to `dx`"""

    gradient = (f(x + delta_x) - f(x - delta_x))/(2*delta_x)
    return gradient

# TODO: Add tests
class OpenCircuitSystem(object):
    """
    An electrical system that is not connected to a load
    """
    def __init__(self, z_index, phi_arr, flux_model, **model_kwargs):
        self.received_t = []
        self.received_z = []
        self.current_t = 0
        self.current_z = 0
        flux_model = fetch_key_from_dictionary(FLUX_MODEL_DICT, flux_model, "Flux model not found.")
        self.flux_model = flux_model(z_index, phi_arr, **model_kwargs)
        self.current_phi = self.flux_model(self.current_z)


    def reset(self):
        self.received_z = []
        self.received_t = []


    def get_emf(self, next_t, next_z):
        self.received_t.append(next_t)
        self.received_z.append(next_z)
        next_phi = self.flux_model(next_z)

        delta_phi = self.current_phi - next_phi
        delta_t = next_t - self.current_t
        delta_z = next_z - self.current_z

        dphi_dz = delta_phi / delta_z
        dz_dt = delta_z / delta_t

        emf = dphi_dz * dz_dt

        self.current_t = next_t
        self.current_z = next_z
        self.current_phi = next_phi

        return emf

    def get_emf_arr(self, t_arr, z_arr):
        phi_at_z = self.flux_model(z_arr)
        dphi_dz = np.gradient(phi_at_z, z_arr)
        dz_dt = np.gradient(z_arr, t_arr)

        emf = dphi_dz * dz_dt
        return emf

    def get_processed_flux_curve(self, z_arr):
        """
        Return the processed flux curve using the flux curve model for magnet
        assembly positions of `z_arr`.
        """
        return self.flux_model(z_arr)

