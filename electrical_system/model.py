import numpy as np


class OpenCircuitSystem(object):
    """
    An electrical system that is not connected to a load
    """
    def __init__(self, z_index, phi_arr, flux_model, **model_kwargs):
        self.received_t = []
        self.received_z = []
        self.current_t = 0
        self.current_z = 0
        self.flux_model = flux_model(z_index, phi_arr, **model_kwargs)
        self.current_phi = self.flux_model(self.current_z)

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
