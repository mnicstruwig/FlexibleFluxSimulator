import numpy as np
from unified_model.electrical_system.flux.model import flux_interpolate
from unified_model.utils.utils import fetch_key_from_dictionary

def _gradient(f, x, delta_x=1e-3):
    """Compute the gradient of function `f` at point `y` relative to `x`"""
    gradient = (f(x + delta_x) - f(x - delta_x))/(2*delta_x)
    if np.isinf(gradient):
       return 0.0
    return gradient


# TODO: Add tests + documentation
class ElectricalSystem:
    """A generic electrical system."""

    def __init__(self, flux_model, load_model, **kwargs):
        self.flux_model = flux_model
        self.load_model = load_model
        self.precompute_gradient = kwargs.pop('precompute_gradient', False)

        if self.precompute_gradient is True:
            self.flux_gradient = self.flux_model.derivative()

    def get_flux_gradient(self, y):
        """Get the gradient of the flux relative to z."""
        x1, x2, x3, x4, x5 = y
        if self.precompute_gradient is True:
            return self.flux_gradient(x3-x1)
        return _gradient(self.flux_model, x3-x1)

    def get_emf(self, y):
        x1, x2, x3, x4, x5 = y
        dphi_dz = self.get_flux_gradient(y)
        emf = dphi_dz * (x4 - x2)
        return emf

    def get_current(self, y):
        x1, x2, x3, x4, x5 = y
        dphi_dz = self.get_flux_gradient(y)
        emf = dphi_dz * (x4-x2)
        return self.load_model.get_current(emf)

