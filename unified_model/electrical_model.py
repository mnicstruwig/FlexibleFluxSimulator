import numpy as np

# TODO: Move to utils
def _gradient(f, x, delta_x=1e-3):
    """Compute the gradient of function `f` at point `y` relative to `x`"""
    gradient = (f(x + delta_x) - f(x - delta_x))/(2*delta_x)
    if np.isinf(gradient):
        return 0.0
    return gradient


class ElectricalModel:
    """A model of an electrical system.

    Attributes
    ----------
    name : str
        String identifier of the electrical model.
    flux_model : fun
        Function that returns the flux linkage of a coil when the position of a
        magnet assembly's bottom edge is passed to it.
    load_model : obj
        A load model.
    flux_gradient : fun
        The gradient of `flux_model` if the `precompute_gradient` argument is
        set to True when using the `set_flux_model` method. Otherwise, None.

    """

    def __init__(self, name):
        """Constructor

        Parameters
        ----------
        name : str
            String identifier of the electrical model.

        """
        self.name = name
        self.flux_model = None
        self.load_model = None
        self.flux_gradient = None
        self.precompute_gradient = False

    def set_flux_model(self, flux_model, precompute_gradient=False):
        """Assign a flux model.

        Parameters
        ----------
        flux_model : fun
            Function that returns the flux linkage of a coil when the position
            of a magnet assembly's bottom edge is passed to it.
        precompute_gradient : bool
            Precompute the gradient of `flux_model`, if supported (such as in
            the case of a `scipy.interpolate` object`).
            Default value is False.

        """
        self.flux_model = flux_model

        if precompute_gradient:
            self.flux_gradient = self.flux_model.derivative()
            self.precompute_gradient = True

    def set_load_model(self, load_model):
        """Assign a load model

        Parameters
        ----------
        load_model : obj
            The load model to set.

        """
        self.load_model = load_model

    def get_flux_gradient(self, y):
        """Return the instantaneous gradient of the flux relative to z.

        Parameters
        ----------
        y : ndarray
            The `y` input vector that is supplied to the set of governing
            equations, with shape (n,), where `n` is the number of equations
            in the set of governing equations.

        Returns
        -------
        ndarray
            The instantaneous flux gradient.

        """
        x1, x2, x3, x4, x5 = y  # TODO: Remove reliance on hard-coding.
        if self.precompute_gradient is True:
            return self.flux_gradient(x3-x1)
        return _gradient(self.flux_model, x3-x1)

    def get_emf(self, y):
        """Return the instantaneous emf produced by the electrical system.

        Note, this is the open-circuit emf and *not* the emf supplied to
        the load.

        Parameters
        ----------
        y : ndarray
            The `y` input vector that is supplied to the set of governing
            equations, with shape (n,), where `n` is the number of equations
            in the set of governing equations.

        Returns
        -------
        ndarray
            The instantaneous emf.

        """
        x1, x2, x3, x4, x5 = y  # TODO: Remove reliance on hard-coding
        dphi_dz = self.get_flux_gradient(y)
        emf = dphi_dz * (x4 - x2)
        return emf

    def get_current(self, y):
        """Return the instantaneous current produced by the electrical system.

        Parameters
        ----------
        y : ndarray
            The `y` input vector that is supplied to the set of governing
            equations, with shape (n,), where `n` is the number of equations
            in the set of governing equations.

        Returns
        -------
        ndarray
            The instantaneous current.

        """
        x1, x2, x3, x4, x5 = y
        dphi_dz = self.get_flux_gradient(y)
        emf = dphi_dz * (x4-x2)
        return self.load_model.get_current(emf)
