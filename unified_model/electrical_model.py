from __future__ import annotations

import warnings

import numpy as np

from unified_model.utils.utils import pretty_str
from unified_model.electrical_components.coil import CoilConfiguration
from unified_model.local_exceptions import ModelError


class ElectricalModel:
    """A model of an electrical system.

    Attributes
    ----------
    flux_model : Callable
        Function that returns the flux linkage of a coil when the position of a
        magnet assembly's bottom edge is passed to it.
    dflux_model : Callable
        The gradient of `flux_model`.
    rectification_drop : float
        The voltage drop (from open-circuit voltage) due to rectification
        by a full-wave bridge rectifier.
    load_model : Object
        A load model object.

    """

    def __init__(self):
        """Constructor."""
        self.flux_model = None
        self.dflux_model = None
        self.coil_config = None
        self.rectification_drop = None
        self.load_model = None

    def __str__(self):
        """Return string representation of the ElectricalModel"""
        return f"""Electrical Model: {pretty_str(self.__dict__, 1)}"""

    def _validate(self):
        """Validate the electrical model.

        Do some basic checks to make sure mandatory components have been set.
        """
        try:
            assert self.flux_model is not None
            assert self.dflux_model is not None
        except AssertionError as e:
            raise ModelError('A flux model and dflux model must be specified.') from e  # noqa
        try:
            assert self.coil_config is not None
        except AssertionError as e:
            raise ModelError('A coil model must be specified.') from e
        try:
            assert self.rectification_drop is not None
        except AssertionError:
            warnings.warn('Rectification drop not specified. Assuming no loss due to rectification.')  # noqa
        try:
            assert self.load_model is not None
        except AssertionError as e:
            raise ModelError('A load model must be specified.') from e

    def set_flux_model(self, flux_model, dflux_model):
        """Assign a flux model.

        Parameters
        ----------
        flux_model : function
            Function that returns the flux linkage of a coil when the position
            of a magnet assembly's bottom edge is passed to it.
        dflux_model : function
            Function that returns the derivative of the flux linkage of a coil
            (relative to `z` i.e. the position of a magnet assembly's bottom
            edge) when the position of a magnet assembly's bottom edge is passed to it.

        """
        self.flux_model = flux_model
        self.dflux_model = dflux_model
        return self

    def set_coil_configuration(self, coil_config: CoilConfiguration
                               ) -> ElectricalModel:
        """Set the coil model"""
        self.coil_config = coil_config
        return self

    def set_rectification_drop(self, v: float) -> ElectricalModel:
        """Set the open-circuit voltage drop due to rectification."""
        self.rectification_drop = v
        return self

    def set_load_model(self, load_model):
        """Assign a load model

        Parameters
        ----------
        load_model : SimpleLoad
            The load model to set.

        """
        self.load_model = load_model
        return self

    def get_load_voltage(self, mag_pos, mag_vel):
        emf = self.get_emf(mag_pos, mag_vel)
        v_load = emf * self.load_model.R / (self.load_model.R
                                            + self.coil_config.coil_resistance)

        return v_load

    def get_emf(self, mag_pos, mag_vel):
        """Return the instantaneous emf produced by the electrical system.

        Note, this is the open-circuit emf and *not* the emf supplied to
        the load.

        Parameters
        ----------
        mag_pos : float
            The position of the center of the first (bottom) magnet in the
            magnet assembly. In metres.
        mag_vel : float
            The velocity of the magnet assembly. In metres per second.

        Returns
        -------
        float
            The instantaneous emf. In volts.

        """
        dphi_dz = self.dflux_model.get(mag_pos)
        emf = dphi_dz * (mag_vel)

        if self.rectification_drop:
            emf = np.abs(emf)
            if emf > self.rectification_drop:
                emf = emf - self.rectification_drop
            else:
                emf = 0

        return emf

    def get_current(self, emf_oc):
        """Return the instantaneous current produced by the electrical system.

        Takes into account the resistance of the coils, as well as the load
        resistance.

        Parameters
        ----------
        emf_oc : float
            The instantaneous open-circuit emf induced in the coil(S). Can be
            calculated by using the `get_emf` method.

        Returns
        -------
        float
            The instantaneous current flowing through the electrical system.

        """
        if not self.load_model:
            return 0

        r_load = self.load_model.R
        r_coil = self.coil_config.coil_resistance
        # V = I/R -> I = V/R
        return emf_oc / (r_load + r_coil)
