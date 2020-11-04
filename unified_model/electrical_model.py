from __future__ import annotations

import numpy as np
from unified_model.utils.utils import pretty_str
from unified_model.electrical_components.coil import CoilModel


class ElectricalModel:
    """A model of an electrical system.

    Attributes
    ----------
    flux_model : fun
        Function that returns the flux linkage of a coil when the position of a
        magnet assembly's bottom edge is passed to it.
    dflux_model : fun
        The gradient of `flux_model`.
    coil_resistance : float
        The resistance of the coil in Ohms.
        Default value is `np.inf`, which is equivalent to an open-circuit
        system.
    rectification_drop : float
        The voltage drop (from open-circuit voltage) due to rectification
        by a full-wave bridge rectifier.
    load_model : obj
        A load model.

    """

    def __init__(self):
        """Constructor."""
        self.flux_model = None
        self.dflux_model = None
        self.coil_model = None
        self.rectification_drop = None
        self.load_model = None

    def __str__(self):
        """Return string representation of the ElectricalModel"""
        return f"""Electrical Model: {pretty_str(self.__dict__, 1)}"""

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

    def set_coil_model(self, coil_model: CoilModel) -> ElectricalModel:
        """Set the coil model"""
        self.coil_model = coil_model
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
        v_load = emf*self.load_model.R / (self.load_model.R
                                          + self.coil_model.coil_resistance)

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
        r_coil = self.coil_model.coil_resistance
        # V = I/R -> I = V/R
        return emf_oc / (r_load + r_coil)
