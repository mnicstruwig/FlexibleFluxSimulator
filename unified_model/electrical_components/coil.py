from typing import Optional

import numpy as np
from unified_model.mechanical_components.magnet_assembly import MagnetAssembly
from unified_model.mechanical_components.magnetic_spring import MagneticSpringInterp


class CoilConfiguration:
    def __init__(
        self,  # pylint: disable=too-many-arguments
        c: int,
        n_z: Optional[int],
        n_w: Optional[int],
        l_ccd_mm: float,
        ohm_per_mm: float,
        tube_wall_thickness_mm: float,
        coil_wire_radius_mm: float,
        coil_center_mm: float,
        inner_tube_radius_mm: float,
        custom_coil_resistance: Optional[float] = None,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        c : int
            Number of coils.
        n_z : int
            Number of windings, per coil, in the axial (vertical) direction.
            Set to `None` if a flux model will be defined using measured data.
        n_w : int
            Number of windings, per coil, in the radial (horizontal) direction.
            Set to `None` if a flux model will be defined using measured data.
        l_ccd_mm : float
            Distance between consecutive coil centers. Has no effect for c == 1.
        ohm_per_mm : float
            The resistance (in Ohms) per mm of the copper wire used for the
            coil.
        tube_wall_thickness_mm : float
            The thickness of the tube wall that the coil will be wound around.
        coil_wire_radius_mm : float
            The radius of the copper wire strands used to wind the coil.
        coil_center_mm : float
            The center of the lowermost coil, in mm, *relative* to the top of
            the fixed magnet.
        inner_tube_radius_mm : float
            The inner-radius of the microgenerator tube, in mm.
        custom_coil_resistance : Optional[float]
            The resistance of the coil. This is optional, and the value is
            intended to be calculated from the other parameters. Use only when
            attempting to override the calculated values.

        """

        self.c = c
        self.n_z = n_z
        self.n_w = n_w
        # TODO: Allow l_ccd_mm to be set to `None` when c == 1.
        self.l_ccd_mm = l_ccd_mm
        self.ohm_per_mm = ohm_per_mm
        self.tube_wall_thickness_mm = tube_wall_thickness_mm
        self.coil_wire_radius_mm = coil_wire_radius_mm
        self.coil_center_mm = coil_center_mm
        self.inner_tube_radius_mm = inner_tube_radius_mm
        self.custom_coil_resistance = custom_coil_resistance

        self._validate()

    def get_coil_resistance(self):
        if not self.custom_coil_resistance:
            return self._calculate_coil_resistance()
        return self.custom_coil_resistance

    def __repr__(self):
        to_print = ", ".join([f"{k}={v}" for k, v in self.__dict__.items()])
        return f"CoilModel({to_print})"

    def _validate(self) -> None:
        """Do some basic validation of parameters."""
        if self.l_ccd_mm < 0:
            raise ValueError("l_ccd_mm must be > 0")

        if self.l_ccd_mm == 0 and self.c > 1:
            raise ValueError("l_ccd_mm = 0, but c > 1")

        if not self.custom_coil_resistance:
            try:
                assert self.n_z is not None
                assert self.n_w is not None
            except AssertionError as e:
                raise ValueError(
                    "`coil_resistance` must be specified if `n_z` and `n_w` are `None`"
                ) from e  # noqa

    def _calculate_coil_resistance(self) -> float:
        return (
            2
            * np.pi
            * self.ohm_per_mm
            * self.n_w
            * self.n_z  # type: ignore
            * (
                self.tube_wall_thickness_mm
                + self.inner_tube_radius_mm
                + self.coil_wire_radius_mm
                + 2 * self.coil_wire_radius_mm * (self.n_w + 1) / 2  # type: ignore
            )
            * self.c
        )

    def _calc_hovering_height(
        self, magnet_assembly: MagnetAssembly, magnetic_spring: MagneticSpringInterp
    ) -> float:
        hover_height_m = magnetic_spring.get_hover_height(
            magnet_assembly=magnet_assembly
        )

        return hover_height_m * 1000  # Must be in mm

    def get_height(self) -> float:
        """Get the height of the coil in metres."""
        assert self.n_z is not None
        return self.coil_wire_radius_mm * 2 * self.n_z / 1000

    def get_width(self) -> float:
        """Get the width of the coil in metres."""
        assert self.n_w is not None
        return self.coil_wire_radius_mm * 2 * self.n_w / 1000

    def set_optimal_coil_center(
        self,
        magnet_assembly: MagnetAssembly,
        magnetic_spring: MagneticSpringInterp,
        l_eps: float = 5.0,
    ) -> None:

        hover_height = self._calc_hovering_height(magnet_assembly, magnetic_spring)
        hover_height = hover_height
        coil_height = self.get_height()
        new_coil_center_mm = (
            hover_height + magnet_assembly.get_length() + l_eps + coil_height / 2
        )

        self.coil_center_mm = new_coil_center_mm

    def to_json(self):
        return {
            "c": self.c,
            "n_z": self.n_z,
            "n_w": self.n_w,
            "l_ccd_mm": self.l_ccd_mm,
            "ohm_per_mm": self.ohm_per_mm,
            "tube_wall_thickness_mm": self.tube_wall_thickness_mm,
            "coil_wire_radius_mm": self.coil_wire_radius_mm,
            "coil_center_mm": self.coil_center_mm,
            "inner_tube_radius_mm": self.inner_tube_radius_mm,
            "custom_coil_resistance": self.custom_coil_resistance,
        }

    def update(self, um):
        """Update the internal state when notified."""
