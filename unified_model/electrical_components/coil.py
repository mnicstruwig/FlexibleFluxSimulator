from typing import Union

import numpy as np


# TODO: Documentation
class CoilConfiguration:  # pylint: disable=too-many-instance-attributes
    """A linear coil model."""

    def __init__(self,  # pylint: disable=too-many-arguments
                 c: int,
                 n_z: Union[int, None],
                 n_w: Union[int, None],
                 l_ccd_mm: float,
                 ohm_per_mm: float,
                 tube_wall_thickness_mm: float,
                 coil_wire_radius_mm: float,
                 coil_center_mm: float,
                 inner_tube_radius_mm: float,
                 coil_resistance: float = None) -> None:
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

        self._validate()

        if not coil_resistance:
            self.coil_resistance = self._calculate_coil_resistance()
        else:  # Override
            self.coil_resistance = coil_resistance

    def __repr__(self):
        to_print = ', '.join([f'{k}={v}' for k, v in self.__dict__.items()])
        return f'CoilModel({to_print})'

    def _validate(self) -> None:
        """Do some basic validation of parameters."""
        if self.l_ccd_mm < 0:
            raise ValueError('l_ccd_mm must be > 0')

        if self.l_ccd_mm == 0 and self.c > 1:
            raise ValueError('l_ccd_mm = 0, but c > 1')

    def _calculate_coil_resistance(self) -> float:
        return (
            2 * np.pi
            * self.ohm_per_mm
            * self.n_w * self.n_z
            * (
                self.tube_wall_thickness_mm
                + self.inner_tube_radius_mm
                + self.coil_wire_radius_mm
                + 2 * self.coil_wire_radius_mm * (self.n_w + 1) / 2
            )
            * self.c
        )

