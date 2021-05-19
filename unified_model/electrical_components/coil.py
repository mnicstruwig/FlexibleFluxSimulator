from typing import Optional

import numpy as np
from unified_model.mechanical_components.magnet_assembly import MagnetAssembly
from unified_model.mechanical_components.magnetic_spring import \
    MagneticSpringInterp


class CoilConfiguration:  # pylint: disable=too-many-instance-attributes
    """A linear coil model configuration"""

    def __init__(self,  # pylint: disable=too-many-arguments
                 c: int,
                 n_z: Optional[int],
                 n_w: Optional[int],
                 l_ccd_mm: float,
                 ohm_per_mm: float,
                 tube_wall_thickness_mm: float,
                 coil_wire_radius_mm: float,
                 coil_center_mm: float,
                 inner_tube_radius_mm: float,
                 coil_resistance: Optional[float] = None) -> None:
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
        coil_resistance : Optional[float]
            The resistance of the coil. This is optional, and the value is
            intended to be calculated from the other parameters. Use only when
            attempting to override the calculate values.

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

        if not coil_resistance:
            self.coil_resistance = self._calculate_coil_resistance()
        else:  # Override
            self.coil_resistance = coil_resistance
        self._validate()

    def __repr__(self):
        to_print = ', '.join([f'{k}={v}' for k, v in self.__dict__.items()])
        return f'CoilModel({to_print})'

    def _validate(self) -> None:
        """Do some basic validation of parameters."""
        if self.l_ccd_mm < 0:
            raise ValueError('l_ccd_mm must be > 0')

        if self.l_ccd_mm == 0 and self.c > 1:
            raise ValueError('l_ccd_mm = 0, but c > 1')

        if not self.coil_resistance:
            try:
                assert self.n_z is not None
                assert self.n_w is not None
            except AssertionError as e:
                raise ValueError('`coil_resistance` must be specified if `n_z` and `n_w` are `None`') from e # noqa

    def _calculate_coil_resistance(self) -> float:
        return (
            2 * np.pi
            * self.ohm_per_mm
            * self.n_w * self.n_z  # type: ignore
            * (
                self.tube_wall_thickness_mm
                + self.inner_tube_radius_mm
                + self.coil_wire_radius_mm
                + 2 * self.coil_wire_radius_mm
                * (self.n_w + 1) / 2  # type: ignore
            )
            * self.c
        )

    def _calc_hovering_height(self,
                              magnet_assembly: MagnetAssembly,
                              magnetic_spring: MagneticSpringInterp) -> float:
        target_force = magnet_assembly.get_weight()
        z_arr = np.linspace(0, 0.1, 1000)
        predict_force = magnetic_spring.get_force(z_arr)
        search_arr = np.abs(predict_force - target_force)
        idx = np.argmin(search_arr)
        hover_height = z_arr[idx]

        return hover_height * 1000  # Must be in mm

    def set_optimal_coil_center(self,
                                magnet_assembly: MagnetAssembly,
                                magnetic_spring: MagneticSpringInterp) -> None:

        hover_height = self._calc_hovering_height(
            magnet_assembly,
            magnetic_spring
        )
        hover_height = hover_height
        l_eps = 5
        coil_height = self.coil_wire_radius_mm * 2 * self.n_z
        new_coil_center_mm = (hover_height
                              + magnet_assembly.get_length()
                              + l_eps
                              + coil_height / 2)

        self.coil_center_mm = new_coil_center_mm
