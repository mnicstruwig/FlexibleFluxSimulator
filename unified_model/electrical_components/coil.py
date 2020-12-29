import numpy as np

# TODO: Documentation
class CoilModel:
    """A coil model."""

    def __init__(self,
                 c: int,
                 n_z: int,
                 n_w: int,
                 l_ccd_mm: float,
                 ohm_per_mm: float,
                 tube_wall_thickness_mm: float,
                 coil_wire_radius_mm: float,
                 coil_center_mm: float,
                 outer_tube_radius_mm: float,
                 coil_resistance: float = None) -> None:

        self.c = c
        self.n_z = n_z
        self.n_w = n_w
        self.l_ccd_mm = l_ccd_mm
        self.ohm_per_mm = ohm_per_mm
        self.tube_wall_thickness_mm = tube_wall_thickness_mm
        self.coil_wire_radius_mm = coil_wire_radius_mm
        self.coil_center_mm = coil_center_mm
        self.outer_tube_radius_mm = outer_tube_radius_mm

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
            2*np.pi
            * self.ohm_per_mm
            * self.n_w*self.n_z
            * (
                self.tube_wall_thickness_mm
                + self.outer_tube_radius_mm
                + self.coil_wire_radius_mm
                + 2*self.coil_wire_radius_mm*(self.n_w+1)/2
            )
            * self.c
        )

