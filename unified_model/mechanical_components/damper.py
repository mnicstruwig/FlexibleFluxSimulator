import numpy as np

from unified_model.utils.utils import pretty_str
from unified_model.mechanical_components.magnet_assembly import MagnetAssembly

def _sigmoid(x, x0=0):
    return 1 / (1 + np.exp(-(5*x - x0)))

class QuasiKarnoppDamper:
    """A damper that is based on a modified Karnopp friction model."""
    def __init__(self,
                 coulomb_damping_coefficient: float,
                 motional_damping_coefficient: float,
                 magnet_assembly: MagnetAssembly,
                 tube_inner_radius_mm: float) -> None:
        """Constructor"""
        # TODO: Documentation
        # TODO: __repr__ and __str__
        self.cdc = coulomb_damping_coefficient
        self.mdc = motional_damping_coefficient
        self.magnet_assembly_length = magnet_assembly.get_length()
        self.magnet_assembly_mass = magnet_assembly.get_mass()
        self.r_t = tube_inner_radius_mm
        self.angle_friction_factor = 2 * self.r_t / self.magnet_assembly_length

    def get_force(self, velocity, velocity_threshold=0.01):
        """Get the force exerted by the damper."""
        coulomb_contribution = self.cdc * velocity * self.magnet_assembly_mass
        shape_contribution = self.mdc * self.angle_friction_factor * _sigmoid(velocity, velocity_threshold)
        return coulomb_contribution + shape_contribution


class ConstantDamper:
    """A constant-damping-coefficient damper.

    The force will be equal to the damping coefficient multiplied by a
    velocity, i.e. F = c * v.

    """

    def __init__(self, damping_coefficient):
        """Constructor

        Parameters
        ----------
        damping_coefficient : float
            The constant-damping-coefficient of the damper.

        """
        self.damping_coefficient = damping_coefficient

    def get_force(self, velocity):
        """Get the force exerted by the damper.

        Parameters
        ----------
        velocity : float
            Velocity of the object attached to the damper. In m/s.

        Returns
        -------
        float
            The force exerted by the damper. In Newtons.

        """
        return self.damping_coefficient * velocity

    def __repr__(self):
        return f'ConstantDamper(damping_coefficient={self.damping_coefficient})'

    def __str__(self):
        """Return string representation of the Damper."""
        return f"""ConstantDamper: {pretty_str(self.__dict__, 1)}"""
