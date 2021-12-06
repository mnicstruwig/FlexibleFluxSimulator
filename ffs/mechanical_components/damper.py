import warnings

import numpy as np

from ..utils.utils import pretty_str
from ..mechanical_components.magnet_assembly import MagnetAssembly


def _sigmoid_shifted(x, x0=0):
    return 1 / (1 + np.exp(-(5 * x - x0))) - 0.5


class QuasiKarnoppDamper:
    """A damper that is based on a modified Karnopp friction model."""

    def __init__(
        self,
        coulomb_damping_coefficient: float,
        motional_damping_coefficient: float,
        magnet_assembly: MagnetAssembly,
        tube_inner_radius_mm: float,
    ) -> None:
        """Constructor

        Parameters
        ----------
        coulomb_damping_coefficient : float
            The damping coefficient that controls the amount of viscous
            friction. Typical values fall in the 1.0 - 6.0 range.
        motional_damping_coefficient : float
            The damping coefficient that controls the amount of friction due to
            the shape of the magnet assembly. Typical values fall in the
            0.0-0.005 range.
        magnet_assembly : MagnetAssembly
            The MagnetAssembly that is being used.
        tube_inner_radius_mm : float
            The inner radius of the microgenerator tube in mm.

        """
        self.cdc = coulomb_damping_coefficient
        self.mdc = motional_damping_coefficient
        self.magnet_assembly_length = magnet_assembly.get_length()
        self.magnet_assembly_mass = magnet_assembly.get_mass()
        self.r_t = tube_inner_radius_mm
        self.angle_friction_factor = 2 * self.r_t / self.magnet_assembly_length

    def __repr__(self):
        return f"QuasiKarnoppDamper(\n  coulomb_damping_coefficient={self.cdc},\n  motional_damping_coefficient={self.mdc},\n  tube_inner_radius_mm={self.r_t}\n)"  # noqa

    def get_force(self, velocity, velocity_threshold=0.01):
        """Get the force exerted by the damper."""
        coulomb_contribution = self.cdc * velocity * self.magnet_assembly_mass
        shape_contribution = (
            self.mdc
            * self.angle_friction_factor
            * _sigmoid_shifted(velocity, velocity_threshold)
        )
        return coulomb_contribution + shape_contribution


class MassProportionalDamper:
    """A mass-dependent constant damper.#!/usr/bin/env python

    The force is equal to the damping coefficient multiplied by the velocity,
    proportional to the mass of the magnet assembly.
    """

    def __init__(
        self, damping_coefficient: float, magnet_assembly: MagnetAssembly
    ) -> None:
        """Constructor.

        Parameters
        ----------
        damping_coefficient : float
            The damping coefficient of the damper.
        magnet_asssembly : MagnetAssembly
            The MagnetAssembly whose mass directly proportions the force of the damper.

        """
        self.damping_coefficient = damping_coefficient
        self.magnet_assembly_mass = magnet_assembly.get_mass()

    def get_force(self, velocity: float) -> float:
        """Get the force exerted by the damper.

        Parameters
        ----------
        velocity : float
            The velocity of the magnet assembly in m/s.

        Returns
        -------
        float
            The force exerted by the damper on the magnet assembly in Newtons.

        """

        return self.magnet_assembly_mass * self.damping_coefficient * velocity

    def to_json(self):
        return {
            "damping_coefficient": self.damping_coefficient,
            "magnet_assembly": "dep:magnet_assembly",
        }

    def update(self, model):
        """Update the internal state when notified."""
        try:
            assert model.magnet_assembly is not None
            self.magnet_assembly_mass = model.magnet_assembly.get_mass()
        except AssertionError:
            warnings.warn(
                "Missing dependency `magnet_assembly` for MassProportionalDamper."
            )

    def __repr__(self) -> str:
        return f"MassProportionalDamper(damping_coefficient={self.damping_coefficient}, magnet_assembly_mass={self.magnet_assembly_mass})"  # noqa


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
        return f"ConstantDamper(damping_coefficient={self.damping_coefficient})"

    def __str__(self):
        """Return string representation of the Damper."""
        return f"""ConstantDamper: {pretty_str(self.__dict__, 1)}"""
