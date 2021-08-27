"""Various mechanical springs for use in the mechanical model."""

from typing import Optional

import numpy as np

from unified_model.mechanical_components.magnet_assembly import MagnetAssembly
from ..local_exceptions import ModelError


class MechanicalSpring:
    """Mechanical, non-attached, spring."""

    def __init__(
        self,
        magnet_assembly: MagnetAssembly,
        strength: float = 1e7,
        damping_coefficient: float = 0,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        magnet_assembly: MagnetAssembly
            The magnet assembly that will be moving in the microgenerator.
        strength : float
            The "strength" of the mechanical spring. It is recommended to use a
            large value, or to leave this at the default value. Default value
            is 1e6.
        damping_coefficient : float
            Controls the amount of energy "lost" upon impact. A value of zero
            indicates an ideal spring.

        """
        self.position: Optional[float] = None
        self.magnet_length = magnet_assembly.l_m_mm / 1000  # Must be in metres
        self.magnet_assembly_length = magnet_assembly.get_length() / 1000
        self.strength = strength
        self.damping_coefficient = damping_coefficient

    def __repr__(self):
        return f"MechanicalSpring(position={self.position}, magnet_length={self.magnet_length}, magnet_assembly_length={self.magnet_assembly_length}, strength={self.strength}, damping_coefficient={self.damping_coefficient})"  # noqa

    def _heaviside_step_function(self, x, boundary):
        """Compute the output of a Heaviside step function"""
        return 0.5 * (np.sign(x - boundary) + 1)

    def set_position(self, position: float) -> None:
        """Set the height at which the mechanical spring acts, in metres."""
        self.position = position

    def get_force(self, x: float, x_dot: float) -> float:
        """Get the force exerted by the spring.

        Parameters
        ----------
        x : float
            Displacement of the object. In metres.
        x_dot : float
            Velocity of the object. In metres per second.

        Returns
        -------
        float
            The force exerted by the mechanical spring. In Newtons.

        """
        try:
            assert self.position is not None
        except AssertionError as e:
            raise ModelError(
                "The position of the mechanical spring has not been defined. Did you call `.set_height` on the UnifiedModel?"  # noqa
            ) from e

        offset = self.magnet_assembly_length - (self.magnet_length / 2)
        force = self._heaviside_step_function(x + offset, self.position) * (
            self.strength * (x - self.position + offset)
            + self.damping_coefficient * x_dot
        )

        return force
