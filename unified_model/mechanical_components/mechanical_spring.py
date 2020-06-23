"""Various mechanical springs for use in the mechanical model."""

import numpy as np


class MechanicalSpring:
    """Mechanical, non-attached, spring."""

    def __init__(self,
                 position: float,
                 magnet_length: float,
                 strength: float = 1e6,
                 damping_coefficient: float = 0) -> None:
        """Constructor.

        Parameters
        ----------
        position : float
            The height at which the mechanical spring acts. In metres.
        magnet_length : float
            The length of the magnet in *metres*. This is used to properly
            offset when the mechanical spring is supposed to begin acting.
        strength : float
            The "strength" of the mechanical spring. It is recommended to use a
            large value, or to leave this at the default value. Default value
            is 1e6.
        damping_coefficient : float
            Controls the amount of energy "lost" upon impact. A value of zero
            indicates an ideal spring.

        """
        self.position = position
        self.magnet_length = magnet_length
        self.strength = strength
        self.damping_coefficient = damping_coefficient

    def __repr__(self):
        return f'MechanicalSpring(position={self.position}, magnet_length={self.magnet_length}, strength={self.strength}, damping_coefficient={self.damping_coefficient})'  # noqa

    def _heaviside_step_function(self, x, boundary):
        """Compute the output of a Heaviside step function"""
        return 0.5 * (np.sign(x - boundary) + 1)

    def get_force(self, x, x_dot):
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
        offset = self.magnet_length / 2
        force = (
            self._heaviside_step_function(x + offset, self.position)
            * (self.strength * (x - self.position)
               + self.damping_coefficient * x_dot)
            )

        return force
