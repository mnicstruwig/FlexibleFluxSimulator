"""Various mechanical springs for use in the mechanical model."""

import numpy as np


class MechanicalSpring(object):
    """Mechanical, non-attached, spring."""

    def __init__(self,
                 position: float,
                 strength: float = 1e6,
                 damper_constant: float = 0) -> None:
        """Constructor.

        Parameters
        ----------
        position : float
            The height at which the mechanical spring acts. In metres.
        strength : float
            The "strength" of the mechanical spring. It is recommended to use a
            large value, or to leave this at the default value. Default value
            is 1e6.
        damper_constant : float
            Controls the amount of energy "lost" upon impact. A value of zero
            indicates an ideal spring.

        """
        self.position = position
        self.k = strength
        self.c = damper_constant

    def __repr__(self):
        return f'MechanicalSpring(position={self.position}, strength={self.k}, damper_constant={self.c})'
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
        force = self._heaviside_step_function(x, self.position) \
            * (self.k * (x - self.position) + self.c * x_dot)

        return force
