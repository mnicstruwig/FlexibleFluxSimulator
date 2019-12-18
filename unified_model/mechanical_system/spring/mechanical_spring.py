"""Mechanical springs available for use."""

import numpy as np


# TODO: Documentation
class MechanicalSpring(object):
    """Mechanical spring with customizable parameters.
    """
    def __init__(self,
                 position: float,
                 strength: float = 1e6,
                 damper_constant: float = 0) -> None:
        self.position = position
        self.k = strength
        self.c = damper_constant

    def _heaviside_step_function(self, x, boundary):
        """Compute the output of a Heaviside step function"""
        return 0.5 * (np.sign(x - boundary) + 1)

    def get_force(self, x, x_dot):
        force = self._heaviside_step_function(x, self.position) \
            * (self.k * (x - self.position) + self.c * x_dot)

        return force
