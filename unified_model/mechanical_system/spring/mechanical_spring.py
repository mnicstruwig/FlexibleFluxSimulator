"""Mechanical springs available for use."""

import numpy as np


class MechanicalSpring(object):
    """Mechanical spring with customizable parameters.

    Currently only non-ideal (non-linear) springs are supported.

    """
    def __init__(self,
                 push_direction: str,
                 position: float,
                 strength: float = 1000,
                 sharpness: float = 0.001,  # TODO: Rename to something more appropriate
                 pure: bool = True,
                 damper_constant: float = None) -> None:
        self.pure = pure
        self.push_direction = push_direction
        self.position = position
        self.strength = strength
        self.sharpness = sharpness
        self.c = damper_constant

        if push_direction is 'down':
            self.direction_modifier = -1
        elif push_direction is 'up':
            self.direction_modifier = 1
        else:
            raise ValueError(f'Incorrect spring direction specified: "{push_direction}".\n\
            Possible values are "up" and "down".')

    def get_force(self, x, x_dot):
        force = self.strength \
            * np.exp(self.direction_modifier*(self.position-x)/self.sharpness)

        if self.pure:
            return force
        return force - np.abs(self.c*x_dot)
