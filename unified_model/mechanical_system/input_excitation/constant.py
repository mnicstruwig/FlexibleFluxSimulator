"""
Constant input excitations that will always return a constant acceleration.
"""


class ConstantAcceleration:
    """A constant acceleration input excitation."""
    def __init__(self, c=0):
        """Constructor.

        Parameters
        ----------
        c : float
            The constant acceleration to return. In m/s^2.
            Default is 0.

        """
        self.c = 0

    def get_acceleration(self, *args, **kwargs):
        return self.c
