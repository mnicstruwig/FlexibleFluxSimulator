import numpy as np


# TODO: Add docstrings to functions
# TODO: Implement OC case + test
class SimpleLoad:
    """A simple resistive load."""

    def __init__(self, R):
        """Constructor."""
        self.R = R

    def get_current(self, emf):
        if np.isinf(self.R):
            return 0
        return emf/self.R

    def get_voltage(self, current):
        pass

    def get_power(self, **kwargs):
        pass
