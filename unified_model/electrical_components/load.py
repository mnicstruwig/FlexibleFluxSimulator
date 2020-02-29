import numpy as np


# TODO: Add docstrings to functions
# TODO: Implement OC case + test
class SimpleLoad:
    """A simple resistive load."""

    def __init__(self, R):
        """Constructor."""
        self.R = R

    def __str__(self):
        return 'SimpleLoad: {} Ohms'.format(self.R)

    def __repr__(self):
        # Couldn't discover a nicer way to do this
        class_path = str(self.__class__).split("'")[1]
        return '{}(R={})'.format(class_path, self.R)

    def get_current(self, emf, coil_resistance):
        """Get the current through the load."""
        if np.isinf(self.R) or np.isinf(coil_resistance):
            return 0
        v_load = emf*self.R/(self.R + coil_resistance)
        return v_load/self.R

    def get_voltage(self, current):
        """Get the voltage over the load."""
        pass

    def get_power(self, **kwargs):
        """Get the power dissipated in the load."""
        pass
