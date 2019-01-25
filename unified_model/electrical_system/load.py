# TODO: Add docstrings to functions
class SimpleLoad:
    """A simple resistive load."""

    def __init__(self, R):
        """Constructor."""
        self.R = R

    def get_current(self, emf):
        return emf/self.R

    def get_voltage(self, current):
        pass

    def get_power(self, **kwargs):
        pass
