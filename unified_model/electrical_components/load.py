import numpy as np


# TODO: Add docstrings to functions
# TODO: Implement OC case + test
class SimpleLoad:
    """A simple resistive load."""

    def __init__(self, R):
        """Constructor."""
        self.R = R

    def __str__(self):
        return f"SimpleLoad: {self.R} Ohms"

    def __repr__(self):
        # Couldn't discover a nicer way to do this
        return f"SimpleLoad(R={self.R})"

    def to_json(self):
        return {'R': self.R}
