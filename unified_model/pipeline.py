"""
This module contains useful and commonly-used post-processing pipelines
for the `UnifiedModel` class.
"""
import numpy as np


def clip_x2(y):
    """
    Clip the x2 to be zero if both x1 < 0 and x2 < 0.
    """
    if y[0] < 0 and y[1] < 0:
        y[0] = 0
        y[1] = 0

    return y
