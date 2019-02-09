import numpy as np
import pandas as pd

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from unified_model.electrical_system.electrical_system import ElectricalSystem
from unified_model.electrical_system.load import SimpleLoad
from unified_model.electrical_system.flux.utils import FluxDatabase
from unified_model.electrical_system.flux.model import flux_univariate_spline

from unified_model.mechanical_system.evaluator import LabeledVideoProcessor


def _build_y_input_vector_at_timestamps(x3, timestamps):
    """
    Build the `y` input vector (at each timestamp!) that is fed to the
    electrical system in order to calculate the emf.

    y[0] --> x1 --> tube displacement
    y[1] --> x2 --> tube velocity (gradient of x1)
    y[2] --> x3 --> magnet displacement
    y[3] --> x4 --> magnet velocity (gradient of x3)
    y[4] --> x5 --> flux linkage

    We can "hack" around this by setting everything to zero, _except_ x3 and x4
    and then calculate the induced EMF by hand.

    Parameters
    ----------
    x3 : array_like
        The relative distance between the top of the fixed magnet and the bottom
    timestamps : array_like
        Timestamps corresponding to `x3`

    Returns
    -------
    (n, 5) ndarray
        y "input vector" where each row is a value of the input vector `y`.
    """

    x1 = np.zeros(len(x3))
    x2 = x1
    # x3 already defined
    x4 = np.gradient(x3) / np.gradient(timestamps)
    x5 = x1  # Not used, but must be present.

    return np.array([x1, x2, x3, x4, x5]).T

def apply_rectification(emf_values):
    """Do a "dumb" simulation of rectification of the EMF values."""

    for i, e in enumerate(emf_values):
        e = np.abs(e)

        if e > 0.2:
            e = e-0.2
        else:
            e = 0

        emf_values[i]= e
    return emf_values

# TODO: Add Docstring
def simulate_electrical_system(y_relative_mm, timestamps, flux_model, load, interpolate=True):
    """Simulate the electrical system using pre-calculated input vectors.

    i.e. By avoiding integration and solving the numerical system."""

    x3 = y_relative_mm

    if interpolate:
        x3_interpolator = interp1d(timestamps, x3)
        timestamps = np.linspace(0, timestamps[-1], 10000)
        x3 = x3_interpolator(timestamps)

    ys = _build_y_input_vector_at_timestamps(x3, timestamps)

    electrical_system = ElectricalSystem(flux_model=flux_model, load_model=load, precompute_gradient=True)
    emf_values = np.array([electrical_system.get_emf(y) for y in ys])

    return emf_values, timestamps

