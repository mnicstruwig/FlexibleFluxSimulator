import numpy as np
from scipy.interpolate import interp1d

from unified_model.electrical_model import ElectricalModel


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


def apply_rectification(emf_values, v=0.2):
    """Do a "dumb" simulation of rectification of the EMF values."""
    emf_values = emf_values.copy()
    for i, e in enumerate(emf_values):
        e = np.abs(e)

        if e > v:
            e = e - v
        else:
            e = 0

        emf_values[i] = e
    return emf_values


# TODO: Add Docstring
def simulate_electrical_system(
    rel_mag_pos,
    rel_mag_vel,
    timestamps,
    flux_model,
    dflux_model,
    load_model=None,
    coil_resistance=np.inf,
):
    """Simulate the electrical system using pre-calculated input vectors.

    i.e. By avoiding integration / solving the numerical system.

    """

    electrical_model = ElectricalModel(name="debug")
    electrical_model.set_flux_model(flux_model, dflux_model)
    electrical_model.set_load_model(load_model)
    electrical_model.set_coil_resistance(coil_resistance)

    emf_values = np.array(
        [
            np.abs(electrical_model.get_emf(rmp, rmv))
            for rmp, rmv in zip(rel_mag_pos, rel_mag_vel)
        ]
    )

    if np.isinf(coil_resistance):
        return timestamps, emf_values
    return timestamps, emf_values * load_model.R / (load_model.R + coil_resistance)
