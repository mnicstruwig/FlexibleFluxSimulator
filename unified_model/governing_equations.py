import numpy as np


def unified_ode_mechanical_only(t,
                                y,
                                mechanical_model,
                                electrical_model,
                                coupling_model):
    spring = mechanical_model.spring
    damper = mechanical_model.damper
    input_ = mechanical_model.input_
    magnet_assembly = mechanical_model.magnet_assembly

    # tube displacement, tube velocity, magnet displacement, magnet velocity
    x1, x2, x3, x4 = y

    # prevent tube from going through floor.
    if x1 <= 0 and x2 <= 0:
        x1 = 0.
        x2 = 0.

    x1_dot = x2
    x2_dot = input_.get_acceleration(t)
    x3_dot = x4

    x4_dot = (spring.get_force(x3 - x1) - magnet_assembly.get_weight() -
              damper.get_force(x4 - x2)) / magnet_assembly.get_mass()

    return [x1_dot, x2_dot, x3_dot, x4_dot]


def _is_assembly_above_max_height(tube_pos, mag_pos, assembly_height, max_height):
    """Determine if top of the magnet assembly extends beyond the max height
    of the tube.

    Parameters
    ----------
    tube_pos : float
        Current position of the bottom of the tube, in metres.
    mag_pos : float
        Current position of the _bottom_ of the magnet assembly, in metres.
    assembly_height : float
        Height of the magnet assembly, in metres.
    max_height : float
        Maximum height that the _top_ of the magnet assembly can have, in
        metres, relative to the bottom of the tube.

    Returns
    -------
    bool
        True if the top of the magnet assembly extends beyond the maximum
        allowed height. False otherwise.

    """
    if (mag_pos - tube_pos) > (max_height - assembly_height):
        return True
    return False


def _get_assembly_max_position(tube_pos, assembly_height, max_height):
    """Get the maximum position the bottom of the magnet assembly may have."""
    return max_height - assembly_height + tube_pos


def unified_ode(t, y, mechanical_model, electrical_model, coupling_model):
    magnetic_spring = mechanical_model.magnetic_spring
    mechanical_spring = mechanical_model.mechanical_spring
    damper = mechanical_model.damper
    input_ = mechanical_model.input_
    magnet_assembly = mechanical_model.magnet_assembly

    # tube displ., tube velocity, magnet displ. , magnet velocity, flux
    x1, x2, x3, x4, x5 = y

    # Make the sudden stop of the tube slightly less harsh
    if x1 <= 0.015 and x2 <= 0:
        x2 = x2/3

    # prevent tube from going through bottom.
    if x1 <= 0 and x2 <= 0:
        x1 = 0
        x2 = 0

    x1_dot = x2
    x2_dot = input_.get_acceleration(t)
    x3_dot = x4

    emf = electrical_model.get_emf(y=[x1, x2, x3, x4, x5])
    current = electrical_model.load_model.get_current(emf,
                                                      electrical_model.coil_resistance)
    coupling_force = coupling_model.get_mechanical_force(current)

    # Ensure coupling mechanical force is *opposite* to direction of motion.
    # Note that the direction is the same as the motion here, since we subtract
    # the force later.
    if (x4 - x2) < 0:
        coupling_force = -np.abs(coupling_force)
    if (x4 - x2) > 0:
        coupling_force = np.abs(coupling_force)

    try:
        mechanical_spring_force = mechanical_spring.get_force(x3-x1, x4-x2)
    except AttributeError:
        mechanical_spring_force = 0

    x4_dot = (+ magnetic_spring.get_force(x3 - x1)
              - mechanical_spring_force
              - magnet_assembly.get_weight()
              - damper.get_force(x4 - x2)
              - coupling_force) / magnet_assembly.get_mass()

    x5_dot = emf

    return [x1_dot, x2_dot, x3_dot, x4_dot, x5_dot]
