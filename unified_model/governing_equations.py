import numpy as np


def unified_ode_coupled(t, y, kwargs):
    """
    A coupled unified electro-mechanical model.
    """

    spring = kwargs['spring']
    damper = kwargs['damper']
    mechanical_input = kwargs['input']
    magnet_assembly = kwargs['magnet_assembly']
    coupling = kwargs['coupling']
    electrical_system = kwargs['electrical_system']

    x1, x2, x3, x4, x5 = y  # tube displacement, tube velocity, magnet displacement, magnet velocity, flux

    if x1 <= 0 and x2 <= 0:
        x1 = 0
        x2 = 0

    x1_dot = x2
    x2_dot = mechanical_input.get_acceleration(t)
    x3_dot = x4
    emf = electrical_system.get_emf(y)
    current = electrical_system.load_model.get_current(emf)
    coupling_force = coupling.get_mechanical_force(current)

    # Ensure coupling mechanical force is *opposite* to direction of motion.
    # Note that the direction is the same as the motion here, since we subtract
    # the force later.
    if (x4 - x2) < 0:
        coupling_force = -np.abs(coupling_force)
    if (x4 - x2) > 0:
        coupling_force = np.abs(coupling_force)

    x4_dot = (spring.get_force(x3 - x1) - magnet_assembly.get_weight() -
              damper.get_force(x4 - x2) - coupling_force) / magnet_assembly.get_mass()

    x5_dot = emf

    return [x1_dot, x2_dot, x3_dot, x4_dot, x5_dot]
