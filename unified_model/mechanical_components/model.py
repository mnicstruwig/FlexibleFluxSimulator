"""
Defines the models for the mechanical system
"""

def _enforce_boundary_condition(boundary_condition, y, lower=True):

    if lower is False:
        raise NotImplementedError('This feature is not yet implemented')

    for (index, value), condition in zip(enumerate(y), boundary_condition):
        try:
            y[index] = condition if value < boundary_condition[index] else value
        except TypeError:  # Handles cases where there is no boundary condition
            if boundary_condition[index] is None:
                continue
            else:
                raise
    return y


def ode_decoupled(t, y, kwargs):
    """
    The ordinary differential equation (ODE) model that is *decoupled* from the influence of the electrical system.
    Alternatively, the electrical system is a pure open-circuit and has no impact on this mechanical model.

    Parameters
    ----------
    :param y: Array containing the values of the four differential equations that make up this model.
    :param t: Instantaneous time
    :param kwargs:  spring : spring object to be used.
                    damper : damper object to be used.
                    mechanical_input : input object to be used.
                    magnet_assembly : magnet_Assembly object to be used
                    lower_boundary_conditions: list, optional
                        Specify the lower boundary conditions for x1, x2, x3, x4. If no boundary condition exists,
                        use `None`.
    :return: Array containing the derivatives of `y`
    """

    spring = kwargs['spring']
    damper = kwargs['damper']
    mechanical_input = kwargs['input']
    magnet_assembly = kwargs['magnet_assembly']

    x1, x2, x3, x4 = y  # tube displacement, tube velocity, magnet-assembly displacement, magnet-assembly velocity

    if x1 < 0 and x2 < 0:
        x1 = 0
        x2 = 0

    x1_dot = x2
    x2_dot = mechanical_input.get_acceleration(t)
    x3_dot = x4
    x4_dot = (spring.get_force(x3 - x1) - magnet_assembly.get_weight() - damper.get_force(
        x4 - x2)) / magnet_assembly.get_mass()

    return [x1_dot, x2_dot, x3_dot, x4_dot]  # Return the derivative of the input
