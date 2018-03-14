"""
Defines the models for the mechanical system
"""


def ode_decoupled(y, t, kwargs):
    spring = kwargs['spring']
    damper = kwargs['damper']
    mechanical_input = kwargs['input']
    magnet_assembly = kwargs['magnet_assembly']

    x1, x2, x3, x4 = y  # tube displacement, tube velocity, magnet-assembly displacement, magnet-assembly velocity

    x1_dot = x2
    x2_dot = mechanical_input.get_acceleration(t)
    x3_dot = x4
    x4_dot = (spring.get_force(x3 - x1) - magnet_assembly.get_weight() - damper.get_force(
        x4 - x2)) / magnet_assembly.get_mass()

    return [x1_dot, x2_dot, x3_dot, x4_dot]  # Return the derivative of the input
