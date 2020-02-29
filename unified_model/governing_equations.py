import numpy as np
import warnings


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


def unified_ode(t, y, mechanical_model, electrical_model, coupling_model):
    magnetic_spring = mechanical_model.magnetic_spring
    mechanical_spring = mechanical_model.mechanical_spring
    damper = mechanical_model.damper
    input_ = mechanical_model.input_
    magnet_assembly = mechanical_model.magnet_assembly

    # tube displ., tube velocity, magnet displ. , magnet velocity, flux
    x1, x2, x3, x4, x5 = y

    # prevent tube from going through bottom.
    if x1 <= 0 and x2 <= 0:
        x1 = 0
        x2 = 0

    x1_dot = x2
    x2_dot = input_.get_acceleration(t)
    x3_dot = x4

    emf = electrical_model.get_emf(x3-x1, x4-x2)
    current = electrical_model.get_current(emf)
    coupling_force = coupling_model.get_mechanical_force(current)

    try:
        mechanical_spring_force = mechanical_spring.get_force(x3-x1, x4-x2)
    except AttributeError:
        mechanical_spring_force = 0
    except TypeError:
        mechanical_spring_force = 0

    x4_dot = (+ magnetic_spring.get_force(x3 - x1)
              - mechanical_spring_force
              - magnet_assembly.get_weight()
              - damper.get_force(x4 - x2)
              - coupling_force) / magnet_assembly.get_mass()

    x5_dot = emf

    return [x1_dot, x2_dot, x3_dot, x4_dot, x5_dot]
