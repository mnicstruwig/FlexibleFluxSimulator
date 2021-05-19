import numpy as np


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

    load_voltage = electrical_model.get_load_voltage(x3 - x1, x4 - x2)

    emf = electrical_model.get_emf(x3 - x1, x4 - x2)
    current = electrical_model.get_current(emf)
    coupling_force = np.sign(x4 - x2) * coupling_model.get_mechanical_force(current)

    try:
        mechanical_spring_force = mechanical_spring.get_force(x3-x1, x4-x2)
    except AttributeError as e:
        mechanical_spring_force = 0
        raise e
    except TypeError:
        print('Type Error')
        mechanical_spring_force = 0

    magnetic_spring_force = magnetic_spring.get_force(x3 - x1)
    assembly_mass = magnet_assembly.get_mass()
    assembly_weight = magnet_assembly.get_weight()
    damper_force = damper.get_force(x4 - x2)

    x4_dot = (+ magnetic_spring_force
              - mechanical_spring_force
              - assembly_weight
              - damper_force
              - coupling_force) / assembly_mass

    x5_dot = load_voltage  # NB <-- we want the EMF 'output' to be the load voltage

    return [x1_dot, x2_dot, x3_dot, x4_dot, x5_dot]
