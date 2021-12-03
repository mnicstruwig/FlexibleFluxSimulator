from typing import Any

import numpy as np
from unified_model.unified import UnifiedModel


def _get_emf(mag_pos: float, mag_vel: float, model: UnifiedModel) -> float:
    if model.flux_model is not None:
        dphi_dz = model.flux_model.get_dflux(mag_pos)
    else:
        raise ValueError("`flux_model` is not defined")
    emf = dphi_dz * mag_vel

    if model.rectification_drop:
        emf = np.abs(emf)
        if emf > model.rectification_drop:
            emf = emf - model.rectification_drop
        else:
            emf = 0

    return emf


def _get_load_voltage(emf: float, model: UnifiedModel) -> float:
    if model.load_model is None:
        raise ValueError("`load_model` is not defined")
    if model.coil_configuration is None:
        raise ValueError("`coil_configuration` is not defined")
    return (
        emf
        * model.load_model.R
        / (model.load_model.R + model.coil_configuration.get_coil_resistance())
    )


def _get_current(emf: float, model: UnifiedModel) -> float:
    if model.load_model is None:
        raise ValueError("`load_model` is not defined")
    if model.coil_configuration is None:
        raise ValueError("`coil_configuration` is not defined")
    return emf / (model.load_model.R + model.coil_configuration.get_coil_resistance())


def unified_ode(t: Any, y: Any, model: UnifiedModel) -> Any:
    # tube displ., tube velocity, magnet displ. , magnet velocity, flux
    x1, x2, x3, x4, x5 = y

    # prevent tube from going through bottom.
    if x1 <= 0 and x2 <= 0:
        x1 = 0
        x2 = 0

    # Calculate all our required values for the model equations
    emf = _get_emf(mag_pos=x3 - x1, mag_vel=x4 - x2, model=model)
    load_voltage = _get_load_voltage(emf=emf, model=model)
    current = _get_current(emf=emf, model=model)

    if model.coupling_model is not None:
        coupling_force = np.sign(x4 - x2) * model.coupling_model.get_mechanical_force(
            current
        )
    else:
        raise ValueError("`coupling_model` is not defined")

    if model.mechanical_spring is not None:
        mechanical_spring_force = model.mechanical_spring.get_force(x3 - x1, x4 - x2)
    else:
        raise ValueError("`mechanical_spring` is not defined")

    if model.magnetic_spring is not None:
        magnetic_spring_force = model.magnetic_spring.get_force(x3 - x1)
    else:
        raise ValueError("`magnetic_spring` is not defined")

    if model.magnet_assembly is not None:
        assembly_mass = model.magnet_assembly.get_mass()
        assembly_weight = model.magnet_assembly.get_weight()
    else:
        raise ValueError("`magnet_assembly` is not defined")

    if model.mechanical_damper is not None:
        damper_force = model.mechanical_damper.get_force(x4 - x2)
    else:
        raise ValueError("`mechanical_damper` is not defined")

    if model.input_excitation is not None:
        tube_acceleration = model.input_excitation.get_acceleration(t)  # type :ignore
    else:
        raise ValueError("`input_excitation` is not defined")

    x1_dot = x2
    x2_dot = tube_acceleration
    x3_dot = x4

    x4_dot = (
        +magnetic_spring_force
        - mechanical_spring_force
        - assembly_weight
        - damper_force
        - coupling_force
    ) / assembly_mass

    x5_dot = load_voltage  # NB <-- we want the EMF 'output' to be the load voltage

    return [x1_dot, x2_dot, x3_dot, x4_dot, x5_dot]


def unified_ode_old(t, y, mechanical_model, electrical_model, coupling_model):
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
        mechanical_spring_force = mechanical_spring.get_force(x3 - x1, x4 - x2)
    except AttributeError as e:
        mechanical_spring_force = 0
        raise e
    except TypeError:
        print("Type Error")
        mechanical_spring_force = 0

    magnetic_spring_force = magnetic_spring.get_force(x3 - x1)
    assembly_mass = magnet_assembly.get_mass()
    assembly_weight = magnet_assembly.get_weight()
    damper_force = damper.get_force(x4 - x2)

    x4_dot = (
        +magnetic_spring_force
        - mechanical_spring_force
        - assembly_weight
        - damper_force
        - coupling_force
    ) / assembly_mass

    x5_dot = load_voltage  # NB <-- we want the EMF 'output' to be the load voltage

    return [x1_dot, x2_dot, x3_dot, x4_dot, x5_dot]
