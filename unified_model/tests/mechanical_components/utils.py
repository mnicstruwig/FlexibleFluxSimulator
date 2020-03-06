from unified_model.mechanical_model import MechanicalModel
from unified_model.mechanical_components.damper import Damper
from unified_model.mechanical_components.input_excitation.footstep import Footstep
from unified_model.mechanical_components.magnetic_spring import MagneticSpring
from unified_model.mechanical_components.magnet_assembly import MagnetAssembly

from unified_model.tests.mechanical_components.test_data import TEST_MAGNET_SPRING_FEA_PATH


def build_test_mechanical_system_model():
    """
    Builds a hard-coded mechanical system model for use in testing
    :return The fully-completed mechanical system
    """
    test_spring = MagneticSpring(fea_data_file=TEST_MAGNET_SPRING_FEA_PATH,
                                 model='coulombs_modified')
    test_magnet_assembly = MagnetAssembly(n_magnet=2,
                                          l_m=10,
                                          l_mcd=5,
                                          dia_magnet=10,
                                          dia_spacer=10)
    test_damper = Damper(model='constant', model_kwargs={'damping_coefficient': 0.1})

    acc_up = 1.5 * 9.81
    acc_dec = -1 * 9.81
    acc_down = -2 * 9.81
    acc_impact = 2 * 9.81

    test_footstep = Footstep(accelerations=[acc_up, acc_dec, acc_down, acc_impact],
                             t_couple_separation=0.05,
                             positive_footstep_displacement=0.15,
                             t_footstep_start=1)

    test_mechanical_system = MechanicalModel()
    test_mechanical_system.set_magnetic_spring(test_spring)
    test_mechanical_system.set_damper(test_damper)
    test_mechanical_system.set_input(test_footstep)
    test_mechanical_system.set_magnet_assembly(test_magnet_assembly)

    initial_conditions = [0, 0, 0.1, 0]
    test_mechanical_system.set_model('ode_decoupled', initial_conditions=initial_conditions)

    return test_mechanical_system
