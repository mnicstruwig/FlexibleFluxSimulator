from mechanical_system.mechanical_system import MechanicalSystem
from mechanical_system.model import ode_decoupled
from mechanical_system.damper.damper import Damper
from mechanical_system.footstep.footstep import Footstep
from mechanical_system.spring.magnetic_spring import MagneticSpring
from mechanical_system.magnet_assembly.magnet_assembly import MagnetAssembly


def build_test_mechanical_system_model():
    """
    Builds a hard-coded mechanical system model for use in testing
    :return The fully-completed mechanical system
    """
    test_spring = MagneticSpring(fea_data_file='../test_data/test_magnetic_spring_fea.csv',
                                 model='coulombs_modified')
    test_magnet_assembly = MagnetAssembly(n_magnet=2,
                                          h_magnet=10,
                                          h_spacer=5,
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

    test_mechanical_system = MechanicalSystem()
    test_mechanical_system.set_spring(test_spring)
    test_mechanical_system.set_damper(test_damper)
    test_mechanical_system.set_input(test_footstep)
    test_mechanical_system.set_magnet_assembly(test_magnet_assembly)

    initial_conditions = [0, 0, 0.1, 0]
    test_mechanical_system.set_model(ode_decoupled, initial_conditions=initial_conditions)

    return test_mechanical_system
