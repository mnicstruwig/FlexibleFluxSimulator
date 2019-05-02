import unittest

# Local imports
from unified_model.mechanical_model import MechanicalModel
from unified_model.mechanical_system.spring.magnetic_spring import MagneticSpring

# Local test helpers
from unified_model.tests.mechanical_system.test_data.test_data import TEST_MAGNET_SPRING_FEA_PATH, TEST_RAW_OUTPUT, TEST_TIME_STEPS



class TestMechanicalModel(unittest.TestCase):
    """
    Tests the MechanicalModel class
    """

    def setUp(self):
        """
        Run before every test.
        """

        self.test_mechanical_system = MechanicalModel(name='test_mechanical_model')
        self.test_model = 'ode_decoupled'
        self.test_initial_conditions = [1, 2, 3, 4]
        self.test_mechanical_system.raw_output = TEST_RAW_OUTPUT
        self.test_mechanical_system.t = TEST_TIME_STEPS

    def test_set_initial_conditions(self):
        """
        Tests if the initial conditions can be set correctly
        """
        self.test_mechanical_system.set_initial_conditions(self.test_initial_conditions)
        self.assertEqual(self.test_mechanical_system.initial_conditions, self.test_initial_conditions)

    def test_set_spring(self):
        """
        Tests if the spring can be set correctly.
        """
        test_spring = MagneticSpring(TEST_MAGNET_SPRING_FEA_PATH, model='coulombs_modified')
        self.test_mechanical_system.set_spring(test_spring)

        self.assertEqual(self.test_mechanical_system.spring, test_spring)
