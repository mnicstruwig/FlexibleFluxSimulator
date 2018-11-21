import unittest

# Local imports
from unified_model.mechanical_system.spring.magnetic_spring import MagneticSpring
from unified_model.mechanical_system.spring.model import coulombs_law, coulombs_law_modified
from unified_model.tests.mechanical_system.test_data import TEST_MAGNET_SPRING_FEA_PATH

class TestMagneticSpring(unittest.TestCase):
    """
    Tests the MagneticSpring class functions
    """

    def setUp(self):
        """
        Test set-up
        """
        self.test_spring = MagneticSpring(TEST_MAGNET_SPRING_FEA_PATH, model='coulombs_modified')

    def test_set_model(self):
        """
        Test if the model function can be set
        """
        self.test_spring._set_model('coulombs_unmodified')
        self.assertEqual(self.test_spring.model, coulombs_law)

    def test_fit_model_parameters(self):
        """
        Test if the model parameters can be fit
        """
        self.test_spring._fit_model_parameters()
        self.assertEqual(len(self.test_spring.model_parameters), 2)

    def test_get_force(self):
        """
        Test if the force exerted by the magnetic spring can be calculated
        """
        test_z_array = [1, 2, 3]
        predicted_force = self.test_spring.get_force_array(test_z_array)

        self.assertTrue(len(predicted_force) == 3)
