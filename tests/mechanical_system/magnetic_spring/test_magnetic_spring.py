import unittest

# Local imports
from mechanical_system.magnetic_spring.magnetic_spring import MagneticSpring
from mechanical_system.magnetic_spring.model import coulombs_law, coulombs_law_modified


class TestMagneticSpring(unittest.TestCase):
    """
    Tests the MagneticSpring class functions
    """

    def setUp(self):
        """
        Test set-up
        """
        self.test_spring = MagneticSpring('../test_data/test_magnetic_spring_fea.csv', model='coulombs_modified')

    def test_set_model(self):
        """
        Tests if the model function can be set
        """
        self.test_spring._set_model('coulombs_unmodified')
        self.assertEqual(self.test_spring.model, coulombs_law)

    def test_fit_model_parameters(self):
        """
        Tests if the model parameters can be fit
        """
        self.test_spring._fit_model_parameters()
        self.assertEqual(len(self.test_spring.model_parameters), 2)

    def test_get_force(self):
        """
        Tests if the force exerted by the magnetic spring can be calculated
        """
        test_z_array = [1, 2, 3]
        predicted_force = self.test_spring.get_force_array(test_z_array)

        self.assertAlmostEqual(predicted_force[0], 23.5, places=1)
        self.assertAlmostEqual(predicted_force[1], 18.1, places=1)
        self.assertAlmostEqual(predicted_force[2], 13.1, places=1)
