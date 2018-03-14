import unittest

from mechanical_system.damper.model import DamperConstant
from mechanical_system.damper import damper

CONSTANT_DAMPING_COEFFICIENT = 3


class TestDamperModels(unittest.TestCase):
    """
    Tests the various Damper models
    """

    @classmethod
    def setUpClass(cls):
        """
        Run once before running tests.
        """
        cls.damper_constant = DamperConstant(damping_coefficient=CONSTANT_DAMPING_COEFFICIENT)

    def test_damper_constant(self):
        velocity = 3
        expected_force = CONSTANT_DAMPING_COEFFICIENT * velocity

        # Tests
        self.assertEqual(self.damper_constant.damping_coefficient, CONSTANT_DAMPING_COEFFICIENT)
        self.assertEqual(self.damper_constant.get_force(velocity), expected_force)


class TestDamper(unittest.TestCase):
    """
    Tests the Damper class
    """

    def setUp(self):
        """
        Runs before each test.
        """
        self.damper_kwargs = {'damping_coefficient': CONSTANT_DAMPING_COEFFICIENT}
        self.test_damper = damper.Damper('constant', self.damper_kwargs)

    def test_set_model(self):
        """
        Tests the `_set_model` method
        """
        new_damper_kwargs = {'damping_coefficient': 5}
        self.test_damper._set_model('constant', new_damper_kwargs)  # Set new model

        # Test
        self.assertEqual(self.test_damper.model.damping_coefficient, 5)

    def test_get_force(self):
        """
        Tests if the force can be calculated using the model
        """
        velocity = 3
        expected_force = velocity * CONSTANT_DAMPING_COEFFICIENT

        self.assertEqual(self.test_damper.get_force(velocity), expected_force)
