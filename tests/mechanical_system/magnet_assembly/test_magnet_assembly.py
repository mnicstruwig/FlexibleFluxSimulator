import unittest

from mechanical_system.magnet_assembly.magnet_assembly import MagnetAssembly


class TestMagnetAssembly(unittest.TestCase):
    """
    Tests the `MagnetAssembly` class functions
    """

    def setUp(self):
        """
        Set-up of tests
        """
        n_magnet = 2
        h_magnet = 10
        h_spacer = 5
        dia_magnet = 5
        dia_spacer = 5

        self.test_magnet_assembly = MagnetAssembly(n_magnet, h_magnet, h_spacer, dia_magnet, dia_spacer)

    def test_set_weight(self):
        """
        Tests if the weight is can be set manually and automatically
        """

        # Manual case
        custom_weight = 0.005
        self.test_magnet_assembly._set_weight(custom_weight)
        self.assertEqual(self.test_magnet_assembly.weight, custom_weight)

        # Automatic case
        self.test_magnet_assembly._set_weight()
        self.assertAlmostEqual(self.test_magnet_assembly.weight, 0.036116043669979545, places=5)

    def test_calculate_weight(self):
        """
        Tests if the weight gets calculated correctly.
        """
        self.assertAlmostEqual(self.test_magnet_assembly.weight, 0.036116043669979545, places=5)

    def test_get_mass(self):
        """
        Tests if the mass is correctly calculated and returned
        """
        self.assertAlmostEqual(self.test_magnet_assembly.get_mass(), self.test_magnet_assembly.weight / 9.81)

    def test_get_weight(self):
        """
        Tests if the weight is correctly returned
        """
        self.test_magnet_assembly.weight = 100
        self.assertEqual(self.test_magnet_assembly.get_weight(), 100)
