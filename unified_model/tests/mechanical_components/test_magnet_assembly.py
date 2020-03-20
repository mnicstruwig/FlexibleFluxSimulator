import unittest

from unified_model.mechanical_components.magnet_assembly import MagnetAssembly


class TestMagnetAssembly(unittest.TestCase):
    """
    Test the `MagnetAssembly` class
    """

    def setUp(self):
        """
        Set-up of tests
        """
        n_magnet = 2
        l_m = 10
        l_mcd = 5
        dia_magnet = 5
        dia_spacer = 5

        self.test_magnet_assembly = MagnetAssembly(n_magnet, l_m, l_mcd, dia_magnet, dia_spacer)

    def test_calculate_weight(self):
        """
        Tests if the weight gets calculated correctly.
        """
        self.assertAlmostEqual(self.test_magnet_assembly.weight, 0.036116043669979545, places=5)

    def test_calculate_contact_surface_area(self):
        """
        Tests if the contact surface area gets calculated correctly
        """
        self.assertAlmostEqual(self.test_magnet_assembly.surface_area, 392.6990817, places=5)

    def test_get_mass(self):
        """
        Tests if the mass is correctly calculated and returned
        """
        self.assertAlmostEqual(self.test_magnet_assembly.get_mass(),
                               self.test_magnet_assembly.weight / 9.81)

    def test_get_weight(self):
        """
        Tests if the weight is correctly returned
        """
        self.test_magnet_assembly.weight = 100
        self.assertEqual(self.test_magnet_assembly.get_weight(), 100)

