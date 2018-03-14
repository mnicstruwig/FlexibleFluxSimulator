import unittest

# Local imports
from mechanical_system.footstep import footstep


class TestFootstep(unittest.TestCase):
    """
    Tests the footstep class
    """
    def setUp(self):
        """
        Set-up for the `TestFootstep` class
        """

        self.accelerations = [1., -2., -3., 4.]
        self.acceleration_couple_time_separation = 0.05
        self.positive_footstep_displacement = 0.15
        self.footstep = footstep.Footstep(self.accelerations, self.acceleration_couple_time_separation,
                                          self.positive_footstep_displacement, )

    def test_calculate_acceleration_couple_times(self):
        """
        Tests the `_calculate_acceleration_couple_times` function
        """

        acc_up = self.accelerations[0]
        acc_dec = self.accelerations[1]
        acc_down = self.accelerations[2]
        acc_impact = self.accelerations[3]

        t_acc_up, t_acc_dec = footstep.calculate_acceleration_couple_times(acc_up, acc_dec,
                                                                           self.positive_footstep_displacement)
        t_acc_down, t_acc_impact = footstep.calculate_acceleration_couple_times(acc_down, acc_impact,
                                                                                -self.positive_footstep_displacement)

        # Tests
        self.assertAlmostEqual(t_acc_up, 0.4472135954999579)
        self.assertAlmostEqual(t_acc_dec, 0.223606797749979)
        self.assertAlmostEqual(t_acc_down, 0.23904572186687872)

    def test_set_accelerations(self):
        """
        Tests if the acceleration values are set correctly
        """

        test_accelerations = [9, -8, 7, -6]
        self.footstep._set_accelerations(test_accelerations)
        self.assertEqual(self.footstep.acc_up, 9)
        self.assertEqual(self.footstep.acc_dec, -8)
        self.assertEqual(self.footstep.acc_down, 7)
        self.assertEqual(self.footstep.acc_impact, -6)

    def test_set_acceleration_times(self):
        """
        Integration test if the acceleration times are set correctly.
        """

        self.assertAlmostEqual(self.footstep.t_acc_up, 0.4472135954999579)
        self.assertAlmostEqual(self.footstep.t_acc_dec, 0.223606797749979)
        self.assertAlmostEqual(self.footstep.t_acc_down, 0.2390457218668787)
        self.assertAlmostEqual(self.footstep.t_acc_impact, 0.17928429140015903)

    def test_get_footstep_acceleration(self):
        """
        Tests the `get_footstep_acceleration` method
        """
        t_before = -1
        t_up = 0.1
        t_dec = 0.5
        t_delay = 0.7
        t_down = 0.8
        t_impact = 1
        t_after = 5

        acc_before = self.footstep.get_acceleration(t_before)
        acc_up = self.footstep.get_acceleration(t_up)
        acc_dec = self.footstep.get_acceleration(t_dec)
        acc_delay = self.footstep.get_acceleration(t_delay)
        acc_down = self.footstep.get_acceleration(t_down)
        acc_impact = self.footstep.get_acceleration(t_impact)
        acc_after = self.footstep.get_acceleration(t_after)

        self.assertEqual(acc_before, 0)
        self.assertEqual(acc_up, 1)
        self.assertEqual(acc_dec, -2)
        self.assertEqual(acc_delay, 0)
        self.assertEqual(acc_down, -3)
        self.assertEqual(acc_impact, 4)
        self.assertEqual(acc_after, 0)
