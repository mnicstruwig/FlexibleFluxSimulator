import unittest

# Local imports
from mechanical_system.footstep import footstep


class TestFootstep(unittest.TestCase):
    def setUp(self):
        """
        Set-up for the `TestFootstep` class
        """

        self.accelerations = [1., -2., 3., -4.]
        self.acceleration_couple_time_separation = 0.05
        self.positive_footstep_displacement = 0.15
        self.footstep = footstep.Footstep(self.accelerations, self.acceleration_couple_time_separation, self.positive_footstep_displacement, )

    def test_calculate_acceleration_couple_times(self):
        """
        Tests the `_calculate_acceleration_couple_times` function
        """

        acc_up = self.accelerations[0]
        acc_dec = self.accelerations[1]
        acc_down = self.accelerations[2]
        acc_impact = self.accelerations[3]
        t_acc_up, t_acc_dec = self.footstep \
            ._calculate_acceleration_couple_times(acc_up, acc_dec,
                                                  self.positive_footstep_displacement)
        t_acc_down, t_acc_impact = self.footstep \
            ._calculate_acceleration_couple_times(acc_down, acc_impact,
                                                  self.positive_footstep_displacement)

        # Tests
        self.assertAlmostEqual(t_acc_up, 0.4472135954999579)
        self.assertAlmostEqual(t_acc_dec, 0.223606797749979)
        self.assertAlmostEqual(t_acc_down, 0.23904572186687872)
        self.assertAlmostEqual(t_acc_impact, 0.17928429140015903)
