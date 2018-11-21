import unittest

# Local imports
from unified_model.mechanical_system.model import _enforce_boundary_condition

class TestOdeDecoupled(unittest.TestCase):
    """
    Test the `ode_decoupled` mechanical model.
    """
    def test_enforce_boundary_condition_met(self):
        """
        Test that boundary conditions are enforced when conditions are met.
        """
        test_boundary_condition = [0, 0, 2, 3]  # This must be enforced for the lower-limit.
        test_y = [-1, -2, 1, 4]
        expected_value = [0, 0, 2, 4]
        actual_value = _enforce_boundary_condition(test_boundary_condition, test_y, lower=True)

        self.assertEqual(expected_value, actual_value)

    def test_enforce_boundary_condition_not_met(self):
        """
        Test that boundary conditions are not enforced when the conditions are not met.
        """
        test_boundary_condition = [0, 0, 0, 0]
        test_y = [1, 1, 1, 1]  # Should not be changed, since not violating boundary
        actual_value = _enforce_boundary_condition(test_boundary_condition, test_y, lower=True)

        self.assertEqual(test_y, actual_value)

    def test_enforce_boundary_condition_no_cond(self):
        """
        Test case when no boundary conditions are specified.
        """
        test_boundary_conditions = [None, None, None, None]
        test_y = [-1, 0, 2, 3]
        actual_value = _enforce_boundary_condition(test_boundary_conditions, test_y, lower=True)

        self.assertEqual(test_y, actual_value)
