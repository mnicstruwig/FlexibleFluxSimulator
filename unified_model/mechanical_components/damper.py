from unified_model.utils.utils import fetch_key_from_dictionary
from unified_model.utils.utils import pretty_str


class DamperSurfaceArea(object):
    """
    A constant damping coefficient damper that is tuned using a tuning parameter and the contact surface
    area of the magnet assembly
    """

    def __init__(self, magnet_assembly_surface_area, tuning_parameter=1):
        """
        Initialization
        :param magnet_assembly_surface_area: The contact surface area of the magnet assembly in mm^2.
        :param tuning_parameter: The tuning parameter to be used to scale the effect of the magnet_assembly_surface_Area
        """

        self.magnet_assembly_surface_area = magnet_assembly_surface_area
        self.tuning_parameter = tuning_parameter

    def get_force(self, velocity):
        """
        Return the force exerted by the damper
        :param velocity:  Velocity of the object attached to the damper in m/s
        :return: The force exerted by the damper in Newtons.
        """
        return self.magnet_assembly_surface_area * self.tuning_parameter * velocity


class ConstantDamper(object):
    """A constant-damping-coefficient damper.

    The force will be equal to the damping coefficient multiplied by a
    velocity, i.e. F = c * v.

    """

    def __init__(self, damping_coefficient):
        """Constructor

        Parameters
        ----------
        damping_coefficient : float
            The constant-damping-coefficient of the damper.

        """
        self.damping_coefficient = damping_coefficient

    def get_force(self, velocity):
        """Get the force exerted by the damper.

        Parameters
        ----------
        velocity : float
            Velocity of the object attached to the damper. In m/s.

        Returns
        -------
        float
            The force exerted by the damper. In Newtons.

        """
        return self.damping_coefficient * velocity

    def __repr__(self):
        return f'ConstantDamper({self.damping_coefficient})'

    def __str__(self):
        """Return string representation of the Damper."""
        return f"""ConstantDamper: {pretty_str(self.__dict__,1)}"""
