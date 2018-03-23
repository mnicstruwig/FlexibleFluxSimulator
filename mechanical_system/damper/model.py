class DamperConstant(object):
    """
    A constant damping coefficient damper
    """

    def __init__(self, damping_coefficient):
        """
        Initialization
        :param damping_coefficient: The constant damping-coefficient of the damper
        """
        self.damping_coefficient = damping_coefficient

    def get_force(self, velocity):
        """
        Return the force exerted by the damper
        :param velocity: Velocity of the object attached to the damper in m/s
        :return: The force exerted by the damper in Newtons.
        """
        return self.damping_coefficient * velocity


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
