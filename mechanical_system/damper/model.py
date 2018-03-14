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

