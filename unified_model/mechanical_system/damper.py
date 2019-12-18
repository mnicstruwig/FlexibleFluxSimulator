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
    """
    A constant damping coefficient damper
    """

    def __init__(self, damping_coefficient):
        """
        Constructor
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

    def __str__(self):
        """Return string representation of the Damper."""
        return f"""ConstantDamper: {pretty_str(self.__dict__,1)}"""


DAMPER_DICT = {
    'constant': ConstantDamper
}


def _get_damper_model(damper_dict, damper_key):
    return fetch_key_from_dictionary(damper_dict, damper_key, "The damper is not defined!")


# TODO: Reformat docstrings
class Damper(object):
    """
    The Damper class
    """

    def __init__(self, model, model_kwargs):
        """
        Initialization
        :param model: The model name as a string. Currently supported: 'constant'
        :param model_kwargs: The model kwargs as a dictionary {parameter_key : value}
        """
        self._set_model(model, model_kwargs)
        self._model_kwargs = model_kwargs

    def _set_model(self, model, model_kwargs):
        """
        Initializes and sets the damper model.
        :param model: The model name as a string.
        :param model_kwargs: The model kwargs as a dictionary
        """
        self.model = _get_damper_model(DAMPER_DICT, model)
        self.model = self.model(**model_kwargs)

    def get_force(self, velocity):
        """
        Return the force generated by the damper
        :param velocity: The velocity of the object attached to the damper
        :return: The force generated by the damper in Newtons
        """
        return self.model.get_force(velocity)


