from scipy.integrate import odeint
import pandas as pd

from unified_model.utils.utils import fetch_key_from_dictionary
from unified_model.mechanical_system.model import ode_decoupled

MODEL_DICT = {'ode_decoupled': ode_decoupled}


def get_mechanical_model(model_dict, model):
    """
    Fetches the mechanical model
    :param model_dict:  Dictionary containing the model definitions {'key' : model_function}
    :param model: The key for the model to load from `model_dict`
    :return: The mechanical system model
    """
    return fetch_key_from_dictionary(model_dict, model, "The mechanical model {} is not defined!".format(model))


class MechanicalSystem(object):
    """
    The mechanical system object that can be used to simulate the mechanical system
    """

    def __init__(self):
        self.model = None
        self.spring = None
        self.magnet_assembly = None
        self.damper = None
        self.input = None
        self.results = None
        self.initial_conditions = None
        self.raw_output = None
        self.output_time_steps = None

    def set_initial_conditions(self, initial_conditions):
        """
        Sets the initial conditions for the model.
        :param initial_conditions: list of initial conditions for each of the differential equations
                                   of the model. [float_1, float_2, float_3, ...]
        """
        self.initial_conditions = initial_conditions

    def set_model(self, model, initial_conditions=None, **additional_model_kwargs):
        """
        Set the model of the mechanical system

        Parameters
        ----------
        model : cls
            The mechanical model class to use as the model for the mechanical system.
        initial_conditions : list
            A list containing the initial conditions for the mechanical model.
        **additional_model_kwargs
            Additional keyword arguments to pass to the model object at simulation time.

        Returns
        -------
        None
        """
        self.model = get_mechanical_model(MODEL_DICT, model)

        self.additional_model_kwargs = additional_model_kwargs

        if initial_conditions is not None:
            self.set_initial_conditions(initial_conditions)

    def set_spring(self, spring):
        """
        Adds a spring to the mechanical system
        """
        self.spring = spring

    def set_damper(self, damper):
        """
        Adds a damper to the mechanical system
        """
        self.damper = damper

    def set_input(self, mechanical_input):
        """
        Adds an input excitation to the mechanical system

        The `mechanical_input` object must implement a
        `get_acceleration(t)` method, where `t` is the current
        time step.

        Parameters
        ----------
        mechanical_input : obj
            The input excitation to add to the mechanical system.
        """
        self.input = mechanical_input

    def set_magnet_assembly(self, magnet_assembly):
        """
        Adds a magnet assembly to the mechanical system
        """
        self.magnet_assembly = magnet_assembly

    def _build_model_kwargs(self):
        """
        Build the model kwargs that will be used when running the simulation.
        """

        kwargs = {'spring': self.spring,
                  'damper': self.damper,
                  'input': self.input,
                  'magnet_assembly': self.magnet_assembly}

        kwargs.update(self.additional_model_kwargs)

        return kwargs

    def solve(self, t_array):
        """
        Run the simulation
        """
        self.output_time_steps = t_array
        t_step = t_array[1] - t_array[0]
        model_kwargs = self._build_model_kwargs()
        psoln = odeint(func=self.model,
                       y0=self.initial_conditions,
                       t=t_array,
                       args=(model_kwargs,),
                       hmin=t_step,
                       hmax=t_step)

        self.raw_output = psoln

    def get_output(self):
        """
        Return the output of the simulation as a dataframe.
        :return: The output as a dataframe
        """
        x1 = self.raw_output[:, 0]  # Tube displacement
        x2 = self.raw_output[:, 1]  # Tube velocity
        x3 = self.raw_output[:, 2]  # Magnet assembly displacement
        x4 = self.raw_output[:, 3]  # Magnet assembly velocity

        x_relative_displacement = x3 - x1
        x_relative_velocity = x4 - x2

        output = pd.DataFrame()
        output['time'] = self.output_time_steps
        output['tube_displacement'] = x1
        output['tube_velocity'] = x2
        output['assembly_displacement'] = x3
        output['assembly_velocity'] = x4
        output['assembly_relative_displacement'] = x_relative_displacement
        output['assembly_relative_velocity'] = x_relative_velocity

        return output
