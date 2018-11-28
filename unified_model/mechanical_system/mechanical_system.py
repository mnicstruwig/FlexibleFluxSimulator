from asteval import Interpreter
from scipy.integrate import odeint, solve_ivp
import pandas as pd

from unified_model.utils.utils import fetch_key_from_dictionary
from unified_model.mechanical_system.model import ode_decoupled
from unified_model.model import unified_ode_coupled

MODEL_DICT = {'ode_decoupled': ode_decoupled,
              'unified_ode_coupled': unified_ode_coupled}


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
        self.t = None

    def set_initial_conditions(self, initial_conditions):
        """
        Sets the initial conditions for the model.
        :param initial_conditions: list of initial conditions for each of the differential equations
                                   of the model. [float_1, float_2, float_3, ...]
        """
        self.initial_conditions = initial_conditions

    def set_model(self, model, **additional_model_kwargs):
        """
        Set the model of the mechanical system

        Parameters
        ----------
        model : string
            The mechanical model to use as the model for the mechanical system.
            Current options: {'ode_decoupled', 'unified_ode_coupled'}
        **additional_model_kwargs
            Additional keyword arguments to pass to the model object at simulation time.

        Returns
        -------
        None
        """
        self.model = get_mechanical_model(MODEL_DICT, model)
        self.additional_model_kwargs = additional_model_kwargs

    def set_spring(self, spring):
        """
        Add a spring to the mechanical system
        """
        self.spring = spring

    def set_damper(self, damper):
        """
        Add a damper to the mechanical system
        """
        self.damper = damper

    def set_input(self, mechanical_input):
        """
        Add an input excitation to the mechanical system

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
        Add a magnet assembly to the mechanical system
        """
        self.magnet_assembly = magnet_assembly

    def attach_electrical_model(self, electrical_model):
        """
        Add a flux model to the system.
        """
        self.electrical_model = electrical_model

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

    def solve(self, t_start, t_end, t_max_step=0.01):
        """
        Run the simulation
        """
        t_span = (t_start, t_end)
        model_kwargs = self._build_model_kwargs()

        psoln = solve_ivp(fun=lambda t, y: self.model(t, y, model_kwargs),
                          t_span = t_span,
                          y0 = self.initial_conditions,
                          max_step = t_max_step)

        self.raw_output = psoln.y
        self.t = psoln.t

    # TODO: Write test
    def _parse_output_expression(self, **kwargs):
        df_out = pd.DataFrame()

        def _populate_asteval_symbol_table(aeval):
            aeval.symtable['t'] = self.t
            for i in range(self.raw_output.shape[0]):
                aeval.symtable['x' + str(i+1)] = self.raw_output[i, :]
            return aeval

        aeval = Interpreter()
        aeval = _populate_asteval_symbol_table(aeval)

        for key, expr in kwargs.items():
            df_out[key] = aeval(expr)

        return df_out

    # TODO: Update documentation
    def get_output(self, **kwargs):
        """
        Return the output of the simulation as a dataframe.
        :return: The output as a dataframe
        """
        return self._parse_output_expression(**kwargs)
