from scipy import integrate
import numpy as np

from unified_model.utils.utils import fetch_key_from_dictionary, parse_output_expression
from unified_model.mechanical_system.model import ode_decoupled
from unified_model.governing_equations import unified_ode_coupled

MODEL_DICT = {'ode_decoupled': ode_decoupled,
              'unified_ode_coupled': unified_ode_coupled}


# TODO: Update documentation
def get_mechanical_model(model_dict, model):
    """Fetch mechanical model

    Parameters
    ----------
    model_dict : dict
        Dict containing the model name and function lookup.
    model : str
        Corresponding key in `model_dict` to lookup.

    Returns
    -------
    function
        The mechanical system model.

    """
    return fetch_key_from_dictionary(model_dict,
                                     model,
                                     "The mechanical model {} is not defined!".format(model))


# TODO: Add example once interface is more stable
class MechanicalModel:
    """A mechanical system of a kinetic microgenerator whose motion can be simulated.

    Attributes
    ----------
    model : function
        The mechanical system model.
    spring : obj
        The spring model attached to the magnet assembly and the tube.
    magnet_assembly : obj
        The magnet assembly model.
    damper : obj
        The damper model that represents losses in the mechanical system.
    input_ : obj
        The mechanical input that is applied to the system.
    initial_condition : array_like
        Initial conditions of the mechanical system model.
    raw_output : array_like
        Raw solution output returned by the numerical solver,
        `scipy.integrate.solve_ivp`.
    t : array_like
        Time values of solution output.
    electrical_system: obj
        Electrical system that has been coupled to the mechanical system.
    coupling : obj
        Coupling that models the interaction between the mechanical and attached
        electrical system.

    """

    def __init__(self):
        self.model = None
        self.spring = None
        self.magnet_assembly = None
        self.damper = None
        self.input_ = None
        self.results = None
        self.initial_conditions = None
        self.raw_output = None
        self.t = None
        self.electrical_system = None
        self.coupling = None

    def set_initial_conditions(self, initial_conditions):
        """Set the initial conditions of the mechanical system.

        Parameters
        ----------
        initial_conditions : (float,) array_like
            Initial conditions for each of the differential equations in the
            mechanical_system model.

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
        """Add a spring to the mechanical system

        Parameters
        ----------
        spring : obj
            The spring model attached to the magnet assembly and the tube.

        """
        self.spring = spring

    def set_damper(self, damper):
        """Add a damper to the mechanical system

        Parameters
        ----------
        damper : obj
            The damper model that represents losses in the mechanical system.

        """
        self.damper = damper

    def set_input(self, mechanical_input):
        """Add an input excitation to the mechanical system

        The `mechanical_input` object must implement a
        `get_acceleration(t)` method, where `t` is the current
        time step.

        Parameters
        ----------
        mechanical_input : obj
            The input excitation to add to the mechanical system.

        """
        self.input_ = mechanical_input

    def set_magnet_assembly(self, magnet_assembly):
        """Add a magnet assembly to the mechanical system.

        Parameters
        ----------
        magnet_assembly : obj
            The magnet assembly model.

        """
        self.magnet_assembly = magnet_assembly

    def attach_electrical_system(self, electrical_system, coupling):
        """Attach an electrical system.

        Parameters
        ----------
        electrical_model : obj
            Electrical system model to attach to the mechanical system
        coupling : obj
            Coupling that models the interaction between the mechanical
            and electrical system.

        """
        self.electrical_system = electrical_system
        self.coupling = coupling

    def _build_model_kwargs(self):
        """
        Build the model kwargs that will be used when running the simulation.
        """

        kwargs = {'spring': self.spring,
                  'damper': self.damper,
                  'input': self.input_,
                  'magnet_assembly': self.magnet_assembly,
                  'electrical_system': self.electrical_system,
                  'coupling': self.coupling}

        kwargs.update(self.additional_model_kwargs)

        return kwargs

    def solve(self, t_start, t_end, t_max_step=0.01):
        """Run the simulation.

        Parameters
        ----------
        t_start : float
            Time at which the simulation should begin.
        t_end : float
            Time at which the simulation should stop.
        t_max_step : float, optional.
            The maximum timestep the solver will take.

        """
        t_span = (t_start, t_end)
        model_kwargs = self._build_model_kwargs()

        psoln = integrate.solve_ivp(fun=lambda t, y: self.model(t, y, model_kwargs),
                          t_span=t_span,
                          y0=self.initial_conditions,
                          max_step=t_max_step)

        self.t = psoln.t
        self.raw_output = psoln.y

    # TODO: Update documentation
    def get_output(self, **kwargs):
        """Parse and evaluate expressions on the raw solution output.

        *Any* reasonable expression is possible. You can refer to each of the
        differential equations that represented by the mechanical system model
        using the letter 'x' with the number appended. For example `x1` refers
        to the first differential equation, `x2` to the second, etc.

        Each expression is available as a column in the returned pandas
        dataframe, with the column name being the key of the kwarg used.

        Parameters
        ----------
        **kwargs
            Each key is the name of the column of the returned dataframe.
            Each value is the expression to be evaluated.

        Returns
        -------
        pandas dataframe
            Output dataframe containing the evaluated expressions.

        See Also
        --------
        _parse_output_expression : helper function that contains the parsing
            logic.

        Example
        --------
        >>> ms = MechanicalModel()
        >>> raw_output =  np.array([[1, 2, 3, 4, 5], [1, 1, 1, 1, 1]])
        >>> print(raw_output)
        [[1 2 3 4 5]
        [1 1 1 1 1]]
        >>> ms.raw_output = raw_output
        >>> df_output = ms.get_output(an_expression='x1-x2', another_expr='x1*x1')
        >>> print(df_output)
           an_expression  another_expr
        0              0             1
        1              1             4
        2              2             9
        3              3            16
        4              4            25

        """
        return parse_output_expression(self.t, self.raw_output, **kwargs)
