from unified_model.mechanical_system.model import ode_decoupled
from unified_model.governing_equations import unified_ode_coupled

MODEL_DICT = {'ode_decoupled': ode_decoupled,
              'unified_ode_coupled': unified_ode_coupled}


# TODO: Add example once interface is more stable
class MechanicalModel:
    """A mechanical model of a kinetic microgenerator whose motion can be simulated.

    Attributes
    ----------
    name : str
        String identifier of the mechanical model.
    spring : obj
        The spring model attached to the magnet assembly and the tube.
    magnet_assembly : obj
        The magnet assembly model.
    damper : obj
        The damper model that represents losses in the mechanical system.
    input_ : obj
        The mechanical input that is applied to the system.
    raw_output : array_like
        Raw solution output returned by the numerical solver,
        `scipy.integrate.solve_ivp`.
    t : array_like
        Time values of solution output.
    electrical_system: obj
        Electrical system that has been coupled to the mechanical system.

    """

    def __init__(self, name):
        """Constructor

        Parameters
        ----------
        name : str
            String identifier of the mechanical model.

        """
        self.name = name
        self.spring = None
        self.magnet_assembly = None
        self.damper = None
        self.input_ = None
        self.results = None
        self.raw_output = None
        self.t = None
        self.electrical_system = None

    def __str__(self):
        def pretty_str(dict_):
            str_ = ""
            for key, val in dict_.items():
                str_ = str_ + f"{key} :" + str(val) + "\n"
            return str_
        return "Mechanical Model:\n" + pretty_str(self.__dict__)

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
