from unified_model.utils.utils import pretty_str


# TODO: Add example once interface is more stable
class MechanicalModel:
    """A mechanical model of a kinetic microgenerator whose motion can be simulated.

    Attributes
    ----------
    name : str
        String identifier of the mechanical model.
    max_height : float
        The maximum height of the microgenerator (in metres). The magnet
        assembly cannot exceed this limit.
    magnetic_spring : obj
        The magnetic spring model that is attached to the magnet assembly
        and the tube.
    mechanical_spring : obj
        The mechanical spring model that is attached to the tube.
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

    """

    def __init__(self, name):
        """Constructor

        Parameters
        ----------
        name : str
            String identifier of the mechanical model.

        """
        self.name = name
        self.max_height = None
        self.magnetic_spring = None
        self.mechanical_spring = None
        self.magnet_assembly = None
        self.damper = None
        self.input_ = None
        self.t = None

    def __str__(self):
        """Return string representation of the MechanicalModel"""
        return "Mechanical Model:\n" + pretty_str(self.__dict__)

    def set_max_height(self, max_height):
        """Set the maximum height of the microgenerator tube.

        Parameters
        ----------
        max_height : float
            The maximum height of the microgenerator tube.

        """
        self.max_height = max_height

    def set_magnetic_spring(self, spring):
        """Add a magnetic spring to the mechanical system.

        Parameters
        ----------
        spring : obj
            The magnetic spring model attached to the magnet assembly and the
            tube.

        """
        self.magnetic_spring = spring

    def set_mechanical_spring(self, spring):
        """Add a mechanical spring to the mechanical system.

        Parameters
        ----------
        spring : obj
            The mechanical spring model attached to the magnet assembly and the
            tube.

        """
        self.mechanical_spring = spring

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
