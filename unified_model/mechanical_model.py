import warnings

from unified_model.utils.utils import pretty_str
from unified_model.local_exceptions import ModelError
from .mechanical_components.mechanical_spring import MechanicalSpring


# TODO: Add example once interface is more stable
class MechanicalModel:
    """A mechanical model of a kinetic microgenerator whose motion can be simulated.

    Attributes
    ----------
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

    """

    def __init__(self):
        """Constructor"""
        self.magnetic_spring = None
        self.mechanical_spring = None
        self.magnet_assembly = None
        self.damper = None
        self.input_ = None

    def __str__(self):
        """Return string representation of the MechanicalModel"""
        return f"""Mechanical Model: {pretty_str(self.__dict__, 1)}"""

    def _validate(self):
        """Validate the mechanical model.

        Do some basic checks to make sure mandatory components have been set.
        """
        try:
            assert self.magnetic_spring is not None
        except AssertionError:
            raise ModelError("A magnetic spring model must be specified.")  # noqa
        try:
            assert self.mechanical_spring is not None
        except AssertionError:
            warnings.warn(
                """A mechanical spring, simulating the impact dynamics
                          of the magnet assembly impacting the top of the
                          microgenerator device, has not been set. The resulting
                          simulation will assume an infinitely-long
                          microgenerator tube."""
            )
        try:
            assert self.damper is not None
        except AssertionError:
            raise ModelError("A friction damper model must be specified.")
        try:
            assert self.magnet_assembly is not None
        except AssertionError:
            raise ModelError("A magnet assembly model must be specified.")
        try:
            assert self.input_ is not None
        except AssertionError:
            raise ModelError("An input excitation must be specified.")

    def set_magnetic_spring(self, spring: MechanicalSpring):
        """Add a magnetic spring to the mechanical system.

        Parameters
        ----------
        spring : MagneticSpring
            The magnetic spring model attached to the magnet assembly and the
            tube.

        """
        self.magnetic_spring = spring
        return self

    def set_mechanical_spring(self, spring):
        """Add a mechanical spring to the mechanical system.

        Parameters
        ----------
        spring : obj
            The mechanical spring model attached to the magnet assembly and the
            tube.

        """
        self.mechanical_spring = spring
        return self

    def set_damper(self, damper):
        """Add a damper to the mechanical system

        Parameters
        ----------
        damper : obj
            The damper model that represents losses in the mechanical system.

        """
        self.damper = damper
        return self

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
        return self

    def set_magnet_assembly(self, magnet_assembly):
        """Add a magnet assembly to the mechanical system.

        Parameters
        ----------
        magnet_assembly : obj
            The magnet assembly model.

        """
        self.magnet_assembly = magnet_assembly
        return self

    def verify(self):
        """Verify the mechanical model, and raise any warnings.

        Note: This only raises warnings, it does _not_ modify
        the mechanical model.
        """
        # Make sure not attributes are None
        for k, v in self.__dict__.items():
            if v is None:
                warnings.warn(f"{k} has not been defined!")
