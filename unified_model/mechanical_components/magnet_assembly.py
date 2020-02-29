from unified_model.utils.utils import fetch_key_from_dictionary
from unified_model.mechanical_components.utils import \
    calc_contact_surface_area_cylinder, calc_volume_cylinder

MATERIAL_DICT = {'NdFeB': 7.5e-6,
                 'iron': 7.5e-6}


def _get_material_density(material_dict, material_key):
    """
    Gets a material from the material dictionary
    :param material_dict: Material density dictionary
    :param material_key: Material key
    :return: Density of material in kg/mm^3
    """
    return fetch_key_from_dictionary(
        material_dict,
        material_key,
        'Material not found!'
    )


class MagnetAssembly:
    """
    The magnet assembly class
    """

    def __init__(
            self,
            n_magnet: int,
            h_magnet: float,
            h_spacer: float,
            dia_magnet: float,
            dia_spacer: float,
            mat_magnet='NdFeB',
            mat_spacer='iron'):
        """
        Constructor.

        Parameters
        ----------
        n_magnet : int
            Number of magnets.
        h_magnet : float
            Height of the magnets in mm.
        h_spacer : float
            Height of spacer in mm.
        dia_magnet: float
            Diameter of magnets in mm.
        dia_spacer : float
            Diameter of spacer in mm
        mat_magnet : str
            Magnet material key.
        mat_spacer: str
            Spacer material key.

        """
        self.n_magnet = n_magnet
        self.h_magnet = h_magnet
        self.h_spacer = h_spacer
        self.dia_magnet = dia_magnet
        self.dia_spacer = dia_spacer
        self.weight = None
        self.surface_area = None
        self.density_magnet = _get_material_density(MATERIAL_DICT, mat_magnet)
        self.density_spacer = _get_material_density(MATERIAL_DICT, mat_spacer)
        self.weight = self._calculate_weight()
        self.surface_area = self._calculate_contact_surface_area()

    def _calculate_weight(self):
        """Calculate the weight of the magnet assembly."""
        volume_magnet = calc_volume_cylinder(self.dia_magnet, self.h_magnet)
        volume_spacer = calc_volume_cylinder(self.dia_spacer, self.h_spacer)

        weight_magnet = volume_magnet * self.density_magnet * 9.81
        weight_spacer = volume_spacer * self.density_spacer * 9.81

        return (
            self.n_magnet
            * weight_magnet
            + (self.n_magnet - 1) * weight_spacer
        )

    def _calculate_contact_surface_area(self):
        """Calculate the contact surface area of the magnet assembly."""
        surface_area_magnet = calc_contact_surface_area_cylinder(
            self.dia_magnet, self.h_magnet
        )
        surface_area_spacer = calc_contact_surface_area_cylinder(
            self.dia_spacer, self.h_spacer
        )

        return (
            self.n_magnet
            * surface_area_magnet
            + (self.n_magnet - 1) * surface_area_spacer
            )

    def get_mass(self):
        """Get the mass of the magnet assembly."""
        return self.weight / 9.81

    def get_weight(self):
        """Get the weight of the magnet assembly."""
        return self.weight

    def get_contact_surface_area(self):
        """Get the contact surface area of the magnet assembly in mm^2."""
        return self.surface_area

    # TODO: Add test
    def get_height(self):
        """Get the height of the magnet assembly in m."""
        return (
            self.n_magnet
            * self.h_magnet
            + (self.n_magnet - 1) * self.h_spacer / 1000
    )
