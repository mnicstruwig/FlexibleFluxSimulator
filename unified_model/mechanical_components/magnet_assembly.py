import numpy as np
from unified_model.utils.utils import fetch_key_from_dictionary

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
    The magnet assembly class.

    Note: Only used for weight-based calculations.

        Parameters
        ----------
        n_magnet : int
            Number of magnets.
        l_m_mm : float
            Height of the magnets in mm.
        l_mcd_mm : float
            Distance between the centers of each magnet, in mm.
        dia_magnet_mm: float
            Diameter of magnets in mm.
        dia_spacer_mm : float
            Diameter of spacer in mm
        mat_magnet : str
            Magnet material key. Optional.
        mat_spacer: str
            Spacer material key. Optional.

    """

    def __init__(
            self,
            m: int,
            l_m_mm: float,
            l_mcd_mm: float,
            dia_magnet_mm: float,
            dia_spacer_mm: float,
            mat_magnet='NdFeB',
            mat_spacer='iron'):
        """Constructor"""

        self.m = m
        self.l_m_mm = l_m_mm
        self.l_mcd_mm = l_mcd_mm
        self.dia_magnet_mm = dia_magnet_mm
        self.dia_spacer_mm = dia_spacer_mm
        self.weight = None
        self.surface_area = None
        self.density_magnet = _get_material_density(MATERIAL_DICT, mat_magnet)
        self.density_spacer = _get_material_density(MATERIAL_DICT, mat_spacer)
        self.weight = self._calculate_weight()

    def __repr__(self):
        to_print_dict = {
            'n_magnet': self.m,
            'l_m_mm': self.l_m_mm,
            'l_mcd_mm': self.l_mcd_mm,
            'dia_magnet_mm': self.dia_magnet_mm,
            'dia_spacer_mm': self.dia_spacer_mm
        }
        to_print = ', '.join([f'{k}={v}' for k, v in to_print_dict.items()])
        return f'MagnetAssembly({to_print})'

    @staticmethod
    def _calc_volume_cylinder(diameter, length):
        return np.pi*(diameter/2)**2*length

    def _calculate_weight(self):
        """Calculate the weight of the magnet assembly."""
        volume_magnet = self._calc_volume_cylinder(self.dia_magnet_mm, self.l_m_mm)
        volume_spacer = self._calc_volume_cylinder(self.dia_spacer_mm, self.l_mcd_mm)

        weight_magnet = volume_magnet * self.density_magnet * 9.81
        weight_spacer = volume_spacer * self.density_spacer * 9.81

        return (
            self.m
            * weight_magnet
            + (self.m - 1) * weight_spacer
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
            self.m
            * self.l_m_mm
            + (self.m - 1) * self.l_mcd_mm / 1000
    )
