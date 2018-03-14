import numpy as np
from utils.utils import fetch_key_from_dictionary
from mechanical_system.magnet_assembly.utils import calc_volume_cylinder

MATERIAL_DICT = {'NdFeB': 7.5e-6,
                 'iron': 7.5e-6}


def _get_material_density(material_dict, material_key):
    """
    Gets a material from the material dictionary
    :param material_dict: Material density dictionary
    :param material_key: Material key
    :return: Density of material in kg/mm^3
    """
    return fetch_key_from_dictionary(material_dict, material_key, 'Material not found!')


class MagnetAssembly(object):
    """
    The magnet assembly class
    """

    def __init__(self, n_magnet, h_magnet, h_spacer, dia_magnet, dia_spacer, mat_magnet='NdFeB', mat_spacer='iron'):
        """
        Initialization of the magnet assembly.
        :param n_magnet: Number of magnets
        :param h_magnet: Height of magnets in mm
        :param h_spacer: Height of spacer in mm
        :param dia_magnet: Diameter of magnets in mm
        :param dia_spacer: Diameter of spacer in mm
        :param mat_magnet: Magnet material key
        :param mat_spacer: Spacer material key
        :return: A magnet assembly object
        """
        self.n_magnet = n_magnet

        self.h_magnet = h_magnet
        self.h_spacer = h_spacer

        self.dia_magnet = dia_magnet
        self.dia_spacer = dia_spacer

        self.weight = None

        self.density_magnet = _get_material_density(MATERIAL_DICT, mat_magnet)
        self.density_spacer = _get_material_density(MATERIAL_DICT, mat_spacer)

        self._set_weight()

    def _set_weight(self, new_weight=None):
        """
        Sets the weight of the magnet assembly
        :param new_weight: The new weight of the magnet assembly. Set to `None` to automatically calculate from
        parameters.
        """
        if new_weight is not None:
            self.weight = new_weight
        else:
            self.weight = self._calculate_weight()

    def _calculate_weight(self):
        """
        Calculate the weight of the magnet assembly from the parameters
        Return: The weight of the magnet assembly
        """
        volume_magnet = calc_volume_cylinder(self.dia_magnet, self.h_magnet)
        volume_spacer = calc_volume_cylinder(self.dia_spacer, self.h_spacer)

        weight_magnet = volume_magnet * self.density_magnet * 9.81
        weight_spacer = volume_spacer * self.density_spacer * 9.81

        return self.n_magnet * weight_magnet + (self.n_magnet - 1) * weight_spacer

    def get_mass(self):
        """
        Get the mass of the magnet assembly
        """
        return self.weight / 9.81

    def get_weight(self):
        """
        Returns the weight of the magnet assembly
        """
        return self.weight
