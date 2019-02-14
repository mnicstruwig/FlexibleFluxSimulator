import numpy as np


def calc_contact_surface_area_cylinder(diameter, height):
    """
    Calculate the outer surface area of a cylinder, excluding the top caps
    :param diameter: Diameter of the cylinder
    :param height: Height of the cylinder
    :return: The outer surface area of the cylinder
    """

    return np.pi * diameter * height


def calc_volume_cylinder(diameter, height):
    """
    Calculate the volume of a cylinder
    :param diameter: The cylinder diameter
    :param height: The cylinder height
    :return: The volume of the cylinder
    """
    radius = diameter / 2
    return np.pi * radius * radius * height

