import numpy as np


def calc_volume_cylinder(diameter, height):
    """
    Calculate the volume of a cylinder
    :param diameter: The cylinder diameter
    :param height: The cylinder height
    :return: The volume of the cylinder
    """
    radius = diameter / 2
    return np.pi*radius*radius*height

