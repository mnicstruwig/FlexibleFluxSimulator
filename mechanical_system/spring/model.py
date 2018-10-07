import numpy as np


def coulombs_law(z, m):
    """
    Coulomb's Law for two identical monopoles
    :param z: The distance between two identical monopoles.
    :param m: The size of the monopole charges.
    :return: The force between the two monopoles at distance `z`
    """
    u0 = 4 * np.pi * 10 ** (-7)
    return u0 * m * m / (4 * np.pi * z * z)


def coulombs_law_modified(z, g, numerator):
    """
    Modified version of Coulomb's Law for two identical monopoles with an added `G` term to the denominator.
    :param z: The distance between two identical monopoles
    :param g: The additional parameter in the denominator
    :param numerator: The numerator representing the multiplication of u0 and the monopole charges.
    :return:
    """
    return numerator / (np.pi * 4 * z ** 2 + g)


def power_series_2(z, a0, a1, a2):
    """
    Power series representation with maximum degree of 2
    :param z:
    :param a0:
    :param a1:
    :param a2:
    :return:
    """
    return a0 + a1 * z + a2 * z * z


def power_series_3(z, a0, a1, a2, a3):
    """
    Power series representation
    :param z:
    :param a0:
    :param a1:
    :param a2:
    :param a3:
    :return:
    """
    # d0 = 0.040150

    part0 = a0
    part1 = a1 * z
    part2 = a2 * z ** 2
    part3 = a3 * z ** 3
    return part0 + part1 + part2 + part3
