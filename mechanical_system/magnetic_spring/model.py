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


def coulombs_law_modified(z, G, numerator):
    """
    Modified version of Coulomb's Law for two identical monopoles with an added `G` term to the denominator.
    :param z: The distance between two identical monopoles
    :param G: The additional parameter in the denominator
    :param numerator: The numerator representing the multiplication of u0 and the monopole charges.
    :return:
    """
    return numerator / (np.pi * 4 * z ** 2 + G)
