import numpy as np
from numpy.testing import assert_almost_equal

from unified_model.electrical_components.flux.model import InterpFluxModel


def generate_fake_flux_curve(z_arr, p_c=0):
    """Generate a fake flux curve"""
    return np.array([1 - np.tanh(z - p_c)**2 for z in z_arr])


class TestInterpFluxModel:
    """Test the InterpFluxModel class."""

    def test_1c_1m_flux(self):
        """Test the 1 coil, 1 magnet flux case for an interpolation model"""
        test_c = 1
        test_m = 1
        test_c_c = 3
        test_z_arr = np.linspace(-10, 10, 100)
        test_phi_values = generate_fake_flux_curve(test_z_arr)

        test_flux_model = InterpFluxModel(
            c=test_c,
            m=test_m,
            c_c=test_c_c
        )

        test_flux_model.fit(test_z_arr, test_phi_values)
        actual_result = [test_flux_model._flux_model(z)
                         for z in test_z_arr]
        expected_result = generate_fake_flux_curve(test_z_arr, p_c=test_c_c)
        assert_almost_equal(actual_result, expected_result, decimal=1)

    def test_1c_1m_dflux(self):
        """Test the 1 coil, 1 magnet dphi/dz case for an interpolation model"""
        test_c = 1
        test_m = 1
        test_c_c = 3
        test_z_arr = np.linspace(-10, 10, 200)
        test_phi_values = generate_fake_flux_curve(test_z_arr)

        test_flux_model = InterpFluxModel(
            c=test_c,
            m=test_m,
            c_c=test_c_c
        )

        test_flux_model.fit(test_z_arr, test_phi_values)
        actual_result = [test_flux_model._dflux_model(z) for z in test_z_arr]
        expected_phi = generate_fake_flux_curve(test_z_arr, p_c=test_c_c)
        expected_result = np.gradient(expected_phi)/np.gradient(test_z_arr)
        assert_almost_equal(actual_result, expected_result, decimal=1)

    def test_1c_2m_flux(self):
        """Test the 1 coil, 2 magnet case for an interpolation model"""
        test_c = 1
        test_m = 2  # Changed to 2
        test_c_c = 10
        test_l_mcd = 3
        test_z_arr = np.linspace(-10, 10, 100)
        test_phi_values = generate_fake_flux_curve(test_z_arr)

        test_flux_model = InterpFluxModel(
            c=test_c,
            m=test_m,
            c_c=test_c_c,
            l_mcd=test_l_mcd
        )

        test_flux_model.fit(test_z_arr, test_phi_values)

        expected_z_arr = np.linspace(0, 40, 1000)
        expected_result = (
            + generate_fake_flux_curve(expected_z_arr, p_c=test_c_c)
            - generate_fake_flux_curve(expected_z_arr,
                                       p_c=test_c_c + test_l_mcd)
        )
        actual_result = [test_flux_model._flux_model(z) for z in expected_z_arr]
        assert_almost_equal(actual_result, expected_result, decimal=1)

    def test_1c_2m_dflux(self):
        """Test the 1 coil, 2 magnet dphi/dz case for an interpolation model"""
        test_c = 1
        test_m = 2
        test_c_c = 10
        test_l_mcd = 3
        test_z_arr = np.linspace(-10, 10, 200)
        test_phi_values = generate_fake_flux_curve(test_z_arr)

        test_flux_model = InterpFluxModel(
            c=test_c,
            m=test_m,
            c_c=test_c_c,
            l_mcd=test_l_mcd
        )

        test_flux_model.fit(test_z_arr, test_phi_values)

        expected_z_arr = np.linspace(0, 40, 1000)
        expected_phi = (
            + generate_fake_flux_curve(expected_z_arr, p_c=test_c_c)
            - generate_fake_flux_curve(expected_z_arr,
                                       p_c=test_c_c + test_l_mcd)
        )
        expected_result = np.gradient(expected_phi)/np.gradient(expected_z_arr)
        actual_result = [test_flux_model._dflux_model(z) for z in expected_z_arr]
        assert_almost_equal(actual_result, expected_result, decimal=1)


    def test_2c_1m_flux(self):
        """Test the 2 coil, 1 magnet case for an interpolation model"""
        test_c = 2
        test_m = 1
        test_c_c = 10
        test_l_ccd = 3
        test_z_arr = np.linspace(-10, 10, 100)
        test_phi_values = generate_fake_flux_curve(test_z_arr)

        test_flux_model = InterpFluxModel(
            c=test_c,
            m=test_m,
            c_c=test_c_c,
            l_ccd=test_l_ccd
        )

        test_flux_model.fit(test_z_arr, test_phi_values)
        expected_z_arr = np.linspace(0, 40, 1000)
        expected_result = (
            generate_fake_flux_curve(expected_z_arr, p_c=test_c_c)
            - generate_fake_flux_curve(expected_z_arr, p_c=test_c_c + test_l_ccd)
        )
        actual_result = [test_flux_model._flux_model(z) for z in expected_z_arr]
        assert_almost_equal(actual_result, expected_result, decimal=1)

    def test_2c_2m_flux(self):
        """Test the 2 coil, 2 magnet case for an interpolation model"""
        test_c = 2
        test_m = 2
        test_c_c = 10
        test_l_ccd = 3
        test_l_mcd = 3
        test_z_arr = np.linspace(-10, 10, 100)
        test_phi_values = generate_fake_flux_curve(test_z_arr)

        test_flux_model = InterpFluxModel(
            c=test_c,
            m=test_m,
            c_c=test_c_c,
            l_ccd=test_l_ccd,
            l_mcd=test_l_mcd
        )

        test_flux_model.fit(test_z_arr, test_phi_values)
        expected_z_arr = np.linspace(0, 40, 1000)

        # Build the expected result
        expected_result = []
        for i in range(test_c):
            for j in range(test_m):
                p_c = test_c_c + i * test_l_ccd + j * test_l_mcd
                phi = (-1) ** (i + j) * generate_fake_flux_curve(expected_z_arr,
                                                                 p_c)
                expected_result.append(phi)
        expected_result = np.sum(expected_result, axis=0)

        actual_result = [test_flux_model._flux_model(z) for z in expected_z_arr]
        assert_almost_equal(actual_result, expected_result, decimal=1)
