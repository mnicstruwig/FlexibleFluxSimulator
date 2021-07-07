import os
from unittest.mock import patch

from scipy.signal import savgol_filter

from unified_model.electrical_components.coil import CoilConfiguration
from unified_model.mechanical_components.magnet_assembly import MagnetAssembly
from unified_model.mechanical_components.magnetic_spring import MagneticSpringInterp

CUR_DIR = os.path.dirname(os.path.abspath(__file__))


def test_coil_calc_hover_height():
    test_coil_config = CoilConfiguration(
        c=1,
        n_z=68,
        n_w=20,
        l_ccd_mm=10,
        ohm_per_mm=2,
        tube_wall_thickness_mm=2,
        coil_wire_radius_mm=0.1,
        coil_center_mm=50,
        inner_tube_radius_mm=5,
    )

    test_magnet_assembly = MagnetAssembly(
        m=2, l_m_mm=10, l_mcd_mm=17, dia_magnet_mm=10, dia_spacer_mm=10
    )

    test_fea_data_file_path = os.path.join(
        CUR_DIR, "../mechanical_components/test_data/magnetic_spring_fea_data.csv"
    )

    test_magnetic_spring = MagneticSpringInterp(
        fea_data_file=test_fea_data_file_path,
        magnet_length=10 / 1000,
        filter_callable=lambda x: savgol_filter(x, window_length=27, polyorder=5),
    )

    actual_hover_height = test_coil_config._calc_hovering_height(  # noqa
        magnet_assembly=test_magnet_assembly, magnetic_spring=test_magnetic_spring
    )

    print(test_magnet_assembly.get_length())
    test_coil_config.set_optimal_coil_center(test_magnet_assembly, test_magnetic_spring)
    print(test_coil_config.coil_center_mm)
    print(actual_hover_height)

    # expected_hover_height = 0.0468 * 1000

    # assert actual_hover_height == pytest.approx(expected_hover_height, 0.1)


def test_coil_set_optimal_coil_center():

    test_n_z = 100
    test_coil_wire_radius_mm = 0.05

    test_coil_config = CoilConfiguration(
        c=1,
        n_z=test_n_z,
        n_w=10,
        l_ccd_mm=10,
        ohm_per_mm=2,
        tube_wall_thickness_mm=2,
        coil_wire_radius_mm=test_coil_wire_radius_mm,
        coil_center_mm=50,
        inner_tube_radius_mm=5,
    )

    test_magnet_assembly = MagnetAssembly(
        m=1, l_m_mm=10, l_mcd_mm=10, dia_magnet_mm=10, dia_spacer_mm=10
    )

    test_fea_data_file_path = os.path.join(
        CUR_DIR, "../mechanical_components/test_data/magnetic_spring_fea_data.csv"
    )

    test_magnetic_spring = MagneticSpringInterp(
        fea_data_file=test_fea_data_file_path,
        magnet_length=10 / 1000,
        filter_callable=lambda x: savgol_filter(x, window_length=27, polyorder=5),
    )

    with patch(
        "unified_model.electrical_components.coil.CoilConfiguration._calc_hovering_height"  # noqa
    ) as mock_calc_hovering_height:  # noqa
        test_hovering_height = 10
        mock_calc_hovering_height.return_value = test_hovering_height

        test_coil_config.set_optimal_coil_center(
            magnet_assembly=test_magnet_assembly, magnetic_spring=test_magnetic_spring
        )

        expected_coil_center = (
            test_hovering_height
            + test_magnet_assembly.get_length()
            + 5  # l_epsilon
            + test_n_z
            * test_coil_wire_radius_mm
            * 2
            / 2  # coil height, divided by 2 (for clarity)
        )

        assert test_coil_config.coil_center_mm == expected_coil_center
