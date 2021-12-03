import os

from unified_model.coupling import CouplingModel
from unified_model.electrical_components.coil import CoilConfiguration
from unified_model.electrical_components.flux.model import (
    FluxModelInterp, FluxModelPretrained)
from unified_model.electrical_components.load import SimpleLoad
from unified_model.governing_equations import unified_ode
from unified_model.mechanical_components import (MagnetAssembly,
                                                 MagneticSpringInterp,
                                                 MassProportionalDamper,
                                                 MechanicalSpring,
                                                 magnet_assembly)
from unified_model.mechanical_components.input_excitation.accelerometer import \
    AccelerometerInput
from unified_model.unified import UnifiedModel

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(CURRENT_DIR, "test_data")

TEST_LOG_ACC_PATH_1 = os.path.join(TEST_DATA_DIR, "test_log_acc_1.csv")
TEST_LOG_ADC_PATH_1 = os.path.join(TEST_DATA_DIR, "test_log_adc_1.csv")
TEST_LOG_VIDEO_LABELS_PATH_1 = os.path.join(TEST_DATA_DIR, "test_log_video_labels_1.csv")

TEST_LOG_ACC_PATH_2 = os.path.join(TEST_DATA_DIR, "test_log_acc_2.csv")
TEST_LOG_ADC_PATH_2 = os.path.join(TEST_DATA_DIR, "test_log_adc_2.csv")
TEST_LOG_VIDEO_LABELS_PATH_2 = os.path.join(TEST_DATA_DIR, "test_log_video_labels_2.csv")

TEST_MAG_SPRING_DATA_PATH = os.path.join(TEST_DATA_DIR, "test_mag_spring_data.csv")
TEST_PRETRAINED_CURVE_MODEL_PATH = os.path.join(
    TEST_DATA_DIR, "test_curve_model_pretrained.model"
)


def build_unified_model_with_standard_components() -> UnifiedModel:
    """Create a `UnifiedModel` with all standard components attached."""
    input_excitation = AccelerometerInput(
        raw_accelerometer_data_path=TEST_LOG_ACC_PATH_1,
        accel_column="z_G",
        time_column="time(ms)",
        accel_unit="g",
        time_unit="ms",
        smooth=True,
        interpolate=True,
    )

    mag_assembly = MagnetAssembly(
        m=1, l_m_mm=10, l_mcd_mm=0, dia_magnet_mm=10, dia_spacer_mm=10
    )

    mag_spring = MagneticSpringInterp(
        fea_data_file=TEST_MAG_SPRING_DATA_PATH,
        magnet_assembly=mag_assembly,
    )

    mech_spring = MechanicalSpring(magnet_assembly=mag_assembly, damping_coefficient=1)

    mech_damper = MassProportionalDamper(
        damping_coefficient=1, magnet_assembly=mag_assembly
    )

    coil_configuration = CoilConfiguration(
        c=1,
        n_z=20,
        n_w=20,
        l_ccd_mm=0,
        ohm_per_mm=100,
        tube_wall_thickness_mm=1,
        coil_wire_radius_mm=0.143 / 2,
        coil_center_mm=50,
        inner_tube_radius_mm=5,
    )

    flux_model = FluxModelPretrained(
        coil_configuration=coil_configuration,
        magnet_assembly=mag_assembly,
        curve_model_path=TEST_PRETRAINED_CURVE_MODEL_PATH,
    )

    load_model = SimpleLoad(R=30)

    coupling_model = CouplingModel(coupling_constant=1)

    test_model = (
        UnifiedModel()
        .with_height(105 / 1000)
        .with_magnet_assembly(mag_assembly)
        .with_magnetic_spring(mag_spring)
        .with_mechanical_spring(mech_spring)
        .with_mechanical_damper(mech_damper)
        .with_input_excitation(input_excitation)
        .with_coil_configuration(coil_configuration)
        .with_flux_model(flux_model)
        .with_rectification_drop(0.01)
        .with_load_model(load_model)
        .with_coupling_model(coupling_model)
        .with_governing_equations(unified_ode)
    )

    return test_model
