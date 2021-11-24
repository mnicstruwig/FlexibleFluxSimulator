import json
import os

import pytest
import numpy as np
import pandas as pd

from unified_model import metrics
from unified_model.coupling import CouplingModel
from unified_model.electrical_components.coil import CoilConfiguration
from unified_model.electrical_components.flux.model import (
    FluxModelInterp,
    FluxModelPretrained,
)
from unified_model.electrical_components.load import SimpleLoad
from unified_model.evaluate import Measurement
from unified_model.governing_equations import unified_ode
from unified_model.mechanical_components.input_excitation.accelerometer import (
    AccelerometerInput,
)

from unified_model.utils.utils import Sample
from unified_model.unified import UnifiedModel
from unified_model.mechanical_components import (
    MagnetAssembly,
    MechanicalSpring,
    MassProportionalDamper,
    MagneticSpringInterp,
    magnet_assembly,
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(CURRENT_DIR, "test_data")

TEST_LOG_ACC_PATH = os.path.join(TEST_DATA_DIR, "test_log_acc.csv")
TEST_LOG_ADC_PATH = os.path.join(TEST_DATA_DIR, "test_log_adc.csv")
TEST_LOG_VIDEO_LABELS_PATH = os.path.join(TEST_DATA_DIR, "test_log_video_labels.csv")
TEST_MAG_SPRING_DATA_PATH = os.path.join(TEST_DATA_DIR, "test_mag_spring_data.csv")
TEST_PRETRAINED_CURVE_MODEL_PATH = os.path.join(TEST_DATA_DIR, "test_curve_model_pretrained.model")

def _build_unified_model_with_standard_components() -> UnifiedModel:
    """Create a `UnifiedModel` with all standard components attached."""
    input_excitation = AccelerometerInput(
        raw_accelerometer_data_path=TEST_LOG_ACC_PATH,
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
        curve_model_path=TEST_PRETRAINED_CURVE_MODEL_PATH
    )

    load_model = SimpleLoad(R=30)

    coupling_model = CouplingModel(coupling_constant=1)

    test_model = (
        UnifiedModel()
        .with_height(105)
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



def test_get_config_for_magnet_assembly_only():
    """Test that the correct config is generated for only a magnetic spring
    attached to the model."""

    test_model = UnifiedModel()
    test_mag_assembly = MagnetAssembly(
        m=1, l_m_mm=10, l_mcd_mm=0, dia_magnet_mm=0, dia_spacer_mm=0
    )
    test_model.with_magnet_assembly(test_mag_assembly)

    actual_config = test_model.get_config()
    assert actual_config["magnet_assembly"] is not None


def test_get_config_for_no_components():
    """Test that the correct config is generated when no components are attached
    to the model."""
    test_model = UnifiedModel()

    actual_config = test_model.get_config()

    # Mechanical components
    assert actual_config["height"] is None
    assert actual_config["magnet_assembly"] is None
    assert actual_config["magnetic_spring"] is None
    assert actual_config["mechanical_spring"] is None
    assert actual_config["mechanical_damper"] is None
    assert actual_config["input_excitation"] is None

    # Electrial components
    assert actual_config["coil_configuration"] is None
    assert actual_config["flux_model"] is None
    assert actual_config["rectification_drop"] is None
    assert actual_config["load_model"] is None

    # Coupling
    assert actual_config["coupling_model"] is None

    # Governing Equations
    assert actual_config["governing_equations"] is None

    # Extras
    assert actual_config["extra_components"] is None


def test_get_config_for_all_standard_components():
    """Test that the correct config is generated when a number of different
    components are attached to the model."""

    test_model = _build_unified_model_with_standard_components()
    actual_config = test_model.get_config()

    # Mechanical components
    assert actual_config["height"] is not None
    assert actual_config["magnet_assembly"] is not None
    assert actual_config["magnetic_spring"] is not None
    assert actual_config["mechanical_spring"] is not None
    assert actual_config["mechanical_damper"] is not None
    assert actual_config["input_excitation"] is not None

    # Electrial components
    assert actual_config["coil_configuration"] is not None
    assert actual_config["flux_model"] is not None
    assert actual_config["rectification_drop"] is not None
    assert actual_config["load_model"] is not None

    # Coupling
    assert actual_config["coupling_model"] is not None

    # Governing equations
    assert actual_config["governing_equations"] is not None

    # Extras
    assert actual_config["extra_components"] is None  # We didn't specify any extras


def test_from_config_for_all_standard_components():
    """Test that the correct model is created from a config definition.

    Here we test `from_config` when all standard components are specified.

    We test this by creating a model, and using its config to create a second
    model. We then compare the two models across some attributes.
    """

    test_model = _build_unified_model_with_standard_components()

    # Make a second model from the config of the first model
    test_config = test_model.get_config()
    assert isinstance(test_config, dict)
    test_model_2 = UnifiedModel.from_config(test_config)

    # Do comparison  -- checking lengths of the __dict__ of attributes (in the
    # case when these are other objects) isn't the *best* check, but at least
    # it's not checking implementation details, just some sort of equivalence.

    assert test_model.height == test_model_2.height

    assert len(test_model.magnet_assembly.__dict__) == len(
        test_model_2.magnet_assembly.__dict__
    )
    assert len(test_model.magnetic_spring.__dict__) == len(
        test_model_2.magnetic_spring.__dict__
    )
    assert len(test_model.mechanical_spring.__dict__) == len(
        test_model_2.mechanical_spring.__dict__
    )
    assert len(test_model.mechanical_damper.__dict__) == len(
        test_model_2.mechanical_damper.__dict__
    )
    assert len(test_model.mechanical_damper.__dict__) == len(
        test_model_2.mechanical_damper.__dict__
    )
    assert len(test_model.input_excitation.__dict__) == len(
        test_model_2.input_excitation.__dict__
    )
    assert len(test_model.coil_configuration.__dict__) == len(
        test_model_2.coil_configuration.__dict__
    )
    assert len(test_model.flux_model.__dict__) == len(test_model_2.flux_model.__dict__)
    assert test_model.rectification_drop == test_model_2.rectification_drop
    assert len(test_model.load_model.__dict__) == len(test_model_2.load_model.__dict__)
    assert len(test_model.coupling_model.__dict__) == len(
        test_model_2.coupling_model.__dict__
    )
    assert callable(test_model_2.governing_equations)


def test_same_json_config_and_dict_config_for_all_standard_components():
    """Test that the `.get_config` returns the correct configuraiton for all
    standard components, when the output is specified to be a JSON string.

    We check this by making sure the parsed JSON config matches the dict config
    that is returned from `.get_config`.
    """

    test_model = _build_unified_model_with_standard_components()

    actual_config_dict = test_model.get_config(kind="dict")
    actual_config_json = test_model.get_config(kind="json")

    # Make sure it's a string
    assert isinstance(actual_config_json, str)

    # Make sure it can be parsed to a dict
    actual_config_parsed = json.loads(actual_config_json)

    # Make sure they're the same
    assert actual_config_dict == actual_config_parsed


def test_solve():
    """Test that we can solve for a solution of a `UnifiedModel`."""

    test_model = _build_unified_model_with_standard_components()

    # Initiate the solve
    test_model.solve(
        t_start=0,
        t_end=8,
        y0=[0.0, 0.0, 0.04, 0.0, 0.0],
        t_eval=np.linspace(0, 8, 1000),
        t_max_step=1e-3,
        method="RK45",
    )


def test_get_results():
    """Test that we can solve for a `UnifiedModel` and get the results."""

    test_model = _build_unified_model_with_standard_components()

    # Initiate the solve
    test_t_eval = np.linspace(0, 8, 1000)

    test_model.solve(
        t_start=0,
        t_end=8,
        y0=[0.0, 0.0, 0.04, 0.0, 0.0],
        t_eval=test_t_eval,
        t_max_step=1e-3,
        method="RK45",
    )

    # Get some results
    results = test_model.get_result(time='t',
                                     mag_pos='x3-x1',
                                     mag_vel='x4-x2',
                                     emf='g(t, x5)')

    # Check we get a DataFrame back
    assert isinstance(results, pd.DataFrame)

    # Check we get the correct columns
    assert 'time' in results
    assert 'mag_pos' in results
    assert 'mag_vel' in results
    assert 'emf' in results

    # Check that we have the correct number samples returned
    assert len(test_t_eval) == len(results)


def test_score_measurement_mech_model():
    """Test the that we can score a solution of a `UnifiedModel` against a
    measurement, for the mechanical component of the model."""

    test_model = _build_unified_model_with_standard_components()

    # Create our measurement
    test_sample = Sample(
        acc_path=TEST_LOG_ACC_PATH,
        adc_path=TEST_LOG_ADC_PATH,
        video_labels_path=TEST_LOG_VIDEO_LABELS_PATH
    )

    test_measurement = Measurement(
        sample=test_sample,
        model_prototype=test_model
    )

    # Score
    actual_score, _ = test_model.score_measurement(
        measurement=test_measurement,
        solve_kwargs=dict(
            t_start=0,
            t_end=8,
            y0=[0.0, 0.0, 0.04, 0.0, 0.0],
            t_eval=np.linspace(0, 8, 1000),
            t_max_step=1e-3,
            method="RK45",
        ),
        mech_pred_expr='x3-x1',
        mech_metrics_dict={'y_diff_dtw': metrics.dtw_euclid_distance}
    )

    assert isinstance(actual_score, dict)
    assert 'y_diff_dtw' in actual_score


def test_score_measurement_elec_model():
    """Test the that we can score a solution of a `UnifiedModel` against a
    measurement, for the electrical component of the model."""

    test_model = _build_unified_model_with_standard_components()

    # Create our measurement
    test_sample = Sample(
        acc_path=TEST_LOG_ACC_PATH,
        adc_path=TEST_LOG_ADC_PATH,
        video_labels_path=TEST_LOG_VIDEO_LABELS_PATH
    )

    test_measurement = Measurement(
        sample=test_sample,
        model_prototype=test_model
    )

    # Score
    actual_score, _ = test_model.score_measurement(
        measurement=test_measurement,
        solve_kwargs=dict(
            t_start=0,
            t_end=8,
            y0=[0.0, 0.0, 0.04, 0.0, 0.0],
            t_eval=np.linspace(0, 8, 1000),
            t_max_step=1e-3,
            method="RK45",
        ),
        elec_pred_expr='g(t, x5)',
        elec_metrics_dict={'rms_perc_diff': metrics.root_mean_square_percentage_diff}
    )

    assert isinstance(actual_score, dict)
    assert 'rms_perc_diff' in actual_score

def test_calculate_metrics():
    """Test that we can calculate metrics for a single prediction expression."""

    test_model = _build_unified_model_with_standard_components()
    # First need to solve
    test_model.solve(
        t_start=0,
        t_end=8,
        y0=[0.0, 0.0, 0.04, 0.0, 0.0],
        t_eval=np.linspace(0, 8, 1000),
        t_max_step=1e-3,
        method="RK45",
    )

    actual_result = test_model.calculate_metrics(prediction_expr='g(t, x5)',
                                                 metric_dict={'mean_load_voltage': lambda x: np.mean(x),
                                                              'max_load_voltage': lambda x: np.max(x)})

    assert isinstance(actual_result, dict)
    assert 'mean_load_voltage' in actual_result
    assert 'max_load_voltage' in actual_result
