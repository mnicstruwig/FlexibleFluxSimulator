import json

import numpy as np
import pandas as pd
from ffs import metrics
from ffs.electrical_components.coil import CoilConfiguration
from ffs.electrical_components.flux import FluxModelInterp, FluxModelPretrained
from ffs.evaluate import Measurement
from ffs.mechanical_components.magnet_assembly import MagnetAssembly
from ffs.mechanical_components.magnetic_spring import MagneticSpringInterp
from ffs.mechanical_components.mechanical_spring import MechanicalSpring
from ffs.mechanical_components.damper import MassProportionalDamper
from ffs.mechanical_components.input_excitation.accelerometer import AccelerometerInput
from . import utils as test_utils
from ffs.unified import UnifiedModel
from ffs.utils.utils import Sample


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

    test_model = test_utils.build_unified_model_with_standard_components()
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

    test_model = test_utils.build_unified_model_with_standard_components()

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

    test_model = test_utils.build_unified_model_with_standard_components()

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

    test_model = test_utils.build_unified_model_with_standard_components()

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

    test_model = test_utils.build_unified_model_with_standard_components()

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
    results = test_model.get_result(
        time="t", mag_pos="x3-x1", mag_vel="x4-x2", emf="g(t, x5)"
    )

    # Check we get a DataFrame back
    assert isinstance(results, pd.DataFrame)

    # Check we get the correct columns
    assert "time" in results
    assert "mag_pos" in results
    assert "mag_vel" in results
    assert "emf" in results

    # Check that we have the correct number samples returned
    assert len(test_t_eval) == len(results)


def test_score_sample_mech_model():
    """Test the that we can score a solution of a `UnifiedModel` against a
    measurement, for the mechanical component of the model."""

    test_model = test_utils.build_unified_model_with_standard_components()

    # Create our sample
    test_sample = Sample(
        acc_path=test_utils.TEST_LOG_ACC_PATH_1,
        adc_path=test_utils.TEST_LOG_ADC_PATH_1,
        video_labels_path=test_utils.TEST_LOG_VIDEO_LABELS_PATH_1,
    )

    # Score
    actual_score, _ = test_model.score_sample(
        sample=test_sample,
        solve_kwargs=dict(
            t_start=0,
            t_end=8,
            y0=[0.0, 0.0, 0.04, 0.0, 0.0],
            t_eval=np.linspace(0, 8, 1000),
            t_max_step=1e-3,
            method="RK45",
        ),
        y_diff_expr="x3-x1",
        mech_metrics_dict={"y_diff_dtw": metrics.dtw_euclid_distance},
    )

    assert isinstance(actual_score, dict)
    assert "y_diff_dtw" in actual_score


def test_score_sample_elec_model():
    """Test the that we can score a solution of a `UnifiedModel` against a
    measurement, for the electrical component of the model."""

    test_model = test_utils.build_unified_model_with_standard_components()

    # Create our sample
    test_sample = Sample(
        acc_path=test_utils.TEST_LOG_ACC_PATH_1,
        adc_path=test_utils.TEST_LOG_ADC_PATH_1,
        video_labels_path=test_utils.TEST_LOG_VIDEO_LABELS_PATH_1,
    )

    # Score
    actual_score, _ = test_model.score_sample(
        sample=test_sample,
        solve_kwargs=dict(
            t_start=0,
            t_end=8,
            y0=[0.0, 0.0, 0.04, 0.0, 0.0],
            t_eval=np.linspace(0, 8, 1000),
            t_max_step=1e-3,
            method="RK45",
        ),
        v_load_expr="g(t, x5)",
        elec_metrics_dict={"rms_perc_diff": metrics.root_mean_square_percentage_diff},
    )

    assert isinstance(actual_score, dict)
    assert "rms_perc_diff" in actual_score


def test_calculate_metrics():
    """Test that we can calculate metrics for a single prediction expression."""

    test_model = test_utils.build_unified_model_with_standard_components()
    # First need to solve
    test_model.solve(
        t_start=0,
        t_end=8,
        y0=[0.0, 0.0, 0.04, 0.0, 0.0],
        t_eval=np.linspace(0, 8, 1000),
        t_max_step=1e-3,
        method="RK45",
    )

    actual_result = test_model.calculate_metrics(
        prediction_expr="g(t, x5)",
        metric_dict={
            "mean_load_voltage": lambda x: np.mean(x),
            "max_load_voltage": lambda x: np.max(x),
        },
    )

    assert isinstance(actual_result, dict)
    assert "mean_load_voltage" in actual_result
    assert "max_load_voltage" in actual_result


# TODO: Test updating an invalid component!
def test_update_single_param_with_no_observers():
    """Test that model components can be updated.

    In this case, we update a model that has no registered observers.
    """

    test_model = UnifiedModel()

    test_m_value = 1
    mag_assembly = MagnetAssembly(
        m=test_m_value, l_m_mm=10, l_mcd_mm=0, dia_magnet_mm=10, dia_spacer_mm=10
    )

    test_model.with_magnet_assembly(mag_assembly)
    new_m_value = 3

    actual_updated_model = test_model.update_params(
        [("magnet_assembly.m", new_m_value)]
    )

    # Check that the returned model has the updated parameter value
    assert actual_updated_model.magnet_assembly.m == new_m_value
    # Check that the old model still has the previous parameter value
    assert test_model.magnet_assembly.m == test_m_value


def test_update_top_level_param():
    """Test that model components can be updated.

    In this case, we test if we can update top-level parameters that are defined
    on the model itself (rather than on a component).
    """

    test_model = UnifiedModel()
    test_height = 100
    test_model.with_height(test_height)

    new_height = 200

    actual_updated_model = test_model.update_params([("height", new_height)])

    # Check that the returned model has the updated parameter value
    assert actual_updated_model.height == new_height
    # Check that the old model still has the previous parameter value
    assert test_model.height == test_height


def test_update_multiple_params():
    """Test that model components can be updated.

    Here we test updating multiple components at the same time.
    """

    test_m = 1
    test_l_mcd_mm = 10
    test_c = 1
    test_height = 100 / 1000

    mag_assembly = MagnetAssembly(
        m=test_m, l_m_mm=10, l_mcd_mm=test_l_mcd_mm, dia_magnet_mm=10, dia_spacer_mm=10
    )

    coil_configuration = CoilConfiguration(
        c=test_c,
        n_z=10,
        n_w=20,
        l_ccd_mm=10,
        ohm_per_mm=10,
        coil_center_mm=50,
        tube_wall_thickness_mm=10,
        coil_wire_radius_mm=0.127 / 2,
        inner_tube_radius_mm=5,
    )

    test_model = (
        UnifiedModel()
        .with_magnet_assembly(mag_assembly)
        .with_coil_configuration(coil_configuration)
        .with_height(test_height)
    )

    new_m = 3
    new_l_mcd_mm = 20
    new_c = 3
    new_height = 200 / 1000

    actual_updated_model = test_model.update_params(
        [
            ("magnet_assembly.m", new_m),
            ("magnet_assembly.l_mcd_mm", new_l_mcd_mm),
            ("coil_configuration.c", new_c),
            ("height", new_height),
        ]
    )

    # Check that the returned model has the updated parameter value
    assert actual_updated_model.magnet_assembly.m == new_m
    assert actual_updated_model.magnet_assembly.l_mcd_mm == new_l_mcd_mm
    assert actual_updated_model.coil_configuration.c == new_c
    assert actual_updated_model.height == new_height
    # Check that the old model still has the previous parameter value
    assert test_model.magnet_assembly.m == test_m
    assert test_model.magnet_assembly.l_mcd_mm == test_l_mcd_mm
    assert test_model.coil_configuration.c == test_c
    assert test_model.height == test_height


def test_update_params_with_observers():
    """Test that when we update the parameter, our subscribed observers also update."""

    class DummyCouplingModel:
        """Record the number of magnets and coils when notified."""

        def __init__(self, m, c):
            self.m = m
            self.c = c

        def update(self, um):  # Make an observer
            self.m = um.magnet_assembly.m
            self.c = um.coil_configuration.c

    test_m = 1
    test_l_mcd_mm = 10
    test_c = 1
    test_height = 100 / 1000

    mag_assembly = MagnetAssembly(
        m=test_m, l_m_mm=10, l_mcd_mm=test_l_mcd_mm, dia_magnet_mm=10, dia_spacer_mm=10
    )

    coil_configuration = CoilConfiguration(
        c=test_c,
        n_z=10,
        n_w=20,
        l_ccd_mm=10,
        ohm_per_mm=10,
        coil_center_mm=50,
        tube_wall_thickness_mm=10,
        coil_wire_radius_mm=0.127 / 2,
        inner_tube_radius_mm=5,
    )

    dummy_coupling_model = DummyCouplingModel(m=None, c=None)

    test_model = (
        UnifiedModel()
        .with_magnet_assembly(mag_assembly)
        .with_coil_configuration(coil_configuration)
        .with_height(test_height)
        .with_coupling_model(dummy_coupling_model)  # Should register the observer
    )

    new_m = 3
    new_l_mcd_mm = 20
    new_c = 3
    new_height = 200 / 1000

    actual_updated_model = test_model.update_params(
        [
            ("magnet_assembly.m", new_m),
            ("magnet_assembly.l_mcd_mm", new_l_mcd_mm),
            ("coil_configuration.c", new_c),
            ("height", new_height),
        ]
    )

    # Assert our dummy coupling model was notified and updated after updating the parameters
    assert (
        actual_updated_model.coupling_model.m == actual_updated_model.magnet_assembly.m
    )
    assert (
        actual_updated_model.coupling_model.c
        == actual_updated_model.coil_configuration.c
    )

    # Assert our old model remained unchanged
    assert test_model.coupling_model.m == test_model.magnet_assembly.m
    assert test_model.coupling_model.c == test_model.coil_configuration.c
