import numpy as np
import shutil
from glob import glob

import pyarrow.parquet as pq
import pytest
import ray
import ffs.tests.utils as test_utils
from ffs import batch, metrics
from ffs.electrical_components.flux import FluxModelInterp, FluxModelPretrained
from ffs.mechanical_components.magnet_assembly import MagnetAssembly
from ffs.mechanical_components.magnetic_spring import MagneticSpringInterp
from ffs.mechanical_components.mechanical_spring import MechanicalSpring
from ffs.mechanical_components.damper import MassProportionalDamper
from ffs.mechanical_components.input_excitation.accelerometer import AccelerometerInput
from ffs.utils.utils import Sample


@pytest.fixture
def build_unified_model():
    test_model = test_utils.build_unified_model_with_standard_components()
    yield test_model

    # Stop ray if it was running
    ray.shutdown()
    # Delete our batch run data
    for f in glob("batch_run*.parquet"):
        shutil.rmtree(f)


def test_solve_for_batch_no_calc_metrics_on_prediction_expr(build_unified_model):
    """Test if we can solve in batch and calculate scores, but not calculate any
    extra metrics."""
    test_model = build_unified_model

    test_sample_1 = Sample(
        acc_path=test_utils.TEST_LOG_ACC_PATH_1,
        adc_path=test_utils.TEST_LOG_ADC_PATH_1,
        video_labels_path=test_utils.TEST_LOG_VIDEO_LABELS_PATH_1,
    )

    test_sample_2 = test_sample_1

    test_base_model_config = test_model.get_config(kind="dict")
    param_path_1 = "coil_configuration.n_z"
    param_path_2 = "coil_configuration.n_w"

    test_param_set_1 = [(param_path_1, 10), (param_path_2, 10)]
    test_param_set_2 = [(param_path_1, 20), (param_path_2, 20)]
    test_param_set_3 = [(param_path_1, 30), (param_path_2, 30)]

    test_params = [test_param_set_1, test_param_set_2, test_param_set_3]
    mech_metric_name = "dtw_euclid_mech"
    elec_metric_name = "rms_perc_diff"

    batch.solve_for_batch(
        base_model_config=test_base_model_config,
        params=test_params,
        samples=[test_sample_1, test_sample_2],
        y_diff_expr="x3-x1",
        mech_metrics={mech_metric_name: metrics.dtw_euclid_distance},
        v_load_expr="g(t, x5)",
        elec_metrics={elec_metric_name: metrics.root_mean_square_percentage_diff},
        output_root_dir=".",
        solve_kwargs={
            "t_start": 0,
            "t_end": 2,
            "y0": [0.0, 0.0, 0.04, 0.0, 0.0],
            "t_eval": np.linspace(0, 2, 100),
            "t_max_step": 1e-2,
            "method": "RK23",
        },
    )

    f = glob("batch_run*.parquet")[0]  # Should only be one result
    df = pq.read_table(f).to_pandas()

    # Check our metrics are in the results
    assert mech_metric_name in df.columns
    assert elec_metric_name in df.columns
    # Check our config is in the results
    assert "config" in df.columns
    # Check our param set variable are in the results
    assert param_path_1 in df.columns
    assert param_path_2 in df.columns
    # Check that the input excitations are in the results
    assert "input" in df.columns
    # Check that the correct number of results are there
    assert len(df) == 6  # 3 param sets X 2 input excitations


def test_solve_for_batch_also_calc_metrics_on_prediction_expr(build_unified_model):
    """Test if we can solve in batch and calculate scores, and also calculate some
    extra metrics."""
    test_model = build_unified_model

    test_sample_1 = Sample(
        acc_path=test_utils.TEST_LOG_ACC_PATH_1,
        adc_path=test_utils.TEST_LOG_ADC_PATH_1,
        video_labels_path=test_utils.TEST_LOG_VIDEO_LABELS_PATH_1,
    )

    test_base_model_config = test_model.get_config(kind="dict")
    param_path_1 = "coil_configuration.n_z"
    param_path_2 = "coil_configuration.n_w"

    test_param_set_1 = [(param_path_1, 10), (param_path_2, 10)]
    test_param_set_2 = [(param_path_1, 20), (param_path_2, 20)]

    test_params = [test_param_set_1, test_param_set_2]
    mech_metric_name = "dtw_euclid_mech"
    elec_metric_name = "rms_perc_diff"
    pred_metric_name = "max_load_voltage"

    batch.solve_for_batch(
        base_model_config=test_base_model_config,
        params=test_params,
        samples=[test_sample_1],
        y_diff_expr="x3-x1",
        mech_metrics={mech_metric_name: metrics.dtw_euclid_distance},
        v_load_expr="g(t, x5)",
        elec_metrics={elec_metric_name: metrics.root_mean_square_percentage_diff},
        prediction_expr="g(t, x5)",
        prediction_metrics={pred_metric_name: lambda x: max(x)},
        output_root_dir=".",
        solve_kwargs={
            "t_start": 0,
            "t_end": 2,
            "y0": [0.0, 0.0, 0.04, 0.0, 0.0],
            "t_eval": np.linspace(0, 2, 100),
            "t_max_step": 1e-2,
            "method": "RK23",
        },
    )

    f = glob("batch_run*.parquet")[0]  # Should only be one result
    df = pq.read_table(f).to_pandas()

    # Check our metrics are in the results
    assert mech_metric_name in df.columns
    assert elec_metric_name in df.columns
    assert pred_metric_name in df.columns
    # Check our config is in the results
    assert "config" in df.columns
    # Check our param set variable are in the results
    assert param_path_1 in df.columns
    assert param_path_2 in df.columns
    # Check that the input excitations are in the results
    assert "input" in df.columns
    # Check that the correct number of results are there
    assert len(df) == 2  # 2 param sets X 1 input excitations
