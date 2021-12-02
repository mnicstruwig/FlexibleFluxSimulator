import numpy as np
import shutil
from glob import glob

import pyarrow.parquet as pq
import pytest
import ray
import unified_model.tests.utils as test_utils
from unified_model import batch, metrics
from unified_model.electrical_components.flux.model import (
    FluxModelInterp, FluxModelPretrained)
from unified_model.mechanical_components import (MagneticSpringInterp,
                                                 MassProportionalDamper,
                                                 MechanicalSpring,
                                                 magnet_assembly)
from unified_model.mechanical_components.input_excitation.accelerometer import \
    AccelerometerInput
from unified_model.utils.utils import Sample


@pytest.fixture
def build_unified_model():
    test_model = test_utils.build_unified_model_with_standard_components()
    yield test_model

    # Stop ray if it was running
    ray.shutdown()
    # Delete our batch run data
    for f in glob('batch_run*.parquet'):
        shutil.rmtree(f)


def test_solve_for_batch(build_unified_model):
    """Test if we can solve in batch"""
    test_model = build_unified_model

    test_sample_1 = Sample(
        acc_path=test_utils.TEST_LOG_ACC_PATH_1,
        adc_path=test_utils.TEST_LOG_ADC_PATH_1,
        video_labels_path=test_utils.TEST_LOG_VIDEO_LABELS_PATH_1
    )

    test_sample_2 = test_sample_1

    test_base_model_config = test_model.get_config(kind='dict')
    param_path_1 = 'coil_configuration.n_z'
    param_path_2 = 'coil_configuration.n_w'

    test_param_set_1 = [
        (param_path_1, 10),
        (param_path_2, 10)
    ]
    test_param_set_2 = [
        (param_path_1, 20),
        (param_path_2, 20)
    ]
    test_param_set_3 = [
        (param_path_1, 30),
        (param_path_2, 30)
    ]

    test_params = [test_param_set_1, test_param_set_2, test_param_set_3]
    mech_metric_name = 'dtw_euclid_mech'
    elec_metric_name = 'rms_perc_diff'

    batch.solve_for_batch(
        base_model_config=test_base_model_config,
        params=test_params,
        samples=[test_sample_1, test_sample_2],
        mech_pred_expr='x3-x1',
        mech_metrics={mech_metric_name: metrics.dtw_euclid_distance},
        elec_pred_expr='g(t, x5)',
        elec_metrics={elec_metric_name: metrics.root_mean_square_percentage_diff},
        output_root_dir='.',
        solve_kwargs={
            't_start': 0,
            't_end': 2,
            'y0': [0., 0., 0.04, 0., 0.],
            't_eval': np.linspace(0, 2, 100),
            't_max_step': 1e-2}
    )

    f = glob('batch_run*.parquet')[0]  # Should only be one result
    df = pq.read_table(f).to_pandas()

    # Check our metrics are in the results
    assert mech_metric_name in df.columns
    assert elec_metric_name in df.columns
    # Check our config is in the results
    assert 'config' in df.columns
    # Check our param set variable are in the results
    assert param_path_1 in df.columns
    assert param_path_2 in df.columns
    # Check that the input excitations are in the results
    assert 'input' in df.columns
    # Check that the correct number of results are there
    assert len(df) == 6  # 3 param sets X 2 input excitations
