import nevergrad as ng
import ffs.tests.utils as test_utils
from ffs import parameter_search
from ffs.evaluate import Measurement
from ffs.utils.utils import Sample

import pytest
import ray


@pytest.fixture
def get_test_models_and_measurements():
    test_model_1 = test_utils.build_unified_model_with_standard_components()
    test_model_2 = test_model_1.update_params(
        [
            ('coil_configuration.n_z', 20),
            ('coil_configuration.n_w', 20)
        ]
    )

    test_measurement_1 = Measurement(
        sample=Sample(
            acc_path=test_utils.TEST_LOG_ACC_PATH_1,
            adc_path=test_utils.TEST_LOG_ADC_PATH_1,
            video_labels_path=test_utils.TEST_LOG_VIDEO_LABELS_PATH_1
        ),
        model_prototype=test_model_1
    )

    test_measurement_2 = Measurement(
        sample=Sample(
            acc_path=test_utils.TEST_LOG_ACC_PATH_2,
            adc_path=test_utils.TEST_LOG_ADC_PATH_2,
            video_labels_path=test_utils.TEST_LOG_VIDEO_LABELS_PATH_2
        ),
        model_prototype=test_model_2
    )

    yield [test_model_1, test_model_2], [test_measurement_1, test_measurement_2]

    ray.shutdown()  # Make sure we shutdown `ray` in case something broke


def test_mean_of_scores_param_search_single_device(get_test_models_and_measurements):
    """Test that we can execute the `mean_of_scores` parameter search
    using only a single device as reference."""

    test_models, test_measurements = get_test_models_and_measurements
    test_model = test_models[0]

    test_instruments = {
        'mech_damping_coefficient': ng.p.Scalar(init=0.1, lower=0, upper=10),
        'coupling_constant': ng.p.Scalar(init=0.1, lower=0, upper=5),
        'mech_spring_damping_coefficient': ng.p.Scalar(init=0.1, lower=0, upper=5)
    }

    results = parameter_search.mean_of_scores(
        models_and_measurements=[(test_model, test_measurements)],
        instruments=test_instruments,
        cost_metric='dtw',
        budget=5,  # Keep it small for the test
        verbose=True,
        log_to_disk=False
    )

    assert results['mech_damping_coefficient'] is not None
    assert results['coupling_constant'] is not None
    assert results['mech_spring_damping_coefficient'] is not None
    assert results['loss'] is not None


def test_mean_of_scores_param_search_multiple_devices(get_test_models_and_measurements):
    """Test that we can execute the `mean_of_scores` parameter search only
    multiple reference devices."""
    test_models, test_measurements = get_test_models_and_measurements
    test_model_1, test_model_2 = test_models
    test_measurement_1, test_measurement_2 = test_measurements

    test_instruments = {
        'mech_damping_coefficient': ng.p.Scalar(init=0.1, lower=0, upper=10),
        'coupling_constant': ng.p.Scalar(init=0.1, lower=0, upper=5),
        'mech_spring_damping_coefficient': ng.p.Scalar(init=0.1, lower=0, upper=5)
    }

    results = parameter_search.mean_of_scores(
        models_and_measurements=[(test_model_1, [test_measurement_1]),
                                 (test_model_2, [test_measurement_2])],
        instruments=test_instruments,
        cost_metric='dtw',
        budget=5,  # Keep it small for the test
        verbose=True,
        log_to_disk=False
    )

    assert results['mech_damping_coefficient'] is not None
    assert results['coupling_constant'] is not None
    assert results['mech_spring_damping_coefficient'] is not None
    assert results['loss'] is not None
