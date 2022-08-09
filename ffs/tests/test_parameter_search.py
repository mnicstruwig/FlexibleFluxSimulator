import nevergrad as ng
import ffs.tests.utils as test_utils
from ffs import parameter_search
from ffs.evaluate import Measurement
from ffs.utils.utils import Sample

import pytest
import ray


@pytest.fixture
def get_test_models_and_samples():
    test_model_1 = test_utils.build_unified_model_with_standard_components()
    test_model_2 = test_model_1.update_params(
        [("coil_configuration.n_z", 20), ("coil_configuration.n_w", 20)]
    )
    test_sample_1 = Sample(acc_path=test_utils.TEST_LOG_ACC_PATH_1,
                           adc_path=test_utils.TEST_LOG_ADC_PATH_1,
                           video_labels_path=test_utils.TEST_LOG_VIDEO_LABELS_PATH_1)

    test_sample_2 = Sample(acc_path=test_utils.TEST_LOG_ACC_PATH_2,
                           adc_path=test_utils.TEST_LOG_ADC_PATH_2,
                           video_labels_path=test_utils.TEST_LOG_VIDEO_LABELS_PATH_2)

    yield [test_model_1, test_model_2], [test_sample_1, test_sample_2]

    ray.shutdown()  # Make sure we shutdown `ray` in case something broke


def test_mean_of_scores_param_search_single_device(get_test_models_and_samples):
    """Test that we can execute the `mean_of_scores` parameter search
    using only a single device as reference."""

    test_models, test_samples = get_test_models_and_samples
    test_model = test_models[0]

    test_instrumented_params = [
        ('mechanical_damper.damping_coefficient', ng.p.Scalar(init=0.1,
                                                              lower=0,
                                                              upper=10)),
        ('coupling_model.coupling_constant', ng.p.Scalar(init=0.1,
                                                         lower=0,
                                                         upper=5)),
        ('mechanical_spring.damping_coefficient', ng.p.Scalar(init=0.1,
                                                              lower=0,
                                                              upper=5))
    ]

    results = parameter_search.mean_of_scores(
        models_and_samples=[(test_model, test_samples)],
        instrumented_params=test_instrumented_params,
        cost_metric="dtw",
        budget=5,  # Keep it small for the test
        verbose=True,
        log_to_disk=False,
    )

    assert results["mechanical_damper.damping_coefficient"] is not None
    assert results["coupling_model.coupling_constant"] is not None
    assert results["mechanical_spring.damping_coefficient"] is not None
    assert results["loss"] is not None


def test_mean_of_scores_param_search_multiple_devices(get_test_models_and_samples):
    """Test that we can execute the `mean_of_scores` parameter search only
    multiple reference devices."""
    test_models, test_samples = get_test_models_and_samples
    test_model_1, test_model_2 = test_models
    test_measurement_1, test_measurement_2 = test_samples

    test_instrumented_params = [
        ("mechanical_damper.damping_coefficient", ng.p.Scalar(init=0.1, lower=0, upper=10)),
        ("coupling_model.coupling_constant", ng.p.Scalar(init=0.1, lower=0, upper=5)),
        ("mechanical_spring.damping_coefficient", ng.p.Scalar(init=0.1, lower=0, upper=5)),
    ]

    results = parameter_search.mean_of_scores(
        models_and_samples=[
            (test_model_1, [test_measurement_1]),
            (test_model_2, [test_measurement_2]),
        ],
        instrumented_params=test_instrumented_params,
        cost_metric="dtw",
        budget=5,  # Keep it small for the test
        verbose=True,
        log_to_disk=False,
    )

    assert results["mechanical_damper.damping_coefficient"] is not None
    assert results["coupling_model.coupling_constant"] is not None
    assert results["mechanical_spring.damping_coefficient"] is not None
    assert results["loss"] is not None
