import argparse
import json
import os

import numpy as np

from wasabi import msg
import nevergrad as ng
from unified_model.evaluate import Measurement
from unified_model.unified import UnifiedModel
from unified_model.utils.utils import collect_samples
from unified_model import parameter_search

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--configs', type=str, nargs='+',
                    help='Paths to the .json model config files.')
parser.add_argument('--sample_dirs', type=str, nargs='+',
                    help='Paths to the measurements that should be used.')
parser.add_argument('--budget', type=int, default=500,
                    help='Evolutionary budget.')

args = parser.parse_args()
model_configs = args.configs
model_sample_dirs = args.sample_dirs
budget = args.budget
model_config_paths = [os.path.abspath(path) for path in model_configs]
model_sample_dirs = [os.path.abspath(path) + '/' for path in model_sample_dirs]

models = []
for p in model_config_paths:
    with open(p) as f:
        json_str = f.read()
        config = json.loads(json_str)
        models.append(UnifiedModel.from_config(config))

samples = []
for base_dir in model_sample_dirs:
    sample_collection = collect_samples(
        base_path=base_dir,
        acc_pattern='*acc*.csv',
        adc_pattern='*adc*.csv',
        video_label_pattern='*labels*.csv'
    )
    samples.append(sample_collection)

measurements = []
for model, sample_collection in zip(models, samples):
    measurement_set = [Measurement(s, model) for s in sample_collection]
    measurements.append(measurement_set)


models_and_measurements = list(zip(models, measurements))

instruments = {
    'mech_damping_coefficient': ng.p.Scalar(init=2, lower=0, upper=10),
    'coupling_constant': ng.p.Scalar(init=0, lower=0, upper=10),
    'mech_spring_damping_coefficient': ng.p.Scalar(init=0, lower=0, upper=10),
}

results = parameter_search.mean_of_scores(
    models_and_measurements=models_and_measurements,
    instruments=instruments,
    cost_metric='power',
    budget=budget
)
