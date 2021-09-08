from collections import namedtuple
from glob import glob
import os
import pandas as pd

Sample = namedtuple("Sample", ["acc_emf_df", "video_labels_df"])


def collect_samples(base_path, labeled_video_pattern, acc_emf_pattern):
    sample_collection = []
    labeled_video_paths = glob(os.path.join(base_path, labeled_video_pattern))
    acc_emf_paths = glob(os.path.join(base_path, acc_emf_pattern))

    for aep, lvp in zip(acc_emf_paths, labeled_video_paths):
        sample_collection.append(Sample(pd.read_csv(aep), pd.read_csv(lvp)))
    return sample_collection
