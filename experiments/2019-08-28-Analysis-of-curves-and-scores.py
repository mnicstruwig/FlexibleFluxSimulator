import ray  # Used for PyArrow (using regular pyarrow sometimes causes crash)
#matplotlib.use('MacOSX')  # Temporary workaround for crash in MPL for Mac OSX 10.14.6
import pandas as pd
from plotnine import *
import numpy as np
import os


# ═════════════════════════════════
which_device = 'A'
which_sample = 0
# ═════════════════════════════════

def make_paths(device, sample, root_dir='.'):
    scores_path = os.path.join(root_dir, f'{device}_{sample}_score.parq')
    curves_path = os.path.join(root_dir, f'{device}_{sample}_curves.parq')

    return scores_path, curves_path


def get_dataframes(device, sample, root_dir='.'):
    scores_path, curves_path = make_paths(device, sample, root_dir)
    return pd.read_parquet(scores_path), pd.read_parquet(curves_path)


def plot_results(df,
                 signal_column,
                 time_target=None,
                 signal_target=None,
                 color_by='label'):

    df_groundtruth = pd.DataFrame()
    df_groundtruth['time'] = time_target
    df_groundtruth[signal_column] = signal_target
    df_groundtruth[color_by] = 'Groundtruth'
    df_groundtruth['idx'] = 'None'



    df['label'] = 'Experiments'

    p = ggplot(aes(x='time', y=signal_column, group='idx', color=color_by), df)
    p = (
        p
        + geom_line(data=df_groundtruth, size=1)
        + geom_line(size=1)  # Curves
        + theme_538()
    )

    p.__repr__()


def get_best_curve(metric_column, df_scores, df_curves):
    best_idx = df_scores.sort_values(by=metric_column).index[0:1]
    best_scores = df_scores.iloc[best_idx, :].copy()
    best_curves = df_curves[df_curves['idx'].isin(best_idx)].copy()
    best_curves['best_in'] = metric_column

    best_scores['idx'] = best_scores.index
    df_best = best_curves.join(best_scores, 'idx', 'inner', lsuffix='_curve')
    df_best = df_best.drop(columns='idx_curve')
    return df_best


df_scores, df_curves = get_dataframes(which_device, which_sample)
df_best = get_best_curve('mech_dtw', df_scores, df_curves)
df_groundtruth = pd.read_parquet(
    f'{which_device}_{which_sample}_groundtruth.parq'
)

# Look at plotting the best curve for each of the metrics
metric_columns = ['mech_dtw', 'elec_dtw', 'elec_rms_perc_diff']
best = [
    get_best_curve(metric, df_scores, df_curves)
    for metric
    in metric_columns
]

df_best = pd.concat(best)
signal = 'emf'

df_target = df_groundtruth.query(f'label == "{signal}"')
time_target = df_target['time'].values
signal_target =df_target['values'].values

plot_results(df_best,
             signal,
             time_target,
             signal_target,
             color_by='best_in')
