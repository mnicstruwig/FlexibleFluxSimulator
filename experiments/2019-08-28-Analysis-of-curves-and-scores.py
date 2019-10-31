import ray  # Used for PyArrow (using regular pyarrow sometimes causes crash)
#matplotlib.use('MacOSX')  # Temporary workaround for crash in MPL for Mac OSX 10.14.6
import pandas as pd
import plotnine as p9
import matplotlib.pyplot as plt
import numpy as np
import os


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


def plot_results_facet(df,
                       signal_column,
                       time_target=None,
                       signal_target=None,
                       color_by='label',
                       facet_by='label',
                       **kwargs):

    try:
        assert(len(time_target) == len(signal_target))
    except AssertionError:
        raise AssertionError('len(time_target) != len(signal_target)')

    df_groundtruth = pd.DataFrame()
    df_groundtruth['time'] = time_target
    df_groundtruth[signal_column] = signal_target
    df_groundtruth['idx'] = 'None'

    p = ggplot(aes(x='time', y=signal_column, color=color_by),
               df)

    p = p \
        + geom_line(data=df_groundtruth, color='black', size=1, alpha=0.5) \
        + geom_line(size=1) \
        + facet_grid(f'{facet_by} ~ .') \
        + theme_538()

    x_limits = kwargs.pop('xlim', None)
    if x_limits:
        p = p + xlim(x_limits)

    p.__repr__()


def get_best_curve(metric_column, df_scores, df_curves):
    """Get the curve with the best score and return in tidy-format dataframe."""
    best_idx = df_scores.sort_values(by=metric_column).index[0:1]
    best_scores = df_scores.iloc[best_idx, :].copy()
    best_curves = df_curves[df_curves['idx'].isin(best_idx)].copy()
    best_curves['best_in'] = metric_column

    best_scores['idx'] = best_scores.index
    df_best = best_curves.join(best_scores, 'idx', 'inner', lsuffix='_curve')
    df_best = df_best.drop(columns='idx_curve')
    return df_best


def get_best_parameter_set(metric_col, df_scores, n=3):
    best = df_scores.reset_index()\
                    .sort_values(by=metric_col)\
                    .iloc[0:n]
    return best


def get_best_params_by_metric(metric_col, df_scores_dict, n):
    """
    Example df_scores_dict :

    df_dict_ = {'A' : {0 : pd.DataFrame(*values),
                       1 : pd.DataFrame(*values),
                       2 : pd.DataFrame(*values)},
                'B' : {0 : pd.DataFrame(*values),
                                             ...}
               }
    """
    best_params = []
    for device, samples in df_scores_dict.items():
        for sample, df in samples.items():
            best_param = get_best_parameter_set(metric_col, df, n)
            best_param['device'] = device
            best_param['sample'] = sample
            best_params.append(best_param)
    best_params = pd.concat(best_params)

    for col in best_params:
        try:
            best_params[col] = best_params[col].astype('float')
        except ValueError:
            continue

    return best_params.reset_index(drop=True)  # Prevent indices with same value


# ═════════════════════════════════
which_device = 'A'
which_sample = 3
# ═════════════════════════════════

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
signal_target = df_target['values'].values

plot_results(df_best,
             signal,
             time_target,
             signal_target,
             color_by='best_in')

plot_results_facet(
    df_best,
    signal,
    time_target,
    signal_target,
    color_by='best_in',
    facet_by='best_in'
)

devices = ['A', 'B', 'C']
samples = [0, 1, 2, 3, 4]

# Build df_scores_dict
df_scores_dict = {}
for d in devices:
    samples_dict = {}
    for s in samples:
        df, _ = get_dataframes(d, s)
        samples_dict[s] = df
    df_scores_dict[d] = samples_dict


def scores_dict_to_wide(df_scores_dict):
    df_wide = pd.DataFrame()
    for device, samples in df_scores_dict.items():
        for sample_number, df in samples.items():
            df_temp = df.copy()
            df_temp['sample'] = sample_number
            df_temp['device'] = device
            df_wide = pd.concat([df_wide, df_temp])
    return df_wide


def scores_dict_to_tidy(df_scores_dict, value_columns):
    df_wide = scores_dict_to_wide(df_scores_dict)
    return df_wide.melt(id_vars=['device', 'sample'], value_vars=value_columns)


# Let's do a parameter analysis
df_mech = get_best_params_by_metric('mech_dtw', df_scores_dict, n=1)
df_elec = get_best_params_by_metric('elec_dtw', df_scores_dict, n=1)

df_mech_melt = df_mech.melt(
    id_vars=['device'],
    value_vars=['friction_damping', 'spring_damping', 'coupling_factor']
)

df_mech_melt['metric'] = 'mech_dtw'

df_elec_melt = df_elec.melt(id_vars=['device'],
                            value_vars=['friction_damping', 'spring_damping', 'coupling_factor'])

df_join_melt = pd.concat([df_mech_melt, df_elec_melt])

p = p9.ggplot(p9.aes(x='device', y='value'), df_mech_melt)
p = p \
    + p9.geom_jitter(height=0, width=0.35) \
    + p9.facet_wrap('variable', scales='free_y') \
    + p9.theme(subplots_adjust={'wspace': 0.3}) \
    + p9.ggtitle('Metric: Mechanical DTW')
p.__repr__()

p = p9.ggplot(p9.aes(x='device', y='value'), df_elec_melt)
p = (
    p
    + p9.geom_jitter(height=0, width=0.35)
    + p9.facet_wrap('variable', scales='free_y')
    + p9.theme(subplots_adjust={'wspace': 0.3})
    + p9.ggtitle('Metric: Electrical DTW')
)
p.__repr__()

p = p9.ggplot(p9.aes(x='device', y='value'), df_join_melt)
p = p \
    + p9.geom_jitter(height=0, width=0.35) \
    + p9.facet_wrap('variable', scales='free_y') \
    + p9.theme(subplots_adjust={'wspace': 0.3}) \
    + p9.ggtitle('Metric: Mech + Electrical DTW')
p.__repr__()

df_params_avg = df_join_melt.groupby('variable').mean()
print('Average parameter values:')
print(df_params_avg)


# Let's do a score analysis
scores_tidy = scores_dict_to_tidy(df_scores_dict,
                                  value_columns=['mech_dtw', 'elec_dtw'])
p = p9.ggplot(p9.aes(x='device', y='value'), scores_tidy)
p = p \
    + p9.geom_sina(alpha=0.1) \
    + p9.facet_wrap('variable', scales='free_y') \
    + p9.theme(subplots_adjust={'wspace': 0.3})
p.__repr__()

# Does mech_dtw correlate with elec_dtw?
scores_wide = scores_dict_to_wide(df_scores_dict)
corr = scores_wide[['mech_dtw', 'elec_dtw', 'device']]\
       .groupby('device')\
       .corr()
corr = corr.iloc[np.arange(0, len(corr), 2), 1]\
           .reset_index()\
           .rename({'elec_dtw': 'corr_coeff'}, axis=1)

corr['corr_coeff'] = np.round(corr['corr_coeff'], 3)


p = p9.ggplot(p9.aes(x='mech_dtw', y='elec_dtw'), scores_wide)
p = p \
    + p9.geom_point(alpha=0.3) \
    + p9.geom_smooth() \
    + p9.geom_text(p9.aes(x=0, y=600, label='corr_coeff'), corr, nudge_x=10) \
    + p9.facet_wrap('device') \
    + p9.xlim(0, 50) \

p.__repr__()

p = p9.ggplot(p9.aes(x='mech_dtw', y='elec_rms_perc_diff', color='sample'), scores_wide)
p = p \
    + p9.geom_point(alpha=0.5) \
    + p9.facet_wrap('device') \
    + p9.xlim(0, 50) \
    + p9.ylim(-10, 10)

p.__repr__()

p = p9.ggplot(p9.aes(x='elec_dtw', y='elec_rms_perc_diff', color='sample'), scores_wide)
p = p \
    + p9.geom_point(alpha=0.5) \
    + p9.facet_wrap('device') \
    + p9.xlim(0, 150) \
    + p9.ylim(-10, 10)

p.__repr__()

# Let's make a selection (based on the above graph) and see what the parameters say
scores_select = scores_wide.query('mech_dtw < 15 and -10 < elec_rms_perc_diff < 10')
