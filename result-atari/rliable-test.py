# @title Score data for Human and random agent
# @title Atari 100k -- List of 26 Games
# @title Imports
import time

from rliable import library as rly
from rliable import metrics
from rliable import plot_utils
import inspect
import json
import os
import os.path as osp
# See warnings only once
import warnings
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatches
from rliable import plot_utils

warnings.filterwarnings('default')

# The answer to life, universe and everything
RAND_STATE = np.random.RandomState(42)

RANDOM_SCORES = {
    'Alien': 227.8,
    'Amidar': 5.8,
    'Assault': 222.4,
    'Asterix': 210.0,
    'BankHeist': 14.2,
    'BattleZone': 2360.0,
    'Boxing': 0.1,
    'Breakout': 1.7,
    'ChopperCommand': 811.0,
    'CrazyClimber': 10780.5,
    'DemonAttack': 152.1,
    'Freeway': 0.0,
    'Frostbite': 65.2,
    'Gopher': 257.6,
    'Hero': 1027.0,
    'Jamesbond': 29.0,
    'Kangaroo': 52.0,
    'Krull': 1598.0,
    'KungFuMaster': 258.5,
    'MsPacman': 307.3,
    'Pong': -20.7,
    'PrivateEye': 24.9,
    'Qbert': 163.9,
    'RoadRunner': 11.5,
    'Seaquest': 68.4,
    'UpNDown': 533.4
}

HUMAN_SCORES = {
    'Alien': 7127.7,
    'Amidar': 1719.5,
    'Assault': 742.0,
    'Asterix': 8503.3,
    'BankHeist': 753.1,
    'BattleZone': 37187.5,
    'Boxing': 12.1,
    'Breakout': 30.5,
    'ChopperCommand': 7387.8,
    'CrazyClimber': 35829.4,
    'DemonAttack': 1971.0,
    'Freeway': 29.6,
    'Frostbite': 4334.7,
    'Gopher': 2412.5,
    'Hero': 30826.4,
    'Jamesbond': 302.8,
    'Kangaroo': 3035.0,
    'Krull': 2665.5,
    'KungFuMaster': 22736.3,
    'MsPacman': 6951.6,
    'Pong': 14.6,
    'PrivateEye': 69571.3,
    'Qbert': 13455.0,
    'RoadRunner': 7845.0,
    'Seaquest': 42054.7,
    'UpNDown': 11693.2
}


def convert_to_matrix(score_dict):
    keys = sorted(list(score_dict.keys()))
    return np.stack([score_dict[k] for k in keys], axis=1)


ATARI_100K_GAMES = [
    'Alien', 'Amidar', 'Assault', 'Asterix', 'BankHeist', 'BattleZone',
    'Boxing', 'Breakout', 'ChopperCommand', 'CrazyClimber', 'DemonAttack',
    'Freeway', 'Frostbite', 'Gopher', 'Hero', 'Jamesbond', 'Kangaroo',
    'Krull', 'KungFuMaster', 'MsPacman', 'Pong', 'PrivateEye', 'Qbert',
    'RoadRunner', 'Seaquest', 'UpNDown'
]

StratifiedBootstrap = rly.StratifiedBootstrap

IQM = lambda x: metrics.aggregate_iqm(x)  # Interquartile Mean
OG = lambda x: metrics.aggregate_optimality_gap(x, 1.0)  # Optimality Gap
MEAN = lambda x: metrics.aggregate_mean(x)
MEDIAN = lambda x: metrics.aggregate_median(x)


# @title Dataframe saving and reading code for Atari 100k

def save_df(df_to_save, name, base_df_path='atari_100k'):
    base_dir = osp.join(base_df_path, name)
    os.makedirs(base_dir)
    for game in df_to_save.keys():
        file_name = osp.join(base_dir, f'{game}.json')
        with open(file_name, 'w') as f:
            df_to_save[game].to_json(f, orient='records')
        print(f'Saved {file_name}')


# @title Helpers for normalizing scores and plotting histogram plots.


def pgs(f):
    print(inspect.getsource(f))


def score_normalization(res_dict, min_scores, max_scores):
    games = res_dict.keys()
    norm_scores = {}
    for game, scores in res_dict.items():
        norm_scores[game] = (scores - min_scores[game]) / (max_scores[game] - min_scores[game])
    return norm_scores


# def plot_score_hist(score_matrix, bins=20, figsize=(28, 14),
#                     fontsize='xx-large', N=6, extra_row=1,
#                     names=None):
#     num_tasks = score_matrix.shape[1]
#     if names is None:
#         names = ATARI_100K_GAMES
#     N1 = (num_tasks // N) + extra_row
#     fig, ax = plt.subplots(nrows=N1, ncols=N, figsize=figsize)
#     for i in range(N):
#         for j in range(N1):
#             idx = j * N + i
#             if idx < num_tasks:
#                 ax[j, i].set_title(names[idx], fontsize=fontsize)
#                 sns.histplot(score_matrix[:, idx], bins=bins, ax=ax[j, i], kde=True)
#             else:
#                 ax[j, i].axis('off')
#             decorate_axis(ax[j, i], wrect=5, hrect=5, labelsize='xx-large')
#             ax[j, i].xaxis.set_major_locator(plt.MaxNLocator(4))
#             if idx % N == 0:
#                 ax[j, i].set_ylabel('Count', size=fontsize)
#             else:
#                 ax[j, i].yaxis.label.set_visible(False)
#             ax[j, i].grid(axis='y', alpha=0.1)
#     return fig


def read_df(name, base_df_path='atari_100k'):
    base_dir = osp.join(base_df_path, name)
    df_to_read = {}
    for game in ATARI_100K_GAMES:
        file_name = osp.join(base_dir, f'{game}.json')
        with open(file_name, 'r') as f:
            df_to_read[game] = pd.read_json(f, orient='records')
    return df_to_read


def remove_additional_evaluations(df_to_filter):
    new_df = {}
    for key in df_to_filter.keys():
        df_a = df_to_filter[key]
        df_a = df_a[df_a.index == '0'].drop(['iteration'], axis=1)
        new_df[key] = df_a
    return new_df


# @title Helpers for loading Atari 100k data

def create_score_dict_atari_100k(main_df, normalization=True,
                                 evaluation_key='eval_average_return'):
    """Creates a dictionary of scores."""
    score_dict = {}
    for key, df in main_df.items():
        score_dict[key] = df[evaluation_key].values
    if normalization:
        score_dict = score_normalization(score_dict, RANDOM_SCORES, HUMAN_SCORES)
    return score_dict


def get_scores(df, normalization=True, eval='Final'):
    score_dict_df = create_score_dict_atari_100k(df, normalization=normalization)
    score_matrix = convert_to_matrix(score_dict_df)
    median, mean = MEDIAN(score_matrix), MEAN(score_matrix)
    print('{}: Median: {}, Mean: {}'.format(eval, median, mean))
    return score_dict_df, score_matrix


def load_and_read_scores(algorithm_name, num_evals=None):
    print(f'Loading scores for {algorithm_name}:')
    df = read_df(algorithm_name)
    if num_evals is None:
        return get_scores(df)
    # Read multiple evals.
    final_scores_df, max_scores_df = {}, {}
    for game, game_df in df.items():
        final_scores_df[game] = game_df[game_df['iteration'] == num_evals - 1]
        max_scores_df[game] = game_df.groupby('run_number').max()
    return get_scores(final_scores_df), get_scores(max_scores_df, eval='Max')


def read_curl_scores():
    print(f'Loading scores for CURL:')
    df = pd.read_json('atari_100k/CURL_10_evals.json', orient='records')
    score_dict = {'Max': {}, 'Final': {}}
    for game in ATARI_100K_GAMES:
        game_df = df[df['game'] == game]
        score_dict['Final'][game] = game_df['HNS'].values
        score_dict['Max'][game] = game_df['Max HNS'].values
    score_matrices = {}
    for key, val in score_dict.items():
        score_matrices[key] = convert_to_matrix(val)
        median, mean = MEDIAN(score_matrices[key]), MEAN(score_matrices[key])
        print('{}: Median: {}, Mean: {}'.format(key, median, mean))
    return (score_dict['Final'], score_matrices['Final']), (
        score_dict['Max'], score_matrices['Max'])


def load_json_scores(algorithm_name, base_path='atari_100k'):
    print(f'Loading scores for {algorithm_name}:')
    path = osp.join(base_path, f'{algorithm_name}.json')
    with open(path, 'r') as f:
        scores = json.load(f)
    scores = {game: np.array(val) for game, val in scores.items()}
    scores = score_normalization(scores, RANDOM_SCORES, HUMAN_SCORES)
    score_matrix = convert_to_matrix(scores)
    median, mean = MEDIAN(score_matrix), MEAN(score_matrix)
    print('{}: Median: {}, Mean: {}'.format(eval, median, mean))
    return scores, score_matrix


# @title Load all score dicts

(score_dict_der, score_der), (_, score_der_max) = load_and_read_scores(
    'DER', num_evals=10)
(score_dict_curl, score_curl), (_, score_curl_max) = read_curl_scores()

score_dict_otr, score_otr = load_json_scores('OTRainbow')
score_dict_drq, score_drq = load_json_scores('DrQ')
score_dict_spr, score_spr = load_json_scores('SPR')
score_dict_simple, score_simple = load_json_scores('SimPLe')
score_dict_rqdia, score_rqida = load_json_scores('DER+rQdia')
# DrQ agent but with standard epsilon values of 0.01/0.001 for training
# and evaluation eps-greedy parameters
score_dict_drq_eps, score_drq_eps = load_json_scores('DrQ(eps)')

score_data_dict = {'CURL': score_curl,
                   'DrQ': score_drq,
                   'DrQ(ε)': score_drq_eps,
                   'DER': score_der,
                   'SimPLe': score_simple,
                   'OTR': score_otr,
                   'SPR': score_spr,
                   'DER+rQdia': score_rqida}
colors = sns.color_palette('colorblind')
xlabels = ['DER', 'OTR', 'CURL', 'DrQ(ε)', 'SPR', 'SimPLe', 'DrQ', 'DER+rQdia']
color_idxs = [0, 3, 4, 2, 1, 7, 8, 9]
ATARI_100K_COLOR_DICT = dict(zip(xlabels, [colors[idx] for idx in color_idxs]))
atari_100k_score_dict = {key: val[:10] for key, val in score_data_dict.items()}


def subsample_scores(score_dict, n=5, replace=False):
    subsampled_dict = {}
    total_samples = len(score_dict[list(score_dict.keys())[0]])
    for game, scores in score_dict.items():
        indices = np.random.choice(range(total_samples), size=n, replace=replace)
        subsampled_dict[game] = scores[indices]
    return subsampled_dict


def subsample_scores_mat(score_mat, num_samples=5, replace=False):
    subsampled_dict = []
    total_samples, num_games = score_mat.shape
    subsampled_scores = np.empty((num_samples, num_games))
    for i in range(num_games):
        indices = np.random.choice(total_samples, size=num_samples, replace=replace)
        subsampled_scores[:, i] = score_mat[indices, i]
    return subsampled_scores


def subsample_seeds(score_mat, num_samples=5, replace=False):
    indices = np.random.choice(
        score_mat.shape[0], size=num_samples, replace=replace)
    return score_mat[indices]


def batch_subsample_seeds(score_mat, num_samples=5, batch_size=100,
                          replace=False):
    indices = [
        np.random.choice(score_mat.shape[0], size=num_samples, replace=replace)
        for _ in range(batch_size)
    ]
    return (score_mat[idx] for idx in indices)


def subsample_scores_mat_with_replacement(score_mat, num_samples=5):
    subsampled_dict = []
    total_samples, num_games = score_mat.shape
    indices = np.random.choice(
        total_samples, size=(num_samples, num_games), replace=True)
    col_indices = np.expand_dims(np.arange(num_games), axis=0)
    col_indices = np.repeat(col_indices, num_samples, axis=0)
    subsampled_scores = score_mat[indices, col_indices]
    return subsampled_scores


# @title Aggregate computation helpers

SIZES = [3, 5, 10, 25, 50, 100]


def calc_aggregate_fn(score_data, num_samples=5, total_n=20000,
                      aggregate_fn=MEDIAN, replace=False):
    subsampled_scores = batch_subsample_seeds(
        score_data, num_samples, batch_size=total_n, replace=replace)
    aggregates = [aggregate_fn(scores) for scores in subsampled_scores]
    return np.array(aggregates)


def calculate_aggregate_varying_sizes(score_matrix, aggregate_fn, total_n=20000,
                                      sizes=None, replace=False):
    agg_dict = {}
    if sizes is None:
        sizes = SIZES
    for size in sizes:
        agg_dict[n] = calc_aggregate_fn(score_matrix, num_samples=size, aggregate_fn=aggregate_fn,
                                        total_n=total_n, replace=replace)
        print('Mean Aggregate: {}'.format(np.mean(agg_dict[n])))
    return agg_dict


def CI(bootstrap_dist, stat_val=None, alpha=0.05, is_pivotal=False):
    """
    Get the bootstrap confidence interval for a given distribution.
    Args:
      bootstrap_distribution: numpy array of bootstrap results.
      stat_val: The overall statistic that this method is attempting to
        calculate error bars for. Default is None.
      alpha: The alpha value for the confidence intervals.
      is_pivotal: if true, use the pivotal (reverse percentile) method.
        If false, use the percentile method.
    Returns:
      (low, high): The lower and upper limit for `alpha` x 100% CIs.
      val: The median value of the bootstrap distribution if `stat_val` is None
        else `stat_val`.
    """
    # Adapted from https://pypi.org/project/bootstrapped
    if is_pivotal:
        assert stat_val is not None, 'Please pass the statistic for a pivotal'
        'confidence interval'
        low = 2 * stat_val - np.percentile(bootstrap_dist, 100 * (1 - alpha / 2.))
        val = stat_val
        high = 2 * stat_val - np.percentile(bootstrap_dist, 100 * (alpha / 2.))
    else:
        low = np.percentile(bootstrap_dist, 100 * (alpha / 2.))
        val = np.percentile(bootstrap_dist, 50)
        high = np.percentile(bootstrap_dist, 100 * (1 - alpha / 2.))
    return (low, high), val


# ====================================== first result================================================================


# ====================================== Start darwing ==============================
suffix = 'w_SPR'

algorithms = ['SimPLe', 'DER', 'OTR', 'CURL', 'DrQ', 'SPR', 'DER+rQdia']


def save_fig(fig, name, suffix=suffix):
    file_name = f'{name}_{suffix}.pdf'
    fig.savefig(file_name, format='pdf', bbox_inches='tight')
    # files.download(file_name)
    return file_name


aggregate_func = lambda x: np.array([MEDIAN(x), IQM(x), MEAN(x), OG(x)])
start = time.time()
aggregate_scores, aggregate_interval_estimates = rly.get_interval_estimates(
    atari_100k_score_dict, aggregate_func, reps=50000)
fig, axes = plot_utils.plot_interval_estimates(
    aggregate_scores,
    aggregate_interval_estimates,
    metric_names=['Median', 'IQM', 'Mean', 'Optimality Gap'],
    algorithms=algorithms,
    colors=ATARI_100K_COLOR_DICT,
    xlabel_y_coordinate=-0.16,
    xlabel='Human Normalized Score')
# axes[-1].legend(loc='upper right')
plt.show()
# this one is tricky, we cannot only see the legend by saving a fig
save_fig(fig, 'atari_aggregates')

from scipy.stats.stats import find_repeats

# ======================================================================================================
# @title Calculate score distributions and average score distributions for for Atari 100k

# algorithms = ['SimPLe', 'DER', 'OTR', 'CURL', 'DrQ(ε)', 'SPR', 'DER+rQdia']
# algorithms = ['SimPLe', 'DER', 'OTR', 'CURL', 'DrQ', 'DER+rQdia']
# algorithms = ['SimPLe', 'DER', 'OTR', 'CURL', 'DrQ', 'SPR', 'DER+rQdia']

score_dict = {key: score_data_dict[key][:10] for key in algorithms}
ATARI_100K_TAU = np.linspace(0.0, 2.0, 201)
# Higher value of reps corresponds to more accurate estimates but are slower
# to computed. `reps` corresponds to number of bootstrap resamples.
reps = 2000

score_distributions, score_distributions_cis = rly.create_performance_profile(
    score_dict, ATARI_100K_TAU, reps=reps)
avg_score_distributions, avg_score_distributions_cis = rly.create_performance_profile(
    score_dict, ATARI_100K_TAU, use_score_distribution=False, reps=reps)
# @title Plot score distributions and contrast with average score distribution
# fig, axes = plt.subplots(ncols=2, figsize=(14.5, 6))

# plot_utils.plot_performance_profiles(
#     score_distributions, ATARI_100K_TAU,
#     performance_profile_cis=score_distributions_cis,
#     colors=ATARI_100K_COLOR_DICT,
#     xlabel=r'Human Normalized Score $(\tau)$',
#     labelsize='xx-large',
#     ax=axes[0])
#
# plot_utils.plot_performance_profiles(
#     avg_score_distributions, ATARI_100K_TAU,
#     performance_profile_cis=avg_score_distributions_cis,
#     colors=ATARI_100K_COLOR_DICT,
#     xlabel=r'Human Normalized Score $(\tau)$',
#     ylabel=r'Fraction of tasks with score > $\tau$',
#     labelsize='xx-large',
#     ax=axes[1])
# # axes[0].legend(loc='upper right')
# axes[1].legend(loc='upper right')
# axes[0].axhline(0.5, ls='--', color='k', alpha=0.4)
# axes[1].axhline(0.5, ls='--', color='k', alpha=0.4)
#
# fake_patches = [mpatches.Patch(color=ATARI_100K_COLOR_DICT[alg],
#                                alpha=0.75) for alg in algorithms]
# legend = fig.legend(fake_patches, algorithms, loc='upper center',
#                     fancybox=True, ncol=len(algorithms),
#                     fontsize='x-large',
#                     bbox_to_anchor=(0.45, 1.1))
# fig.subplots_adjust(top=0.92, wspace=0.24)
#
# plt.show()
# ======================================================================================================
# @title Performance profiles for Atari 100K

fig, ax = plt.subplots(ncols=2, figsize=(7 * 2, 4))
plot_utils.plot_performance_profiles(
    score_distributions, ATARI_100K_TAU,
    performance_profile_cis=score_distributions_cis,
    colors=ATARI_100K_COLOR_DICT,
    xlabel=r'Human Normalized Score $(\tau)$',
    labelsize='xx-large',
    ax=ax[0])
ax[0].set_title('Score Distributions ', size='x-large')
xticks = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0]
plot_utils.plot_performance_profiles(
    score_distributions, ATARI_100K_TAU,
    performance_profile_cis=score_distributions_cis,
    colors=ATARI_100K_COLOR_DICT,
    xlabel=r'Human Normalized Score $(\tau)$',
    ylabel=r'Fraction of tasks with score > $\tau$',
    labelsize='xx-large',
    use_non_linear_scaling=True,
    xticks=xticks,
    ax=ax[1])
ax[1].set_title('Score Distributions with Non Linear Scaling', size='x-large')
# ax[0].legend(loc='upper right')
fake_patches = [mpatches.Patch(color=ATARI_100K_COLOR_DICT[alg],
                               alpha=0.75) for alg in algorithms]
legend = fig.legend(fake_patches, algorithms, loc='upper center',
                    fancybox=True, ncol=len(algorithms),
                    fontsize='x-large',
                    bbox_to_anchor=(0.45, 1.15))
# fig.subplots_adjust(wspace=0.24)
save_fig(fig, 'atari-performance-profiles')
plt.show()
# ======================================================================================================
# @title Compute Probability of Improvement for all pair of algorithms

all_keys = atari_100k_score_dict.keys()

# algorithms = ['SimPLe', 'DER', 'CURL', 'DrQ(ε)', 'SPR', 'DER+rQdia']
# algorithms = ['SimPLe', 'DER', 'OTR', 'CURL', 'DrQ', 'SPR', 'DER+rQdia']
all_pairs = {}
for alg in algorithms:
    all_pairs[alg] = {}
    for alg2 in (algorithms):
        pair_name = f'{alg}_{alg2}'
        all_pairs[alg][pair_name] = (atari_100k_score_dict[alg], atari_100k_score_dict[alg2])

probabilities, probability_cis = {}, {}
# reps = 1000 is quite slow as the probabilities are computed for all
# 36 pairs
# Use a smaller reps if running interactively
# reps = 1000
reps = 50
algorithms = ['SimPLe', 'DER', 'OTR', 'CURL', 'DrQ', 'SPR', 'DER+rQdia']
for alg in algorithms:
    probabilities[alg], probability_cis[alg] = rly.get_interval_estimates(
        all_pairs[alg], metrics.probability_of_improvement, reps=reps)
# @title Plot probabilities of improvement
# algorithms = ['SimPLe', 'DER', 'OTR', 'CURL', 'DrQ(ε)', 'SPR', 'DER+rQdia']
nrows = 2
ncols = 6 // nrows
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(9, 8))
h = 0.6

for idx, alg in enumerate(algorithms[1:]):
    row, col = idx // 3, idx % 3
    ax = axes[row, col]
    cis = probability_cis[alg]
    probs = probabilities[alg]
    for i, (alg_pair, prob) in enumerate(probs.items()):
        _, alg1 = alg_pair.split('_')
        (l, u) = cis[alg_pair]
        ax.barh(y=i, width=u - l, height=h,
                left=l, color=ATARI_100K_COLOR_DICT[alg1],
                alpha=0.75)
        ax.vlines(x=prob, ymin=i - 7.5 * h / 16, ymax=i + (6 * h / 16),
                  color='k', alpha=0.85)
    ax.set_yticks(range(len(algorithms)))
    if idx % ncols == 0:
        ax.set_yticklabels(algorithms)
    else:
        ax.set_yticklabels([])

    ax.set_ylabel(r'Algorithm $X$', size='xx-large')
    ax.set_title(fr'P({alg} > $X$)', size='xx-large')
    plot_utils._annotate_and_decorate_axis(ax, labelsize='xx-large', ticklabelsize='xx-large')
    ax.xaxis.set_major_locator(MaxNLocator(4))
fig.subplots_adjust(wspace=0.25, hspace=0.45)
plt.show()
save_fig(fig, 'atari-which-better')
print()
