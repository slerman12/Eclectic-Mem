import collections
import json
import numpy as np
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatches
import seaborn as sns

# @title Plot Mean Scores
IQM = lambda x: metrics.aggregate_iqm(x)  # Interquartile Mean
OG = lambda x: metrics.aggregate_optimality_gap(x, 1.0)  # Optimality Gap
MEAN = lambda x: metrics.aggregate_mean(x)
MEDIAN = lambda x: metrics.aggregate_median(x)
DMC_ENVS = sorted(['ball_in_cup_catch',
                   'cartpole_swingup',
                   'cheetah_run',
                   'finger_spin',
                   'reacher_easy',
                   'walker_walk'])


def decorate_axis(ax, wrect=10, hrect=10, labelsize='large'):
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    # Deal with ticks and the blank space at the origin
    ax.tick_params(length=0.1, width=0.1, labelsize=labelsize)
    # Pablos' comment
    ax.spines['left'].set_position(('outward', hrect))
    ax.spines['bottom'].set_position(('outward', wrect))


def save_fig(fig, name):
    file_name = '{}.pdf'.format(name)
    fig.savefig(file_name, format='pdf', bbox_inches='tight')
    # files.download(file_name)
    return file_name


def convert_to_matrix(score_dict):
    keys = sorted(list(score_dict.keys()))
    return np.stack([score_dict[k] for k in keys], axis=1)


# algorithm.steps.exps = list of runs

def read_dmc_json(algorithm, steps='100k'):
    assert steps in ['100k', '500k']
    with open(f'reliable/{algorithm}.json', 'r') as f:
        out = json.load(f)
    out = out[steps]
    # out['finger_spin'] = out['finger_spin'][:10]
    print(algorithm, steps, {k: np.round(np.mean(v), 1) for k, v in out.items()})
    return {k: np.round(v, 1) for k, v in out.items()}


# ==============================TABLE=================================================

# algs = ['DrQ+rQdia', 'DrQ', 'SLAC', 'SAC+AE', 'PISAC', 'RAD', 'Dreamer']


DMC_ENVS = sorted(['ball_in_cup_catch',
                   'cartpole_swingup',
                   'cheetah_run',
                   'finger_spin',
                   'reacher_easy',
                   'walker_walk'])
# 'ball_in_cup_catch',
#  'cartpole_swingup',
#  'cheetah_run',
#  'finger_spin',
#  'reacher_easy',
#  'walker_walk'

convert = lambda x: {step: {game: score for game, score in zip(DMC_ENVS, x[step])} for step in ['100k', '500k']}
CURL = {
    '500k': [958, 861, 500, 874, 904, 906],
    '100k': [772, 592, 307, 779, 517, 344]
}
SACS = {
    '500k': [979, 870, 772, 929, 975, 964],
    '100k': [957, 812, 228, 672, 919, 604]
}
# PlaNet = {
#     '500k': [939, 787, 568, 718, 588, 478],
#     '100k': [718, 563, 165, 563, 82, 221]
# }
algs = ['DrQ+rQdia', 'DrQ', 'CURL', 'RAD', 'SAC+AE', 'SACS']
raw_score = {}
for steps in ['500k', '100k']:
    buff = {}
    for alg in algs:
        if alg == 'CURL':
            res = convert(CURL)[steps]
        elif alg == 'SACS':
            res = convert(SACS)[steps]
        else:
            res = read_dmc_json(alg, steps)
        buff[alg] = res
    raw_score[steps] = buff

# PlaNet = convert(PlaNet)
for step, subtable in raw_score.items():
    for game in DMC_ENVS:
        row = [np.round(np.mean(i[game]), 2) for i in subtable.values()]
        max_s = max(row[:-1])
        means = [
            f"{mean:0.3f}".rstrip('0').rstrip('.') if mean != max_s else "$\mathbf{" + f'{mean:0.3f}'.rstrip(
                '0').rstrip('.') + "}$"
            for mean in row]
        game = ' '.join([i.title() for i in game.split('_')])
        print(f"{game}&" + ' & '.join(means) + '\\\\')
    print('====================================')
# ================================= PLOT FOR riable ======================================
algs = ['SLAC', 'SAC+AE', 'PISAC', 'RAD', 'DrQ', 'Dreamer', 'DrQ+rQdia']
dmc_scores = {steps: {alg: convert_to_matrix(read_dmc_json(alg, steps)) for alg in algs} for steps in ['100k', '500k']}
normalized_dmc_scores = {steps: {alg: scores / 1000 for alg, scores in dmc_scores[steps].items()} for steps in
                         ['100k', '500k']}
# @title setup colors

# colors = sns.color_palette("Paired")
# algs = ['SLAC', 'SAC+AE', 'PISAC', 'RAD', 'DrQ',  'SUNRISE', 'Dreamer', 'CURL-D2RL', 'PlaNet', 'DrQ+rQdia']
# color_idxs = [0, 3, 4, 2, 1] + list(range(9, 4, -1)) + [10]
colors = sns.color_palette('colorblind')
algs = ['SLAC', 'SAC+AE', 'PISAC', 'RAD', 'DrQ', 'Dreamer', 'DrQ+rQdia']
color_idxs = [3, 4, 2, 1, 7, 8, 9]
DMC_COLOR_DICT = dict(zip(algs, [colors[idx] for idx in color_idxs]))
# @title Calculate score distributions on DM Control
dmc_tau = np.linspace(0.0, 1.0, 21)
perf_prof_dmc = {}
perf_prof_dmc_cis = {}
for steps in normalized_dmc_scores.keys():
    perf_prof_dmc[steps], perf_prof_dmc_cis[steps] = rly.create_performance_profile(
        normalized_dmc_scores[steps], dmc_tau, reps=5000)

# @title Performance profiles on DM Control

algorithms = ['SLAC', 'SAC+AE', 'Dreamer', 'PISAC', 'RAD', 'DrQ', 'DrQ+rQdia']
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(5, 4))
color_dict = dict(zip(algorithms, sns.color_palette('colorblind')))
steps = '500k'
for i, steps in enumerate(['100k', '500k']):
    plot_utils.plot_performance_profiles(
        perf_prof_dmc[steps], dmc_tau,
        performance_profile_cis=perf_prof_dmc_cis[steps],
        colors=DMC_COLOR_DICT,
        ylabel='',
        xlabel=r'Normalized Score $(\tau)$',
        labelsize='xx-large',
        ax=ax[i])
    ax[i].set_title(f'{steps} steps', size='x-large')
ax[0].set_xlabel('')
fig.subplots_adjust(hspace=0.6)
fig.text(x=-0.05, y=0.1, s=r'Fraction of runs with score $> \tau$', rotation=90, size=15)
save_fig(fig, 'mujoco-performance-profiles')


# ====================================================
# @title Rank Computation Helpers

def subsample_scores_mat(score_mat, num_samples=5, replace=False):
    subsampled_dict = []
    total_samples, num_games = score_mat.shape
    subsampled_scores = np.empty((num_samples, num_games))
    for i in range(num_games):
        indices = np.random.choice(total_samples, size=num_samples, replace=replace)
        subsampled_scores[:, i] = score_mat[indices, i]
    return subsampled_scores


def get_rank_matrix(score_dict, n=100000, algorithms=None):
    arr = []
    if algorithms is None:
        algorithms = sorted(score_dict.keys())
    print(f'Using algorithms: {algorithms}')
    for alg in algorithms:
        arr.append(subsample_scores_mat(
            score_dict[alg], num_samples=n, replace=True))
    X = np.stack(arr, axis=0)
    num_algs, _, num_tasks = X.shape
    all_mat = []
    for task in range(num_tasks):
        # Sort based on negative scores as rank 0 corresponds to minimum value,
        # rank 1 corresponds to second minimum value when using lexsort.
        task_x = -X[:, :, task]
        # This is done to randomly break ties.
        rand_x = np.random.random(size=task_x.shape)
        # Last key is the primary key,
        indices = np.lexsort((rand_x, task_x), axis=0)
        mat = np.zeros((num_algs, num_algs))
        for rank in range(num_algs):
            cnts = collections.Counter(indices[rank])
            mat[:, rank] = np.array([cnts[i] / n for i in range(num_algs)])
        all_mat.append(mat)
    all_mat = np.stack(all_mat, axis=0)
    return all_mat


# @title Test where Algo 4 > 3 > 2 > 1 > 0.
dmc_score_dict = dmc_scores['100k']

sdc = {k: dmc_score_dict['DrQ'] + k * 100 for k in range(7)}
all_ranks = get_rank_matrix(sdc, 100000)
mean_ranks = np.mean(all_ranks, axis=0)
mean_ranks_all = {}
all_ranks_individual = {}
for key in ['100k', '500k']:
    dmc_score_dict = dmc_scores[key]
    algs = ['SLAC', 'SAC+AE', 'Dreamer', 'PISAC', 'RAD', 'DrQ', 'DrQ+rQdia']
    all_ranks = get_rank_matrix(dmc_score_dict, 200000, algorithms=algs)
    mean_ranks_all[key] = np.mean(all_ranks, axis=0)
    all_ranks_individual[key] = all_ranks
# @title Plot individual ranks on 6 tasks

keys = algs
labels = list(range(1, len(keys) + 1))
width = 1.0  # the width of the bars: can also be len(x) sequence

fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(14, 3.5))
all_ranks = all_ranks_individual['100k']
for task in range(6):
    bottom = np.zeros_like(mean_ranks[0])
    for i, key in enumerate(keys):
        ranks = all_ranks[task]
        ax = axes[task]
        ax.bar(labels, ranks[i], width, color=DMC_COLOR_DICT[key], bottom=bottom, alpha=0.9)
        bottom += ranks[i]
        ax.set_title(DMC_ENVS[task], fontsize='large')
    if task == 0:
        ax.set_ylabel('Distribution', size='x-large')
    ax.set_xlabel('Ranking', size='x-large')
    ax.set_xticks(labels)
    ax.set_ylim(0, 1)
    ax.set_xticklabels(labels, size='large')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='both', which='both', bottom=False, top=False,
                   left=False, right=False, labeltop=False,
                   labelbottom=True, labelleft=False, labelright=False)

fake_patches = [mpatches.Patch(color=DMC_COLOR_DICT[m], alpha=0.75)
                for m in keys]
legend = fig.legend(fake_patches, keys, loc='upper center',
                    fancybox=True, ncol=len(keys), fontsize='x-large')
fig.subplots_adjust(top=0.78, wspace=0.1, hspace=0.05)
plt.show()
save_fig(fig, 'mujoco-rank-per-game')

# ======================================================================

mean_func = lambda x: np.array([MEAN(x)])
all_mean_CIs, score_dmc_all = {}, {}
for steps in ['100k', '500k']:
    score_dmc_all[steps], all_mean_CIs[steps] = rly.get_interval_estimates(
        normalized_dmc_scores[steps], mean_func, reps=50000)
# @title Manually list scores

means_100k, means_500k = {}, {}
stds_100k, stds_500k = {}, {}
ENVS = ['finger_spin', 'cartpole_swingup', 'reacher_easy',
        'cheetah_run', 'walker_walk', 'ball_in_cup_catch']

# CURL https://arxiv.org/pdf/2004.04136.pdf
means_500k['CURL'] = [926, 841, 929, 518, 902, 959]
means_100k['CURL'] = [767, 582, 538, 299, 403, 769]

stds_500k['CURL'] = [45, 45, 44, 28, 43, 27]
stds_100k['CURL'] = [56, 146, 233, 48, 24, 43]

# SUNRISE
means_500k['SUNRISE'] = [983, 876, 982, 678, 953, 969]
means_100k['SUNRISE'] = [905, 591, 722, 413, 667, 663]

stds_500k['SUNRISE'] = [1, 4, 3, 46, 13, 5]
stds_100k['SUNRISE'] = [57, 55, 50, 35, 147, 241]

# CURL-D2RL
means_100k['CURL-D2RL'] = [837, 836, 754, 253, 540, 880]
means_500k['CURL-D2RL'] = [970, 859, 929, 386, 931, 955]

stds_500k['CURL-D2RL'] = [14, 8, 62, 115, 24, 15]
stds_100k['CURL-D2RL'] = [18, 34, 168, 57, 153, 48]

import scipy


def mean_CI(mean, std, n, confidence=0.95):
    se = std / np.sqrt(n)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return np.array([mean - h, mean + h]) / 1000


ci_algs = {'100k': {}, '500k': {}}
mean_algs = {'100k': {}, '500k': {}}
for alg in ['CURL', 'SUNRISE', 'CURL-D2RL']:
    if alg == 'SUNRISE':
        num_seeds = 5
    else:
        num_seeds = 10
    for steps in ['100k', '500k']:
        if steps == '100k':
            std = np.sqrt(np.mean(np.square(stds_100k[alg])))
            mean = np.mean(means_100k[alg])
        else:
            std = np.sqrt(np.mean(np.square(stds_500k[alg])))
            mean = np.mean(means_500k[alg])
        ci_algs[steps][alg] = mean_CI(mean, std, num_seeds)
        mean_algs[steps][alg] = mean / 1000

for step in mean_algs.keys():
    all_mean_CIs[step].update(ci_algs[step])
    score_dmc_all[step].update(mean_algs[step])

steps = ['100k', '500k']
fig, axes = plt.subplots(nrows=1, ncols=len(steps), figsize=(8, 2.8))
h = 0.6
algs = ['SLAC', 'SAC+AE', 'Dreamer', 'PISAC', 'RAD', 'DrQ', 'DrQ+rQdia']
for idx, step in enumerate(steps):
    perf_res = all_mean_CIs[step]
    score_dmc_step = score_dmc_all[step]
    ax = axes[idx]
    for i, alg in enumerate(algs):
        (l, u), p = perf_res[alg], score_dmc_step[alg]
        ax.barh(y=i, width=u - l, height=h,
                left=l, color=DMC_COLOR_DICT[alg],
                alpha=0.75, label=alg, )
        ax.vlines(x=p, ymin=i - h / 2, ymax=i + (6 * h / 16),
                  label=alg, color='k', alpha=0.75)
    ax.set_yticks(list(range(len(algs))))
    if idx != 0:
        ax.set_yticklabels([])
    else:
        ax.set_yticklabels(algs, fontsize='x-large')
    ax.set_title(steps[idx] + ' steps', fontsize='xx-large')
    ax.tick_params(axis='both', which='major', labelsize='x-large')
    ax.set_xlabel('Normalized Scores', fontsize='xx-large')
    decorate_axis(ax, wrect=5, labelsize='xx-large')
    ax.spines['left'].set_visible(False)
    ax.grid(alpha=0.2, axis='x')
plt.subplots_adjust(wspace=0.05)
plt.show()
save_fig(fig, 'mujoco-sampled-mean-scores')
