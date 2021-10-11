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

DMC_ENVS = sorted(['ball_in_cup_catch',
                   'cartpole_swingup',
                   'cheetah_run',
                   'finger_spin',
                   'reacher_easy',
                   'walker_walk', 'DrQ+rQdia'])


def convert_to_matrix(score_dict):
    keys = sorted(list(score_dict.keys()))
    return np.stack([score_dict[k] for k in keys], axis=1)


# algorithm.steps.exps = list of runs

def read_dmc_json(algorithm, steps='100k'):
    assert steps in ['100k', '500k']
    with open(f'reliable/{algorithm}.json', 'r') as f:
        out = json.load(f)
    out = out[steps]
    out['finger_spin'] = out['finger_spin'][:10]
    print(algorithm, steps, {k: np.round(np.mean(v), 1) for k, v in out.items()})
    return {k: np.round(v, 1) for k, v in out.items()}


algs = ['SLAC', 'SAC+AE', 'Dreamer', 'PISAC', 'RAD', 'DrQ', 'DrQ+rQdia']
dmc_scores = {steps: {alg: convert_to_matrix(read_dmc_json(alg, steps)) for alg in algs} for steps in ['100k', '500k']}
normalized_dmc_scores = {steps: {alg: scores / 1000 for alg, scores in dmc_scores[steps].items()} for steps in
                         ['100k', '500k']}
# @title setup colors

colors = sns.color_palette("Paired")
algs = ['SLAC', 'SAC+AE', 'PISAC', 'RAD', 'DrQ', 'CURL', 'SUNRISE', 'Dreamer', 'CURL-D2RL', 'PlaNet', 'DrQ+rQdia']
color_idxs = [0, 3, 4, 2, 1] + list(range(9, 4, -1)) + [10]
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
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
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
    ax[i].legend(loc='lower left')
ax[0].set_xlabel('')
fig.subplots_adjust(hspace=0.6)
fig.text(x=-0.05, y=0.1, s=r'Fraction of runs with score $> \tau$', rotation=90, size=15)
fig.show()
