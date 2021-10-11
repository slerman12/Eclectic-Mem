import re
from collections import defaultdict
from pathlib import Path

import json
from collections import defaultdict


def extract():
    envs = defaultdict(dict)
    data = json.load(open('rQdia-drq.json'))
    for i in data:
        # find the right name
        env_name = re.findall('cheetah|ball_in_cup|finger|walker|cartpole|reacher', i['name'])[0]
        # find the right seed
        seed = i['name'].replace(env_name, '').replace('_', '-').split('-')[-1]
        if seed.isnumeric(): envs[env_name][seed] = i
    return envs


result = extract()
for name, v in result.items():
    print(name, len(v.values()))

test_data = result['ball_in_cup']

print()

# dfs = []
# dfs4plot = []
# for method, experiment in results.items():
#     print(method)
#     results_step = defaultdict(list)
#     df = []
#     df4plot = []
#     for exp_name, result in experiment.items():
#         # CURL vs dqr= CURL vs dqr[diff exp]=[(seed,CURL vs dqr),()]
#         result_mean = [sum(i) / len(i) for i in zip(*[i[1]['y'] for i in result])]
#         for step, score in enumerate(result_mean):
#             df4plot.append({'envs': exp_name, 'step': step * 10000, method: score})
#
#         score_100k = max(result_mean[:11])
#         score_500k = max(result_mean)
#         results_step["100k"].append((exp_name, score_100k))
#         results_step["500k"].append((exp_name, score_500k))
#     dfs4plot.append(pd.DataFrame(df4plot))
#     for k, v in results_step.items():
#         print('\t', k)
#         for exp_name, score in sorted(v):
#             row = {'envs': exp_name, 'step': k, method: score}
#             df.append(row)
#             # print('\t\t',f"{exp_name:25} {score:.2f}")
#     dfs.append(pd.DataFrame(df))
# dfs = pd.merge(dfs[0], dfs[1], how='outer', on=['envs', 'step'])
# dfs = dfs.sort_values(by=['envs', 'step'])
# dfs.reset_index(drop=True, inplace=True)
# # dfs.to_markdown('mujuco_result.md',index=False)
#
#
# dfs4plot = pd.merge(dfs4plot[0], dfs4plot[1], how='outer', on=['envs', 'step'])
# dfs4plot = dfs4plot.sort_values(by=['envs', 'step'])
# dfs4plot.reset_index(drop=True, inplace=True)
# print()

# root = Path('CURL vs dqr')
# result = defaultdict(list)
# # , 'finger'
# for exp in ['ballincup', 'cartpole', 'cheetah', 'reacher', 'walker']:
#     for name in ['our', 'CURL']:
#         expname = f"{exp}-{name}"
#         with open(root / expname, 'r') as file:
#             for line in file.readlines():
#                 data = line.split('|')
#                 try:
#                     if 'train' in data[1]:
#                         if int(data[3].split(':')[1]) > 1000 and int(data[3].split(':')[1]) < 100000:
#                             result[expname].append(float(data[5].split(':')[1]))
#                 except:
#                     pass
#
# from clearml import Task, Logger
#
# snapshots_path = Path('./experiments')
# task = Task.init(project_name="Eclectic-Mem", task_name="resultplot", output_uri=str(snapshots_path))
# logger = task.get_logger()
# for expname, rewards in result.items():
#     for step, reward in enumerate(rewards):
#         logger.report_scalar(expname.split('-')[0], expname,
#                              iteration=step, value=reward)
#
# == == == == == == == == =
#
#
# import json
# import random
# from pathlib import Path
# import pandas
# import pandas as pd
# import plotly.express as px
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go
#
# data = []
# for file in Path(__file__).parent.glob('*.csv'):
#     data.append((file.stem, pd.read_csv(file)))
# methods = ['rQdia-sac', 'rQdia-dqr']
# palette = random.sample(px.colors.qualitative.G10, len(methods) + 1)
# method_color = {method: palette[i] for i, method in enumerate(methods)}
# groud_legend = {method: True for method in methods}
#
# exp_num = len(data)
# col = 3
# row = exp_num // col
# fig = make_subplots(rows=row, cols=col, subplot_titles=[' ' for i in range(row * col)])
# for idx, (exp_name, exp_data) in enumerate(data):
#     r_i = idx // col
#     c_i = idx % col
#     x = exp_data['step']
#     for method in methods:
#         fig.add_trace(
#             go.Scatter(x=x, y=exp_data[method], name=method, mode='lines+markers',
#                        showlegend=groud_legend[method],
#                        marker={'color': method_color[method]}),
#             row=r_i + 1, col=c_i + 1
#         )
#         groud_legend[method] = False
#     fig.layout.annotations[r_i * col + c_i]['text'] = exp_name
# fig.show()
# print()
