import json
import re
import socket

# os.environ['CLEARML_CONFIG_FILE'] = str(Path.home() / f"clearml-{socket.getfqdn()}.conf")
from collections import defaultdict

import tqdm
from clearml import Task

update = True

project = "Eclectic-Mem"

file = 'rQdia-drq.json'
exp_done = json.load(open(file))
previous_len = len(exp_done)
if update:
    tasks = Task.get_tasks(project_name=project)
    print('Done getting task')
    for task in tqdm.tqdm(tasks):
        if task.get_last_iteration() < 490 * 1000: continue
        # reports could be more then one
        # ['Training', 'eval', ':monitor:gpu', ':monitor:machine', 'train_actor', 'train_alpha', 'train_critic']
        report = task.get_reported_scalars()
        if 'eval' not in report.keys() or task.name in [i['name'] for i in exp_done]: continue
        exp_done.append({'name': task.name, 'result': report['eval']})
print(f'updated {len(exp_done) - previous_len}')
json.dump(exp_done, open(file, 'w'))

# =============== prepare 4 reliable ==============================
standard_name = ['ball_in_cup_catch',
                 'cartpole_swingup',
                 'cheetah_run',
                 'finger_spin',
                 'reacher_easy',
                 'walker_walk']
steps = ['100k', '500k']
envs = defaultdict(lambda: defaultdict(list))
for exp in exp_done:
    # find the right name
    env_name = re.findall('cheetah|ball_in_cup|finger|walker|cartpole|reacher', exp['name'])[0]
    env_name = [i for i in standard_name if env_name in i][0]
    # find the right seed
    seed = exp['name'].replace(env_name, '').replace('_', '-').split('-')[-1]
    if seed.isnumeric():
        print(env_name)
        reward = exp['result']['mean_episode_reward']
        #
        # envs['100k'][env_name].append(max(reward['y'][:reward['x'].index(100000) + 1]))
        # envs['500k'][env_name].append(max(reward['y'][:]))
        envs['100k'][env_name].append(reward['y'][:reward['x'].index(100000) + 1][-1])
        envs['500k'][env_name].append(reward['y'][:][-1])
for name, exps in envs['500k'].items():
    print(f"{name:>20}->{len(exps)}")
    json.dump(envs, open('reliable/DrQ+rQdia-backup.json', 'w'))
smallest_len = min([len(envs['100k'][i]) for i in standard_name])
print(f'Trim to {smallest_len}')
envs = {step: {env: envs[step][env][:smallest_len] for env in standard_name} for step in steps}
json.dump(envs, open('reliable/DrQ+rQdia.json', 'w'))
print()
