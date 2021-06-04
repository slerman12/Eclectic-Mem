import json
import time
from collections import defaultdict

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from pathlib import Path
folder=Path('.')
data=[]
for file in folder.glob('Eclectic-*.json'):
    with open(file,'r') as f:
        data.extend(json.load(f))
results=defaultdict(lambda: defaultdict(list))
for task in data:
    if 'rQdia' not in task['name']: continue
    token = task['name'].split('-')
    method = "-".join(token[:2])
    experiment="-".join(token[2:-1])
    seed = token[-1]
    if not seed.isnumeric(): continue
    # print(method,experiment,seed)
    results[method][experiment].append((seed,task['result']['mean_episode_reward']))
    # print(i['name'])
dfs = []
dfs4plot=[]
for method,experiment in results.items():
    print(method)
    results_step = defaultdict(list)
    df=[]
    df4plot = []
    for exp_name,result in experiment.items():
        # result= result[diff exp]=[(seed,result),()]
        result_mean=[sum(i) / len(i) for i in zip(*[i[1]['y'] for i in result])]
        for step,score in enumerate(result_mean):
            df4plot.append({'envs': exp_name, 'step': step*10000, method: score})


        score_100k = max(result_mean[:11])
        score_500k= max(result_mean)
        results_step["100k"].append((exp_name, score_100k))
        results_step["500k"].append((exp_name, score_500k))
    dfs4plot.append(pd.DataFrame(df4plot))
    for k,v in results_step.items():
        print('\t',k)
        for exp_name,score in sorted(v):
            row={'envs':exp_name,'step':k, method:score}
            df.append(row)
            # print('\t\t',f"{exp_name:25} {score:.2f}")
    dfs.append(pd.DataFrame(df))
dfs=pd.merge(dfs[0],dfs[1],  how='outer',on=['envs','step'])
dfs=dfs.sort_values(by=['envs','step'])
dfs.reset_index(drop=True, inplace=True)
# dfs.to_markdown('mujuco_result.md',index=False)


dfs4plot = pd.merge(dfs4plot[0],dfs4plot[1],how='outer',on=['envs','step'])
dfs4plot=dfs4plot.sort_values(by=['envs','step'])
dfs4plot.reset_index(drop=True, inplace=True)
print()