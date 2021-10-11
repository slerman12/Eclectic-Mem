import json
import random
from collections import defaultdict

# with open('rQdia-rainbow.json', 'r') as f:
#     data = json.load(f)
# games_res=defaultdict(list)
# for k,v in data.items():
#     tokens=k.split('-')
#     method=f"{tokens[0]}-{tokens[1]}"
#     game=tokens[-2]
#     game = game[:-1] if not game[0].isnumeric() else game[1:]
#     for i in v:
#         games_res[game].append(i)
# res4print=[]
# for game, results in games_res.items():
#     mean_score = []
#     for CURL vs dqr in results:
#         for line in [i for i in CURL vs dqr if i['name'] == 'Mean']:
#             mean_score.append(line['y'])
#     mean_score = [sum(i) / len(i) for i in zip(*mean_score)]
#     res4print.append((game,mean_score))
# for name,score in sorted(res4print):
#     print(f"{name:15}| {max(score):0.2f}")

with open('curl-rainbow.json', 'r') as f:
    data = json.load(f)
games_res = defaultdict(list)
for k, v in data.items():
    tokens = k.split('-')
    method = f"{tokens[0]}-{tokens[1]}"
    # print(tokens)
    game = tokens[-2].replace('curlr', '')
    for i in v:
        games_res[game].append(i)
res4print = []
for game, results in games_res.items():
    mean_score = []
    for result in results:
        for line in [i for i in result if i['name'] == 'Mean']:
            mean_score.append(line['y'])
    mean_score = [sum(i) / len(i) for i in zip(*mean_score)]
    res4print.append((game, mean_score))
for name, score in sorted(res4print):
    print(f"{name:15}| {max(score):0.2f}")
