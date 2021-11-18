import collections
import json
import random

ATARI_100K_GAMES = [
    'Alien', 'Amidar', 'Assault', 'Asterix', 'BankHeist', 'BattleZone',
    'Boxing', 'Breakout', 'ChopperCommand', 'CrazyClimber', 'DemonAttack',
    'Freeway', 'Frostbite', 'Gopher', 'Hero', 'Jamesbond', 'Kangaroo',
    'Krull', 'KungFuMaster', 'MsPacman', 'Pong', 'PrivateEye', 'Qbert',
    'RoadRunner', 'Seaquest', 'UpNDown'
]
with open('rQdia-Rainbow.json', 'r') as f:
    scores = json.load(f)
final = collections.defaultdict(list)
# trim to same length
trim_len = min([len(v) for v in scores.values()])
print(f'we trim to {trim_len}')
# unify name
for name, score in scores.items():
    game = ''.join([i.title() for i in name.split('-')[2].split('_')])
    for result in random.sample(score, trim_len):
        steps = [i for i in result if i['name'] == 'Max'][0]['y']
        # filter = lambda x: x[-1]
        filter = lambda x: x[-1]
        final[game].append(filter(steps))

with open('atari_100k/DER+rQdia.json', 'w') as f:
    json.dump(final, f)
print()
