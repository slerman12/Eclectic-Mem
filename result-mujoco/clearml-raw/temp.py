import json
from pathlib import Path

dqr = []
for file in Path(__file__).parent.glob('*.json'):
    data = json.load(open(file))
    for i in data:
        # print(i['name'])

        if '-dqr-' in i['name']:
            dqr.append(i)
for i in dqr:
    print(i['name'])
print()
