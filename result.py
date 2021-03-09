from collections import defaultdict
from pathlib import Path

root = Path('result')
result = defaultdict(list)
# , 'finger'
for exp in ['ballincup', 'cartpole', 'cheetah', 'reacher', 'walker']:
    for name in ['our', 'CURL']:
        expname = f"{exp}-{name}"
        with open(root / expname, 'r') as file:
            for line in file.readlines():
                data = line.split('|')
                try:
                    if 'train' in data[1]:
                        if int(data[3].split(':')[1]) > 1000 and int(data[3].split(':')[1]) < 100000:
                            result[expname].append(float(data[5].split(':')[1]))
                except:
                    pass

from clearml import Task, Logger

snapshots_path = Path('./experiments')
task = Task.init(project_name="Eclectic-Mem", task_name="resultplot", output_uri=str(snapshots_path))
logger = task.get_logger()
for expname, rewards in result.items():
    for step, reward in enumerate(rewards):
        logger.report_scalar(expname.split('-')[0], expname,
                             iteration=step, value=reward)
