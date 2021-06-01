import os
from collections import defaultdict
from pathlib import Path
import socket

import json

os.environ['CLEARML_CONFIG_FILE'] = str(Path.home() / f"clearml-{socket.getfqdn()}.conf")
from clearml import Task

project = "Eclectic-Mem"
tasks = Task.get_tasks(project_name=project)
server_data = []
for task in tasks:
    if task.get_status() != 'completed': continue
    report = task.get_reported_scalars()
    if 'eval' not in report.keys(): continue
    result = report['eval']
    server_data.append({'name': task.name, 'result': result})
print(len(server_data))
with open(f"{project}-{socket.getfqdn()}.json", 'w') as f:
    print("saving")
    json.dump(server_data, f)
print()
