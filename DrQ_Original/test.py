import socket
from pathlib import Path
import os

os.environ['LD_LIBRARY_PATH'] = str(Path.home() / '.mujoco/mujoco200_linux/bin:/usr/lib/nvidia-440')
os.environ['MJLIB_PATH'] = str(Path.home() / '.mujoco/mujoco200_linux/bin/libmujoco200.so')
os.environ['MJKEY_PATH'] = str(Path.home() / f'.mujoco/mjkey_{socket.getfqdn()}.txt')

os.environ['CLEARML_CONFIG_FILE'] = str(Path.home() / f"clearml-{socket.getfqdn()}.conf")
from clearml import Task

results_dir = 'results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
Task.init(project_name=f"DrQ", task_name=f"Original", output_uri=str(results_dir))
print()
