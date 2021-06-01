import os
from pathlib import Path
import socket

os.environ['CLEARML_CONFIG_FILE'] = str(Path.home() / f"clearml-{socket.getfqdn()}.conf")
from clearml import Task
from clearml import Logger as TrainLogger

Task.init(project_name=f"rQdia-Rainbow", task_name=f"curl")

logger = TrainLogger.current_logger()

avg_r = [527.0, 582.0, 971.0, 957.0, 1144.0, 2083.0, 1795.0, 1898.0, 1828.0, 1956.0]
for i, v in enumerate(avg_r, 1):
    logger.report_scalar('Evaluating', 'Avg. reward', iteration=i * 10000, value=v)
avg_Q = [5.082374048948288, 7.235004305839539, 7.778314247131347, 8.619482944488526, 8.264973501205445,
         8.52652338027954, 8.501409417629242, 8.27165054655075, 8.19569793844223, 8.48986261177063]
for i, v in enumerate(avg_r, 1):
    logger.report_scalar('Evaluating', 'Avg. Q', iteration=i * 10000, value=v)
