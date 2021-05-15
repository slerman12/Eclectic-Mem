import os
import time

os.system("module load python3/3.8.3")
for game in ["ms_pacman", "pong"]:
    os.system("python3 sbatch.py --cpu --params '--game {}'".format(game))
    time.sleep(2)
