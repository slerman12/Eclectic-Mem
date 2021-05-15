import os
import time


for game in ["ms_pacman", "pong"]:
    os.system("python3 sbatch.py --cpu --params '--game {}'".format(game))
    time.sleep(2)
