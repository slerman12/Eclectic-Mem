#!/bin/sh
module load python3/3.8.3
for game in "ms_pacman" "pong"
do
	python3 sbatch.py --cpu --params "--game $game"
	sleep 2
done
