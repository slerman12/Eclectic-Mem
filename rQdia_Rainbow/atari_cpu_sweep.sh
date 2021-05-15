#!/bin/sh
module load python3/3.8.3
seed=1
for game in "ms_pacman" "pong"
do
	python3 sbatch.py --cpu --name $game$seed --params "--game $game --seed $seed"
	sleep 2
done
