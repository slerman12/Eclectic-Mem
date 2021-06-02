#!/bin/sh
module load python3/3.8.3
seed=1
for game in "alien" "amidar" "assault" "asterix" "bank_heist" "battle_zone" "boxing" "breakout" "chopper_command" "crazy_climber" "demon_attack" "freeway" "frostbite" "gopher" "hero" "jamesbond" "kangaroo" "krull" "kung_fu_master" "ms_pacman" "pong" "private_eye" "qbert" "road_runner" "seaquest" "up_n_down"
#for game in "breakout" "jamesbond" "krull"
#for game in "chopper_command" "asterix" "battle_zone" "bank_heist" "amidar" "assault" "alien"
do
	python3 sbatch.py --cpu --name curlre$game$seed --params "--game $game --seed $seed --expname curlr$game$seed"
	sleep 2
done
