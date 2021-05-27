#!/usr/bin/env python3
import numpy as np
import numpy.random as npr

import time
import os
import sys
import argparse
from subprocess import Popen, DEVNULL


class Overrides(object):
    def __init__(self):
        self.kvs = dict()

    def add(self, key, values):
        value = ','.join(str(v) for v in values)
        assert key not in self.kvs
        self.kvs[key] = value

    def cmd(self):
        cmd = []
        for k, v in self.kvs.items():
            cmd.append(f'{k}={v}')
        return cmd


DQN_ENVS = [
    'Pong',
    'Breakout',
    'Seaquest',
    'SpaceInvaders',
    'BeamRider',
]

DOPAMINE_ENVS = [
    'AirRaid', 'Alien', 'Amidar', 'Assault', 'Asterix', 'Asteroids',
    'Atlantis', 'BankHeist', 'BattleZone', 'BeamRider', 'Berzerk', 'Bowling',
    'Boxing', 'Breakout', 'Carnival', 'Centipede', 'ChopperCommand',
    'CrazyClimber', 'DemonAttack', 'DoubleDunk', 'ElevatorAction', 'Enduro',
    'FishingDerby', 'Freeway', 'Frostbite', 'Gopher', 'Gravitar', 'Hero',
    'IceHockey', 'Jamesbond', 'JourneyEscape', 'Kangaroo', 'Krull',
    'KungFuMaster', 'MontezumaRevenge', 'MsPacman', 'NameThisGame', 'Phoenix',
    'Pitfall', 'Pong', 'Pooyan', 'PrivateEye', 'Qbert', 'Riverraid',
    'RoadRunner', 'Robotank', 'Seaquest', 'Skiing', 'Solaris', 'SpaceInvaders',
    'StarGunner', 'Tennis', 'TimePilot', 'Tutankham', 'UpNDown', 'Venture',
    'VideoPinball', 'WizardOfWor', 'YarsRevenge', 'Zaxxon'
]

SMALL_RAINBOW_ENVS = [
    'Alien',
    'Breakout',
    'CrazyClimber',
    'Frostbite',
    'Pong'
]

RAINBOW_ENVS = [
    'Alien', 'Amidar', 'Assault', 'Asterix', 'BankHeist', 'BattleZone',
    'Boxing', 'Breakout', 'ChopperCommand', 'CrazyClimber', 'DemonAttack',
    'Freeway', 'Frostbite', 'Gopher', 'Hero', 'Jamesbond', 'Kangaroo', 'Krull',
    'KungFuMaster', 'MsPacman', 'Pong', 'PrivateEye', 'Qbert', 'RoadRunner',
    'Seaquest', 'UpNDown'
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=str)
    parser.add_argument('--dry', action='store_true')
    args = parser.parse_args()

    overrides = Overrides()
    overrides.add('hydra/launcher', ['submitit'])
    overrides.add('env', RAINBOW_ENVS)
    overrides.add('seed', list(range(1, 6)))
    overrides.add('experiment', [args.experiment])
    overrides.add('log_save_tb', ['false'])
    
    # sweep
    #overrides.add('beta_1', [0.9, 0.0])
    #overrides.add('beta_2', [0.999, 0.9])
    #overrides.add('weight_decay', [0.0, 1e-5])
    #overrides.add('max_grad_norm', [10.0])
    overrides.add('agent.params.multistep_return', [5, 10])
    overrides.add('num_exploration_steps', [2500, 5000, 1000])
    overrides.add('intensity_scale', [0.05, 0.1, 0.01, 0.15])
    
    # eval 
    #overrides.add('num_train_steps', [200001])
    #overrides.add('eval_frequency', [10000])
    overrides.add('aug_type', ['crop_intensity'])

    cmd = ['python', 'train.py', '-m']
    cmd += overrides.cmd()

    if args.dry:
        print(' '.join(cmd))
    else:
        env = os.environ.copy()
        p = Popen(cmd, env=env)
        p.communicate()


if __name__ == '__main__':
    main()
