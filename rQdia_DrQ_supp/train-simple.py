import argparse
import json
import os
# import gym
import time

import dmc2gym
import numpy as np
import torch

import curl_utils
from rqdia_drq import DrQAgent
from logger import Logger
from video import VideoRecorder
import torch.distributed as dist
from torch.distributed.rpc import RRef, rpc_async, remote, rpc_sync

import torch.distributed.rpc as rpc
from storage import DistributedReplayBuffer


def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--domain_name', default='cartpole')
    parser.add_argument('--task_name', default='swingup')
    parser.add_argument('--pre_transform_image_size', default=100, type=int)

    parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--action_repeat', default=8, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    # train
    parser.add_argument('--agent', default='curl_sac', type=str)
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--num_train_steps', default=250000, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    # eval
    parser.add_argument('--eval_freq', default=10000, type=int)
    parser.add_argument('--num_eval_episodes', default=10, type=int)
    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float)  # try 0.05 or 0.1
    parser.add_argument('--critic_target_update_freq', default=2,
                        type=int)  # try to change it to 1 and retain 0.01 above
    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # encoder
    parser.add_argument('--encoder_type', default='pixel', type=str)
    parser.add_argument('--encoder_feature_dim', default=50, type=int)
    parser.add_argument('--encoder_lr', default=1e-3, type=float)
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    parser.add_argument('--curl_latent_dim', default=128, type=int)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--alpha_beta', default=0.5, type=float)
    # misc
    parser.add_argument('--seed', default=2, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--detach_encoder', default=False, action='store_true')

    parser.add_argument('--log_interval', default=20, type=int)
    args = parser.parse_args()
    return args


def call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)


def remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc_sync(rref.owner(), call_method, args=args, kwargs=kwargs)


def make_agent(obs_shape, action_shape, args, device):
    if args.agent == 'curl_sac':
        return DrQAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            log_interval=args.log_interval,
            detach_encoder=args.detach_encoder,
            curl_latent_dim=args.curl_latent_dim

        )
    else:
        assert 'agent is not supported: %s' % args.agent


class Observer:

    def __init__(self, args):

        self.id = rpc.get_worker_info().id
        # print(f'starting worker.{self.id}', args)
        env = dmc2gym.make(domain_name=args.domain_name, task_name=args.task_name, seed=args.seed,
                           visualize_reward=False,
                           from_pixels=(args.encoder_type == 'pixel'), height=args.pre_transform_image_size,
                           width=args.pre_transform_image_size, frame_skip=args.action_repeat)
        self.env = curl_utils.FrameStack(env, k=args.frame_stack)
        self.args = args

    def run_episode(self, agent_rref, buffer_rref):
        obs, ep_reward = self.env.reset(), 0
        for step in range(2000):
            # send the state to the agent to get an action
            if step < self.args.init_steps:
                action = self.env.action_space.sample()
            else:
                with curl_utils.eval_mode(self.agent):
                    action = remote_method(DrQAgent.select_action, agent_rref, self.id, obs)

            # apply the action to the environment, and get the reward
            next_obs, reward, done, _ = self.env.step(action)

            # report the reward to the agent for training purpose
            remote_method(DistributedReplayBuffer.append, buffer_rref, self.id, obs, action, reward, next_obs, done)
            obs = next_obs
            if done:
                break


def run(rank, world_size, args, L):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    # os.environ['MUJOCO_GL'] = 'osmesa'
    os.environ['LD_LIBRARY_PATH'] = '/u/jbi5/.mujoco/mujoco200/bin:/usr/lib/nvidia-440'
    os.environ['MJLIB_PATH'] = '/u/jbi5/.mujoco/mujoco200_linux/bin/libmujoco200.so'
    os.environ['MJKEY_PATH'] = '/u/jbi5/.mujoco/mjkey.txt'
    observer_pool = []
    if rank == 0:
        device = 1
        rpc.init_rpc('master', rank=rank, world_size=world_size)
        # curl_utils.set_seed_everywhere(args.seed)
        action_shape = (1,)
        obs_shape = (3 * args.frame_stack, args.image_size, args.image_size)
        pre_aug_obs_shape = (3 * args.frame_stack, args.pre_transform_image_size, args.pre_transform_image_size)
        agent = make_agent(obs_shape=obs_shape, action_shape=action_shape, args=args, device=device)
        replay_buffer = DistributedReplayBuffer(obs_shape=pre_aug_obs_shape, action_shape=action_shape,
                                                capacity=args.replay_buffer_capacity, batch_size=args.batch_size,
                                                world_size=world_size,
                                                image_size=args.image_size, )

        # for ob_rank in range(1, world_size):

        # ========== log ================
        start_time = time.time()
        # ========== log ================
        for step in range(args.num_train_steps):
            futs = []
            for ob_rref in observer_pool:
                # make async RPC to kick off an episode on all observers
                futs.append(
                    rpc_async(ob_rref.owner(),
                              call_method,
                              args=(Observer.run_episode, ob_rref, RRef(agent), RRef(replay_buffer))))
                # wait until all obervers have finished this episode
            for fut in futs:
                fut.wait()
            # run training update
            if step >= args.init_steps:
                num_updates = world_size - 1
                for _ in range(num_updates):
                    agent.update(replay_buffer, L, step, device)
            # ========== log ================
            if step % args.log_interval == 0:
                L.log('train/duration', time.time() - start_time, step)
                L.dump(step * (world_size - 1))
                start_time = time.time()
    else:
        rpc.init_rpc(f"observer_{rank}", rank=rank, world_size=world_size)
        ob_info = rpc.get_worker_info(f"observer_{rank}")
        # print(ob_info.id)
        observer_pool.append(remote(ob_info, Observer, args=(args,)))
    rpc.shutdown()


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    args = parse_args()
    WORLDSIZE = 3
    L = Logger(args.work_dir, use_tb=args.save_tb)
    torch.multiprocessing.spawn(run,
                                args=(WORLDSIZE, args, L),
                                nprocs=WORLDSIZE,
                                join=True)
