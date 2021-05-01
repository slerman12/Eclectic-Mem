from itertools import count

import os
from collections import defaultdict

import dmc2gym
import torch
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef, rpc_async, remote, rpc_sync
from torch.utils.data import Dataset
import numpy as np


# purpose: can multi agent append to the same replay buffer with remote call?

def call_method(method, rref, *args, **kwargs):
    # call observer member method, method(self, args)
    # e.g. def run_episode(self, args)
    return method(rref.local_value(), *args, **kwargs)


def remote_method(method, rref, *args, **kwargs):
    # rref.func()
    # rpc_sync(rref, method, args) = rref.method(args)
    args = [method, rref] + list(args)
    return rpc_sync(rref.owner(), call_method, args=args, kwargs=kwargs)


class Observer:
    def __init__(self):
        self.id = rpc.get_worker_info().id
        self.env = dmc2gym.make(domain_name="cartpole", task_name="swingup", seed=100, visualize_reward=False,
                                from_pixels=True,
                                height=100, width=84, frame_skip=8)

    def run_episode(self, agent_rref, buffer_rref):
        obs, ep_reward = self.env.reset(), 0
        for step in range(100):
            action = self.env.action_space.sample()
            next_obs, reward, done, _ = self.env.step(action)
            remote_method(DistributedReplayBuffer.append, buffer_rref, self.id, obs, action, reward, next_obs, done)
            if done:
                break
            obs = next_obs


class Agent:
    def __init__(self, world_size):
        self.ob_rrefs = []
        self.agent_rref = RRef(self)
        self.replay = DistributedReplayBuffer((3 * 3 * 100, 100), (1,), 100000, 128, 84)
        self.replay_rref = RRef(self.replay)
        for ob_rank in range(1, world_size):
            ob_info = rpc.get_worker_info(f"observer_{ob_rank}")
            self.ob_rrefs.append(remote(ob_info, Observer))

    def run_episode(self):
        futs = []
        for ob_rref in self.ob_rrefs:
            futs.append(rpc_async(ob_rref.owner(), call_method,
                                  args=(Observer.run_episode, ob_rref, self.agent_rref, self.replay_rref)))
        for fut in futs:
            fut.wait()

    def finish_episode(self):
        print(self.replay.sample_one())


class DistributedReplayBuffer():
    def __init__(self, obs_shape, action_shape, capacity, batch_size, image_size=84):
        self.capacity = capacity
        self.batch_size = batch_size
        self.image_size = image_size
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8
        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.buffer = defaultdict(list)
        self.idx = 0
        self.full = False
        self.ob_rrefs = []

    def append(self, ob_id, obs, action, reward, next_obs, done):
        self.buffer[ob_id].append((obs, action, reward, next_obs, done))

    def sample_one(self):
        for k, v in self.buffer:
            for data in v:
                obs, action, reward, next_obs, done = data
                np.copyto(self.obses[self.idx], obs)
                np.copyto(self.actions[self.idx], action)
                np.copyto(self.rewards[self.idx], reward)
                np.copyto(self.next_obses[self.idx], next_obs)
                np.copyto(self.not_dones[self.idx], not done)
                self.idx = (self.idx + 1) % self.capacity
                self.full = self.full or self.idx == 0
            self.buffer[k] = []
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )
        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        actions = self.actions[idxs]
        rewards = self.rewards[idxs]
        not_dones = self.not_dones[idxs]
        return obses, actions, rewards, next_obses, not_dones


def run(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    # os.environ['MUJOCO_GL'] = 'osmesa'
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    os.environ['LD_LIBRARY_PATH'] = '/u/jbi5/.mujoco/mujoco200/bin:/usr/lib/nvidia-440'
    os.environ['MJLIB_PATH'] = '/u/jbi5/.mujoco/mujoco200_linux/bin/libmujoco200.so'
    os.environ['MJKEY_PATH'] = '/u/jbi5/.mujoco/mjkey.txt'
    if rank == 0:
        rpc.init_rpc('master', rank=rank, world_size=world_size)
        agent = Agent(world_size)
        for i_episode in count(1):
            agent.run_episode()
    else:
        rpc.init_rpc(f"observer_{rank}", rank=rank, world_size=world_size)
    rpc.shutdown()


if __name__ == '__main__':
    WORLDSIZE = 3
    torch.multiprocessing.spawn(run, args=(WORLDSIZE,),
                                nprocs=WORLDSIZE,
                                join=True)
