from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from curl_utils import random_crop
from collections import defaultdict


class DistributedReplayBuffer(Dataset):
    """Buffer to store environment transitions."""

    def __init__(self, obs_shape, action_shape, capacity, batch_size, world_size, image_size=84, transform=None):
        self.capacity = capacity
        self.batch_size = batch_size
        self.image_size = image_size
        self.transform = transform
        # the proprioceptive obs is stored as float32, pixels obs as uint8
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

    def sample_cpc(self, device):

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
        pos = obses.copy()

        obses = random_crop(obses, self.image_size)
        next_obses = random_crop(next_obses, self.image_size)
        pos = random_crop(pos, self.image_size)

        obses = torch.as_tensor(obses, device=device).float()
        next_obses = torch.as_tensor(
            next_obses, device=device
        ).float()
        actions = torch.as_tensor(self.actions[idxs], device=device)
        rewards = torch.as_tensor(self.rewards[idxs], device=device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=device)

        pos = torch.as_tensor(pos, device=device).float()
        cpc_kwargs = dict(obs_anchor=obses, obs_pos=pos,
                          time_anchor=None, time_pos=None)

        return obses, actions, rewards, next_obses, not_dones, cpc_kwargs

    def __getitem__(self, idx):
        idx = np.random.randint(
            0, self.capacity if self.full else self.idx, size=1
        )
        idx = idx[0]
        obs = self.obses[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        next_obs = self.next_obses[idx]
        not_done = self.not_dones[idx]

        if self.transform:
            obs = self.transform(obs)
            next_obs = self.transform(next_obs)

        return obs, action, reward, next_obs, not_done

    def __len__(self):
        return self.capacity
