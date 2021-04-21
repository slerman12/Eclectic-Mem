import torch
import numpy as np
import os
from torch.utils.data import Dataset
from torch.nn import Module
from utils import random_crop


class EclecticMem(Dataset, Module):
    """Buffer to store environment transitions."""

    def __init__(self, obs_shape, c_size, action_shape, capacity, batch_size, device, image_size=84, transform=None,
                 key_size=32, num_heads=1, delta=None, k=80, N=5000):
        super().__init__()

        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.image_size = image_size
        self.transform = transform
        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = torch.float32 if len(obs_shape) == 1 else torch.uint8

        self.obses = torch.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = torch.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.c = torch.empty((capacity, c_size), dtype=torch.float32)
        self.next_c = torch.empty((capacity, c_size), dtype=torch.float32)
        self.actions = torch.empty((capacity, *action_shape), dtype=torch.float32)
        self.rewards = torch.empty((capacity, 1), dtype=torch.float32)
        self.q = torch.empty((capacity, 1), dtype=torch.float32)
        self.not_dones = torch.empty((capacity, 1), dtype=torch.float32)
        self.times = torch.empty((capacity, 1), dtype=torch.float32)
        self.episode_steps = torch.empty((capacity, 1), dtype=torch.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False
        self.time = 0.001
        self.retrieved = None

        self.key_size = key_size
        # TODO can set c_size automatically in self.add()
        self.head_size = c_size
        self.c_prime_size = c_size
        self.num_heads = num_heads

        self.delta = delta
        self.k = k
        self.N = N

        # This is for dynamic sized value_size in case metadata includes or doesn't include current action
        # TODO just use predefined value size and project metadata to that size; maybe even reuse for q-value & action
        self.metadata_encoder = {
            # "embed_metadata": lambda metadata: torch.nn.Sequential(torch.nn.Linear(metadata.shape[-1],
            #                                                                        self.value_size),
            #                                                        torch.nn.ReLU(),
            #                                                        torch.nn.Linear(self.value_size,
            #                                                                        self.value_size)
            #                                                        ).to(self.device),
            "value_size": lambda metadata: metadata.shape[-1],
            "qkv_size": lambda x: 2 * self.key_size + self.value_size,
            "total_size": lambda x: self.qkv_size * self.num_heads,  # Denote as F.
            "qkv_encoder": lambda x: torch.nn.Linear(self.value_size, self.total_size).to(self.device),
            "layer_norm": lambda x: torch.nn.LayerNorm(self.total_size).to(self.device),
            "layer_norm_mem": lambda x: torch.nn.LayerNorm(self.value_size).to(self.device),
            "attention_mlp": lambda x: torch.nn.Sequential(torch.nn.Linear(self.value_size, self.value_size),
                                                           torch.nn.ReLU(), torch.nn.Linear(self.value_size,
                                                                                            self.value_size)
                                                           ).to(self.device),
            # TODO maybe superfluous
            "project_output": lambda x: torch.nn.Sequential(torch.nn.Linear(self.value_size, self.value_size),
                                                            torch.nn.ReLU(), torch.nn.Linear(self.value_size,
                                                                                             self.c_prime_size)
                                                            ).to(self.device)}

    def add(self, obs, c, action, reward, q, next_obs, next_c, done, episode_step):

        self.obses[self.idx] = torch.from_numpy(obs).detach()
        self.c[self.idx] = c.detach()
        self.actions[self.idx] = torch.from_numpy(action).detach()
        self.rewards[self.idx] = reward
        self.q[self.idx] = q.detach()
        self.next_obses[self.idx] = torch.from_numpy(next_obs).detach()
        self.next_c[self.idx] = next_c.detach()
        self.not_dones[self.idx] = not done
        self.times[self.idx] = self.time
        self.episode_steps[self.idx] = episode_step

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0
        self.time += 0.001

    def sample_proprio(self):

        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]

        obses = torch.as_tensor(obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        return obses, actions, rewards, next_obses, not_dones

    def sample_cpc(self):

        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size)

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        pos = obses.clone().detach()

        obses = random_crop(obses.numpy(), self.image_size)
        next_obses = random_crop(next_obses.numpy(), self.image_size)
        pos = random_crop(pos.numpy(), self.image_size)

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        pos = torch.as_tensor(pos, device=self.device).float()
        cpc_kwargs = dict(obs_anchor=obses, obs_pos=pos,
                          time_anchor=None, time_pos=None)

        return obses, actions, rewards, next_obses, not_dones, cpc_kwargs

    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.c[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.next_c[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.q[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx],
            self.times[self.last_save:self.idx],
            self.episode_steps[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.c[start:end] = payload[1]
            self.next_obses[start:end] = payload[2]
            self.next_c[start:end] = payload[3]
            self.actions[start:end] = payload[4]
            self.rewards[start:end] = payload[5]
            self.q[start:end] = payload[6]
            self.not_dones[start:end] = payload[7]
            self.times[start:end] = payload[8]
            self.episode_steps[start:end] = payload[9]
            self.idx = end

    def __getitem__(self, idx):
        idx = np.random.randint(
            0, self.capacity if self.full else self.idx, size=1
        )
        idx = idx[0]
        obs = self.obses[idx]
        c = self.c[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        q = self.q[idx]
        next_obs = self.next_obses[idx]
        next_c = self.next_c[idx]
        not_done = self.not_dones[idx]
        time = self.times[idx]
        episode_step = self.episode_steps[idx]

        if self.transform:
            obs = self.transform(obs)
            next_obs = self.transform(next_obs)

        return obs, c, action, reward, q, next_obs, next_c, not_done, time, episode_step

    def __len__(self):
        return self.capacity

    def _query(self, c, weigh_q=False, detach_deltas=False, action=None):
        '''
        c should have batch dimension
        c: concept to query
        k: num memories to retrieve
        delta: CL embed function
        returns batch_size x k
        '''

        n = self.capacity if self.full else self.idx

        # TODO might also consider just subsampling
        start = self.idx - self.N if self.full or self.idx >= self.N else 0
        end = self.idx

        def get_last_N(x):
            if self.full and self.idx < self.N:
                # TODO preallocate
                return torch.cat((x[end:], x[:start]), dim=0)
            else:
                return x[start:end]

        # TODO set top K to parameters to enable updates, and then retroactively update the stored data
        # TODO however, keep in mind that this can potentially corrupt/modify the original replay actions
        k = min(self.k, n)
        # TODO maybe the computation should just happen on CPU since copying memory to GPU might be very expensive
        past_c = torch.as_tensor(get_last_N(self.c), device=self.device).float()
        deltas = self.delta(c, past_c)  # B x n
        if detach_deltas:
            deltas = deltas.detach()
        # TODO instead of top k consider retrieving all above certain threshold, then subsampling those by past q value
        deltas, indices = torch.topk(deltas, k=k, dim=1, sorted=False)  # B x k

        # print(n, indices.min(), indices.max())

        # TODO recompute c?
        # self.c[indices] = compute_c(self.obses[indices])
        # self.next_c[indices] = compute_c(self.next_obses[indices])
        # TODO compute q value from reward and next_c/next_obs?
        # self.q[indices] = self.rewards[indices] + compute_q(self.next_c[indices])

        result = [deltas.unsqueeze(dim=2), self.time - get_last_N(self.times)[indices]]
        print(result[1].max() * 1000, result[1].min() * 1000)
        # TODO compute q from traces up to done or horizon
        # TODO compute q from stored next_c or next_obs (with no grad?), update q in this way
        # TODO compute q from cumulative discounted sum of next indices until done?
        # TODO the current concept c_t and the memory action a_m can be fed into critic
        #  and the output similarity weighted averaged and maximized! (To increase similarity between concepts
        #  with related best actions!) (the critic part could be with no grad)
        #  Some filtering of negative samples would be good too by min cutoff
        # TODO or rather convolve taken action across memory conceots and similarity weighted average critic q vals
        #  of each to predict q val. maybe multiply/divide by weight beta to allow model to learn softmax temp
        # TODO get similar above certain threshold (.9), get top q-val mem
        for key in ["actions", "rewards", "not_dones", "episode_steps", "q"]:
            metadata = get_last_N(getattr(self, key))[indices]  # B x k x mem_size
            result.append(metadata.to(self.device))
        if action is not None:
            result.append(action[:, None, :].expand(-1, k, -1))

        expected_q = None
        if weigh_q:
            # B x k x 1, B * k -> B x 1
            # TODO compute q value from reward and next_c/next_obs?
            expected_q = (get_last_N(self.q)[indices].squeeze(-1) * torch.softmax(deltas, dim=1)).sum(-1).unsqueeze(-1)

        return torch.cat(result, dim=2), expected_q

    def _mhdpa(self, metadata):
        """Perform multi-head attention from 'Attention is All You Need'.
        Implementation of the attention mechanism from
        https://arxiv.org/abs/1706.03762.
        As implemented in:
        https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/relational_memory.py
        Args:
          metadata: Memory tensor to perform attention on.
        Returns:
          new_memory: New memory tensor.
        """
        # qkv = [B, N, F]
        qkv = self.qkv_encoder(metadata)
        # should probably be per q, k, and v, but whatever
        qkv = self.layer_norm(qkv)

        mem_slots = metadata.shape[1]  # Denoted as N.

        # [B, N, F] -> [B, N, H, F/H]
        qkv_reshape = torch.reshape(qkv, [qkv.shape[0], mem_slots, self.num_heads, self.qkv_size])

        # [B, N, H, F/H] -> [B, H, N, F/H]
        qkv_transpose = qkv_reshape.permute(0, 2, 1, 3)
        query, key, value = torch.split(qkv_transpose, [self.key_size, self.key_size, self.value_size], dim=-1)

        query *= self.key_size ** -0.5
        dot_product = torch.matmul(query, key.transpose(2, 3))  # [B, H, N, N]
        weights = torch.softmax(dot_product, dim=-1)

        output = torch.matmul(weights, value)  # [B, H, N, V]

        # [B, H, N, V] -> [B, N, H, V]
        output_transpose = output.permute(0, 2, 1, 3)

        # [B, N, H, V] -> [B, N, H * V]
        new_memory = torch.nn.Flatten(start_dim=2)(output_transpose)
        return new_memory

    def _attend_over_metadata(self, metadata, embed_metadata=False, project_output=False):
        """Perform multiheaded attention over `memory`.
        As implemented in:
        https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/relational_memory.py
        Args:
          metadata: Current relational memory.
        Returns:
          The attended-over memory.
        """
        # TODO project input to a predefined value_size instead of using metadata directly
        if embed_metadata:
            metadata = self.embed_metadata(metadata)
        attended_memory = self._mhdpa(metadata)
        # Add a skip connection to the multiheaded attention's input.
        metadata = self.layer_norm_mem(metadata + attended_memory)
        # print(self.n)  # TODO delete; just debugging check
        # Add a skip connection to the attention_mlp's input.
        metadata = self.layer_norm_mem(self.attention_mlp(metadata) + metadata)
        metadata = torch.mean(metadata, dim=1)
        if project_output:
            metadata = self.project_output(metadata)
        return metadata

    def set_metadata_encoder(self, metadata, action=None, id=""):
        id += "action_" if action is None else "q_value_"
        for module in self.metadata_encoder:
            encoder_module = getattr(self, id + module, self.metadata_encoder[module](metadata))
            setattr(self, module, encoder_module)

    def forward(self, c, weigh_q=False, action=None, detach_deltas=False, return_expected_q=False, detach=False):
        '''
        c should have batch dimension
        c: concept to query
        k: num memories to retrieve
        delta: CL embed function
        '''
        if self.idx == 0 and not self.full:
            return torch.zeros(c.shape[0], 1).to(self.device) if return_expected_q else c
        # TODO can get rid of weigh_q; just need to update em_sac.py calls to be without it
        metadata, expected_q = self._query(c, weigh_q or return_expected_q, action=action, detach_deltas=detach_deltas)
        if return_expected_q:
            return expected_q

        # TODO can save and reuse retrieved
        # TODO for one, don't have to query memory redundantly for both actor and critic
        # self.retrieved = metadata

        self.set_metadata_encoder(metadata, action=action)
        # TODO embed metadata
        c_prime = self._attend_over_metadata(metadata, embed_metadata=False, project_output=True)

        if detach:
            c_prime = c_prime.detach()

        return c_prime