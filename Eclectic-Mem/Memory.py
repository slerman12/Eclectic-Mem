import time
import torch
from torch.nn import Module


class Memory(Module):
    def __init__(self, N, c_size, j=2, key_size=32, num_heads=1):
        '''
        N: Memory max size
        j: time between updates
        '''
        super().__init__()
        self.N = N
        self.n = 0  # Current memory size
        self.memory = {}
        self.j = j
        self._j = 0  # Counts updates
        self.retrieved = None
        self.key_size = key_size
        self.head_size = c_size
        self.c_size = c_size
        # print(c_size, total_size, self.qkv_size)
        self.qkv_encoder = None
        self.num_heads = num_heads

    def add(self, **kwargs):
        '''
        Memory content (kwargs values) should be tensors with batch dimension
        '''
        try:
            batch_size = kwargs["c"].shape[0]
        except TypeError:
            raise TypeError("add() missing 1 required tensor argument: c")

        if "t" not in kwargs:
            kwargs["t"] = torch.tensor([time.time()] * batch_size)
        for key in kwargs:
            # if key != 'step':
            assert kwargs[key].shape[0] == batch_size
            memory = getattr(self, key, torch.empty([self.N] + list(kwargs[key].shape)[1:]))
            # torch.cat supposedly faster:
            # (https://stackoverflow.com/questions/51761806/is-it-possible-to-create-a-fifo-queue-with-pytorch)
            new_memory = torch.cat((kwargs[key].to(memory.device), memory[:-batch_size])).to('cuda:0')
            setattr(self, key, new_memory)
            self.memory[key] = self.__dict__[key]
        # print(self.c.shape[0], self.n)
        assert self.c.shape[0] >= self.n  # todo debugging check, can delete

        #  todo raise error if not all memory keys included in kwargs
        if self.n < self.N:
            self.n = min(self.N, self.n + batch_size)

    def _query(self, c, k, delta, weigh_q=False):
        '''
        c should have batch dimension
        c: concept to query
        k: num memories to retrieve
        delta: CL embed function
        returns batch_size x k
        '''

        k = min(k, self.n)
        tau = delta(c, self.c[:self.n])
        if weigh_q:
            tau = self.q[None, :self.n] * tau
        deltas, indices = torch.topk(tau, k=k, dim=1, sorted=False)
        # deltas.shape[0]
        # print(deltas.shape[0], c.shape[0], deltas.shape[1], self.c.shape[0], self.c.shape)
        assert deltas.shape[0] == c.shape[0] and deltas.shape[1] == k  # todo debugging check, can delete

        result = [deltas.unsqueeze(dim=2)]
        for key in self.memory:
            result.append(self.memory[key][indices])  # B x k x mem_size
        result[-1] = result[-1].unsqueeze(dim=2)

        self._j = (self._j + 1) % self.j

        return torch.cat(result, dim=2)

    def _mhdpa(self, memory):
        """Perform multi-head attention from 'Attention is All You Need'.
        Implementation of the attention mechanism from
        https://arxiv.org/abs/1706.03762.
        As implemented in:
        https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/relational_memory.py
        Args:
          memory: Memory tensor to perform attention on.
        Returns:
          new_memory: New memory tensor.
        """
        # qkv = [B, N, F]
        qkv = self.qkv_encoder(memory)
        # should probably be per q, k, and v, but whatever
        qkv = self.layer_norm(qkv)

        mem_slots = memory.shape[1]  # Denoted as N.

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

    def _attend_over_memory(self, memory):
        """Perform multiheaded attention over `memory`.
        As implemented in:
        https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/relational_memory.py
        Args:
          memory: Current relational memory.
        Returns:
          The attended-over memory.
        """

        attended_memory = self._mhdpa(memory)
        # Add a skip connection to the multiheaded attention's input.
        memory = self.layer_norm_mem(memory + attended_memory)

        # Add a skip connection to the attention_mlp's input.
        memory = self.layer_norm_mem(self.attention_mlp(memory) + memory)
        memory = torch.mean(memory, dim=1)
        return memory

    def forward(self, c, k, delta, weigh_q, encode_c=True):
        '''
        c should have batch dimension
        c: concept to query
        k: num memories to retrieve
        delta: CL embed function
        '''
        if self.n == 0:
            return c

        if self._j == 0 or c.shape[0] == 1:
            mems = self._query(c, k, delta, weigh_q)
            if not self.qkv_encoder:
                self.value_size = mems.shape[-1]
                self.qkv_size = 2 * self.key_size + self.value_size  # 32*2+107 = 171
                self.total_size = self.qkv_size * self.num_heads  # Denote as F.
                self.layer_norm = torch.nn.LayerNorm(self.total_size).to('cuda:0')
                self.layer_norm_mem = torch.nn.LayerNorm(mems.shape[-1]).to('cuda:0')
                self.qkv_encoder = torch.nn.Linear(mems.shape[-1], self.total_size).to('cuda:0')
                self.attention_mlp = torch.nn.Sequential(torch.nn.Linear(mems.shape[-1], mems.shape[-1]),
                                                         torch.nn.ReLU(),
                                                         torch.nn.Linear(mems.shape[-1], mems.shape[-1])).to('cuda:0')
                self.project_mlp = torch.nn.Sequential(torch.nn.Linear(mems.shape[-1], mems.shape[-1]), torch.nn.ReLU(),
                                                       torch.nn.Linear(mems.shape[-1], self.c_size)).to('cuda:0')
            memory = self._attend_over_memory(mems)
            c_prime = self.project_mlp(memory)
            self.retrieved = c_prime
        else:
            c_prime = self.retrieved
        # print('_j', self._j, c.shape, c_prime.shape)
        return c_prime
