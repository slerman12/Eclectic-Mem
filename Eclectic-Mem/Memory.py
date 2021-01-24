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

        self.key_size = key_size
        self.head_size = c_size
        self.value_size = c_size
        self.qkv_size = 2 * key_size + self.value_size
        total_size = self.qkv_size * num_heads  # Denote as F.
        self.qkv_encoder = torch.nn.Linear(c_size, total_size)
        self.layer_norm = torch.nn.LayerNorm(total_size)
        self.attention_mlp = torch.nn.Sequential(torch.nn.Linear(c_size, c_size), torch.nn.ReLU(),
                                                 torch.nn.Linear(c_size, c_size))

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
            assert kwargs[key].shape[0] == batch_size
            memory = getattr(self, key, __default=torch.tensor([self.N] + kwargs[key].shape[1:]))
            # torch.cat supposedly faster:
            # (https://stackoverflow.com/questions/51761806/is-it-possible-to-create-a-fifo-queue-with-pytorch)
            new_memory = torch.cat((kwargs[key], memory[:-batch_size]))
            setattr(self, key, new_memory)
            self.memory[key] = self.__dict__[key]
        assert self.c.shape[0] > self.n  # todo debugging check, can delete

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
        assert deltas.shape[0] == c.shape[0] and deltas.shape[1] == self.c.shape[0]  # todo debugging check, can delete

        self.retrieved = [deltas.unsqueeze(dim=2)]
        for key in self.memories:
            self.retrieved.append(self.memory[key][indices])  # B x k x mem_size
        self.retrieved = torch.cat(self.retrieved, dim=2)

        self._j = (self._j + 1) % self.j

        return self.retrieved

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

        qkv = self.qkv_encoder(memory)
        # should probably be per q, k, and v, but whatever
        qkv = self.layernorm(qkv)

        mem_slots = memory.shape[1]  # Denoted as N.

        # [B, N, F] -> [B, N, H, F/H]
        qkv_reshape = torch.reshape(qkv, [mem_slots, self.num_heads, self.qkv_size])

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

        print(attended_memory.shape)

        # Add a skip connection to the multiheaded attention's input.
        memory = self.layer_norm(memory + attended_memory)

        # Add a skip connection to the attention_mlp's input.

        memory = self.layer_norm(self.attention_mlp(memory) + memory)
        memory = self.layer_norm_mem(self.attention_mlp(memory) + memory)
        memory = self.skip_mlp(memory)

        return memory

    def forward(self, c, k, delta, weigh_q, encode_c=True):
        '''
        c should have batch dimension
        c: concept to query
        k: num memories to retrieve
        delta: CL embed function
        '''

        mems = self._query(c, k, delta, weigh_q) if self._j == self.j else self.retrieved
        if encode_c:
            mems["c_cxt"] = c.view(mems["c"].shape)

        if self.n == 0:
            return c
        mems = self._query(c, k, delta, weigh_q) if self._j == 0 else self.retrieved
        if not self.qkv_encoder:
            self.value_size = mems.shape[-1]
            self.qkv_size = 2 * self.key_size + self.value_size  # 32*2+107 = 171
            self.total_size = self.qkv_size * self.num_heads  # Denote as F.
            self.layer_norm = torch.nn.LayerNorm(self.total_size).to('cuda:0')
            self.layer_norm_mem = torch.nn.LayerNorm(mems.shape[-1]).to('cuda:0')
            self.qkv_encoder = torch.nn.Linear(mems.shape[-1], self.total_size).to('cuda:0')
            self.attention_mlp = torch.nn.Sequential(torch.nn.Linear(mems.shape[-1], mems.shape[-1]), torch.nn.ReLU(),
                                                     torch.nn.Linear(mems.shape[-1], mems.shape[-1])).to('cuda:0')
            self.skip_mlp = torch.nn.Sequential(torch.nn.Linear(mems.shape[-1], mems.shape[-1]), torch.nn.ReLU(),
                                                torch.nn.Linear(mems.shape[-1], self.c_size)).to('cuda:0')

        c_prime = self._attend_over_memory(mems)
        return c_prime
