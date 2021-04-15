from torch.nn import Module
from torch.nn.parameter import Parameter
import torch


class Memory(Module):
    def __init__(self, N, c_size, j=1, key_size=32, num_heads=1):
        '''
        N: Memory max size
        j: time between updates
        '''
        super().__init__()
        self.N = N
        self.n = 0  # Current memory size
        self.memory = {}
        self.j = j
        self._j = 0  # Counts time since last query
        self.retrieved = None
        self.key_size = key_size
        self.head_size = c_size
        self.c_size = c_size
        self.qkv_encoder = None
        self.num_heads = num_heads
        self.time = 0.001
        self.device = 'cuda:0'

        # This is for dynamic sized value_size in case metadata includes current action
        # TODO just use predefined value size and project metadata to that size; maybe even reuse for q-value & action
        self.metadata_encoder = {"value_size": lambda metadata: metadata.shape[-1],
                                 "qkv_size": lambda metadata: 2 * self.key_size + self.value_size,
                                 "total_size": lambda metadata: self.qkv_size * self.num_heads,  # Denote as F.
                                 "layer_norm": lambda metadata: torch.nn.LayerNorm(self.total_size),
                                 "layer_norm_mem": lambda metadata: lambda: torch.nn.LayerNorm(self.value_size),
                                 "qkv_encoder": lambda metadata: torch.nn.Linear(self.value_size, self.total_size),
                                 "attention_mlp": lambda metadata: torch.nn.Sequential(torch.nn.Linear(self.value_size,
                                                                                                       self.value_size),
                                                                                       torch.nn.ReLU(),
                                                                                       torch.nn.Linear(self.value_size,
                                                                                                       self.value_size)),
                                 # TODO maybe superfluous
                                 "project_output": lambda metadata: torch.nn.Sequential(torch.nn.Linear(self.value_size,
                                                                                                        self.value_size),
                                                                                        torch.nn.ReLU(),
                                                                                        torch.nn.Linear(self.value_size,
                                                                                                        self.c_size))}

    def add(self, **kwargs):
        '''
        Memory content (kwargs values) should be tensors with batch dimension
        '''
        try:
            batch_size = kwargs["c"].shape[0]
        except TypeError:
            raise TypeError("add() missing 1 required tensor argument: c")

        if "t" not in kwargs:
            kwargs["t"] = torch.tensor([self.time] * batch_size).unsqueeze(dim=1)
        self.time += 0.001
        for key in kwargs:
            assert kwargs[key].shape[0] == batch_size  # batches only
            assert len(kwargs[key].shape) >= 2  # include non-batch dim
            # get current memories for key or set default
            # TODO parameter memory
            # memory = getattr(self, key, Parameter(torch.Tensor([self.N] + list(kwargs[key].shape)[1:])).to(self.device))
            memory = getattr(self, key, torch.empty([self.N] + list(kwargs[key].shape)[1:]).to(self.device))
            # append new memories to them
            new_memory = torch.cat((kwargs[key].to(self.device), memory[:-batch_size])).to(self.device)
            # TODO parameter memory
            # new_memory = Parameter(torch.cat((Parameter(kwargs[key]).to(self.device), memory[:-batch_size]))).to(self.device)
            # with torch.no_grad():
            #     param.copy_(torch.randn(10, 10))
            setattr(self, key, new_memory)
            self.memory[key] = self.__dict__[key]
        assert self.c.shape[0] >= self.n  # todo debugging check, can delete

        #  todo raise error if not all memory keys included in kwargs
        if self.n < self.N:
            self.n = min(self.N, self.n + batch_size)

    def _query(self, c, k, delta, weigh_q=False, detach_deltas=True, action=None):
        '''
        c should have batch dimension
        c: concept to query
        k: num memories to retrieve
        delta: CL embed function
        returns batch_size x k
        '''

        k = min(k, self.n)
        deltas = delta(c.to(self.device), self.c[:self.n])  # B x n
        if detach_deltas:
            deltas = deltas.detach()
        deltas, indices = torch.topk(deltas, k=k, dim=1, sorted=False)  # B x k
        expected_q = None
        if weigh_q:
            # B x k x 1, B * k -> B x 1
            expected_q = (self.q[indices].squeeze(-1) * torch.softmax(deltas, dim=1)).sum(-1).unsqueeze(-1)
        assert deltas.shape[0] == c.shape[0] and deltas.shape[1] == k  # todo debugging check, can delete

        result = [deltas.unsqueeze(dim=2)]
        for key in self.memory:
            if key != "c":
                metadata = self.memory[key][indices]  # B x k x mem_size
                result.append(metadata)
        if action is not None:
            result.append(action[:, None, :].expand(-1, k, -1))

        self._j = (self._j + 1) % self.j

        return torch.cat(result, dim=2), expected_q

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

    def _attend_over_memory(self, memory, project_input=False, project_output=False):
        """Perform multiheaded attention over `memory`.
        As implemented in:
        https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/relational_memory.py
        Args:
          memory: Current relational memory.
        Returns:
          The attended-over memory.
        """
        # TODO project input to a predefined value_size instead of using metadata directly
        if project_input:
            memory = self.project_metadata(memory)
        attended_memory = self._mhdpa(memory)
        # Add a skip connection to the multiheaded attention's input.
        memory = self.layer_norm_mem(memory + attended_memory)
        # print(self.n)  # TODO delete; just debugging check
        # Add a skip connection to the attention_mlp's input.
        memory = self.layer_norm_mem(self.attention_mlp(memory) + memory)
        memory = torch.mean(memory, dim=1)
        if project_output:
            memory = self.project_output(memory)
        return memory

    # def set_metadata_encoder(self, metadata, action=None):
    #     if not self.qkv_encoder:
    #         self.value_size = metadata.shape[-1]
    #         self.qkv_size = 2 * self.key_size + self.value_size  # 32*2+107 = 171
    #         self.total_size = self.qkv_size * self.num_heads  # Denote as F.
    #         self.layer_norm = torch.nn.LayerNorm(self.total_size).to('cuda:0')
    #         self.layer_norm_mem = torch.nn.LayerNorm(metadata.shape[-1]).to('cuda:0')
    #         self.qkv_encoder = torch.nn.Linear(metadata.shape[-1], self.total_size).to('cuda:0')
    #         self.attention_mlp = torch.nn.Sequential(torch.nn.Linear(metadata.shape[-1], metadata.shape[-1]),
    #                                                  torch.nn.ReLU(),
    #                                                  torch.nn.Linear(metadata.shape[-1], metadata.shape[-1])).to('cuda:0')
    #         self.project_mlp = torch.nn.Sequential(torch.nn.Linear(metadata.shape[-1], metadata.shape[-1]), torch.nn.ReLU(),
    #                                                torch.nn.Linear(metadata.shape[-1], self.c_size)).to('cuda:0')

    def set_metadata_encoder(self, metadata, action=None):
        prefix = "action_" if action is None else "q_value_"
        for module in self.metadata_encoder:
            encoder_module = getattr(self, prefix + module, self.metadata_encoder[module](metadata))
            setattr(self, module, encoder_module)

    def forward(self, c, k, delta, weigh_q, action=None, detach_deltas=True, return_expected_q=False):
        '''
        c should have batch dimension
        c: concept to query
        k: num memories to retrieve
        delta: CL embed function
        '''
        if self.n == 0:
            if return_expected_q:
                return torch.zeros(c.shape[0], 1).to(self.device)
            else:
                return c
        if self._j == 0 or c.shape[0] == 1 or return_expected_q:
            metadata, expected_q = self._query(c, k, delta, weigh_q or return_expected_q, action=action)
            if return_expected_q:
                return expected_q
            self.set_metadata_encoder(metadata, action=action)
            c_prime = self._attend_over_memory(metadata, project_output=True)
            self.retrieved = c_prime
        else:
            c_prime = self.retrieved
        return c_prime
