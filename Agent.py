import torch


class Memory:
    def __init__(self):
        self.c = []
        self.a = []
        self.r = []
        self.deltas = []


class Agent:
    def __init__(self, N, k, m, actions):
        self.k = k
        self.m = m
        self.N = N
        self.n = 0
        self.i = 0
        self.mem = Memory()
        self.deltas = []
        self.A = actions

    def act(self, o_t):
        c_t = self.embed(o_t)

        _K = self.mem.deltas[self.i][:self.k]

        delta_K = []
        a_K = []
        r_K = []

        for l, _ in _K:
            delta_K.append(self.delta(c_t, self.mem.c[l]))
            a_K.append(self.mem.a[l])
            r_K.append(self.mem.r[l])

            self.deltas.append((l, self.delta(c_t, self.mem.c[l])))
            # todo maybe update m_deltas[i][l]

        Q = []
        for a in self.A:
            Q_o_t_a = torch.sum(torch.nn.Softmax()(delta_K) * torch.tensor(r_K) * (torch.tensor(a_K) == a), dim=-1)
            Q.append(Q_o_t_a)


    def delta(self, o_t, o_t_plus_1):
        # todo substitute with module
        return 0

    def embed(self, o_t):
        # todo substitute with module
        return 0

    def learn(self):
# todo contrastive learning
# todo Q learning






