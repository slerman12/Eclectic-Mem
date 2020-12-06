import torch
import secrets
from copy import copy
from math import inf


class Memory:
    def __init__(self, N=inf, memories=None):
        self.N = N

        if memories is None:
            self.n = 0
            self.memories = {}
        else:
            self.memories = copy(memories)


class MemoryUnit:
    def __init__(self, concept, reward, action):
        self.id = secrets.token_bytes()
        self.c = concept
        self.r = reward
        self.a = action
        self.futures = Memory()
        self.pasts = Memory()


class Agent:
    def __init__(self, N, embed, delta, policy):
        self.Memory = Memory(N)
        self.M_t = Memory()
        self.M_t_minus_1 = Memory()

        self.embed = embed
        self.delta = delta
        self.policy = policy

    def act(self, o_t, r_t):
        new
        c_t = self.embed(o_t)

        margin = 0.5
        self.M_t = Memory(memories={m: self.M_t_minus_1.memories[m] for m in self.M_t_minus_1.futures()
                                    if self.delta(c_t, self.M_t_minus_1.memories[m].c()) > margin})

        if len(self.M_t.memories) == 0:
            self.M_t = self.traverse(self.M_t_minus_1,)

        self.M_t = self.update(self.M_t, c_t)

        a_t = self.policy(c_t, self.M_t)

        if len(self.M_t.memories) == 0 or a_t not in self.M_t.a():
            self.store(c_t, r_t, a_t)



    def store(self, c, r, a):
        if self.n == self.N:
            self.delete_LRA()
        m = MemoryUnit(c, r, a)
        self.Memory.add(m)
        self.M_t.add(m)

    def learn(self):
# todo contrastive learning
# todo Q learning






