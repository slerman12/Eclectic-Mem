import torch
import secrets
from copy import copy
from math import inf


class Memory:
    def __init__(self, N=inf):
        self.N = N
        self.n = 0
        self.memories = {}

    def add(self, memories):
        if not isinstance(memories, list):
            memories = [memories]

        for m in memories:
            if self.n == self.N:
                self.delete_LRA()

            self.memories[m.id] = m

    def retrieve(self, ids):
        if isinstance(ids, list):
            return [self.memories[ID] for ID in ids]
        else:
            return self. memories[ids]


class MemoryUnit:
    def __init__(self, concept, reward, action):
        self.id = secrets.token_bytes()
        self.concept = concept
        self.reward = reward
        self.action = action
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

        self.M_t_minus_1 = self.M_t
        margin = 0.5
        self.M_t = Memory()
        self.M_t.add([m for m in self.M_t_minus_1.get_futures() if self.delta(c_t, m.concept) > margin])

        if len(self.M_t.memories) == 0:
            n
            self.M_t = self.traverse(self.M_t_minus_1,)

        self.M_t = self.update(self.M_t, c_t)

        a_t = self.policy(c_t, self.M_t)

        if self.M_t.n == 0 or a_t not in self.M_t.get_a():
            nn
            m = MemoryUnit(c_t, r_t, a_t)
            self.Memory.add(m)
            self.M_t.add(m)

        if

    def learn(self):
# todo contrastive learning
# todo Q learning






