import secrets
import time
from math import inf


class Memory:
    def __init__(self, N=inf):
        self.N = N
        self.n = 0
        self.memories = {}

    def add(self, memories):
        if isinstance(memories, Memory):
            memories = memories.get_memories_list()

        if not isinstance(memories, list):
            memories = [memories]

        for m in memories:
            if m.id not in self.memories:
                if self.n == self.N:
                    self.remove(self.get_LRA())

                self.memories[m.id] = m
                self.n += 1

    def remove(self, memory):
        del self.memories[memory.id]
        self.n -= 1

    def retrieve(self, ids):
        if isinstance(ids, list):
            return [self.memories[ID] for ID in ids]
        else:
            return self.memories[ids]

    def get_memories_list(self):
        return list(self.memories.values())

    def get_futures(self):
        future_ids = {}
        for m in self.memories.values():
            future_ids.update(m.futures.get_ids())
        return self.retrieve(future_ids)

    def get_actions(self):
        # Note: if action is a tensor, python might not be able to do "set" operation on it
        return list(set([m.action for m in self.memories.values()]))



class MemoryUnit:
    def __init__(self, concept, reward, action):
        self.id = secrets.token_bytes()
        self.concept = concept
        self.reward = reward
        self.action = action
        self.futures = Memory()
        self.pasts = Memory()
        self.access_date = time.time()


class Agent:
    def __init__(self, N, embed, delta, policy):
        self.Memory = Memory(N)
        self.M_t = Memory()
        self.M_t_minus_1 = Memory()

        # CNN
        self.embed = embed
        # Contrastive learning (perhaps with time-discounted probabilities)
        self.delta = delta
        # Can be: attention over memories, NEC-style DQN, PPO, SAC, etc.
        self.policy = policy

    def act(self, o_t, r_t):
        new_connection = False

        # Embedding/delta would ideally be recurrent and capture trajectory of at least two observations
        c_t = self.embed(o_t)

        self.M_t_minus_1 = self.M_t
        self.M_t = Memory()
        margin = 0.75
        self.M_t.add([m for m in self.M_t_minus_1.get_futures() if self.delta(c_t, m.concept) > margin])

        if self.M_t.n == 0:
            new_connection = True
            self.traverse(c_t)

        self.update(c_t)

        a_t = self.policy(c_t, self.M_t).sample()

        if self.M_t.n == 0 or a_t not in self.M_t.get_actions():
            new_connection = True
            source = MemoryUnit(c_t, r_t, a_t)
            self.Memory.add(source)
            self.M_t.add(source)

        if new_connection:
            for source in self.M_t_minus_1.get_memories_list():
                # TODO faster index by action
                if source.action == a_t:
                    source.futures.add(self.M_t)
                    for sink in self.M_t.get_memories_list():
                        sink.pasts.add(source)
                    break

        pasts = {}
        for m in self.M_t.get_memories_list():
            pasts = pasts.keys() | m.pasts.memories.keys()
        for m in self.Memory.retrieve(self.M_t_minus_1.memories.keys() & pasts):
            m.access_date = time.time()


    def learn(self):
        # todo contrastive learning
        # todo Q learning






