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
            memories = memories.memories.values()

        if isinstance(memories, MemoryUnit):
            memories = [memories]

        for m in memories:
            if m.id not in self.memories:
                if self.n == self.N:
                    self.remove(self.get_LRA())

                self.memories[m.id] = m
                self.n += 1

    def remove(self, memory, de_reference=True):
        del self.memories[memory.id]
        if de_reference:
            for m in memory.futures.memories.values():
                m.pasts.remove(memory, de_reference=False)
            for m in memory.pasts.memories.values():
                m.futures.remove(memory, de_reference=False)
        self.n -= 1

    def retrieve(self, ids):
        if isinstance(ids, (list, set)):
            return [self.memories[ID] for ID in ids]
        else:
            return self.memories[ids]

    def get_futures(self):
        future_ids = {}
        for m in self.memories.values():
            future_ids.update(m.futures.get_ids())
        return self.retrieve(future_ids)

    def get_memory_by_action(self, action):
        # TODO can do faster by indexing by action
        for m in self.memories.values():
            if m.action == action:
                return m

    def get_LRA(self):
        # Least recently accessed memory
        # TODO can make much faster by keeping cache of memories sorted by access time
        return min(self.memories.values(), key=lambda m: m.access_time)


class MemoryUnit:
    def __init__(self, concept, reward, action, access_time=None):
        self.id = secrets.token_bytes()
        self.concept = concept
        self.reward = reward
        self.action = action
        self.futures = Memory()
        self.pasts = Memory()
        self.access_time = time.time() if access_time is None else access_time

    def merge(self, memory):
        self.futures.memories.update(memory.futures.memories)
        self.futures.n += memory.futures.n
        self.pasts.memories.update(memory.pasts.memories)
        self.pasts.n += memory.pasts.n
        for m in memory.futures.memories.values():
            m.pasts.add(self)
        for m in memory.pasts.memories.values():
            m.futures.add(self)


class Agent:
    def __init__(self, N, embed, delta, policy, max_traversal_steps=inf, delta_margin=0.75):
        self.Memory = Memory(N)
        self.M_t = Memory()
        self.M_t_minus_1 = Memory()
        self.a_t_minus_1 = None

        # CNN
        self.embed = embed
        # Contrastive learning (perhaps with time-discounted probabilities)
        self.delta = delta
        # Can be: attention over memories, NEC-style DQN, PPO, SAC, etc.
        self.policy = policy

        self.max_traversal_steps = max_traversal_steps
        self.delta_margin = delta_margin

    def act(self, o_t, r_t):
        # Embedding/delta would ideally be recurrent and capture trajectory of at least two observations
        c_t = self.embed(o_t)

        self.M_t_minus_1 = self.M_t
        self.M_t = Memory()
        self.M_t.add([m for m in self.M_t_minus_1.get_futures() if self.delta(c_t, m.concept) >= self.delta_margin])

        new_connection = False

        if self.M_t.n == 0:
            self.traverse(c_t)
            new_connection = True

        # Merge memories that delta deems "the same"
        # Note: maybe memories with different rewards should be kept unmerged
        # Note: if action is a tensor, python might not be able to use it as key
        actions = {}
        for m in self.M_t.memories.values():
            m.concept = c_t
            if m.action in actions:
                m.reward = max(m.reward, actions[m.action])
                m.access_time = max(m.access_time, actions[m.action].access_time)
                # TODO can be made slightly more efficient by iterating merge and remove together
                m.merge(actions[m.action])
                self.Memory.remove(actions[m.action])
                self.M_t.remove(actions[m.action], de_reference=False)
            actions[m.action] = m

        a_t = self.policy(c_t, self.M_t).sample()

        access_time = time.time()

        # Store memory
        if self.M_t.n == 0 or a_t not in actions:
            m = MemoryUnit(c_t, r_t, a_t, access_time=access_time)
            self.Memory.add(m)
            self.M_t.add(m)
            new_connection = True

        # Update memory futures/pasts
        if new_connection and self.a_t_minus_1 is not None:
            past_memory = self.M_t_minus_1.get_memory_by_action(self.a_t_minus_1)
            for m in self.M_t.memories.values():
                past_memory.futures.add(m)
                m.pasts.add(past_memory)

        # Update memory access times
        # Todo can improve efficiency by computing on the fly
        pasts = {}
        for m in self.M_t.memories.values():
            pasts = pasts.keys() | m.pasts.memories.keys()
        for m in self.Memory.retrieve(self.M_t_minus_1.memories.keys() & pasts):
            m.access_time = access_time

        self.a_t_minus_1 = a_t  # If terminal, set to None

        return a_t

    def traverse(self, concept):
        steps = 0
        max_delta = 0
        current_positions = Memory()
        current_positions.add(self.M_t_minus_1)
        explored = Memory()
        while steps < self.max_traversal_steps:
            if current_positions.n == 0:
                break
            new_max_delta = max_delta
            for m in current_positions.memories.values():
                if m.id not in explored.memories:
                    delta = self.delta(concept, m.concept)
                    if delta >= self.delta_margin:
                        self.M_t.add(m)
                    if delta > max_delta:
                        new_max_delta = min(delta, self.delta_margin)
                        current_positions.add(m.futures)
                        current_positions.add(m.pasts)
                    explored.add(m)
                    steps += 1
                current_positions.remove(m, de_reference=False)
            max_delta = new_max_delta
        # TODO in rare cases, can also do full lookup

    def learn(self, trajectories):
        self.delta.train(trajectories)
        self.policy.train(trajectories)






