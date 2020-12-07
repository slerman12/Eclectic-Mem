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
            memories = memories.get_memories()

        if isinstance(memories, MemoryUnit):
            memories = [memories]

        for memory in memories:
            if memory not in self:
                if self.n == self.N:
                    self.remove(self.get_LRA())

                self.memories[memory.id] = memory
                self.n += 1

    def remove(self, memory, de_reference=True):
        if memory in self:
            del self.memories[memory.id]
            if de_reference:
                for future in memory.futures.get_memories():
                    future.pasts.remove(memory, de_reference=False)
                for past in memory.pasts.get_memories():
                    past.futures.remove(memory, de_reference=False)
            self.n -= 1

    def get_memories(self):
        return self.memories.values()

    def get_LRA(self):
        # Least recently accessed memory
        # TODO can make much faster by keeping cache of memories sorted by access time
        return min(self.get_memories(), key=lambda memory: memory.access_time)

    def __contains__(self, memory):
        assert isinstance(memory, MemoryUnit)
        return memory.id in self.memories


class MemoryUnit:
    def __init__(self, concept, reward, action, access_time=None, terminal=False):
        self.id = secrets.token_bytes()
        self.concept = concept
        self.reward = reward
        self.action = action
        self.access_time = time.time() if access_time is None else access_time
        self.terminal = terminal
        self.futures = Memory()
        self.pasts = Memory()
        self.future_discounted_reward = -inf

    def merge(self, memory):
        self.futures.memories.update(memory.futures.memories)
        self.futures.n = len(self.futures.memories)
        self.pasts.memories.update(memory.pasts.memories)
        self.pasts.n = len(self.pasts.memories)
        for m in memory.futures.get_memories():
            m.pasts.add(self)
        for m in memory.pasts.get_memories():
            m.futures.add(self)


class TrajectoryHead:
    def __init__(self):
        self.units = {}
        self.memories = Memory()
        self.trace = None
        self.action_unit = None
        self.n = 0

    def add(self, unit, past=None):
        if unit not in self:
            if isinstance(unit, MemoryUnit):
                self.memories.add(unit)
                self.units[unit.id] = TrajectoryUnit(unit)
            elif isinstance(unit, TrajectoryUnit):
                self.memories.add(unit.memory)
                self.units[unit.id] = unit
        if past is not None:
            assert isinstance(past, TrajectoryUnit)
            self.units[unit.id].pasts.add(past)
        self.n = self.memories.n

    def get_units(self):
        return self.units.values()

    def get_memories(self):
        return self.memories.get_memories()

    def __contains__(self, unit):
        return unit.id in self.units


class TrajectoryUnit:
    def __init__(self, memory):
        self.id = memory.id
        self.memory = memory
        self.pasts = TrajectoryHead()
        self.trace = None


class Trace:
    def __init__(self, observation, concept, reward, action, memories, access_time, past_trace, terminal=False):
        self.observation = observation
        self.concept = concept
        self.reward = reward
        self.action = action
        self.memories = memories
        self.access_time = access_time
        self.past_trace = past_trace
        self.future_discounted_reward = None
        self.terminal = terminal


class Agent:
    def __init__(self, embed, delta, policy, N=inf, max_traversal_steps=inf, delta_margin=0.75, T=inf, gamma=1):
        # CNN
        self.embed = embed
        # Contrastive learning (perhaps with time-discounted probabilities)
        self.delta = delta
        # Can be: attention over memories, NEC-style DQN, PPO, SAC, etc.
        self.policy = policy

        self.Memory = Memory(N)
        self.Head = TrajectoryHead()
        self.Traces = []

        self.max_traversal_steps = max_traversal_steps
        self.delta_margin = delta_margin
        # Reward time horizon
        self.T = T
        # Reward discount factor
        self.gamma = gamma

    # Note: incompatible with batches
    def act(self, o_t, r_t):
        return self.update(o_t, r_t)

    # Note: incompatible with batches
    def add_terminal(self, o_t, r_t):
        return self.update(o_t, r_t, terminal=True)

    # Note: incompatible with batches
    def update(self, o_t, r_t, terminal=False):
        # Embedding/delta would ideally be recurrent and capture trajectory of at least two observations
        c_t = self.embed(o_t)

        access_time = time.time()

        new_head = TrajectoryHead()
        for u in self.Head.get_units():
            for m_future in u.memory.futures.get_memories():
                if self.delta(c_t, m_future.concept) >= self.delta_margin:
                    new_head.add(m_future, u)
                    m_future.access_time = access_time

        # If existing connections do not yield similar memories, traverse to find similar memories
        if new_head.n == 0:
            self.traverse(new_head, c_t)

        # Merge memories that delta deems "the same" by action
        actions = self.merge_memories_by_action(new_head, c_t)

        a_t = "terminal" if terminal else self.policy(c_t, new_head.memories).sample()

        if a_t in actions:
            new_head.action_unit = actions[a_t]
        else:
            # Store memory
            m = MemoryUnit(c_t, r_t, a_t, access_time, terminal)
            self.Memory.add(m)
            # Create new connection
            self.connect_memory(new_head, m)
            new_head.action_unit = new_head.units[m.id]

        new_head.trace = Trace(o_t, c_t, r_t, a_t, new_head.memories, access_time, self.Head.trace, terminal=terminal)
        self.Traces.append(new_head.trace)
        self.Head = TrajectoryHead() if terminal else new_head

        return a_t

    def connect_memory(self, new_head, memory):
        if self.Head.action_unit is not None:
            # Update memory futures/pasts
            past_unit = self.Head.action_unit
            past_unit.memory.futures.add(memory)
            memory.pasts.add(past_unit.memory)
            new_head.add(memory, past_unit)

    def traverse(self, new_head, concept):
        steps = 0
        max_delta = 0
        current_positions = Memory()
        current_positions.add(self.M_t_minus_1)
        explored = Memory()
        while steps < self.max_traversal_steps:
            if current_positions.n == 0:
                break
            new_max_delta = max_delta
            for m in current_positions.get_memories():
                if m.id not in explored.memories:
                    delta = self.delta(concept, m.concept)
                    if delta >= self.delta_margin:
                        # Create new connection
                        self.connect_memory(new_head, m)
                    if delta > max_delta:
                        new_max_delta = min(delta, self.delta_margin)
                        current_positions.add(m.futures)
                        current_positions.add(m.pasts)
                    explored.add(m)
                    steps += 1
                current_positions.remove(m, de_reference=False)
            max_delta = new_max_delta
        # TODO in rare cases, can also do full lookup

    def merge_memories_by_action(self, new_head, concept):
        # Note: maybe memories with different rewards should be kept unmerged
        # Note: if action is a tensor, python might not be able to use it as key
        actions = {}
        for m in self.M_t.get_memories():
            m.concept = concept
            if m.action in actions:
                m.reward = max(m.reward, actions[m.action])
                m.access_time = max(m.access_time, actions[m.action].access_time)
                # TODO can be made slightly more efficient by iterating merge and remove together
                m.merge(actions[m.action])
                self.Memory.remove(actions[m.action])
                self.M_t.remove(actions[m.action], de_reference=False)
            actions[m.action] = m
        return actions

    def add_terminal(self, o_t, r_t):

    def propogate_rewards(self):

    def learn(self):
        self.delta.train(self.Traces)
        self.policy.train(self.Traces)
        self.Traces = []






