import secrets
import time
from copy import copy
from math import inf


class Memory:
    def __init__(self, N=inf):
        self.N = N
        self.n = 0
        self.memories = {}

    def __contains__(self, memory):
        assert isinstance(memory, MemoryUnit)
        return memory.id in self.memories

    def add(self, memories):
        if isinstance(memories, Memory):
            self.memories.update(memories.memories)
            self.n = len(self.memories)
            assert self.n <= self.N
            return
            # memories = memories.get_memories()

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

    def merge_connections(self, memory):
        # Merge futures/pasts with another memory
        self.futures.add(memory.futures)
        self.pasts.add(memory.pasts)
        for m in memory.futures.get_memories():
            m.pasts.add(self)
        for m in memory.pasts.get_memories():
            m.futures.add(self)


class TrajectoryHead:
    def __init__(self, T=inf, gamma=1):
        self.units = {}
        self.memories = Memory()
        self.trace = None
        self.action_unit = None
        self.n = 0

        # Reward time horizon
        self.T = T
        # Reward discount factor
        self.gamma = gamma

    def __contains__(self, unit):
        assert isinstance(unit, (MemoryUnit, TrajectoryUnit))
        return unit.id in self.units

    def add(self, unit, past=None):
        assert isinstance(unit, (MemoryUnit, TrajectoryUnit))
        if unit not in self:
            if isinstance(unit, MemoryUnit):
                self.memories.add(unit)
                self.units[unit.id] = TrajectoryUnit(unit, self.T, self.gamma)
            elif isinstance(unit, TrajectoryUnit):
                self.memories.add(unit.memory)
                self.units[unit.id] = unit
        if past is not None:
            assert isinstance(past, TrajectoryUnit)
            self.units[unit.id].pasts.add(past)
        self.n = self.memories.n

    def remove(self, unit):
        if unit in self:
            memory = self.units[unit.id].memory
            del self.units[unit.id]
            self.memories.remove(memory, de_reference=False)
            self.n -= 1

    def get_units(self):
        return self.units.values()

    def get_memories(self):
        return self.memories.get_memories()

    def set_trace(self, trace):
        self.trace = trace
        for unit in self.units.values():
            unit.trace = trace

    def propogate_reward(self, running_future_discounted_reward=0, steps=0, propogated=None):
        if propogated is None:
            propogated = {}
        future_discounted_reward = None
        steps += 1
        pasts_propogated = {}
        for unit in [self.units[ID] for ID in self.units if ID not in propogated]:
            propogated.update(unit.id)
            if future_discounted_reward is None:
                future_discounted_reward = unit.trace.r + self.gamma * running_future_discounted_reward
                unit.trace.future_discounted_reward = future_discounted_reward
            unit.memory.future_discounted_reward = max(future_discounted_reward, unit.memory.future_discounted_reward)
            if steps < self.T:
                unit.pasts.propogate_reward(future_discounted_reward, steps, pasts_propogated)
            else:
                unit.pasts = TrajectoryHead(self.T, self.gamma)


class TrajectoryUnit:
    def __init__(self, memory, T=inf, gamma=1):
        self.id = memory.id
        self.memory = memory
        self.pasts = TrajectoryHead(T, gamma)
        self.trace = None

    def merge_connections(self, unit):
        # Merge pasts and memory futures/pasts with another unit
        assert isinstance(unit, TrajectoryUnit)
        self.pasts.units.update(unit.pasts.units)
        self.pasts.memories.add(unit.pasts.memories)
        self.pasts.n = self.pasts.memories.n
        self.memory.merge_connections(unit.memory)


class Trace:
    def __init__(self, observation, concept, reward, action, memories, access_time, terminal=False):
        self.observation = observation
        self.concept = concept
        self.reward = reward
        self.action = action
        self.memories = memories
        self.access_time = access_time
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
        self.Head = TrajectoryHead(T, gamma)
        self.Traces = []

        self.max_traversal_steps = max_traversal_steps
        self.delta_margin = delta_margin

    # Note: incompatible with batches
    def act(self, o_t, r_t):
        a_t, _ = self.update(o_t, r_t)
        return a_t

    # Note: incompatible with batches
    def add_terminal(self, o_t, r_t):
        return self.update(o_t, r_t, terminal=True)

    # Note: incompatible with batches
    def update(self, o_t, r_t, terminal=False):
        # Embedding/delta would ideally be recurrent and capture trajectory of at least two observations
        c_t = self.embed(o_t)

        access_time = time.time()

        new_head = TrajectoryHead(self.Head.T, self.Head.gamma)
        for u in self.Head.get_units():
            for m_future in u.memory.futures.get_memories():
                if self.delta(c_t, m_future.concept) >= self.delta_margin:
                    new_head.add(m_future, u)
                    m_future.access_time = access_time

        # If existing connections do not yield similar memories, traverse to find similar memories
        if new_head.n == 0:
            self.traverse(new_head, c_t, access_time)

        # Merge memories that delta deems "the same" by action
        actions = self.merge_memories_by_action(new_head, c_t)

        a_t = "terminal" if terminal else self.policy(c_t, new_head.get_memories()).sample()

        if a_t in actions:
            new_head.action_unit = actions[a_t]
        else:
            # Store memory
            m = MemoryUnit(c_t, r_t, a_t, access_time, terminal)
            self.Memory.add(m)
            # Create new connection
            self.connect_memory(new_head, m, access_time)
            new_head.action_unit = new_head.units[m.id]

        new_head.set_trace(Trace(o_t, c_t, r_t, a_t, [copy(m) for m in new_head.get_memories()], access_time, terminal))
        self.Traces.append(new_head.trace)
        self.Head = TrajectoryHead(self.Head.T, self.Head.gamma) if terminal else new_head

        return a_t, new_head

    def connect_memory(self, new_head, memory, access_time):
        if self.Head.action_unit is not None:
            # Update memory futures/pasts
            past_unit = self.Head.action_unit
            past_unit.memory.futures.add(memory)
            memory.pasts.add(past_unit.memory)
            new_head.add(memory, past_unit)
            memory.access_time = access_time

    def traverse(self, new_head, concept, access_time):
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
                        self.connect_memory(new_head, m, access_time)
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
        # Note: maybe memories with different rewards/future-discounted-rewards should be kept unmerged
        # Note: if action is a tensor, python might not be able to use it as key
        actions = {}

        for u in new_head.get_units():
            u.memory.concept = concept
            if u.memory.action in actions:
                u.memory.reward = max(u.memory.reward, actions[u.memory.action].reward)
                u.memory.future_discounted_reward = max(u.memory.future_discounted_reward, actions[u.memory.action].future_discounted_reward)
                u.memory.access_time = max(u.memory.access_time, actions[u.memory.action].access_time)

                # TODO can be made slightly more efficient by iterating merge and remove together
                u.merge_connections(actions[u.memory.action])
                self.Memory.remove(actions[u.memory.action].memory)
                new_head.remove(actions[u.memory.action])

            actions[u.memory.action] = u

        return actions

    def learn(self):
        self.delta.train(self.Traces)
        self.policy.train(self.Traces)
        self.Traces = []







