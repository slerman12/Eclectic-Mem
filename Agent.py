import secrets
import time
from math import inf


class Memory:
    def __init__(self, N=inf, track_actions=False):
        self.N = N
        self.n = 0
        self.memories = {}

        self.track_actions = track_actions
        if track_actions:
            self.action_memories = {}

    def __contains__(self, memory):
        return memory.id in self.memories

    def __add__(self, memories):
        return type(self)().add(self).add(memories)

    def add(self, memories):
        if isinstance(memories, Memory):
            if self.n + memories.n <= self.N and not self.track_actions:
                self.memories.update(memories.memories)
                self.n = len(self.memories)
                return self
            memories = memories.get_memories()

        if isinstance(memories, MemoryUnit):
            memories = [memories]

        for memory in memories:
            if memory not in self:
                if self.n == self.N:
                    self.remove(self.get_LRA())

                self.memories[memory.id] = memory
                self.n += 1
                if self.track_actions:
                    if memory.action in self.action_memories:
                        self.action_memories[memory.action].append(memory)
                    else:
                        self.action_memories[memory.action] = [memory]

        return self

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


class Trace:
    def __init__(self, observation, memory, memories, past_trace=None):
        self.observation = observation
        self.memory = memory
        self.memories = memories
        self.past_trace = past_trace

    def propagate_reward(self, value=0, gamma=1, T=inf):
        self.memory.future_discounted_reward = self.memory.reward + gamma * value
        if T > 1 and self.past_trace is not None:
            self.past_trace.propagate_reward(self.memory.future_discounted_reward, gamma, T - 1)


class Agent:
    def __init__(self, embed, delta, policy, N=inf, k=inf, max_traversal_steps=inf, delta_margin=0.75, T=inf, gamma=1):
        # CNN
        self.embed = embed
        # Contrastive learning (perhaps with time-discounted probabilities)
        self.delta = delta
        # Can be: attention over memories, NEC-style DQN, PPO, SAC, etc.
        self.policy = policy

        self.Memory = Memory(N)
        self.Head = Memory(track_actions=True)
        self.Traces = []

        self.k = k
        self.max_traversal_steps = max_traversal_steps
        self.delta_margin = delta_margin
        # Reward time horizon
        self.T = T
        # Reward discount factor
        self.gamma = gamma

        # Data collection variables just for stats/graphing
        self.lookup_count = 0
        self.traversal_time = 0
        self.head_count = 0
        self.explored_count = 0

    # Note: incompatible with batches
    def act(self, o_t, r_t):
        return self.update(o_t, r_t)

    # Note: incompatible with batches
    def add_terminal(self, o_t, r_t):
        self.update(o_t, r_t, terminal=True)

    # Note: incompatible with batches
    def update(self, o_t, r_t, terminal=False):
        # Embedding/delta would ideally be recurrent and capture trajectory of at least two observations
        c_t = self.embed(o_t)

        new_head = Memory(track_actions=True)

        access_time = time.time()

        # Traverse existing immediate connections
        explored = Memory()
        max_delta = -inf
        for m in self.Head.get_memories():
            m_max_delta = self.traverse(new_head, c_t, access_time, current_positions=m.futures, explored=explored)
            max_delta = max(max_delta, m_max_delta)

        # If existing connections do not yield similar memories, traverse to find similar memories; make new connections
        # Alternatively: if new_head.n < self.k:
        if new_head.n == 0:
            self.traverse(new_head, c_t, access_time, "search_from_explored", explored, max_delta, traverse=True)

        self.traversal_time += time.time() - access_time
        self.head_count += new_head.n
        self.explored_count += explored.n

        a_t = "terminal" if terminal else self.policy(c_t, new_head.get_memories()).sample().item()

        # Store memory
        m = MemoryUnit(c_t, r_t, a_t, access_time, terminal)
        self.Memory.add(m)
        # Create new connection
        self.connect_memory(new_head, m, access_time)

        past_trace = self.Traces[-1] if len(self.Traces) > 0 else None
        trace = Trace(o_t, m, new_head.get_memories(), past_trace)
        self.Traces.append(trace)

        new_head.action_memories = new_head.action_memories[a_t]
        self.Head = Memory(track_actions=True) if terminal else new_head

        return a_t

    def connect_memory(self, new_head, memory, access_time):
        new_head.add(memory)
        memory.access_time = access_time
        if self.Head.action_memories is not None:
            # TODO should new memories be connected only to last action ones like this or to all or to most predictive?
            for past_memory in self.Head.action_memories:
                # Update memory futures/pasts
                past_memory.futures.add(memory)
                memory.pasts.add(past_memory)

    def traverse(self, new_head, concept, access_time, current_positions, explored, max_delta=-inf, traverse=False):
        if current_positions == "search_from_explored":
            current_positions = Memory()
            for e in explored.get_memories():
                current_positions.add(e.futures).add(e.pasts)
        for m in current_positions.get_memories():
            if explored.n >= self.max_traversal_steps or explored.n >= self.Memory.n or new_head.n >= self.k:
                return max_delta
            if m not in explored:
                explored.add(m)
                # Note: not traversing terminal observations or current trajectory
                if m.action != "terminal" and m.future_discounted_reward != -inf:
                    delta = self.delta(concept, m.concept)
                    max_delta = min(max(delta, max_delta), self.delta_margin)
                    if delta >= self.delta_margin:
                        # Create new connection
                        self.connect_memory(new_head, m, access_time)
                    elif delta > max_delta and traverse:
                        # Greedy traversal
                        return self.traverse(new_head, concept, access_time, m.futures + m.pasts, explored, delta, True)

        if explored.n < self.Memory.n and explored.n < self.max_traversal_steps and new_head.n < self.k and traverse:
            self.lookup_count += 1
            # TODO should be randomly shuffled Memory e.g. via O(1) random sampling
            return self.traverse(new_head, concept, access_time, self.Memory, explored, max_delta, True)

        return max_delta

    def learn(self):
        # TODO should use Value of last trace's observation
        self.Traces[-1].propagate_reward(value=0, gamma=self.gamma, T=self.T)

        self.delta.train(self.Traces)
        self.policy.train(self.Traces)

        self.Traces = []
        self.lookup_count = self.traversal_time = self.head_count = self.explored_count = 0
