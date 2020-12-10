import gym
from Agent import Agent
import numpy as np
import torch
from torch.distributions import Categorical
import os
import matplotlib.pyplot as plt


class Embed:
    def __call__(self, observation):
        return observation


class Delta:
    def train(self, traces):
        pass

    def __call__(self, c_i, c_j):
        return 1 / (np.linalg.norm(c_i - c_j) + 0.001)


class Policy:
    def __init__(self, num_actions, softmax_temp=0.1):
        self.num_actions = num_actions
        self.softmax_temp = softmax_temp

    def train(self, traces):
        pass

    def __call__(self, c, memories):
        logits = np.empty(self.num_actions)
        for m in memories:
            if m.action != "terminal" and m.future_discounted_reward != -np.inf:
                if np.isnan(logits[m.action]):
                    logits[m.action] = m.future_discounted_reward
                else:
                    logits[m.action] = max(logits[m.action], m.future_discounted_reward)
        np.nan_to_num(logits, copy=False, nan=float(np.nanmean(logits)) if len(memories) > 0 else 0)
        logits = logits * self.softmax_temp
        # TODO can make deterministic instead (max Q-value) and just use exploration rate for exploration
        return Categorical(logits=torch.from_numpy(logits))


def print_stats(*stats):
    for stat in stats:
        print(stat[0] + ": {}".format(stat[1]))
    print()


def plot():
    if not os.path.exists("Figures"):
        os.makedirs("Figures")

    _, ax = plt.subplots()

    ax.plot(avg_head_sizes, label="Avg Head Size")
    ax.plot(avg_num_explored, label="Avg Num Explored")
    plt.legend()
    ax.set_xlabel('Episode')
    ax.set_title('Traversal Stats: delta_margin=15, N=100000, k=100, max_traversal_steps=10000')
    ax.set_ylabel('Count')
    plt.savefig('Figures/traversal_stats.png', bbox_inches='tight')
    plt.close()

    plt.plot(avg_traversal_time)
    plt.xlabel("Episode")
    plt.ylabel("Avg Traversal Time")
    plt.title('Traversal Times: delta_margin=15, N=100000, k=100, max_traversal_steps=10000')
    plt.savefig('Figures/traversal_times.png', bbox_inches='tight')
    plt.close()

    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title('Rewards: delta_margin=15, N=100000, k=100, max_traversal_steps=10000')
    plt.savefig('Figures/rewards.png', bbox_inches='tight')
    plt.close()

    plt.plot(avg_lookup_count)
    plt.xlabel("Episode")
    plt.ylabel("Avg Lookup Count")
    plt.title('Lookups: delta_margin=15, N=100000, k=100, max_traversal_steps=10000')
    plt.savefig('Figures/lookups.png', bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    env_id = 'CartPole-v0'

    env = gym.make(env_id)
    env.seed(0)

    o = env.reset()
    r = 0

    inputs_dim = np.prod(o.shape)
    outputs_dim = env.action_space.n

    embed = Embed()
    delta = Delta()
    policy = Policy(outputs_dim)

    agent = Agent(embed, delta, policy, delta_margin=15, N=100000, k=100, max_traversal_steps=10000)

    rewards = []
    avg_head_sizes = []
    avg_num_explored = []
    avg_traversal_time = []
    avg_lookup_count = []

    episode_len = np.inf
    num_episodes = 2

    for episode in range(num_episodes):
        total_reward = 0
        steps = 0
        while True:
            a = agent.act(o.flatten(), r)
            o, r, done, _ = env.step(a)
            total_reward += r
            steps += 1
            if done or steps == episode_len:
                agent.add_terminal(o.flatten(), r)
                steps += 1
                o = env.reset()
                r = 0
                break

        print_stats(("Episode", episode), ("Steps", steps), ("Memory Size", agent.Memory.n),
                    ("Avg Head Size", agent.head_count / steps), ("Avg Num Explored", agent.explored_count / steps),
                    ("Lookup Count", agent.lookup_count), ("Avg Traversal Time", agent.traversal_time / steps),
                    ("Reward", total_reward))
        rewards.append(total_reward)
        avg_head_sizes.append(agent.head_count / steps)
        avg_num_explored.append(agent.explored_count / steps)
        avg_traversal_time.append(agent.traversal_time / steps)
        avg_lookup_count.append(agent.lookup_count / steps)
        agent.learn()

    plot()