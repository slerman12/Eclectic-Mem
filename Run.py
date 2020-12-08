import gym
from Agent import Agent
import numpy as np
from torch.distributions import Categorical


class Embed:
    def __call__(self, observation):
        return observation


class Delta:
    def train(self, traces):
        pass

    def __call__(self, c_i, c_j):
        return 1 / np.linalg.norm(c_i - c_j)


class Policy:
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def train(self, traces):
        pass

    def __call__(self, c, memories):
        logits = np.empty(self.num_actions)
        for m in memories:
            logits[m.action] = m.future_discounted_reward
        np.nan_to_num(logits, copy=False, nan=float(np.nanmean(logits)) if len(memories) > 0 else 0)
        return Categorical(logits=logits)


if __name__ == "__main__":
    env_id = 'CartPole-v0'

    env = gym.make(env_id)
    env.seed(0)

    o = env.reset()
    r = 0

    inputs_dim = np.prod(o.shape)
    outputs_dim = env.action_space.n

    agent = Agent(Embed(), Delta(), Policy(outputs_dim), delta_margin=1/0.02)

    rewards = []

    episode_len = np.inf
    num_episodes = 100

    for episode in range(num_episodes):
        total_reward = 0
        steps = 0
        while True and steps < episode_len:
            a = agent.act(o.flatten(), r)
            o, r, done, _ = env.step(a)
            total_reward += r
            steps += 1
            if done:
                agent.add_terminal(o.flatten(), r)
                o = env.reset()
                r = 0
                break
        rewards.append(total_reward)
        print("\nEpisode: ", episode)
        print("Reward: ", total_reward)