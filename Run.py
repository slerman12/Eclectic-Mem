import gym
from Agent import Agent
import numpy as np
import torch
from torch.distributions import Categorical
import argparse
from pathlib import Path

snapshots_path = Path('./experiments')
snapshots_path.mkdir(exist_ok=True)
from clearml import Task, Logger


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
        logits = np.zeros(self.num_actions)
        for m in memories:
            if m.action != "terminal" and m.future_discounted_reward != -np.inf:
                logits[m.action] = max(logits[m.action], m.future_discounted_reward)
        logits = logits * self.softmax_temp
        # TODO can make deterministic instead (max Q-value) and just use exploration rate for exploration
        return Categorical(logits=torch.tensor(logits.tolist()))


def print_stats(*stats):
    for stat in stats:
        print(stat[0] + ": {}".format(stat[1]))
    print()


if __name__ == "__main__":
    task = Task.init(project_name="Eclectic-Mem", task_name="trains_plot", output_uri=str(snapshots_path))
    parser = argparse.ArgumentParser(description='Parameter for agent')
    parser.add_argument('--delta-margin', type=int, default=15, metavar='N')
    parser.add_argument('-n', type=int, default=100000, metavar='N')
    parser.add_argument('-k', type=int, default=100, metavar='N')
    parser.add_argument('-max-traversal-steps', type=int, default=10000, metavar='N')
    parser.add_argument('-num-episodes', type=int, default=150, metavar='N')
    args = parser.parse_args()
    logger = task.get_logger()
    env_id = 'CartPole-v0'

    env = gym.make(env_id)

    env.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    o = env.reset()
    r = 0

    inputs_dim = np.prod(o.shape)
    outputs_dim = env.action_space.n

    embed = Embed()
    delta = Delta()
    policy = Policy(outputs_dim)

    agent = Agent(embed, delta, policy, delta_margin=args.delta_margin, N=args.n, k=args.k,
                  max_traversal_steps=args.max_traversal_steps)

    rewards = []
    avg_head_sizes = []
    avg_num_explored = []
    lookup_count = []

    episode_len = np.inf

    for episode in range(args.num_episodes):
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

        rewards.append(total_reward)
        avg_head_sizes.append(agent.head_count / steps)
        avg_num_explored.append(agent.explored_count / steps)
        lookup_count.append(agent.lookup_count / steps)

        print_stats(("Episode", episode), ("Steps", steps), ("Memory Size", agent.Memory.n),
                    ("Avg Head Size", agent.head_count / steps), ("Avg Num Explored", agent.explored_count / steps),
                    ("Avg Num Futures", agent.futures_count / steps), ("Lookup Count", agent.lookup_count),
                    ("Avg Traversal Time", agent.traversal_time / steps), ("Reward", total_reward))

        logger.report_scalar("Explored vs Headsize", "Avg Num Explored",
                             iteration=episode, value=agent.head_count / steps)
        logger.report_scalar("Explored vs Headsize", "Avg Head Size",
                             iteration=episode, value=agent.explored_count / steps)
        logger.report_scalar("Reward vs Lookup", "Reward", iteration=episode, value=total_reward)
        logger.report_scalar("Reward vs Lookup", "Lookup Count",
                             iteration=episode, value=agent.lookup_count / steps)
        logger.report_scalar("Avg Head Size / Avg Num Explored", "Average",
                             iteration=episode, value=agent.head_count / agent.explored_count)
        logger.report_scalar("Avg Num Explored", "Average", iteration=episode, value=agent.explored_count / steps)
        logger.report_scalar("Avg Head Size", "Average", iteration=episode, value=agent.explored_count / steps)
        logger.report_scalar("Avg Traversal Time", "Average", iteration=episode, value=agent.traversal_time / steps)
        logger.report_scalar("Avg Num Futures", "Average", iteration=episode, value=agent.futures_count / steps)

        agent.learn()

    make_scatter = lambda x: list(zip(*x))
    logger.report_scatter2d("Avg Head Size/Avg Num Explored", "series_markers", iteration=episode,
                            scatter=make_scatter([avg_head_sizes, avg_num_explored]),
                            xaxis="Avg Head Size", yaxis="Avg Num Explored", mode='markers')
    logger.report_scatter2d("Avg Head Size/Rewards", "series_markers", iteration=episode,
                            scatter=make_scatter([avg_head_sizes, rewards]), xaxis="Avg Head Size",
                            yaxis="Reward", mode='markers')
    logger.report_scatter2d("Avg Num Explored/Rewards", "series_markers", iteration=episode,
                            scatter=make_scatter([avg_num_explored, rewards]), xaxis="Avg Num Explored",
                            yaxis="Reward", mode='markers')
    logger.report_scatter2d("Lookup Count/Rewards", "series_markers", iteration=episode,
                            scatter=make_scatter([lookup_count, rewards]), xaxis="Lookup Count",
                            yaxis="Reward", mode='markers')
