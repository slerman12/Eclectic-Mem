import argparse
import subprocess
from itertools import count
from tensorboard_logger import configure, log_value

from models import DQN
from nec_agent import NECAgent
from utils.atari_wrapper import make_atari, wrap_deepmind


def main(env_id, embedding_size):
    env = wrap_deepmind(make_atari(env_id), scale=True)
    embedding_model = DQN(embedding_size)
    agent = NECAgent(env, embedding_model)
    #
    subprocess.Popen(["tensorboard", "--logdir", "runs"])
    configure("runs/pong-run")

    for t in count():
        if t == 0:
            reward = agent.warmup()
        else:
            reward = agent.episode()
        print("Episode {}: Total Reward: {}".format(t, reward))
        log_value('score', reward, t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID',
                        default='PongNoFrameskip-v4')
    parser.add_argument('--embedding_size', help='embedding size', default=64)
    args = parser.parse_args()
    main(args.env, args.embedding_size)
