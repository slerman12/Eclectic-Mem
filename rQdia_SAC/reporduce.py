import random
import time

from clearml import Task
import torch.multiprocessing as mp


def main(seed):
    print(f"{seed} Starting")
    Task.init(project_name="BugTest", task_name=f"test-{seed}")
    for i in range(5):
        time.sleep(1)
        print(f'something {seed}')


if __name__ == '__main__':
    WORLDSIZE = 3
    mp.set_start_method('spawn', force=True)
    seeds = [1, 2, 3]
    with mp.Pool(3) as pool:
        pool.map(main, seeds)
