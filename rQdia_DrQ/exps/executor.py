import itertools
import logging
import time
from functools import partial
from multiprocessing import Pool

import GPUtil
import numpy as np
from pathlib import Path
from yaml import Loader
from yaml import load_all

logging.basicConfig(level=logging.INFO)
import argparse


class Executor:
    def __init__(self, args):
        self.task = None
        self.args = args
        configs = list(load_all(open(self.args.config, 'r'), Loader=Loader))

        self.tasks = [self.args]

        for config in configs:
            seeds = set()
            while len(seeds) != 5:
                seeds.add(np.random.randint(1, 1000000))
            for seed in seeds:
                task = argparse.Namespace(**vars(args))
                config['seed'] = seed
                config['exp_name'] = config['domain_name'] + '_' + str(seed)
                task.__dict__.update(config)
                print(f"added exp: {config['exp_name']}")
                self.tasks.append(task)

    def run(self, func):
        gpus = GPUtil.getAvailable(order='memory', maxMemory=0.9, limit=8)
        print(f'FOUND GPUS {gpus}')
        high_order_func = partial(reports, func)
        proc_num = 6 if 6 / len(gpus) <= 3 else len(gpus) * 3
        with Pool(processes=proc_num) as pool:
            for idx, (gpu_id, task) in enumerate(zip(itertools.cycle(gpus), self.tasks)):
                pool.apply_async(high_order_func, (gpu_id, task))
            pool.close()
            pool.join()

    # processes = []
    # for idx, (gpu_id, task) in enumerate(zip(itertools.cycle(gpus), self.tasks)):
    #     p = mp.Process(target=high_order_func, args=(gpu_id, task))
    #     p.start()
    #     processes.append(p)
    #     if idx % len(gpus) == len(gpus) - 1:
    #         for p in processes:
    #             p.join()
    #             p.close()
    #         processes.clear()
    # for p in processes:
    #     p.join()
    #     p.close()

    #
    # pool = MyPool(len(gpus))
    # pool.starmap(high_order_func,zip(itertools.cycle(gpus),self.tasks))
    # pool.close()
    # pool.join()
    # # with MyPool(processes=) as p:


#
# class NoDaemonProcess(mp.Process):
#     # make 'daemon' attribute always return False
#     def _get_daemon(self):
#         return False
#
#     def _set_daemon(self, value):
#         pass
#
#     daemon = property(_get_daemon, _set_daemon)
#
#
# # We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# # because the latter is only a wrapper function, not a proper class.
# class MyPool(mp.Pool):
#     Process = NoDaemonProcess

def reports(func, gpu_id, args):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    args.__dict__['gpu_id'] = gpu_id
    # exps.setup(config_dict)
    start = time.time()
    logging.info(f'@GPU:{gpu_id} {args.exp_name} START')
    func(args)
    logging.info(f'@GPU:{gpu_id} {args.exp_name} END pip : {time.time() - start:.2f}s')
