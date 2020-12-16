import torch

use_cuda = torch.cuda.is_available()


def move_to_gpu(var):
  if use_cuda:
    return var.cuda()
  else:
    return var
