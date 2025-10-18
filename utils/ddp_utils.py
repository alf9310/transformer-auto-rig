import os

import torch.distributed as dist
from rich import print


def print_rank0(*args, **kwargs):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args, **kwargs)
    else:
        print(*args, **kwargs)


def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    else:
        return 0


def get_world_size():
    if dist.is_initialized():
        return dist.get_world_size()
    else:
        return 1


def get_local_rank():
    if dist.is_initialized():
        return int(os.environ["LOCAL_RANK"])
    else:
        return 0
