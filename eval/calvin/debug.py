from asyncio import gather
import os
import sys
import tempfile
from unittest import result
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from open_flamingo.train.distributed import init_distributed_device, world_info_from_env
from torch.nn.parallel import DistributedDataParallel as DDP

# On Windows platform, the torch.distributed package only
# supports Gloo backend, FileStore and TcpStore.
# For FileStore, set init_method parameter in init_process_group
# to a local file. Example as follow:
# init_method="file:///f:/libtmp/some_file"
# dist.init_process_group(
#    "gloo",
#    rank=rank,
#    init_method=init_method,
#    world_size=world_size)
# For TcpStore, same way as on Linux.

def setup():
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic():
    setup()
    local_rank, rank, world_size = world_info_from_env()
    # create model and move it to GPU with id rank
    results = [local_rank] * 5
    if rank == 0:
        gathered_results = [None for _ in range(world_size)]
    else:
        gathered_results = None
    dist.gather_object(results, gathered_results, dst=0)

    if rank == 0:
        combined_results = [item for sublist in gathered_results for item in sublist]
        print(f"Combined Results: {combined_results}")
        
    dist.destroy_process_group()


if __name__ == "__main__":
    # n_gpus = torch.cuda.device_count()
    demo_basic()
    # run_demo(demo_checkpoint, world_size)
    # world_size = n_gpus//2
    # run_demo(demo_model_parallel, world_size)