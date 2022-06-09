# -*-coding: utf-8 -*-
import os

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

def example(rank, world_size):
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    model = nn.Linear(10, 10).to(rank)
    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.001)

    outputs = ddp_model(torch.randn(20, 10).to(rank))
    labels = torch.randn(20,10).to(rank)

    loss = loss_fn(outputs, labels)
    print(f"[{rank}]: {loss.item():.4f}, ({labels.size()})")
    loss.backward()

    optimizer.step()

if __name__ == '__main__':
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    world_size = 3
    mp.spawn(example, args=(world_size, ), nprocs=world_size, join=True)