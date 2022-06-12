# -*-coding: utf-8 -*-
import os
import argparse
import time

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.transforms as TF

class MyModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(MyModel, self).__init__()
        self.ext_feature = nn.Sequential(nn.Linear(input_size, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, 256),
                                    nn.ReLU(),
                                    nn.Linear(256, 128),
                                    nn.ReLU())
        self.classifier = nn.Linear(128, output_size)

    def forward(self, inputs):
        inputs = torch.flatten(inputs, start_dim=1)
        features = self.ext_feature(inputs)
        outputs = self.classifier(features)
        return outputs

def example_basic(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    print(f"Current rank: {rank} / {world_size}")

    input_size = 28 * 28
    output_size = 10
    batch_size = 32

    train_transforms = TF.Compose([TF.ToTensor()])
    train_dataset = torchvision.datasets.MNIST(root="/workspace/doyoung/lecture_ddp/dataset/",
                                               train=True, download=True, transform=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = MyModel(input_size, output_size).to(rank)
    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.001)

    ddp_model.train()

    for step, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(rank)
        targets = targets.to(rank)

        outputs = ddp_model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        tensor_list = [torch.zeros_like(loss).to(rank) for _ in range(world_size)]
        t = loss.clone().detach()

        if dist.get_rank() == 0:
            print(f"==== Step {step:2d} ====")
        dist.barrier()
        dist.all_gather(tensor_list, t)

        print(f"\tloss list in total rank [{rank}]: {tensor_list}")

        dist.barrier()
        if dist.get_rank() == 0:
            t_mean = torch.mean(torch.tensor(tensor_list))
            print(f"\taverage loss = {t_mean}")

        if step > 5:
            break

    dist.destroy_process_group()

if __name__ == '__main__':
    world_size = 8
    mp.spawn(example_basic, args=(world_size, ), nprocs=world_size, join=True)