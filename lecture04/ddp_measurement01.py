# -*-coding: utf-8 -*-
# Accuracy for classification
import os

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.transforms as TF

def setup_for_distributed(is_master):
    import builtins as __builtin__

    builtins_print = __builtin__.print
    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtins_print(*args, ** kwargs)

    __builtin__.print = print

class MyModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(MyModel, self).__init__()
        self.ext_feature = nn.Sequential(nn.Linear(input_size, 512), nn.ReLU(),
                                         nn.Linear(512, 256), nn.ReLU(),
                                         nn.Linear(256, 128), nn.ReLU())
        self.classifier = nn.Linear(128, output_size)

    def forward(self, inputs):
        inputs = torch.flatten(inputs, start_dim=1)
        features = self.ext_feature(inputs)
        outputs = self.classifier(features)
        return outputs

def trainer(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    print(f"Current rank: {rank} / {world_size}")
    setup_for_distributed(rank == 0)

    input_size = 28 * 28
    output_size = 10
    batch_size = 32

    train_transforms = TF.Compose([TF.ToTensor()])
    train_dataset = torchvision.datasets.MNIST(root="/workspace/doyoung/lecture_ddp/dataset/",
                                               train=True, download=True, transform=train_transforms)
    train_sampler = torch.utils.data.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler)

    model = MyModel(input_size, output_size).to(rank)
    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.1)

    ddp_model.train()
    for epoch in range(100):
        print(f"Epoch [{epoch:2d}/100]")
        correct = torch.zeros(1).to(rank)
        count = torch.zeros(1).to(rank)
        for step, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(rank)
            targets = targets.to(rank)

            outputs = ddp_model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            predictions = torch.argmax(outputs, dim=1)
            correct += torch.eq(targets, predictions).sum()
            count += outputs.size(0)

            t = loss.clone().detach()
            dist.barrier()
            dist.all_reduce(t)
            if step % 100 == 0:
                print(f"\tAverage loss [{step}]: {t / world_size}")

        dist.barrier()
        dist.all_reduce(correct)
        dist.all_reduce(count)
        acc = (correct/count)*100
        print(f"Accuracy: {acc.item():.2f}%")
        print(f"correct: {correct.item()}")
        print(f"Train dataset: {count.item()}, {len(train_dataset)}")

    dist.destroy_process_group()


if __name__ == '__main__':
    world_size = 8
    mp.spawn(trainer, args=(world_size, ), nprocs=world_size, join=True)
