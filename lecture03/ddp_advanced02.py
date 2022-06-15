# -*-coding: utf-8 -*-
# model checkpoint save
import os
import errno
import argparse

import torch
import torch.optim
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler

import torchvision
import torchvision.transforms as TF


def setup_for_distributed(is_master):
    import builtins as __builtin__

    builtins_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtins_print(*args, **kwargs)

    __builtin__.print = print


def set_up(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    print(f"Current rank: {rank} / {world_size}")
    setup_for_distributed(rank == 0)


def clean_up():
    dist.destroy_process_group()


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


def train_one_epoch(rank, model, train_loader, criterion, optimizer, **kwargs):
    print("  Training  ")
    print_freq = kwargs.pop("print_freq", 10)
    world_size = dist.get_world_size()

    model.train()
    for step, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(rank)
        targets = targets.to(rank)

        outputs = model(inputs)
        ave_loss = criterion(outputs, targets)

        ave_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        step_loss = ave_loss.clone().detach()
        dist.barrier()
        dist.all_reduce(step_loss)
        step_loss = step_loss / world_size

        if step % print_freq == 0:
            print(f"\t[ {step:4d} ] Average Loss: {step_loss:.8f}")


def validation(rank, model, valid_loader, criterion):
    print("  Validation  ")
    world_size = dist.get_world_size()

    total_loss = torch.zeros(1).to(rank)
    correct = torch.zeros(1).to(rank)
    count = torch.zeros(1).to(rank)

    with torch.no_grad():
        for step, (inputs, targets) in enumerate(valid_loader):
            inputs = inputs.to(rank)
            targets = targets.to(rank)

            outputs = model(inputs)
            predictions = torch.argmax(outputs, 1)
            correct += torch.eq(targets, predictions).sum()
            count += outputs.size(0)

            ave_loss = criterion(outputs, targets)

            step_loss = ave_loss.clone().detach()
            dist.barrier()
            dist.all_reduce(step_loss)
            total_loss += step_loss

    total_loss = total_loss / (world_size * step)
    dist.barrier()
    dist.all_reduce(correct)
    dist.all_reduce(count)
    acc = (correct / count) * 100
    print(f"\tValid Loss: {total_loss.item():.8f}")
    print(f"\tAccuracy: {acc.item():.2f}")

    return acc, total_loss


def trainer(rank, cfg):
    set_up(rank, cfg.world_size)

    # Call the Train and Validation dataset & loader with sampler
    train_transforms = TF.Compose([TF.ToTensor()])
    train_dataset = torchvision.datasets.MNIST(root=cfg.data_path, transform=train_transforms,
                                               train=True, download=True)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, sampler=train_sampler)

    valid_transforms = TF.Compose([TF.ToTensor()])
    valid_dataset = torchvision.datasets.MNIST(root=cfg.data_path, transform=valid_transforms,
                                               train=False, download=True)
    valid_sampler = DistributedSampler(valid_dataset)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size, sampler=valid_sampler)

    # Call the Model & DDP
    model = MyModel(28 * 28, 10).to(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    model_without_ddp = model.module

    # Call the Criterion & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model_without_ddp.parameters(), lr=cfg.lr)

    best_acc = 0
    for epoch in range(cfg.epochs):
        print(f"Epoch [ {epoch:4d} ]")
        train_sampler.set_epoch(epoch)
        train_one_epoch(rank, model, train_loader, criterion, optimizer, print_freq=cfg.print_freq)
        valid_acc, valid_loss = validation(rank, model, valid_loader, criterion)

        best_acc = valid_acc if best_acc < valid_acc else best_acc

        print("\n" + "+" * 50)

    print(f"Validation Best accuracy: {best_acc.item():2f}")
    clean_up()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", required=True, type=int)
    parser.add_argument("--print_freq", default=2, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--data_path", default="/workspace/doyoung/lecture_ddp/dataset/", type=str)
    parser.add_argument("--epochs", default=100, type=int)

    config = parser.parse_args()

    mp.spawn(trainer, args=(config,), nprocs=config.world_size)
