# -*-coding: utf-8 -*-
import torch.cuda
import torch.nn as nn
import torchvision.datasets
import torchvision.transforms as TF
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

class MyModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(MyModel, self).__init__()
        self.fc = nn.Sequential(nn.Linear(input_size, output_size),
                                nn.Sigmoid(),
                                nn.Linear(output_size, output_size),
                                nn.Sigmoid())

    def forward(self, inputs):
        inputs = torch.flatten(inputs,start_dim=1)
        outputs = self.fc(inputs)
        return outputs


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        raise RuntimeError("CPU don't support")

    input_size = 28*28
    output_size = 10
    batch_size = 32

    train_transforms = TF.Compose([TF.ToTensor()])
    train_dataset = torchvision.datasets.MNIST(root="/workspace/doyoung/lecture_ddp/dataset/",
                                               train=True, download=True, transform=train_transforms)

    rand_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model = MyModel(input_size, output_size).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()
    scaler = GradScaler()

    for inputs, targets in rand_loader:
        print("=" * 5 + " One Step " + "=" * 5)
        optimizer.zero_grad()
        with autocast():
            inputs = inputs.to(device)
            targets = targets.to(device)
            print(f"input type outside in autocast: {inputs.dtype}")
            outputs = model(inputs)
            print(f"output type outside in autocast: {outputs.dtype}")
            loss = loss_fn(outputs, targets)
            print(f"loss type: {loss.dtype}")

        print(f"loss: {loss.item()}")
        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()
        break

