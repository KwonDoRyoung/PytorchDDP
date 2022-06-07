# -*-coding: utf-8 -*-
import torch.cuda
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.len


class MyModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(MyModel, self).__init__()
        self.fc = nn.Sequential(nn.Linear(input_size, output_size),
                                nn.Sigmoid(),
                                nn.Linear(output_size, output_size),
                                nn.Sigmoid())

    def forward(self, inputs):
        outputs = self.fc(inputs)
        print(f"\tIn model: input size: {inputs.size()}")
        print(f"\t          output size: {outputs.size()}")
        return outputs


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        raise RuntimeError("CPU don't support")

    input_size = 100
    output_size = 10
    data_size = 5000
    batch_size = 32

    rand_loader = DataLoader(RandomDataset(input_size, data_size), batch_size=batch_size, shuffle=True)
    model = MyModel(input_size, output_size).to(device)

    for data in rand_loader:
        # one step
        print("=" * 5 + " One Step " + "=" * 5)
        inputs = data.to(device)
        outputs = model(inputs)
        print(f"Outside: input size: {inputs.size()}")
        print(f"         output size: {outputs.size()}")
