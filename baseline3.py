# -*- coding: utf-8 -*-
"""基线重复测量：batch_size=64, epochs=1, 跑 3 次。

用途：给实验对照提供"基线的均值和波动"，而不是只用一次 98.27% 当参照。
不改动 mnist.py。
"""
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

BATCH = 64
EPOCHS = 1
REPEATS = 3

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_dataset = datasets.MNIST(root="./data", train=True, download=False, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=False, transform=transform)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def run_once():
    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    t0 = time.time()
    model.train()
    running = 0.0
    for data, target in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        optimizer.step()
        running += loss.item()
    train_time = time.time() - t0
    avg_loss = running / len(train_loader)

    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            pred = model(data).argmax(dim=1)
            correct += (pred == target).sum().item()

    return train_time, avg_loss, correct / len(test_dataset) * 100


print(f"batch={BATCH}  epochs={EPOCHS}  repeats={REPEATS}  threads={torch.get_num_threads()}")
print("-" * 56)
print(f"{'run':>3} | {'time(s)':>8} | {'avg_loss':>9} | {'acc(%)':>7}")
print("-" * 56)

accs = []
for i in range(1, REPEATS + 1):
    tt, al, acc = run_once()
    accs.append(acc)
    print(f"{i:>3} | {tt:>8.2f} | {al:>9.4f} | {acc:>7.2f}")

mean = sum(accs) / len(accs)
print("-" * 56)
print(f"mean acc = {mean:.2f}%   range = {max(accs) - min(accs):.2f}pp   "
      f"({min(accs):.2f} ~ {max(accs):.2f})")
