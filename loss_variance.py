# -*- coding: utf-8 -*-
"""看单个 batch 的 loss 到底有多飘。

只训练 1 个 epoch，把所有 batch 的 loss 全记下来，看分布。
用途：说明"打印出来的某一批 loss"为什么不能当指标。
"""
import statistics

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_dataset = datasets.MNIST(root="./data", train=True, download=False, transform=transform)


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


def collect(bs):
    torch.manual_seed(0)
    loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    losses = []
    for data, target in loader:
        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def pct(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


for bs in (64, 125):
    L = collect(bs)
    mean = statistics.mean(L)
    sd = statistics.pstdev(L)
    print(f"===== batch_size = {bs}  ({len(L)} batches) =====")
    print(f"  mean  = {mean:.4f}   sd = {sd:.4f}   (relative sd = {sd / mean * 100:.0f}%)")
    print(f"  min   = {min(L):.4f}   max = {max(L):.4f}   (max/min = {max(L) / min(L):.1f}x)")
    print("  percentiles: " + "  ".join(f"p{p}={pct(L, p):.3f}" for p in (5, 25, 50, 75, 95)))
    print(f"  低于 0.05 的 batch 占比: {sum(1 for x in L if x < 0.05) / len(L) * 100:.1f}%")
    print(f"  高于 0.30 的 batch 占比: {sum(1 for x in L if x > 0.30) / len(L) * 100:.1f}%")
    print()
