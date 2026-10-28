# -*- coding: utf-8 -*-
"""验收 4 第二小问：epochs 1 -> 3，看准确率能否到 99%。

不改 mnist.py，参数写死在这里。跑完把数字填进 experiments.md。
"""
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

BATCH = 64
EPOCHS = 3

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


torch.manual_seed(0)
train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"batch={BATCH}  epochs={EPOCHS}")
t0 = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    running = 0.0
    for data, target in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        optimizer.step()
        running += loss.item()
    print(f"epoch {epoch}  avg_loss={running / len(train_loader):.4f}  (+{time.time() - t0:.1f}s)")

train_time = time.time() - t0

model.eval()
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        correct += (model(data).argmax(dim=1) == target).sum().item()

print(f"train_time = {train_time:.2f} s")
print(f"test_acc   = {correct / len(test_dataset) * 100:.2f}%")
