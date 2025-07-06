import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class NetDebug(Net):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.flatten(x, 1)
        w = self.fc1(x)
        x = F.relu(w)
        x = self.fc2(x)
        return x, w
