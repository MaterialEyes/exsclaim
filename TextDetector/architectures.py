import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CNN1(nn.Module):
    def __init__(self):
        super(CNN1, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(128*49, 500)
        self.fc2 = nn.Linear(500, 409)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 128*49)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNN2(nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 7, padding=3)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(3920, 500)
        self.fc2 = nn.Linear(500, 409)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 3920)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
