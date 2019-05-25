import torch
from torch import nn
import torch.nn.functional as F

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        conv1_num = 3
        conv2_num = 6
        self.conv1 = nn.Sequential(nn.Conv2d(1, conv1_num, 3),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, 2))
        self.conv2 = nn.Sequential(nn.Conv2d(conv1_num, conv2_num, 3),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, 2))
        self.wc1 = nn.Linear(conv2_num, 2)
        self.softmax = nn.Softmax(dim=1)

    def to_one_dim(self, x):
        return x.view(x.size(0), -1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.to_one_dim(x)
        x = self.wc1(x)
        x = self.softmax(x)
        return x
