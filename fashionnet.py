import torch
from torch import nn
import torch.nn.functional as F
from auto_gpu import auto_gpu


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        conv1_num = 10
        conv2_num = 18
        conv3_num = 27
        class_num = 10
        self.conv1 = nn.Sequential(nn.Conv2d(1, conv1_num, 3),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, 2))
        self.conv2 = nn.Sequential(nn.Conv2d(conv1_num, conv2_num, 3),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, 2))
        self.conv3 = nn.Sequential(nn.Conv2d(conv2_num, conv3_num, 3),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, 2))
        self.wc1 = nn.Linear(conv3_num, class_num)
        self.inv_eyes = {}

    def to_one_dim(self, x):
        return x.view(x.size(0), -1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.to_one_dim(x)
        x = self.wc1(x)
        return x

class Net1_activate(nn.Module):
    def __init__(self):
        super(Net1_activate, self).__init__()
        conv1_num = 10
        conv2_num = 18
        conv3_num = 27
        class_num = 10
        self.conv1 = nn.Conv2d(1, conv1_num, 3)
        self.conv2 = nn.Conv2d(conv1_num, conv2_num, 3)
        self.conv3 = nn.Conv2d(conv2_num, conv3_num, 3)
        self.wc1 = nn.Linear(conv3_num, class_num)
        self.inv_eyes = {}

    def to_one_dim(self, x):
        return x.view(x.size(0), -1)

    def forward(self, x):
        m = nn.MaxPool2d(2, stride=2)
        x = self.conv1(x)
        x = abs_relu6(x)
        x = m(x)
        x = self.conv2(x)
        x = abs_relu6(x)
        x = m(x)
        x = self.conv3(x)
        x = abs_relu6(x)
        x = m(x)
        x = self.to_one_dim(x)
        x = self.wc1(x)
        return x

class FixedNet(nn.Module):
    def __init__(self):
        super(FixedNet, self).__init__()
        self.filters1 = []
        self.bias1 = []
        self.init_conv1()
        conv1_num = 9
        conv2_num = 18
        conv3_num = 27
        class_num = 10
        self.conv2 = nn.Sequential(nn.Conv2d(conv1_num, conv2_num, 3),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, 2))
        self.conv3 = nn.Sequential(nn.Conv2d(conv2_num, conv3_num, 3),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, 2))
        self.wc1 = nn.Linear(conv3_num, class_num)

    def conv1(self, x):
        x = F.conv2d(x, self.filters1, self.bias1)
        x = bothway_relu6(x)
        x = abs_maxpool_2d(x)
        return x

    def to_one_dim(self, x):
        return x.view(x.size(0), -1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.to_one_dim(x)
        x = self.wc1(x)
        return x

    def init_conv1(self):
        filters = [[]] * 9
        filters[0] = [[0.5, 0.5, 0.5],
                      [0.5, 0.5, -1],
                      [0.5, -1, -1]]
        filters[1] = [[0.5, 0.5, 0.5],
                      [-1, 0.5, 0.5],
                      [-1, -1, 0.5]]
        filters[2] = [[0.5, 0.5, 0.5],
                      [0.5, 0.5, 0.5],
                      [-1, -1, -1]]
        filters[3] = [[0.5, 0.5, -1],
                      [0.5, 0.5, -1],
                      [0.5, 0.5, -1]]
        filters[4] = [[0.25, -2, 0.25],
                      [0.25, 0.25, 0.25],
                      [0.25, 0.25, 0.25]]
        filters[5] = [[0.25, 0.25, 0.25],
                      [0.25, 0.25, 0.25],
                      [0.25, -2, 0.25]]
        filters[6] = [[0.25, 0.25, 0.25],
                      [-2, 0.25, 0.25],
                      [0.25, 0.25, 0.25]]
        filters[7] = [[0.25, 0.25, 0.25],
                      [0.25, 0.25, -2],
                      [0.25, 0.25, 0.25]]
        filters[8] = [[0.25, 0.25, 0.25],
                      [0.25, -2, 0.25],
                      [0.25, 0.25, 0.25]]
        bias = [[0]] * 9
        for i in range(len(filters)):
            self.filters1.append(torch.FloatTensor(filters[i]).unsqueeze(0).unsqueeze(0))
            bias[i] = torch.FloatTensor(bias[i])
            bias[i].requires_grad_(True)
            self.bias1.append(bias[i])
        self.filters1 = auto_gpu(torch.cat(self.filters1, dim=0))
        self.bias1 = auto_gpu(torch.cat(self.bias1, dim=0))
        self.bias1.requires_grad_(True)


def bothway_relu6(x):
    index = torch.where(torch.abs(x) < 0.5)
    x[index] = 0
    x = torch.min(x, auto_gpu(torch.Tensor([6])))
    x = torch.max(x, auto_gpu(torch.Tensor([-6])))
    return x

def abs_relu6(x):
    x = torch.abs(x)
    # index = torch.where(x < 0.1)
    # x[index] = 0
    return x

def abs_maxpool_2d(x):
    x_pool, index = F.max_pool2d_with_indices(torch.abs(x), (2, 2), 2)
    x_flat = x.reshape([x.shape[0], x.shape[1], -1])
    index_flat = index.reshape([index.shape[0], index.shape[1], -1])
    x_selected = torch.gather(x_flat, 2, index_flat).reshape(x_pool.shape)
    x_pool = x_pool * torch.sign(x_selected)
    return x_pool