import torch
from torch import nn


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
        self.inv_eyes = {}

    def to_one_dim(self, x):
        return x.view(x.size(0), -1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.to_one_dim(x)
        x = self.wc1(x)
        x = self.softmax(x)
        return x


    def det_loss(self, det_ratio=0.001):
        det_loss = 0
        det_losses = []
        count = 0
        for weight_name, weight_value in self.named_parameters():
            if count == 0:
                count += 1
                continue
            if weight_name.startswith('conv') and weight_name.endswith('weight'):
                weight2d = weight_value.view(weight_value.shape[0], -1)
                weight_norm = torch.unsqueeze(torch.norm(weight2d, dim=1), 0)
                weight_norm_muti = weight_norm.t().mm(weight_norm)
                muti_weight = weight2d.mm(weight2d.t())
                if weight_name not in self.inv_eyes:
                    self.inv_eyes[weight_name] = (1 - torch.eye(muti_weight.shape[0], muti_weight.shape[1])).cuda()
                curr_det_loss = torch.sum(torch.abs(muti_weight * self.inv_eyes[weight_name] / weight_norm_muti)) *\
                    det_ratio / weight_value.shape[0]
                det_loss += curr_det_loss
                det_losses.append(curr_det_loss)
        return det_loss, det_losses


class DogCatNet(nn.Module):
    def __init__(self):
        super(DogCatNet, self).__init__()
        conv1_num = 32
        conv2_num = 64
        conv3_num = 64
        conv4_num = 128
        conv5_num = 64
        conv6_num = 32
        class_num = 2


        self.conv1 = nn.Sequential(nn.Conv2d(3, conv1_num, 3),
                                   nn.BatchNorm2d(conv1_num),
                                   nn.ReLU6(),
                                   nn.MaxPool2d(2, 2))
        self.conv2 = nn.Sequential(nn.Conv2d(conv1_num, conv2_num, 3),
                                   nn.BatchNorm2d(conv2_num),
                                   nn.ReLU6(),
                                   nn.MaxPool2d(2, 2))
        self.conv3 = nn.Sequential(nn.Conv2d(conv2_num, conv3_num, 3),
                                   nn.BatchNorm2d(conv3_num),
                                   nn.ReLU6(),
                                   nn.MaxPool2d(2, 2))
        self.conv4 = nn.Sequential(nn.Conv2d(conv3_num, conv4_num, 3),
                                   nn.BatchNorm2d(conv4_num),
                                   nn.ReLU6(),
                                   nn.MaxPool2d(2, 2))
        self.conv5 = nn.Sequential(nn.Conv2d(conv4_num, conv5_num, 3),
                                   nn.BatchNorm2d(conv5_num),
                                   nn.ReLU6(),
                                   nn.MaxPool2d(2, 2))
        self.conv6 = nn.Sequential(nn.Conv2d(conv5_num, conv6_num, 3),
                                   nn.BatchNorm2d(conv6_num),
                                   nn.ReLU6(),
                                   nn.MaxPool2d(2, 2))

        self.wc1 = nn.Linear(128, class_num)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.reshape([x.shape[0], -1])
        x = self.wc1(x)
        x = self.softmax(x)
        return x

    def get_feature_maps(self, x):
        feature_maps = []
        x = self.conv1(x)
        feature_maps.append(x)
        x = self.conv2(x)
        feature_maps.append(x)
        x = self.conv3(x)
        feature_maps.append(x)
        x = self.conv4(x)
        feature_maps.append(x)
        x = self.conv5(x)
        feature_maps.append(x)
        x = self.conv6(x)
        feature_maps.append(x)
        return feature_maps