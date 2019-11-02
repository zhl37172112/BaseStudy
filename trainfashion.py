from torchvision.datasets import FashionMNIST
import numpy as np
import cv2
from torch.utils.data import DataLoader
from torchvision import transforms
from fashionmnist import *
from fashionnet import Net1
import torch.nn as nn
import torch.optim as optim
import os
from torch.autograd import Variable
import torch


def auto_gpu(*args):
    ret_args = []
    if torch.cuda.is_available():
        for arg in args:
            arg = arg.cuda()
            ret_args.append(arg)
    else:
        ret_args = args
    if len(ret_args) == 1:
        ret_args = ret_args[0]
    return ret_args


def test(model, data_loader):
    right_num = torch.Tensor([0])
    all_num = torch.Tensor([0])
    right_num, all_num = auto_gpu(right_num, all_num)
    for batched_sample in data_loader:
        imgs = auto_gpu(batched_sample[0])
        labels = auto_gpu(batched_sample[1].long())
        pred_lables = model(imgs)
        pred_lables = torch.argmax(pred_lables, 1, keepdim=True)
        right_num += (pred_lables.squeeze(1) == labels).sum()
        all_num += labels.shape[0]
    accuracy = (right_num / all_num).cpu()
    return accuracy


def train_step(model, optimizer, criterion, batched_sample, batched_target, class_num):
    imgs = auto_gpu(batched_sample)
    labels = batched_target.long().unsqueeze(1)
    temp_batch_size = labels.size()[0]
    ont_hot_labels = Variable(auto_gpu(torch.zeros(temp_batch_size, class_num).scatter_(1, labels, 1).float()))
    imgs = Variable(imgs)
    pre_labels = model(imgs)
    classify_loss = criterion(pre_labels, ont_hot_labels)
    optimizer.zero_grad()
    loss = classify_loss
    loss.backward()
    optimizer.step()
    return loss, classify_loss


def main():
    data_root = 'd:/zhaolin/temp/data'
    ckpt_dir = './fashion_ckpt'
    labelmap = {0: 'T-shirt', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal',
                6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}
    batch_size = 32
    class_num = 10
    lr = 0.01
    composed = transforms.Compose([Normalize(),
                                   ToTensor()])
    train_dataset = FashionMNIST(root=data_root, transform=composed, download=True)
    test_dataset = FashionMNIST(root=data_root, transform=composed, train=False, download=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    model = auto_gpu(Net1())
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    epoch_size = 100
    for epoch in range(epoch_size):
        for ibatch, batched_sample_label in enumerate(train_dataloader):
            batched_sample = batched_sample_label[0]
            batched_target = batched_sample_label[1]
            loss, classify_loss = train_step(model, optimizer, criterion, batched_sample, batched_target, class_num)
            if ibatch % 20 == 0:
                print('Epoch [{} / {}] Batch [{}], loss: {:.6f}, classify_loss: {:.6f}'
                      .format(epoch + 1, epoch_size, ibatch, loss.item(), classify_loss.item()))
            if ibatch % 100 == 0:
                accuracy = test(model, test_dataloader)
                print('Epoch [{} / {}] Batch [{}], Accuracy: {}'.
                      format(epoch + 1, epoch_size, ibatch, accuracy.item()))
        if (epoch + 1) % 20 == 0:
            if not os.path.isdir(ckpt_dir):
                os.makedirs(ckpt_dir)
            model_path = os.path.join(ckpt_dir, 'base_model_{}.pth'.format(epoch + 1))
            torch.save(model, model_path)
            accuracy = test(model, test_dataloader)
            print('Epoch [{} / {}], accuracy: {:.3f}'
                  .format(epoch + 1, epoch_size, accuracy.item()))


if __name__ == '__main__':
    main()
