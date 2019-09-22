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

def test(model, data_loader):
    right_num = torch.Tensor([0]).cuda()
    all_num = torch.Tensor([0]).cuda()
    for batched_sample in data_loader:
        imgs = batched_sample[0].cuda()
        labels = batched_sample[1].long().cuda()
        pred_lables = model(imgs)
        pred_lables = torch.argmax(pred_lables, 1, keepdim=True)
        right_num += (pred_lables.squeeze(1) == labels).sum()
        all_num += labels.shape[0]
    accuracy = (right_num / all_num).cpu()
    return accuracy

def train_step(model, optimizer, criterion, batched_sample, batched_target, class_num):
    imgs = batched_sample.cuda()
    labels = batched_target.long().unsqueeze(1)
    temp_batch_size = labels.size()[0]
    ont_hot_labels = Variable(torch.zeros(temp_batch_size, class_num).scatter_(1, labels, 1).float().cuda())
    imgs = Variable(imgs)
    pre_labels = model(imgs)
    classify_loss = criterion(pre_labels, ont_hot_labels)
    optimizer.zero_grad()
    loss = classify_loss
    loss.backward()
    optimizer.step()
    return loss, classify_loss

def main():
    data_root = 'E:\\Data\\fashionmnist'
    ckpt_dir = './fashion_ckpt'
    labelmap = {0:'T-shirt', 1:'Trouser', 2:'Pullover', 3:'Dress', 4:'Coat', 5:'Sandal',
                6:'Shirt', 7:'Sneaker', 8:'Bag', 9:'Ankle boot'}
    batch_size = 32
    class_num = 10
    lr = 0.01
    composed = transforms.Compose([Normalize(),
                                   ToTensor()])
    train_dataset = FashionMNIST(root=data_root, transform=composed)
    test_dataset = FashionMNIST(root=data_root, transform=composed, train=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    model = Net1().cuda()
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

