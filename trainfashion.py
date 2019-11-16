from torchvision.datasets import FashionMNIST
import numpy as np
import cv2
from torch.utils.data import DataLoader
from torchvision import transforms
from fashionmnist import *
from fashionnet import Net1, FixedNet
import torch.nn as nn
import torch.optim as optim
import os
from torch.autograd import Variable
import torch
from torch.utils.tensorboard import SummaryWriter
import time
from auto_gpu import auto_gpu


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
    ont_hot_labels = Variable(auto_gpu(torch.zeros(temp_batch_size, class_num).scatter_(1, labels, 1).long()))
    imgs = Variable(imgs)
    pre_labels = model(imgs)
    # classify_loss = criterion(pre_labels, ont_hot_labels)
    labels = labels.squeeze()
    classify_loss = criterion(pre_labels, labels.cuda())
    optimizer.zero_grad()
    loss = classify_loss
    loss.backward()
    optimizer.step()
    return loss, classify_loss


def main():
    data_root = 'E:\\Data\\fashionmnist'
    ckpt_dir = './fashion_ckpt_fixed'
    labelmap = {0: 'T-shirt', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal',
                6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}
    class_num = 10
    #train params
    batch_size = 128
    lr = 1
    epoch_size = 60
    save_every_epoch = 10
    # summary params
    log_dir = 'log/fashion_log_cross_entropy_1'
    running_loss = 0
    log_step = 20

    composed = transforms.Compose([Normalize(),
                                   ToTensor()])
    train_dataset = FashionMNIST(root=data_root, transform=composed)
    test_dataset = FashionMNIST(root=data_root, transform=composed, train=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    model = auto_gpu(FixedNet())
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    T_max = int(len(train_dataset) / batch_size) * epoch_size
    schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max)
    writer = SummaryWriter(log_dir)


    for epoch in range(epoch_size):
        for ibatch, batched_sample_label in enumerate(train_dataloader):
            batched_sample = batched_sample_label[0]
            batched_target = batched_sample_label[1]
            loss, classify_loss = train_step(model, optimizer, criterion, batched_sample, batched_target, class_num)
            schedule.step()
            curr_learning_rate = schedule.get_lr()[0]
            running_loss += loss.item()
            if ibatch % log_step == 0:
                curr_time = time.strftime('%Y-%m-%d %H:%M:%S')
                print('{}: Epoch [{} / {}] Batch [{}], loss: {:.6f}, classify_loss: {:.6f}'
                      .format(curr_time, epoch + 1, epoch_size, ibatch, loss.item(), classify_loss.item()))
                writer.add_scalar('loss', running_loss / log_step, epoch * len(train_dataset) + ibatch)
                writer.add_scalar('learning  rate', curr_learning_rate, epoch * len(train_dataset) + ibatch)
                running_loss = 0
            if ibatch % 100 == 0:
                accuracy = test(model, test_dataloader)
                curr_time = time.strftime('%Y-%m-%d %H:%M:%S')
                print('{}: Epoch [{} / {}] Batch [{}], Accuracy: {}'.
                      format(curr_time, epoch + 1, epoch_size, ibatch, accuracy.item()))
                writer.add_scalar('accuracy', accuracy, epoch * len(train_dataset) + ibatch)
        if (epoch + 1) % save_every_epoch == 0:
            if not os.path.isdir(ckpt_dir):
                os.makedirs(ckpt_dir)
            model_path = os.path.join(ckpt_dir, 'base_model_{}.pth'.format(epoch + 1))
            torch.save(model, model_path)
            accuracy = test(model, test_dataloader)
            print('Epoch [{} / {}], accuracy: {:.3f}'
                  .format(epoch + 1, epoch_size, accuracy.item()))


if __name__ == '__main__':
    main()

