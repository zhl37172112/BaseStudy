import Net
from LineDataset import *
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
from torchvision import transforms
from torch.autograd import Variable
import os

def train_step(model, batched_sample, class_num):
    imgs = batched_sample['image'].cuda()
    labels = batched_sample['label'].long()
    temp_batch_size = labels.size()[0]
    ont_hot_labels = Variable(torch.zeros(temp_batch_size, class_num).scatter_(1, labels, 1).float().cuda())
    imgs = Variable(imgs)
    pre_labels = model(imgs)
    loss = criterion(pre_labels, ont_hot_labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def test(model, data_loader):
    right_num = torch.Tensor([0]).cuda()
    all_num = torch.Tensor([0]).cuda()
    for batched_sample in data_loader:
        imgs = batched_sample['image'].cuda()
        labels = batched_sample['label'].long().cuda()
        pred_lables = model(imgs)
        pred_lables = torch.argmax(pred_lables, 1, keepdim=True)
        right_num += (pred_lables == labels).sum()
        all_num += labels.shape[0]
    accuracy = (right_num / all_num).cpu()
    return accuracy

if __name__ == '__main__':
    trainset_path = 'samples_0.txt'
    testset_path = 'samples_1.txt'
    ckpt_dir = './check_point'
    composed = transforms.Compose([Normalize(),
                                   ToTensor()])
    batch_size = 32
    class_num = 2
    train_dataset = LineDataset(trainset_path, transform=composed)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataset = LineDataset(testset_path, transform=composed)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    model = Net.Net1().cuda()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    epoch_size = 100
    for epoch in range(epoch_size):
        for ibatch, batched_sample in enumerate(train_dataloader):
            loss = train_step(model, batched_sample, class_num)
            # params = model.state_dict()
            if ibatch % 20 == 0:
                print('Epoch [{} / {}] Batch [{}], loss: {:.6f}'
                      .format(epoch + 1, epoch_size, ibatch, loss.item()))
        if (epoch + 1) % 20 == 0:
            if not os.path.isdir(ckpt_dir):
                os.makedirs(ckpt_dir)
            model_path = os.path.join(ckpt_dir, 'base_model_{}.pth'.format(epoch + 1))
            torch.save(model, model_path)
            accuracy = test(model, test_dataloader)
            print('Epoch [{} / {}], accuracy: {:.3f}'
                      .format(epoch + 1, epoch_size, accuracy.item()))



