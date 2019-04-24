import Net
from LineDataset import *
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
from torchvision import transforms
from torch.autograd import Variable
import os

if __name__ == '__main__':
    dataset_file_path = 'samples_0.txt'
    ckpt_dir = './check_point'
    composed = transforms.Compose([Normalize(),
                                   ToTensor()])
    batch_size = 32
    class_num = 2
    line_dataset = LineDataset(dataset_file_path, transform=composed)
    line_dataloader = DataLoader(line_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    model = Net.Net1().cuda()

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    epoch_size = 100
    for epoch in range(epoch_size):
        for ibatch, sample_batched in enumerate(line_dataloader):
            imgs = sample_batched['image'].cuda()
            labels = sample_batched['label'].long()
            temp_batch_size = labels.size()[0]
            ont_hot_labels = Variable(torch.zeros(temp_batch_size, class_num).scatter_(1, labels, 1).float().cuda())
            imgs = Variable(imgs)
            pre_labels = model(imgs)
            loss = criterion(pre_labels, ont_hot_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # params = model.state_dict()
            if ibatch % 20 == 0:
                print('Epoch [{} / {}] Batch [{}], loss: {:.6f}'
                      .format(epoch + 1, epoch_size, ibatch, loss.item()))
        if (epoch + 1) % 20 == 0:
            if not os.path.isdir(ckpt_dir):
                os.makedirs(ckpt_dir)
            model_path = os.path.join(ckpt_dir, 'base_model_{}.pth'.format(epoch + 1))
            torch.save(model, model_path)

