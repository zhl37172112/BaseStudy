from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from auto_gpu import auto_gpu
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import datetime
import torch.optim as optim
from torch.autograd import Variable
import torch
import time
import os
import numpy as np


class ClassifyTrainer:
    def __init__(self, class_num, save_dir):
        self.learning_rate = 0.01
        self.epoch_size = 80
        self.batch_size = 32
        self.log_step = 20
        self.save_every_epoch = 20
        self.model = None
        self.class_num = class_num
        self.save_dir = save_dir
        self.train_dataset = None
        self.test_dataset = None
        self.transform = None
        self.train_dataloader = None
        self.test_dataloader = None
        self.criterion = nn.CrossEntropyLoss()
        time_dir = datetime.datetime.now().strftime('%Y-%m-%d')
        log_dir = './log/' + time_dir
        self.writer = SummaryWriter(log_dir)

    @abstractmethod
    def set_model(self):
        pass

    @abstractmethod
    def set_dataset(self):
        pass

    def init_params(self):
        self.set_model()
        self.set_dataset()
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=True)
        if self.test_dataset is not None:
            self.test_dataloader = DataLoader(self.test_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=False)
        self.optimizer = optim.SGD(self.model.parameters(),momentum=0.9, lr=self.learning_rate)
        T_max = int(len(self.train_dataset) / self.batch_size) * self.epoch_size
        self.schedule = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max)

    def show_img(self, batched_sample, batched_target):
        imgs = batched_sample.cpu().numpy()
        imgs = (imgs + 1) * 127.5
        imgs = imgs.astype(np.uint8)
        labels = batched_target.cpu().numpy()
        for i in range(imgs.shape[0]):
            pass




    def train_step(self, batched_sample, batched_target):
        imgs = auto_gpu(batched_sample)
        labels = auto_gpu(batched_target.long())
        labels = labels.squeeze()
        imgs = Variable(imgs)
        pre_labels = self.model(imgs)
        # classify_loss = criterion(pr
        # e_labels, ont_hot_labels)
        classify_loss = self.criterion(pre_labels, labels)
        self.optimizer.zero_grad()
        loss = classify_loss
        loss.backward()
        grads = [x.grad for x in self.optimizer.param_groups[0]['params']]
        self.optimizer.step()
        return loss, classify_loss

    def test(self):
        if self.test_dataloader is None:
            return None
        right_num = torch.Tensor([0])
        all_num = torch.Tensor([0])
        right_num, all_num = auto_gpu(right_num, all_num)
        for batched_sample in self.test_dataloader:
            imgs = auto_gpu(batched_sample['image'])
            labels = auto_gpu(batched_sample['label'].long())
            pred_lables = self.model(imgs)
            pred_lables = torch.argmax(pred_lables, 1, keepdim=True)
            right_num += (pred_lables == labels).sum()
            all_num += labels.shape[0]
        accuracy = (right_num / all_num).cpu()
        return accuracy

    def train(self):
        self.init_params()
        best_acc = 0
        for epoch in range(self.epoch_size):
            for ibatch, batched_sample_label in enumerate(self.train_dataloader):
                batched_sample = batched_sample_label['image']
                batched_target = batched_sample_label['label']
                loss, classify_loss = self.train_step(batched_sample, batched_target)
                self.schedule.step()
                curr_learning_rate = self.schedule.get_lr()[0]
                if ibatch % self.log_step == 0:
                    curr_time = time.strftime('%Y-%m-%d %H:%M:%S')
                    print('{}: Epoch [{} / {}] Batch [{}], loss: {:.6f}, classify_loss: {:.6f}'
                          .format(curr_time, epoch + 1, self.epoch_size, ibatch, loss.item(), classify_loss.item()))
                    self.writer.add_scalar('loss', loss.item(), epoch * len(self.train_dataset) + ibatch)
                    self.writer.add_scalar('learning  rate', curr_learning_rate, epoch * len(self.train_dataset) + ibatch)
                if ibatch % 100 == 0:
                    accuracy = self.test()
                    last_acc = accuracy.item()
                    curr_time = time.strftime('%Y-%m-%d %H:%M:%S')
                    print('{}: Epoch [{} / {}] Batch [{}], Accuracy: {}'.
                          format(curr_time, epoch + 1, self.epoch_size, ibatch, accuracy.item()))
                    self.writer.add_scalar('accuracy', accuracy, epoch * len(self.train_dataset) + ibatch)
            if (epoch + 1) % self.save_every_epoch == 0:
                if not os.path.isdir(self.save_dir):
                    os.makedirs(self.save_dir)
                model_path = os.path.join(self.save_dir, 'base_model_{}.pth'.format(epoch + 1))
                torch.save(self.model.state_dict(), model_path)
                accuracy = self.test()
                if accuracy > best_acc:
                    best_acc = accuracy
                    print('Best model updated.')
                print('Epoch [{} / {}], accuracy: {:.3f}'
                      .format(epoch + 1, self.epoch_size, accuracy.item()))


