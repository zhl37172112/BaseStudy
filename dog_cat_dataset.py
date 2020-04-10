from torch.utils.data import Dataset
import cv2
import numpy as np
import torch
import os
import random
from PIL import Image
from torchvision import transforms as tfs

class DogCatDataset(Dataset):
    def __init__(self, data_dir, transform=None, size=(214, 214)):
        self.imgpath_label = []
        self.size = size
        self.load_datafile(data_dir)
        self.transform = transform

    def load_datafile(self, data_dir):
        cat_dir = os.path.join(data_dir, 'cats')
        dog_dir = os.path.join(data_dir, 'dogs')
        if not os.path.isdir(cat_dir) or not os.path.isdir(dog_dir):
            raise IOError
        cat_path_labels = [(os.path.join(cat_dir, path), 0) for path in os.listdir(cat_dir) if path.endswith('.jpg')]
        dog_path_labels = [(os.path.join(dog_dir, path), 1) for path in os.listdir(dog_dir) if path.endswith('.jpg')]
        self.imgpath_label.extend(cat_path_labels)
        self.imgpath_label.extend(dog_path_labels)
        random.shuffle(self.imgpath_label)

    def __len__(self):
        return len(self.imgpath_label)

    def __getitem__(self, item):
        img_path = self.imgpath_label[item][0]
        img_label = np.array([(self.imgpath_label[item][1])], np.int32)
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
        sample = {'image': img, 'label': img_label}
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        return sample


class Normalize(object):
    def __init__(self, mean=127.5, std=255):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        sample = np.array(sample).astype(np.float32)
        sample = sample - self.mean
        sample = sample / self.std
        return sample


class ToTensor(object):
    def __call__(self, sample):
        sample = np.transpose(sample, [2, 0, 1])
        return torch.from_numpy(sample)

class Resize(object):
    def __init__(self,size):
        self.size = size
    def __call__(self, sample):
        sample = cv2.resize(sample,self.size)
        return sample

class DataAug(object):
    def __call__(self, sample):
        im_aug = tfs.Compose([
            tfs.RandomHorizontalFlip(),
            # tfs.RandomResizedCrop((128,128))
        ])

        sample = Image.fromarray(cv2.cvtColor(sample, cv2.COLOR_BGR2RGB))
        sample = im_aug(sample)
        sample = cv2.cvtColor(np.asarray(sample), cv2.COLOR_RGB2BGR)
        return sample


