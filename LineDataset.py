from torch.utils.data import Dataset
import cv2
import numpy as np
import torch

class LineDataset(Dataset):
    def __init__(self, datafile_path, transform=None):
        self.imgpath_label = []
        self.load_datafile(datafile_path)
        self.transform = transform

    def load_datafile(self, datafile_path):
        with open(datafile_path) as file:
            for line in file.readlines():
                items = line.strip().split('\t')
                if len(items) != 2:
                    print('datafile is not right')
                else:
                    self.imgpath_label.append(items)

    def __len__(self):
        return len(self.imgpath_label)

    def __getitem__(self, item):
        img_path = self.imgpath_label[item][0]
        img_label = np.array([(self.imgpath_label[item][1])], np.int32)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32)
        sample = {'image':img, 'label':img_label}
        if self.transform:
            sample = self.transform(sample)
        return sample

class Normalize(object):
    def __init__(self, mean=128, std=255):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = image - self.mean
        image = image / self.std
        return {'image':image, 'label':label}

class ToTensor(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = np.expand_dims(image, 0)
        return {'image':torch.from_numpy(image),
                'label':torch.from_numpy(label)}