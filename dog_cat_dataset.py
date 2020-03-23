from torch.utils.data import Dataset
import cv2
import numpy as np
import torch
import os

class DogCatDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.imgpath_label = []
        self.load_datafile(data_dir)
        self.transform = transform


    def load_datafile(self, data_dir):
        cat_dir = os.path.join(data_dir, 'cats')
        dog_dir = os.path.join(data_dir, 'dogs')
        if not os.path.isdir(cat_dir) or not os.path.isdir(dog_dir):
            raise IOError
        cat_path_labels = [(path, 0) for path in os.listdir(cat_dir) if path.endswith('.jpg')]
        dog_path_labels = [(path, 0) for path in os.listdir(dog_dir) if path.endswith('.jpg')]
        self.imgpath_label.extend(cat_path_labels)
        self.imgpath_label.extend(dog_path_labels)

    def __len__(self):
        return len(self.imgpath_label)

    def __getitem__(self, item):
        img_path = self.imgpath_label[item][0]
        img_label = np.array([(self.imgpath_label[item][1])], np.int32)
        img = cv2.imread(img_path)
        img = img.astype(np.float32)
        sample = {'image':img, 'label':img_label}
        if self.transform:
            sample = self.transform(sample)
        return sample