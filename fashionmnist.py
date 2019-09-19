import numpy as np
import torch


class Normalize(object):
    def __init__(self, mean=127.5, std=255):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, label = sample[0], sample[1]
        image = np.array(image).astype(np.float32)
        image = image - self.mean
        image = image / self.std
        return {'image':image, 'label':label}

class ToTensor(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = np.expand_dims(image, 0)
        return {'image':torch.from_numpy(image),
                'label':torch.from_numpy(label)}