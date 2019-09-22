import numpy as np
import torch


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
        sample = np.expand_dims(sample, 0)
        return torch.from_numpy(sample)