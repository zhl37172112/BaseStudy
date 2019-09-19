from torchvision.datasets import FashionMNIST
import numpy as np
import cv2
from torch.utils.data import DataLoader
from torchvision import transforms
from fashionmnist import *



def main():
    data_root = 'E:\\Data\\fashionmnist'
    dataset = FashionMNIST(root=data_root, download=True)
    labelmap = {0:'T-shirt', 1:'Trouser', 2:'Pullover', 3:'Dress', 4:'Coat', 5:'Sandal',
                6:'Shirt', 7:'Sneaker', 8:'Bag', 9:'Ankle boot'}
    for i in range(len(dataset)):
        image, label = dataset[i]
        image = np.array(image)
        a = 0
    batch_size = 32
    class_num = 2
    lr = 0.1
    train_dataset = FashionMNIST(root=data_root)
    test_dataset = FashionMNIST(root=data_root, train=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    composed = transforms.Compose([Normalize(),
                                   ToTensor()])


if __name__ == '__main__':
    main()

