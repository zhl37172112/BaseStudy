import fashionnet
import torch
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from fashionmnist import *
from fashionnet import *


def main():
    model_path = 'fashion_ckpt_fixed/base_model_60.pth'
    data_root = 'E:\\Data\\fashionmnist'
    composed = transforms.Compose([Normalize(),
                                   ToTensor()])
    test_dataset = FashionMNIST(root=data_root, transform=composed, train=False)
    model = FixedNet()
    model.load_state_dict(model_path)
    params = model.state_dict()
    a = 1

if __name__ == '__main__':
    main()