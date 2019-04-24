import Net
from LineDataset import *
from torchvision import transforms
from torch.autograd import Variable
import os

if __name__ == '__main__':
    dataset_file_path = 'samples_1.txt'
    ckpt_path = './check_point/base_model_100.pth'
    composed = transforms.Compose([Normalize(),
                                   ToTensor()])
    line_dataset = LineDataset(dataset_file_path, transform=composed)
    model = Net.Net1().cuda()
    model.eval()
    for sample in line_dataset:
        img = sample['image']
        img = torch.unsqueeze(img, 0).cuda()
        label = sample['label']
        pre_label = model(img)
        pass