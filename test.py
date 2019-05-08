import Net
from LineDataset import *
from torch.utils.data import DataLoader
from train import test
from torchvision import transforms
from torch.autograd import Variable
import os
from Visualization import ModelExplainer

if __name__ == '__main__':
    dataset_file_path = 'samples_1.txt'
    ckpt_path = './check_point/base_model_100.pth'
    composed = transforms.Compose([Normalize(),
                                   ToTensor()])
    batch_size = 32
    test_dataset = LineDataset(dataset_file_path, transform=composed)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    model = torch.load(ckpt_path)
    model.eval()
    model_explainer = ModelExplainer()
    state_dict = model.state_dict()
    model_explainer.write_weights('conv.txt', state_dict)
    for weight_name, weight_value in state_dict.items():
        model_explainer.write_conv_weights(weight_name, weight_value.cpu().numpy(), 'conv.txt', cover=True)
        pass

    for sample in test_dataset:
        img = sample['image']
        img = torch.unsqueeze(img, 0).cuda()
        label = sample['label']
        pre_label = torch.argmax(model(img), 1)
        print('label: {}, prediction: {}'.format(label.item(), pre_label.item()))
        pass