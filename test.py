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
    ckpt_path = './saved_model/det_0.pth'
    weight_path = 'weight.txt'
    feature_path = 'feature.txt'
    composed = transforms.Compose([Normalize(),
                                   ToTensor()])
    batch_size = 32
    test_dataset = LineDataset(dataset_file_path, transform=composed)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    model = torch.load(ckpt_path)
    model.eval()
    model_explainer = ModelExplainer()
    for param in model.parameters():
        param = torch.zeros_like(param)
        pass
    state_dict = model.state_dict()
    model_explainer.write_weights(weight_path, state_dict)
    for i, sample in enumerate(test_dataset):
        if i < 20:
            continue
        img = sample['image']
        # noise = (torch.rand_like(img) - 0.5) * 0.01
        # img = img + noise
        img = torch.unsqueeze(img, 0).cuda()
        model_explainer.write_feature_map(feature_path, 'input', img, cover=True)
        conv1_feature = model.conv1(img)
        model_explainer.write_feature_map(feature_path, 'filters1', conv1_feature.cpu().detach().numpy())
        conv2_feature = model.conv2(conv1_feature)
        model_explainer.write_feature_map(feature_path, 'conv2', conv2_feature.cpu().detach().numpy())
        conv2_feature = conv2_feature.view(conv2_feature.size(0), -1)
        model_explainer.write_feature_map(feature_path, 'wc1_input', conv2_feature.cpu().detach().numpy())
        wc1_feature = model.wc1(conv2_feature)
        model_explainer.write_feature_map(feature_path, 'wc1', wc1_feature.cpu().detach().numpy())
        softmax_feature = model.softmax(wc1_feature)
        model_explainer.write_feature_map(feature_path, 'softmax', softmax_feature.cpu().detach().numpy())
        label = sample['label']
        pre_label = torch.argmax(model(img), 1)
        print('No.{} label: {}, prediction: {}'.format(i, label.item(), pre_label.item()))
        pass