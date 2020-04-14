import os
import cv2
import torch
import numpy as np
from auto_gpu import auto_gpu
from net import DogCatNet
from dog_cat_dataset import Resize,ToTensor,Normalize
from torchvision import transforms as tfs

preprocessing_ops = tfs.Compose([Resize((256, 256)), Normalize(), ToTensor()])

def load_model(model_path):
    model = auto_gpu(DogCatNet())
    model.load_state_dict(torch.load(model_path))
    return model

def preprocessing(image):
    return preprocessing_ops(image)

def process_feature(feature,resize=(256,256)):
    feature=torch.squeeze(feature)
    feature_np=feature.cuda().data.cpu().numpy()
    for i in range(feature_np.shape[0]):
        feature_i=feature_np[i]
        pmin = np.min(feature_i)
        pmax = np.max(feature_i)
        img = ((feature_i - pmin) / (pmax - pmin + 0.000001)) * 255
        img=img.astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        # img= cv2.resize(feature_np,resize)
        cv2.namedWindow('feature',flags=0)
        cv2.imshow('feature',img)
        cv2.waitKey(0)


def test(image, model):
    image = preprocessing(image)
    image = torch.unsqueeze(image,0).cuda()
    pred_lable = model(image)
    feature_maps = model.get_feature_maps(image)
    process_feature(feature_maps[2])
    pred_lable = torch.argmax(pred_lable, 1, keepdim=True)
    if pred_lable == 0:
        pred_class = 'cat'
    elif pred_lable == 1:
        pred_class = 'dog'
    return pred_class


def test_main(input_dir, model_path):
    model = load_model(model_path)
    for class_name in os.listdir(input_dir):
        for image_name in os.listdir(os.path.join(input_dir, class_name)):
            image_path = os.path.join(input_dir, class_name, image_name)
            image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
            pred_class=test(image, model)
            print(image_path,pred_class)


if __name__ == '__main__':
    input_dir = 'E:\\temp\\concat'
    model_path = 'F:\\models_test\\cat_dog_test\\BaseStudy\\dog_cat_ckpt\\base_model_30.pth'
    test_main(input_dir,model_path)
