from torch.utils.data import Dataset
import cv2
import numpy as np

class LineDataset(Dataset):
    def __init__(self, datafile_path):
        self.imgpath_label = []
        self.load_datafile(datafile_path)

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
        img_label = int(self.imgpath_label[item][1])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32)
        img = np.expand_dims(img, 0)
        sample = {'image':img, 'label':img_label}
        return sample