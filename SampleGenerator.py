import cv2
import random
import numpy as np
import os
import sys

class SampleGenerator:
    def __init__(self):
        self.type_func = {}
        self.type_func['horizontal_line'] = self.generate_horizontal_lines

    def generate(self, sample_path, img_num, img_size, *img_types):
        self.sample_path = sample_path
        self.img_num = img_num
        self.img_size = img_size
        self.gen_types = img_types
        for type in self.gen_types:
            if type in self.type_func:
                self.type_func[type]()
            else:
                print('>>> type ' + type + ' not in this class.')
                sys.pause()


    def generate_horizontal_lines(self):
        background = 0
        foreground = 255
        min_length_ratio = 0.7
        thickness_ratio = 0.3
        save_dir = os.path.join(self.sample_path, 'horizontal_line')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        for i in range(self.img_num):
            print(str(i + 1) + ' / ' + str(self.img_num))
            img = np.zeros(shape=(self.img_size[1], self.img_size[0]), dtype=np.uint8)
            line_x = random.randrange(0, self.img_size[0])
            line_y1 = random.randrange(0, self.img_size[1] * (1 - min_length_ratio))
            line_y2 = random.randrange(line_y1 + self.img_size[1] * min_length_ratio,
                                       self.img_size[1])
            thickness = random.randrange(1, self.img_size[0] * thickness_ratio)
            cv2.line(img, (line_x, line_y1), (line_x, line_y2), foreground, thickness)
            img_path = os.path.join(save_dir, str(i) + '.jpg')
            cv2.imwrite(img_path, img)

if __name__ == '__main__':
    pass