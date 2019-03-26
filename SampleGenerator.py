import cv2
import random
import numpy as np
import os
import sys

class SampleGenerator:
    def __init__(self, sample_path, img_size, img_num):
        self.sample_path = sample_path
        self.img_size = img_size
        self.img_num = img_num

    def generate_horizontal_lines(self):
        foreground = 255
        min_length_ratio = 0.7
        thickness_ratio = 0.3
        save_dir = os.path.join(self.sample_path, 'horizontal_lines')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        for i in range(self.img_num):
            print(str(i + 1) + ' / ' + str(self.img_num))
            img = np.zeros(shape=(self.img_size[1], self.img_size[0]), dtype=np.uint8)
            line_x = random.randrange(0, self.img_size[0])
            line_y1 = random.randrange(0, int(self.img_size[1] * (1 - min_length_ratio)))
            line_y2 = random.randrange(int(line_y1 + self.img_size[1] * min_length_ratio),
                                       self.img_size[1])
            thickness = int(random.randrange(1, int(self.img_size[0] * thickness_ratio)))
            cv2.line(img, (line_x, line_y1), (line_x, line_y2), foreground, thickness)
            img_path = os.path.join(save_dir, str(i) + '.jpg')
            cv2.imwrite(img_path, img)

    def generate_vertical_lines(self):
        foreground = 255
        min_length_ratio = 0.7
        thickness_ratio = 0.3
        save_dir = os.path.join(self.sample_path, 'vertical_lines')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        for i in range(self.img_num):
            print(str(i + 1) + ' / ' + str(self.img_num))
            img = np.zeros(shape=(self.img_size[1], self.img_size[0]), dtype=np.uint8)
            line_y = random.randrange(0, self.img_size[0])
            line_x1 = random.randrange(0, int(self.img_size[1] * (1 - min_length_ratio)))
            line_x2 = random.randrange(int(line_x1 + self.img_size[1] * min_length_ratio),
                                       self.img_size[1])
            thickness = int(random.randrange(1, int(self.img_size[0] * thickness_ratio)))
            cv2.line(img, (line_x1, line_y), (line_x2, line_y), foreground, thickness)
            img_path = os.path.join(save_dir, str(i) + '.jpg')
            cv2.imwrite(img_path, img)

if __name__ == '__main__':
    img_path = 'E:\\Networks\\Samples'
    img_size = (10, 10)
    img_num = 100
    sample_generator = SampleGenerator(img_path, img_size, img_num)
    # sample_generator.generate_horizontal_lines()
    sample_generator.generate_vertical_lines()