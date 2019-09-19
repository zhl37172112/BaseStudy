import os
import random

def gen_sample_txt(root_path, sample_ratio):
    summary = sum(sample_ratio)
    sample_value = [single_ratio / summary for single_ratio in sample_ratio]
    sample_value = [sum(sample_value[0:i + 1]) for i in range(len(sample_value))]
    datafiles = [open('samples_{}.txt'.format(i), 'w') for i in range(len(sample_ratio))]
    class_names = os.listdir(root_path)
    for i, class_name in enumerate(class_names):
        class_path = os.path.join(root_path, class_name)
        for root, dirs, files in os.walk(class_path):
            for file in files:
                rand_value = random.random()
                data_line = '{}\t{}\n'.format(os.path.join(root, file), str(i))
                for j in range(len(sample_ratio)):
                    if rand_value < sample_value[j]:
                        datafiles[j].write(data_line)
                        break

if __name__ == '__main__':
    sample_path = 'E:\\Networks\\Samples'
    datafile_path = 'E:\\temp\\Samples\\samples.txt'
    gen_sample_txt(sample_path, (0.5, 0.5))