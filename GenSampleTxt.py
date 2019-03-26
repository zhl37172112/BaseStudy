import os

def gen_sample_txt(root_path, datafile_path):
    class_names = os.listdir(root_path)
    datafile = open(datafile_path, 'w')
    for i, class_name in enumerate(class_names):
        class_path = os.path.join(root_path, class_name)
        for root, dirs, files in os.walk(class_path):
            for file in files:
                data_line = '{}\t{}\n'.format(os.path.join(root, file), str(i))
                datafile.write(data_line)

if __name__ == '__main__':
    sample_path = 'E:\\Networks\\Samples'
    datafile_path = 'E:\\Networks\\Samples\\samples.txt'
    gen_sample_txt(sample_path, datafile_path)
