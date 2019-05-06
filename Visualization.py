import math

def write_segment_line(file, line_width):
    segment_line = '-' * line_width + '\n'
    file.write(segment_line)

def write_1num(file, value, decimals_num):
    num_format = '{:.' + str(decimals_num) + 'e}'
    if value >= 0:
        file.write(' ')
    file.write((' ' + num_format).format(value))

def write_title(file, weight_name, weight_value):
    title = weight_name + ', shape=[{}'.format(weight_value.shape[0])
    for i in range(1, len(weight_value.shape)):
        title += ', ' + str(weight_value.shape[i])
    title += ']\n'
    file.write(title)

def write_1line(file, weight_line, decimals_num):
    num_format = '{:.' + str(decimals_num) + 'e}'
    if len(weight_line.shape) == 1:
        write_1num(file, weight_line[0], decimals_num)
        for i in range(1, weight_line.shape[0]):
            write_1num(file, weight_line[i], decimals_num)
    elif len(weight_line.shape) == 2:
        for i in range(weight_line.shape[0]):
            write_1num(file, weight_line[i, 0], decimals_num)
            for j in range(1, weight_line.shape[1]):
                write_1num(file, weight_line[i, j], decimals_num)
            file.write(' ')
    file.write('\n')


def write_conv_weights(file_path, weight_name, weight_value, line_width, decimals_num, cover=False):
    file = open(file_path, 'a+') if not cover else open(file_path, 'w')
    write_segment_line(file, line_width)
    write_title(file, weight_name, weight_value)
    conv_width = weight_value.shape[1] * (weight_value.shape[3] * (decimals_num + 7) + 1) - 1
    split_num = conv_width // line_width + 1
    conv_per_line = math.ceil(weight_value.shape[1] / split_num)
    for i in range(weight_value.shape[0]):
        for j in range(split_num):
            for k in range(weight_value.shape[2]):
                conv_start_index = conv_per_line * j
                conv_end_index = min(conv_per_line * (j+1), weight_value.shape[1])
                weight_line = weight_value[i, conv_start_index:conv_end_index, k, :]
                write_1line(file, weight_line, decimals_num)
            file.write('\n')
        file.write('\n')




class ModelExplainer:
    def __init__(self, line_width=200, decimals_num=4):
        self.line_width = line_width
        self.decimals_num = decimals_num
        self.file_path = None

    def write_conv_weights(self, weight_name, weight_value, file_path=None, cover=False):
        file_path = file_path if file_path is not None else self.file_path
        write_conv_weights(file_path,
                           weight_name,
                           weight_value,
                           self.line_width,
                           self.decimals_num,
                           cover=cover)
