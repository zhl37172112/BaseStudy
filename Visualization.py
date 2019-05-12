import math

NUM_ADD_LENGTH = 7

def write_segment_line(file, line_width):
    segment_line = '-' * line_width + '\n'
    file.write(segment_line)

def write_one_num(file, value, decimals_num):
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

def write_one_line(file, weight_line, decimals_num):
    if len(weight_line.shape) == 1:
        write_one_num(file, weight_line[0], decimals_num)
        for i in range(1, weight_line.shape[0]):
            write_one_num(file, weight_line[i], decimals_num)
    elif len(weight_line.shape) == 2:
        for i in range(weight_line.shape[0]):
            write_one_num(file, weight_line[i, 0], decimals_num)
            for j in range(1, weight_line.shape[1]):
                write_one_num(file, weight_line[i, j], decimals_num)
            file.write(' ')
    file.write('\n')

def write_wc_weights(file_path, weight_name, weight_value, line_width, decimals_num, cover=False):
    file = open(file_path, 'a+') if not cover else open(file_path, 'w')
    write_segment_line(file, line_width)
    write_title(file, weight_name, weight_value)
    fl_width = weight_value.shape[1] * (decimals_num + NUM_ADD_LENGTH) - 1
    split_num = math.ceil(fl_width / line_width)
    num_per_line = math.ceil(weight_value.shape[1] / split_num)
    for i in range(weight_value.shape[0]):
        for j in range(split_num):
            conv_start_index = num_per_line * j
            conv_end_index = min(num_per_line * (j+1), weight_value.shape[1])
            weight_line = weight_value[i, conv_start_index:conv_end_index]
            write_one_line(file, weight_line, decimals_num)
        file.write('\n')

def write_bias(file_path, weight_name, weight_value, line_width, decimals_num, cover=False):
    file = open(file_path, 'a+') if not cover else open(file_path, 'w')
    write_segment_line(file, line_width)
    write_title(file, weight_name, weight_value)
    bias_width = weight_value.shape[0] * (decimals_num + NUM_ADD_LENGTH) + 1
    split_num = math.ceil(bias_width / line_width)
    bias_per_line = math.ceil(weight_value.shape[0] / split_num)
    for j in range(split_num):
        start_index = bias_per_line * j
        end_index = min(bias_per_line * (j+1), weight_value.shape[0])
        if end_index == start_index:
            continue
        weight_line = weight_value[start_index:end_index]
        write_one_line(file, weight_line, decimals_num)
    file.write('\n')

def write_wc_bias(file_path, weight_name, weight_value, line_width, decimals_num, cover=False):
    write_bias(file_path, weight_name, weight_value, line_width, decimals_num, cover)

def write_conv_weights(file_path, weight_name, weight_value, line_width, decimals_num, cover=False):
    file = open(file_path, 'a+') if not cover else open(file_path, 'w')
    write_segment_line(file, line_width)
    write_title(file, weight_name, weight_value)
    conv_width = weight_value.shape[1] * (weight_value.shape[3] * (decimals_num + NUM_ADD_LENGTH) + 1) - 1
    split_num = math.ceil(conv_width / line_width)
    conv_per_line = math.ceil(weight_value.shape[1] / split_num)
    for i in range(weight_value.shape[0]):
        for j in range(split_num):
            for k in range(weight_value.shape[2]):
                conv_start_index = conv_per_line * j
                conv_end_index = min(conv_per_line * (j+1), weight_value.shape[1])
                if conv_end_index == conv_start_index:
                    continue
                weight_line = weight_value[i, conv_start_index:conv_end_index, k, :]
                write_one_line(file, weight_line, decimals_num)
            file.write('\n')
        file.write('\n')

def write_conv_bias(file_path, weight_name, weight_value, line_width, decimals_num, cover=False):
    write_bias(file_path, weight_name, weight_value, line_width, decimals_num, cover)

def write_weights(file_path, weight_name, weight_value, line_width, decimals_num, cover=False):
    if weight_name.startswith('conv'):
        if weight_name.endswith('weight'):
            write_conv_weights(file_path, weight_name, weight_value, line_width, decimals_num, cover)
        elif weight_name.endswith('bias'):
            write_conv_bias(file_path, weight_name, weight_value, line_width, decimals_num, cover)
    elif weight_name.startswith('wc'):
        if weight_name.endswith('weight'):
            write_wc_weights(file_path, weight_name, weight_value, line_width, decimals_num, cover)
        elif weight_name.endswith('bias'):
            write_wc_bias(file_path, weight_name, weight_value, line_width, decimals_num, cover)

def write_feature_map(file_path, feature_name, feature_value, line_width, decimals_num, cover=False):
    file = open(file_path, 'a+') if not cover else open(file_path, 'w')
    write_segment_line(file, line_width)
    write_title(file, feature_name, feature_value)
    if len(feature_value.shape) == 4:
        for i in range(feature_value.shape[1]):
            for j in range(feature_value.shape[2]):
                line_value = feature_value[0, i, j, :]
                write_one_line(file, line_value, decimals_num)
            file.write('\n')
    elif len(feature_value.shape) == 2:
        line_value = feature_value[0, :]
        write_one_line(file, line_value, decimals_num)
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

    def write_conv_bias(self, weight_name, weight_value, file_path=None, cover=False):
        file_path = file_path if file_path is not None else self.file_path
        write_conv_bias(file_path,
                           weight_name,
                           weight_value,
                           self.line_width,
                           self.decimals_num,
                           cover=cover)

    def write_fl_weights(self, weight_name, weight_value, file_path=None, cover=False):
        file_path = file_path if file_path is not None else self.file_path
        write_wc_weights(file_path,
                         weight_name,
                         weight_value,
                         self.line_width,
                         self.decimals_num,
                         cover=cover)

    def write_weights(self, weight_path, dict_state, cover=True):
        if cover:
            open(weight_path, 'w').close()
        for weight_name, weight_value in dict_state.items():
            write_weights(weight_path,
                          weight_name,
                          weight_value,
                          self.line_width,
                          self.decimals_num)

    def write_feature_map(self, file_path, feature_name, feature_value, cover=False):
        write_feature_map(file_path, feature_name, feature_value, self.line_width, self.decimals_num, cover)

