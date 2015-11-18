# Converts
import os
import numpy as np
import h5py
import csv
import sys


data_path = "/home/users/wxie/activity/data/actitracker.txt"
output_name = 'actitracker'

percent_overlap = 0.50

file_dir = os.path.dirname(os.path.abspath(__file__))

# Caffe blobs have the dimensions (n_samples, n_channels, height, width)
# Count lines
num_lines = 0
num_samples = 0
num_channels = 1
height = 1
width = 64
# num_samples = num_lines / width

with open(data_path, 'rt') as f:
    reader = csv.reader(f)
    prev_label = " "
    line_count = 0
    data_index = 0
    x_list = []
    for row in reader:
        line_count += 1
        label = row[1]
        # Reset when new label is found
        if (prev_label != label):
            prev_label = label
            x_list = []
        x_list.append(1)

        # Store when we have the length
        if (len(x_list) >= width):
            data_index += 1
            x_list = x_list[int(len(x_list) * (1.0-percent_overlap)):] ###
    num_samples = data_index
    num_lines = line_count

print("Number of data points: {}".format(num_lines))
print("Number of data samples: {}".format(num_samples))

total_size = num_samples * num_channels * height * width

data = np.arange(total_size)
data = data.reshape(num_samples, num_channels, height, width)
data = data.astype('float32')

data_label = 1 + np.arange(num_samples)[:, np.newaxis]
data_label = data_label.astype('int32')

label_dict =  {
                "Walking":0,
                "Jogging":1,
                "Sitting":2,
                "Standing":3,
                "Upstairs":4,
                "Downstairs":5 }

with open(data_path, 'rt') as f:
    reader = csv.reader(f)
    prev_label = " "
    data_index = 0
    channel_index = 0
    counter = 0
    x_list = []
    y_list = []
    z_list = []
    for row in reader:
        if not row:
            continue
        # print(counter)
        counter += 1
        user_id = int(row[0])
        label = row[1]
        x = float(row[3])
        y = float(row[4])
        z = float(row[5])
        # print("user_id: {}\t label: {}\t x: {}\t y: {}\t z: {}".format(user_id, label, x, y, z))

        # Reset when new label is found
        if (prev_label != label):
            prev_label = label
            x_list = []

        x_list.append(x)

        # Store when we have the length
        if (len(x_list) >= width):
            data[data_index][channel_index][0] = np.array(x_list)
            data_label[data_index] = np.array([label_dict[label]])
            data_index += 1
            x_list = x_list[int(len(x_list) * (1.0-percent_overlap)):] ###
    print(data_index)

print(data.shape)
print(data_label.shape)

with h5py.File(file_dir + '/../' + output_name + '_data.h5', 'w') as f:
    f['data'] = data
    f['label'] = data_label

with open(file_dir + '/../' + output_name + '_data_list.txt', 'w') as f:
    f.write(file_dir + '/../' + output_name + '_data.h5\n')

print("Done.")
