# Converts
import os
import numpy as np
import h5py
import csv
import sys

data_path = "/home/users/wxie/activity/data/actitracker_test_2.txt"

# Caffe blobs have the dimensions (n_samples, n_channels, height, width)
# Count lines
num_lines = sum(1 for line in open(data_path, 'rt'))
print("Number of data points: {}".format(num_lines))

height = 1
width = 2
num_samples = num_lines / width
num_channels = 1
total_size = num_samples * num_channels * height * width

data = np.arange(total_size)
data = data.reshape(num_samples, num_channels, height, width)
data = data.astype('float32')

data_label = 1 + np.arange(num_samples)[:, np.newaxis]
data_label = data_label.astype('int32')

# print(label[1][0])

# print(label)

# print(data)

with open(data_path, 'rt') as f:
    reader = csv.reader(f)
    prev_label = " "
    data_index = 0
    channel_index = 0
    x_list = []
    y_list = []
    z_list = []
    for row in reader:
        user_id = int(row[0])
        label = row[1]
        x = float(row[3])
        y = float(row[4])
        z = float(row[5])
        print("user_id: {}\t label: {}\t x: {}\t y: {}\t z: {}".format(user_id, label, x, y, z))

        # Reset when new label is found
        if (prev_label != label):
            prev_label = label
            x_list = []

        x_list.append(x)

        # Store when we have the length
        if (len(x_list) >= width):
            data[data_index][channel_index][0] = np.array(x_list)
            data_label[data_index] = np.array([hash(label)])
            data_index += 1
            x_list = [] ###

print(data)
print(data_label)
