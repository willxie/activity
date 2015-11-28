# Converts actitracker dataset into HDF5 format to be fed into caffe
import os
import numpy as np
import h5py
import csv
import sys
from random import shuffle

label_dict =  {
                "Walking":0,
                "Jogging":1,
                "Sitting":2,
                "Standing":3,
                "Upstairs":4,
                "Downstairs":5 }
channels = ['x', 'y', 'z']
data_path = "/home/users/wxie/activity/data/actitracker.txt"
dataset_prefix = 'actitracker_'

# Percent overlap between subsequent samples
percent_overlap = 0.50

file_dir = os.path.dirname(os.path.abspath(__file__))

# Caffe blobs have the dimensions (n_samples, n_channels, height, width)
num_lines = 0
num_samples = 0
num_channels = 1
height = 1
width = 64

# Count the number of samples to allocate memory
with open(data_path, 'rt') as f:
    reader = csv.reader(f)
    prev_label = " "
    line_count = 0
    data_index = 0
    current_sample = []
    for row in reader:
        line_count += 1
        label = row[1]
        # Reset when new label is found
        if (prev_label != label):
            prev_label = label
            current_sample = []
        current_sample.append(1)

        # Store when we have the length
        if (len(current_sample) >= width):
            data_index += 1
            current_sample = current_sample[int(len(current_sample) * (1.0-percent_overlap)):] ###
    num_samples = data_index
    num_lines = line_count

print("Number of data points: {}".format(num_lines))
print("Number of data samples: {}".format(num_samples))

total_size = num_samples * num_channels * height * width

# Shuffle! Not applied until end of the for-loop
shuffled_index = range(num_samples)
shuffle(shuffled_index)

# Process each channel
for channel in channels:
    print("Channel " + channel)
    output_name = dataset_prefix + channel

    data = np.arange(total_size)
    data = data.reshape(num_samples, num_channels, height, width)
    data = data.astype('float32')

    data_label = 1 + np.arange(num_samples)[:, np.newaxis]
    data_label = data_label.astype('int32')

    # This list stores all current_sample (huge!)
    sample_list = [] # List of lists
    label_list  = [] # List

    # Segment points to samples
    with open(data_path, 'rt') as f:
        reader = csv.reader(f)
        prev_label = " "
        current_sample = []
        for row in reader:
            if not row:
                continue
            user_id = int(row[0])
            label = row[1]
            x = float(row[3])
            y = float(row[4])
            z = float(row[5])

            # Reset when new label is found
            if (prev_label != label):
                prev_label = label
                current_sample = []

            if (channel == 'x'):
                current_sample.append(x)
            elif (channel == 'y'):
                current_sample.append(y)
            elif (channel == 'z'):
                current_sample.append(z)
            else:
                print("ERROR: no invalid channel")
                sys.exit()

            # Store when we have the length
            if (len(current_sample) >= width):
                sample_list.append(current_sample)
                label_list.append(label)
                # Shrink current sample based on the overlap ratio
                current_sample = current_sample[int(len(current_sample) * (1.0-percent_overlap)):]
        # print(data_index)

    data_index = 0
    channel_index = 0

    # Shuffle after segmenting points into samples
    for i in shuffled_index:
        data[data_index][channel_index][0] = np.array(sample_list[i])
        data_label[data_index] = np.array([label_dict[label_list[i]]])
        data_index += 1

    print(data.shape)
    print(data_label.shape)

    # Save
    with h5py.File(file_dir + '/../' + output_name + '_data.h5', 'w') as f:
        f['data_' + channel] = data
        f['label_' + channel] = data_label

    with open(file_dir + '/../' + output_name + '_data_list.txt', 'w') as f:
        f.write(file_dir + '/../' + output_name + '_data.h5\n')



print("Done.")
