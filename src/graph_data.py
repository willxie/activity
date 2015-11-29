


import numpy as np
import os
import sys
import argparse
import glob
import time
import csv
import caffe
import matplotlib.pyplot as plt

label_dict =  {
                "Walking":0,
                "Jogging":1,
                "Sitting":2,
                "Standing":3,
                "Upstairs":4,
                "Downstairs":5 }

# Percent overlap between subsequent samples
percent_overlap = 0.50

# Filter param
height = 1
width = 64


def process_data_ours(file_path):
    sample_list_x = [] # List of lists
    sample_list_y = [] # List of lists
    sample_list_z = [] # List of lists

    current_sample_x = []
    current_sample_y = []
    current_sample_z = []
    with open(file_path, 'r') as f:
        num_line = 0
        for line in f:
            # Skip the first 10 lines
            if num_line < 20 * 4:
                num_line +=1
                continue
            row = line.split(' ')
            # Skip the indoor localization service data
            if row[0] == '+':
                continue
            # Skip empty line
            if row[0] == '\n':
                continue

            current_sample_x.append(float(row[1]) * 10)
            current_sample_y.append(float(row[2]) * 10)
            current_sample_z.append(float(row[3]) * 10)
            if (len(current_sample_x) >= width):
                ax = plt.gca()
                ax.set_ylim([-15, 15])
                plt.plot(current_sample_x)
                plt.plot(current_sample_y)
                plt.plot(current_sample_z)
                plt.show()
                current_sample_x = current_sample_x[int(len(current_sample_x) * (1.0-percent_overlap)):]
                current_sample_y = current_sample_y[int(len(current_sample_y) * (1.0-percent_overlap)):]
                current_sample_z = current_sample_z[int(len(current_sample_z) * (1.0-percent_overlap)):]



def process_data_theirs(data_path):

    current_sample_x = []
    current_sample_y = []
    current_sample_z = []

    with open(data_path, 'rt') as f:
        reader = csv.reader(f)
        prev_label = " "
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
                current_sample_x = []
                current_sample_y = []
                current_sample_z = []

            current_sample_x.append(x)
            current_sample_y.append(y)
            current_sample_z.append(z)


            # Store when we have the length
            if (len(current_sample_x) >= width):
                print(label)
                if label == "Walking":
                    ax = plt.gca()
                    ax.set_ylim([-15, 15])
                    plt.plot(current_sample_x)
                    plt.plot(current_sample_y)
                    plt.plot(current_sample_z)
                    plt.show()
                # sample_list.append(current_sample)
                # label_list.append(label)
                # Shrink current sample based on the overlap ratio
                current_sample_x = current_sample_x[int(len(current_sample_x) * (1.0-percent_overlap)):]
                current_sample_y = current_sample_y[int(len(current_sample_y) * (1.0-percent_overlap)):]
                current_sample_z = current_sample_z[int(len(current_sample_z) * (1.0-percent_overlap)):]


def main():
    pycaffe_dir = os.path.dirname(__file__)
    caffe_root = '../'
    test_data_path  = caffe_root + "data/test/will/walking.txt"
    train_data_path = caffe_root + "data/actitracker.txt"

    process_data_ours(test_data_path)
    # process_data_theirs(train_data_path)

if __name__ == '__main__':
    main()
