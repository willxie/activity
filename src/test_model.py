import numpy as np
import os
import sys
import argparse
import glob
import time
import csv
import caffe

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

def process_data(file_path):
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
            if num_line < 10:
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

                sample_list_x.append(current_sample_x)
                sample_list_y.append(current_sample_y)
                sample_list_z.append(current_sample_z)
                current_sample_x = current_sample_x[int(len(current_sample_x) * (1.0-percent_overlap)):]
                current_sample_y = current_sample_y[int(len(current_sample_y) * (1.0-percent_overlap)):]
                current_sample_z = current_sample_z[int(len(current_sample_z) * (1.0-percent_overlap)):]

    return sample_list_x, sample_list_y, sample_list_z


def process_data_theirs(data_path):

    sample_list_x = [] # List of lists
    sample_list_y = [] # List of lists
    sample_list_z = [] # List of lists

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
                if label == "Walking":
                    sample_list_x.append(current_sample_x)
                    sample_list_y.append(current_sample_y)
                    sample_list_z.append(current_sample_z)
                current_sample_x = current_sample_x[int(len(current_sample_x) * (1.0-percent_overlap)):]
                current_sample_y = current_sample_y[int(len(current_sample_y) * (1.0-percent_overlap)):]
                current_sample_z = current_sample_z[int(len(current_sample_z) * (1.0-percent_overlap)):]


    return sample_list_x, sample_list_y, sample_list_z


def main(argv):
    pycaffe_dir = os.path.dirname(__file__)
    caffe_root = '../'

    test_data_path = caffe_root + "data/test/will/"

    # Grab all the files from test dir
    file_name_list = []
    for file in os.listdir(test_data_path):
        if file.endswith(".txt"):
            file_name_list.append(file)

    # Setup net and weight
    net = caffe.Net(caffe_root + 'activitynet_deploy.prototxt',
                    caffe_root + 'model_1024_30_6_with_dropout/actitracker_iter_210000.caffemodel',
                    caffe.TEST)

    # For testing the activitynet only
    # file_name_list = ["1"]
    for file_name in file_name_list:
        sample_list_x, sample_list_y, sample_list_z = process_data(test_data_path + file_name)
        # sample_list_x, sample_list_y, sample_list_z = process_data_theirs(caffe_root + "data/actitracker.txt")

        print("====================")
        print(file_name + "     |     sample size: {}".format(len(sample_list_x)))
        print("====================")


        # Trim the first and last 3 seconds (2 samples)
        sample_list_x = sample_list_x[2:len(sample_list_x) - 2]
        sample_list_y = sample_list_y[2:len(sample_list_y) - 2]
        sample_list_z = sample_list_z[2:len(sample_list_z) - 2]

        counter = 0

        # Load data
        # net.blobs['data_x'].reshape(50,1,1,64)
        for i in range(len(sample_list_x)):
            net.blobs['data_x'].data[...] = np.array(sample_list_x[i])
            net.blobs['data_y'].data[...] = np.array(sample_list_y[i])
            net.blobs['data_z'].data[...] = np.array(sample_list_z[i])

            out = net.forward()
            # print("Predicted class is #{}.".format(out['prob'][0].argmax()))
            # print(out['prob'][0])

            for key, value in label_dict.iteritems():
                if value == out['prob'][0].argmax():
                    # print("Predicted class is #{}.".format(out['prob'][0].argmax()))
                    print("Predicted class is " + key)
                    if key == "Walking":
                        counter += 1
                    # print(out['prob'][0])
        print(counter)

if __name__ == '__main__':
    main(sys.argv)
