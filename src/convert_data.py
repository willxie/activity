# Converts actitracker dataset into HDF5 format to be fed into caffe
import os
import numpy as np
import h5py
import csv
import sys
import random
import caffe

SEED = 163

label_dict =  {"Walking":0,
               "Jogging":1,
               "Sitting":2,
               "Standing":3,
               "Upstairs":4,
               "Downstairs":5 }

channels = ['x', 'y', 'z']
data_path = "/home/users/wxie/activity/data/actitracker.txt"
dataset_prefix = 'actitracker_'
to_shuffle = True
zero_mean = True
merge_datasets = False

# Percent overlap between subsequent samples
percent_overlap = 0.50

file_dir = os.path.dirname(os.path.abspath(__file__))

# Caffe blobs have the dimensions (n_samples, n_channels, height, width)
num_channels = 1
height = 1
width = 64

# Trim the front of our dataset a few seconds because it time to take phone in / out of pocket
num_seconds_skipped = 3

# Evenly slice a list into equal proportions
# Source: http://stackoverflow.com/questions/4119070/how-to-divide-a-list-into-n-equal-parts-python
def slice_list(input, size):
    input_size = len(input)
    slice_size = input_size / size
    remain = input_size % size
    result = []
    iterator = iter(input)
    for i in range(size):
        result.append([])
        for j in range(slice_size):
            result[i].append(iterator.next())
        if remain:
            result[i].append(iterator.next())
            remain -= 1
    return result

# Process the data in our iphone app format
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
            if num_line < 5 + 20 * num_seconds_skipped:
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

# Parse actitracker dataset
def process_data_theirs(data_path):
    sample_list_x = [] # List of lists
    sample_list_y = [] # List of lists
    sample_list_z = [] # List of lists

    current_sample_x = []
    current_sample_y = []
    current_sample_z = []

    label_list  = []

    with open(data_path, 'rt') as f:
        reader = csv.reader(f)
        prev_label = " "
        for row in reader:
            if not row:
                continue
            user_id = int(row[0])
            label = label_dict[row[1]]
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
            if (len(current_sample_x) == width):
                sample_list_x.append(current_sample_x)
                sample_list_y.append(current_sample_y)
                sample_list_z.append(current_sample_z)
                label_list.append(label)

                # Retain only certain overlap percentages of the sample
                current_sample_x = current_sample_x[int(len(current_sample_x) * (1.0-percent_overlap)):]
                current_sample_y = current_sample_y[int(len(current_sample_y) * (1.0-percent_overlap)):]
                current_sample_z = current_sample_z[int(len(current_sample_z) * (1.0-percent_overlap)):]

    return sample_list_x, sample_list_y, sample_list_z, label_list

# Find the mean for each dimension of the samples
# Returns a vector of means with dim same as the data
def find_mean(sample_list):
    if not sample_list:
        return []

    mean_list = [0] * len(sample_list[0])
    for sample in sample_list:
        # Element-wise sum
        mean_list = map(sum, zip(mean_list, sample))

    return (np.array(mean_list) / len(sample_list)).tolist()


def main():
    # Load our data stuff
    project_root = "/home/users/wxie/activity/"
    hdf5_path = project_root + 'hdf5/'
    our_data_path = project_root + "data/test/"

    # Load in our data
    our_data_dict = {
        our_data_path + "will/walking.txt": label_dict["Walking"],
        our_data_path + "will/stairs_down_down_in_gdc_side.txt": label_dict["Downstairs"],
        our_data_path + "will/stairs_up_down_in_gdc_side.txt": label_dict["Upstairs"],
        our_data_path + "will/using_phone.txt":label_dict["Standing"],
        our_data_path + "richard/walking_long.txt":label_dict["Walking"],
        our_data_path + "richard/walking_long_2.txt":label_dict["Walking"],
        our_data_path + "richard/stairs_up_long.txt":label_dict["Upstairs"],
        our_data_path + "richard/stairs_down_long.txt":label_dict["Downstairs"],
    }

    our_sample_list_x = []
    our_sample_list_y = []
    our_sample_list_z = []
    our_label_list    = []
    for key, value in our_data_dict.iteritems():
        # Each file as data for only one class
        sample_list_x, sample_list_y, sample_list_z = process_data_ours(key)
        our_sample_list_x.extend(sample_list_x)
        our_sample_list_y.extend(sample_list_y)
        our_sample_list_z.extend(sample_list_z)
        our_label_list.extend([value] * len(sample_list_x))
    # End of our data

    # Load actitracker data
    sample_list_x, sample_list_y, sample_list_z, label_list = process_data_theirs(data_path)

    # print(len(sample_list_x))
    # print(len(sample_list_y))
    # print(len(sample_list_z))
    # print(len(label_list))
    # print()


    ##############################
    # Testing code
    ##############################

    caffe_root = '../'

    # Setup net and weight
    net = caffe.Net(caffe_root + 'activitynet_deploy_1024_30_6_dropout.prototxt',
                    caffe_root + 'model_1024_30_6_with_dropout/actitracker_iter_300000.caffemodel',
                    caffe.TEST)

    class_bin = {}
    for i in range(len(our_label_list)):
        # Create bins
        if our_label_list[i] not in class_bin:
            class_bin[our_label_list[i]] = [0.0] * len(label_dict)

        net.blobs['data_x'].data[...] = np.array(our_sample_list_x[i])
        net.blobs['data_y'].data[...] = np.array(our_sample_list_y[i])
        net.blobs['data_z'].data[...] = np.array(our_sample_list_z[i])

        out = net.forward()
        # print("Predicted class is #{}.".format(out['prob'][0].argmax()))
        # print(out['prob'][0])

        for key, value in label_dict.iteritems():
            if value == out['prob'][0].argmax():
                # print("Predicted class is #{}.".format(out['prob'][0].argmax()))
                # print("Predicted class is " + str(key))
                class_bin[our_label_list[i]][value] += 1
                # print(out['prob'][0])

    for key, value in class_bin.iteritems():
        class_dist = np.array(value)
        class_dist = class_dist / np.sum(class_dist)
        value[:] = class_dist.tolist()

    print (class_bin)

    # exit(0)
    ##############################

    # Merge both datasets
    if (merge_datasets):
        print("Merge two datasets")
        sample_list_x.extend(our_sample_list_x)
        sample_list_y.extend(our_sample_list_y)
        sample_list_z.extend(our_sample_list_z)
        label_list.extend(our_label_list)

    num_samples = len(label_list)

    # print(len(sample_list_x))
    # print(len(sample_list_y))
    # print(len(sample_list_z))
    # print(len(label_list))

    # Calculate HDF5 shape
    total_size = num_samples * num_channels * height * width
    print("Total size: {} \t HDF5 shape: ({}, {}, {}, {})".format(total_size, num_samples, num_channels, height, width))

    # Zero mean
    if zero_mean:
        mean_x = find_mean(sample_list_x)
        mean_y = find_mean(sample_list_y)
        mean_z = find_mean(sample_list_z)
        print("Subtracting mean...")
        for i in range(len(sample_list_x)):
            sample_list_x[i][:] = np.subtract(sample_list_x[i], mean_x).tolist()
            sample_list_y[i][:] = np.subtract(sample_list_y[i], mean_y).tolist()
            sample_list_z[i][:] = np.subtract(sample_list_z[i], mean_z).tolist()

    # Shuffle!
    random.seed(SEED)
    shuffled_index = range(num_samples)
    if to_shuffle:
        print("Shuffling...")
        random.shuffle(shuffled_index)

    sample_list_shuffled_x = []
    sample_list_shuffled_y = []
    sample_list_shuffled_z = []
    label_list_shuffled  = []
    for i in shuffled_index:
        sample_list_shuffled_x.append(sample_list_x[i])
        sample_list_shuffled_y.append(sample_list_y[i])
        sample_list_shuffled_z.append(sample_list_z[i])
        label_list_shuffled.append(label_list[i])


    print("Shuffled size: {} \t Shape: ({}, {}, {}, {})".format(total_size, num_samples, num_channels, height, width))

    # Do 5-fold divide
    sample_list_folds_x = slice_list(sample_list_shuffled_x, 5)
    sample_list_folds_y = slice_list(sample_list_shuffled_y, 5)
    sample_list_folds_z = slice_list(sample_list_shuffled_z, 5)
    label_list_folds  = slice_list(label_list_shuffled, 5)


    # Save data in HDFS format
    print("{} fold cross validation setup".format(len(label_list_folds)))
    for i in range(len(label_list_folds)):
        print("Fold {}".format(i))
        train_fold_index_list = range(len(label_list_folds))
        train_fold_index_list.pop(i)

        # Merging all training sets together
        train_x = []
        train_y = []
        train_z = []
        train_label = []
        for j in train_fold_index_list:
            train_x.extend(sample_list_folds_x[j])
            train_y.extend(sample_list_folds_y[j])
            train_z.extend(sample_list_folds_z[j])
            train_label.extend(label_list_folds[j])


        test_x = []
        test_y = []
        test_z = []
        test_label = []

        test_x.extend(sample_list_folds_x[i])
        test_y.extend(sample_list_folds_y[i])
        test_z.extend(sample_list_folds_z[i])
        test_label.extend(label_list_folds[i])

        print("Train size: {}\t Test size: {}".format(len(train_label), len(test_label)))

        channel_index = 0

        ##############################
        # Testing code
        ##############################
        # caffe_root = '../'

        # # Setup net and weight
        # net = caffe.Net(caffe_root + 'hdf5/0/activitynet_deploy_1024_30_6_dropout.prototxt',
        #                 caffe_root + 'hdf5/0/snapshots/actitracker_iter_50000.caffemodel',
        #                 caffe.TEST)
        # for i in range(len(test_label)):
        #     net.blobs['data_x'].data[...] = np.array(test_x[i])
        #     net.blobs['data_y'].data[...] = np.array(test_y[i])
        #     net.blobs['data_z'].data[...] = np.array(test_z[i])

        #     out = net.forward()
        #     # print("Predicted class is #{}.".format(out['prob'][0].argmax()))
        #     # print(out['prob'][0])

        #     for key, value in label_dict.iteritems():
        #         if value == out['prob'][0].argmax():
        #             # print("Predicted class is #{}.".format(out['prob'][0].argmax()))
        #             print("Predicted class is " + key)
        #             # print(out['prob'][0])

        # exit(0)

        ##############################

        # Formatting to HDF5
        # Train
        # (num_samples, num_channels, height, width)
        data = np.arange(len(train_label) * num_channels * height * width)
        data = data.reshape(len(train_label), num_channels, height, width)
        data = data.astype('float32')

        data_label = np.arange(len(train_label))[:, np.newaxis]
        data_label = data_label.astype('int32')

        channel_dict = {
            'x':train_x,
            'y':train_y,
            'z':train_z
        }
        for channel, channel_list in channel_dict.iteritems():
            output_path = str(i) + '/' + dataset_prefix + channel
            for j in range(len(train_label)):
                data[j][channel_index][0] = np.array(channel_list[j])
                data_label[j] = np.array([train_label[j]])

            with h5py.File(hdf5_path + output_path + '_data.h5', 'w') as f:
                f['data_' + channel] = data
                f['label_' + channel] = data_label
            with open(hdf5_path + output_path + '_data_list.txt', 'w') as f:
                f.write(hdf5_path + output_path + '_data.h5\n')
            print(data.shape)
            print(data_label.shape)


        # Test
        # (num_samples, num_channels, height, width)
        print(len(test_label))
        test = np.arange(len(test_label) * num_channels * height * width)
        test = test.reshape(len(test_label), num_channels, height, width)
        test = test.astype('float32')

        test_label = np.arange(len(test_label))[:, np.newaxis]
        test_label = test_label.astype('int32')

        channel_dict = {
            'x':test_x,
            'y':test_y,
            'z':test_z
        }
        for channel, channel_list in channel_dict.iteritems():
            output_path = str(i) + '/' + dataset_prefix + channel
            for j in range(len(test_label)):
                test[j][channel_index][0] = np.array(channel_list[j])
                test_label[j] = np.array([test_label[j]])

            with h5py.File(hdf5_path + output_path + '_data.h5', 'w') as f:
                f['data_' + channel] = test
                f['label_' + channel] = test_label
            with open(hdf5_path + output_path + '_data_list.txt', 'w') as f:
                f.write(hdf5_path + output_path + '_data.h5\n')
            print(test.shape)
            print(test_label.shape)

        # # Test
        # # (num_samples, num_channels, height, width)
        # test = np.arange(len(label_list_folds[i]) * num_channels * height * width)
        # test = test.reshape(len(label_list_folds[i]), num_channels, height, width)
        # test = test.astype('float32')

        # test_label = np.arange(len(label_list_folds[i]))[:, np.newaxis]
        # test_label = test_label.astype('int32')

        # channel_dict = {
        #     'x':sample_list_folds_x[i],
        #     'y':sample_list_folds_y[i],
        #     'z':sample_list_folds_z[i]
        # }
        # for channel, channel_list in channel_dict.iteritems():
        #     output_path = str(i) + '/' + dataset_prefix + channel
        #     for j in range(len(label_list_folds)):
        #         test[j][channel_index][0] = np.array(channel_list[j])
        #         test_label[j] = np.array([label_list_folds[i][j]])

        #     with h5py.File(hdf5_path + output_path + '_test.h5', 'w') as f:
        #         f['data_' + channel] = test
        #         f['label_' + channel] = test_label
        #     with open(hdf5_path + output_path + '_test_list.txt', 'w') as f:
        #         f.write(hdf5_path + output_path + '_test.h5\n')
        #     print(test.shape)
        #     print(test_label.shape)


    # # Save
    # with h5py.File(file_dir + '/../' + output_name + '_data.h5', 'w') as f:
    #     f['data_' + channel] = data
    #     f['label_' + channel] = data_label

    # with open(file_dir + '/../' + output_name + '_data_list.txt', 'w') as f:
    #     f.write(file_dir + '/../' + output_name + '_data.h5\n')

    print("Done.")

if __name__ == "__main__":
    main()
