# Convolutional neural network based activity classifier using accelerometer data

Process actitracker dataset into hdf5 format:

`
$ python src/convert_data.py
`

Run classifier:

`
$ /path/to/caffe/build/tools/caffe train --solver=activitynet_solver.prototxt
`
For me:

`
$ ~/caffe/build/tools/caffe train --solver=activitynet_solver.prototxt
`
