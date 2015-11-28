# Convolutional neural network based activity classifier using accelerometer data

Install caffe by following the instructions:

http://caffe.berkeleyvision.org/installation.html

Make sure that `solver_mode` in `activitynet_solver.prototxt` is set to the correct hardware configuration (CPU or GPU).

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
