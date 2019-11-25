# -*- coding=utf-8 -*-
"""
You can get the enhanced data set, and the current data set that can be called is

cifar10, cifar100, imagenet, coco.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
from matplotlib import pyplot as plt
import numpy as np
import random

def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)

    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

def _get_subset(num, X, Y):
    if num == None:
        return None, None
    zipped = list(zip(X, Y))
    sli = random.sample(zipped, num)
    sli = list(zip(*sli))
    sub_x = np.array(sli[0])
    sub_y = np.array(sli[1])
    return sub_x, sub_y

def _cifar10(path, num_dev, num_train, format):
    import CIFAR10_process as cf10
    (X_train, Y_train), (X_test, Y_test) = cf10.load_data()
    X_train = X_train.astype('float32') / 255
    Y_train = to_categorical(Y_train, 10)

    x_test = X_test.astype('float32') / 255
    y_test = to_categorical(Y_test, 10)
    x_train, y_train = _get_subset(num_train, X_train, Y_train)
    x_dev, y_dev = _get_subset(num_dev, X_train, Y_train)

    return x_train, y_train, x_test, y_test, x_dev, y_dev

def _cifar100(path, num_dev, num_train, format):
    import CIFAR100_process as cf100
    (X_train, Y_train), (X_test, Y_test) = cf100.load_data()
    X_train = X_train.astype('float32') / 255
    Y_train = to_categorical(Y_train, 100)

    x_test = X_test.astype('float32') / 255
    y_test = to_categorical(Y_test, 100)
    x_train, y_train = _get_subset(num_train, X_train, Y_train)
    x_dev, y_dev = _get_subset(num_dev, X_train, Y_train)
    return x_train, y_train, x_test, y_test, x_dev, y_dev

def _imagenet(path, num_dev, num_train, format):
    pass

def _coco(path, num_dev, num_train, format):
    pass

_switch = {
    'cifar10' : _cifar10,
    'cifar100' : _cifar100,
    'imagenet' : _imagenet,
    'coco' : _coco
}

def data_augment(data_set, path, num_train, num_dev = 500, image_data_format = 'channels_first'):
    """Get enhanced data sets for training, testing

    Get a subset of the target dataset based on the parameters entered.This subset
    is made up of train data, test data, development data. All data has been processed
    by data enhancement.

    Args:
        data_set: string, the name of data set.("cifar10, cifar100, imagenet, coco").
        path: string, relative path of the required data set(cifar10 does not need to provide a path).
        num_train: integer, the size(number of data) of train set.
        num_dev: integer, the size(number of data) of dev set, default is 500.
        image_data_format:string, determine the storage format of the image in the dataset, default is channels_first.

    Returns:
        A list of divided data sets, has the following order:
        x_train, y_train, x_test, y_test, x_dev, y_dev.
        The x of data is a ndarray of picture, the shape is N * C * H * W(channels_first) or N * H * W * C(channels_last).
        The y data is one-hot code, the dimension is determined by the number of species in the data set.
        If the param num_dev is zero, x_dev and y_dev will be None.
    """
    return _switch[data_set](path, num_train, num_dev, image_data_format)

def view_detail(data_set, path = None, show_image = False):
    """Test the type and shape of the data obtained.

    The default output is the following data:

    Data type, data shape, specific data of a training set, test set, and dev set.
    You can also choose whether to output the picture.

    Args:
        data_set:string, the name of data set.("cifar10, cifar100, imagenet, coco").
        path: string, relative path of the required data set(cifar10 does not need to provide a path).
        show_image: bool, decide whether to output images, default is flase.
    """
    x_train, y_train, x_test, y_test, x_dev, y_dev = data_augment(data_set, path, num_train=10, num_dev=10)
    print('The type of data:')
    print('x_train type: %s, x_test type: %s, x_dev type: %s'%(type(x_train), type(x_test), type(x_dev)))
    print('y_train type: %s, y_test type: %s, y_dev type: %s\n'%(type(y_train), type(y_test), type(y_dev)))
    print('The shape of data:')
    print('x_train shape: %s\ny_train shape: %s'%(x_train.shape, y_train.shape))
    print('x_test shape: %s\ny_test shape: %s'%(x_test.shape, y_test.shape))
    print('x_dev shape: %s\ny_dev shape: %s'%(x_dev.shape, y_dev.shape))
    if show_image:
        image = x_train[0]
        red = image[0].reshape(1024, 1)
        green = image[1].reshape(1024, 1)
        blue = image[2].reshape(1024, 1)
        pic = np.hstack((red, green, blue)).reshape(32, 32, 3)
        plt.imshow(pic)
        plt.show()