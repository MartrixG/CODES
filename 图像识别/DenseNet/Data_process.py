# Python 3.6.9 64-bit(tensorflow-gpu)

import random
import numpy as np
import keras.backend as K
from keras.utils import np_utils
from keras.datasets import cifar10
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

K.set_image_data_format('channels_first')
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


def load_data(data_set="cifar10", num_dev=500):
    """
    get the chosen data set

    Arguments:
    data_set --- the name of data set, default is cifar10
    num_dev --- the size(number of data) of dev set, default is 500

    returns: x_tr, y_tr, x_te, y_te, x_de, y_de
    """
    if data_set == "cifar10":
        x_tr, y_tr = load_cifar10_train()
        x_te, y_te = load_cifar10_test()
        x_de, y_de = load_dev(num_dev, x_tr, y_tr)
        return x_tr, y_tr, x_te, y_te, x_de, y_de
    else:
        pass


def load_cifar10_train():
    x_tr = x_train.astype('float32') / 255
    y_tr = np_utils.to_categorical(y_train, 10)
    return x_tr, y_tr


def load_cifar10_test():
    x_te = x_test.astype('float32') / 255
    y_te = np_utils.to_categorical(y_test, 10)
    return x_te, y_te


def load_dev(num_dev, x_tr, y_tr):
    """
    According to the param get the dev set.

    Arguments:
    num_dev --- the size of dev set
    x_tr --- train data of x
    y_tr --- train data of y

    returns: x_dev, y_dev
    """
    x_dev = []
    y_dev = []
    for i in range(num_dev):
        num = random.randint(0, x_tr.shape[0])
        x_dev.append(x_tr[num])
        y_dev.append(y_tr[num])
    x_dev = np.array(x_dev)
    y_dev = np.array(y_dev)
    return x_dev, y_dev
