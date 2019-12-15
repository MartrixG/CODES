# -*- coding=utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

file_batch = 'C:/AutoML/CIFAR10/data_batch_'
file_test = 'C:/AutoML/CIFAR10/test_batch'


def unpickle(file):
    import pickle
    f = open(file, 'rb')
    dict = pickle.load(f, encoding='iso-8859-1')
    f.close()
    return dict


def load_data():
    x_train = np.empty([0, 3072])
    y_train = np.empty([0, ])
    for i in range(1, 6):
        file_name = file_batch + str(i)
        dic = unpickle(file_name)
        x_train = np.concatenate((x_train, np.array(dic['data'])), axis=0)
        y_train = np.concatenate((y_train, np.array(dic['labels'])), axis=0)
    dic = unpickle(file_test)
    x_test, y_test = np.array(dic['data']), np.array(dic['labels'])
    return (x_train.reshape(50000, 3, 32, 32), y_train), (x_test.reshape(10000, 3, 32, 32), y_test)
