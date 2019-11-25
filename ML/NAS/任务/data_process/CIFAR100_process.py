# -*- coding=utf-8 -*-
import numpy as np

train_PATH = 'C:/AutoML/CIFAR100/train'
test_PATH = 'C:/AutoML/CIFAR100/test'
#'filenames', 'batch_label', 'fine_labels', 'coarse_labels', 'data'
def unpickle(file):
    import pickle
    f = open(file, 'rb')
    dict = pickle.load(f, encoding = 'iso-8859-1')
    f.close()
    return dict

def load_data():
    dic = unpickle(train_PATH)
    raw_x = np.array(dic['data'])
    raw_y = np.array(dic['fine_labels'])
    length = len(raw_y)
    x_train = raw_x.reshape(length, 3, 32, 32)
    y_train = raw_y
    dic = unpickle(test_PATH)
    raw_x = np.array(dic['data'])
    raw_y = np.array(dic['fine_labels'])
    length = len(raw_y)
    x_test = raw_x.reshape(length, 3, 32, 32)
    y_test = raw_y
    return (x_train, y_train), (x_test, y_test)
