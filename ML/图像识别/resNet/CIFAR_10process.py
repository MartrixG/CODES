#Python 3.6.9 64-bit(tensorflow-gpu)
'''
    CIFAR-10数据处理。
    数据说明：
    每一个字典里有4组键值对
    1.batch_label ：表明batch的位置,没什么用
    2.data ：32*32图片的数值化数组，是一个10000*3072的numpy二维数组,
              每一行代表一张图片，一行分3段(红绿蓝通道)，每段1024个元素。
    3.labels ：data每一行对应的标签（数字0-9），是个一维数组，10000个元素
    4.filenames ： data每一行对应的文件名，同是一个一维数组，10000个元素
'''
import numpy as np
from matplotlib import pyplot as plt
import pickle

file_batch = 'D:/LEARNING/CODES/ML/图像识别/data/CIFAR-10/cifar-10-batches-py/data_batch_'
file_meta = 'D:/LEARNING/CODES/ML/图像识别/data/CIFAR-10/cifar-10-batches-py/batches.meta'
file_test = 'D:/LEARNING/CODES/ML/图像识别/data/CIFAR-10/cifar-10-batches-py/test_batch'

def unpickle(file):
    f = open(file, 'rb')
    dict = pickle.load(f, encoding = 'iso-8859-1')
    f.close()
    return dict
'''
num_batch: Number of batch to train
return: data, labels
'''
def load_train_data(num_batch):
    file_name = file_batch + str(num_batch)
    dic = unpickle(file_name)
    return dic['data'], dic['labels']

'''
return: all 5 batches data and labels
'''
def load_all_train_data():
    data, label = load_train_data(1)
    for i in range(2, 6):
        tmp_data, tmp_label = load_train_data(i)
        data = np.vstack((data, tmp_data))
        label.extend(tmp_label)
    return data, label

'''
to load test data and labels
return: data, labels
'''
def load_test_data():
    dic = unpickle(file_test)
    return dic['data'], dic['labels']

'''
view the picture num_pic of batch num_ba
num_pic: the number of picture to view
num_ba: the number of batch choose
'''
def view_pic(num_pic, num_ba):
    meta = unpickle(file_meta)
    data, labels = load_train_data(num_ba)
    x = data[num_pic]
    x = x.reshape(3, 32, 32)
    red = x[0].reshape(1024, 1)
    green = x[1].reshape(1024, 1)
    blue = x[2].reshape(1024, 1)
    pic = np.hstack((red, green, blue)).reshape(32, 32, 3)
    print(meta['label_names'][labels[num_pic]])
    plt.imshow(pic)
    plt.show()

view_pic(10, 1)