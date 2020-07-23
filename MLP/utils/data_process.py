from copy import deepcopy
import random

import datasets
from torchvision import datasets
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils import util
from utils.util import load_config

class_num = {'cifar10': 10,
             'cifar100': 100,
             'HAPT': 12,
             'UJI': 5}
x_shape = {'cifar10': (32, 32, 3),
           'cifar100': (32, 32, 3),
           'HAPT': 561,
           'UJI': 520}


class SearchDataset(Dataset):
    def __init__(self, name, src_data, train_split, valid_split, check=True):
        self.name = name
        self.feature = deepcopy(src_data[0])
        self.label = deepcopy(src_data[1])
        self.train_split = deepcopy(train_split)
        self.valid_split = deepcopy(valid_split)
        if check:
            intersection = set(train_split).intersection(set(valid_split))
            assert len(intersection) == 0, 'the split train and validation sets should have no intersection'
        self.length = len(self.train_split)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        assert 0 <= index < self.length, 'invalid index = {:}'.format(index)
        train_index = self.train_split[index]
        valid_index = random.choice(self.valid_split)
        train_image, train_label = self.feature[train_index], self.label[train_index]
        valid_image, valid_label = self.feature[valid_index], self.label[valid_index]
        return train_image, train_label, valid_image, valid_label


class NormalDataset(Dataset):
    def __init__(self, name, src_data):
        self.name = name
        self.feature = deepcopy(src_data[0])
        self.label = deepcopy(src_data[1])
        self.length = len(self.feature)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        assert 0 <= index < self.length, 'invalid index = {:}'.format(index)
        chosen_feature, chosen_label = self.feature[index], self.label[index]
        return chosen_feature, chosen_label


def get_src_dataset(root, name, cutout_length=None):
    """
    get the source data of the given dataset
    :param root: data root
    :param name: dataset name(hapt, uji, cifar10, cifar100).
    :param cutout_length: if the dataset is cnn dataset, may use cutout to enhance the dataset
    :return: train_data:
            if dataset is dnn dataset, return the a tuple of numpy array:(source train data, train target)
            if dataset is cnn dataset, return a 'datasets' type in pytorch with transform of train data.
            test_data:
            if dataset is dnn dataset, return the a tuple of numpy array:(source test data, test target)
            if dataset is cnn dataset, return a 'datasets' type in pytorch with transform of test data.
            x_shape:
            the shape of a single data of the dataset
            class_num:
            the number of class
    """
    if name.lower() == 'hapt':
        x_train_file = [line.split(' ') for line in open('{:}HAPT/Train/X_train.txt'.format(root)).readlines()]
        y_train_file = [line for line in open('{:}HAPT/Train/Y_train.txt'.format(root)).readlines()]
        x_test_file = [line.split(' ') for line in open('{:}HAPT/Test/X_test.txt'.format(root)).readlines()]
        y_test_file = [line for line in open('{:}HAPT/Test/Y_test.txt'.format(root)).readlines()]

        x_train_data = torch.tensor([list(map(float, line)) for line in x_train_file], dtype=torch.float32)
        y_train_data = torch.tensor(list(map(int, y_train_file)), dtype=torch.long) - 1

        x_test_data = torch.tensor([list(map(float, line)) for line in x_test_file], dtype=torch.float32)
        y_test_data = torch.tensor(list(map(int, y_test_file)), dtype=torch.long) - 1

        assert len(x_train_data) == 7767 and len(y_train_data) == 7767
        assert len(x_test_data) == 3162 and len(y_test_data) == 3162

        train_data = (x_train_data, y_train_data)
        test_data = (x_test_data, y_test_data)
    elif name.lower() == 'uji':
        train_file = [line.split(',') for line in open('{:}/UJIndoorLoc/trainingData.csv'.format(root)).readlines()]
        test_file = [line.split(',') for line in open('{:}/UJIndoorLoc/validationData.csv'.format(root)).readlines()]

        x_train_data = torch.tensor([list(map(float, line))[:520] for line in train_file], dtype=torch.float32) / 100.0
        y_train_data = torch.tensor([int(line[522]) for line in train_file], dtype=torch.long)

        x_test_data = torch.tensor([list(map(float, line))[:520] for line in test_file], dtype=torch.float32) / 100.0
        y_test_data = torch.tensor([int(line[522]) for line in test_file], dtype=torch.long)

        assert len(x_train_data) == 19937 and len(y_train_data) == 19937
        assert len(x_test_data) == 1111 and len(y_test_data) == 1111

        train_data = (x_train_data, y_train_data)
        test_data = (x_test_data, y_test_data)
    elif name.lower() == 'cifar10':
        train_transform, valid_transform = util.get_data_transforms_cifar10(cutout_length)
        train_data = datasets.CIFAR10(
            root=root, train=True, download=False, transform=train_transform)
        test_data = datasets.CIFAR10(
            root=root, train=False, download=False, transform=valid_transform)
        assert len(train_data) == 50000 and len(test_data) == 10000
    else:
        raise ValueError
    return train_data, test_data, x_shape[name], class_num[name]


def get_search_loader(train_data, test_data, name, config_root, workers, batch_size=None):
    """
    cifar10和cifar100的数据分割方式不同。
    cifar10: search_loader 加载了完整的训练集，并且使用了SearchDataset这个datasets获取每个epoch的数据，一个用于更新模型参数，一个用于更新架构参数
                           两个部分都使用cutout
             train_loader 加载了完整的训练集，但是使用的是CIFAR10自带的datasets类型，并且在dataloader中使用了SubsetRandomSampler，
                          参数是划分search_loader中使用的train_split。使用cutout
             valid_loader 加载了完整的训练集，使用CIFAR10自带的datasets类型，并且在dataloader中使用了SubsetRandomSampler，
                          参数是划分search_loader中使用的valid_split。没有使用cutout（使用的是test_data自带的transform）
    cifar100：search_loader 加载了全部的训练集和测试集，并且使用了SearchDataset这个datasets获取每个epoch的数据。其中测试集的transform复制了训练集
                            的transform为了获取cutout，因此两部分都使用了cutout
              train_loader 加载了完整的训练集，并且使用CIFAR100自带的datasets类型。并且在dataloader中使用了SubsetRandomSampler，
                           参数是划分search_loader中使用的train_split。使用cutout
              valid_loader 加载了全部的测试集，并且使用CIFAR100自带的datasets类型，并且在dataloader中使用了SubsetRandomSampler，
                           参数是另外一个划分方式（划分了原始测试集的一半）。没有使用cutout（使用的是test_data自带的transform）

    :param batch_size:
    :param train_data: source train data of dataset with type of numpy ndarray(dnn dataset)
                       source train data of dataset with type of 'datasets' in pytorch(cnn dataset)
    :param test_data: source test data of dataset with type of numpy ndarray(dnn dataset)
                      source test data of dataset with type of 'datasets' in pytorch(cnn dataset)
    :param name: dataset name in (hapt, uji, cifar10, cifar100)
    :param config_root: the split file of the search datasets for train data
    :param workers: multi-thread to read data
    :return: search_loader: a dataloader using in searching architecture of MLP(need to divide train data to two parts: train_data, valid_data)
             train_loader: a dataloader to train the final architecture which was found(use the train_split's train data)
             valid_loader: a dataloader for checking architecture loss of MLP(use the valid_split's train data)
             test_loader: a dataloader for testing the final architecture which was found(use the total test data)
    """
    if name.lower() in ['hapt', 'uji']:
        if name.lower() == 'hapt':
            config_root += 'HAPT-split.txt'
        else:
            config_root += 'UJI-split.txt'
        dnn_split = load_config('{:}'.format(config_root))
        train_split, valid_split = dnn_split.train, dnn_split.valid
        valid_data = deepcopy(train_data)
        search_data = SearchDataset(name, train_data, train_split, valid_split)
        search_loader = DataLoader(search_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=workers,
                                   pin_memory=True)
        train_loader = DataLoader(NormalDataset(name, train_data),
                                  batch_size=batch_size,
                                  sampler=torch.utils.data.sampler.SubsetRandomSampler(train_split),
                                  num_workers=workers,
                                  pin_memory=True
                                  )
        valid_loader = DataLoader(NormalDataset(name, valid_data),
                                 batch_size=batch_size,
                                 sampler=torch.utils.data.sampler.SubsetRandomSampler(valid_split),
                                 num_workers=workers,
                                 pin_memory=True
                                 )
        test_loader = DataLoader(NormalDataset(name, test_data),
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=workers,
                                 pin_memory=True
                                 )
    elif name.lower() == 'cifar10':
        assert batch_size is not None
        cifar_split = load_config('{:}cifar10-split.txt'.format(config_root))
        train_split, valid_split = cifar_split.train, cifar_split.valid
        valid_data = deepcopy(train_data)
        valid_data.transform = deepcopy(train_data.transform)
        search_data = SearchDataset(name, train_data, train_split, valid_split)
        search_loader = DataLoader(search_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=workers,
                                   pin_memory=True
                                   )
        train_loader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  sampler=torch.utils.data.sampler.SubsetRandomSampler(train_split),
                                  num_workers=workers,
                                  pin_memory=True
                                  )
        valid_loader = DataLoader(valid_data,
                                  batch_size=batch_size,
                                  sampler=torch.utils.data.sampler.SubsetRandomSampler(valid_split),
                                  num_workers=workers,
                                  pin_memory=True
                                  )
        test_loader = DataLoader(test_data,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=workers,
                                 pin_memory=True)
    elif name.lower() == 'cifar100':
        raise ValueError
    else:
        raise ValueError
    return search_loader, train_loader, valid_loader, test_loader
