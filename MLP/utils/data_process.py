from copy import deepcopy
import random

from torchvision import datasets
import torch
from torch.utils.data import Dataset, DataLoader

from utils import util
from utils.util import load_config

class_num = {'cifar10': 10,
             'cifar100': 100,
             'hapt': 12,
             'uji': 5}
x_shape = {'cifar10': [1, 3, 32, 32],
           'cifar100': [1, 3, 32, 32],
           'hapt': [1, 561, 1, 1],
           'uji': [1, 520, 1, 1]}


# 搜索时使用的Dataset类型
class SearchDataset(Dataset):
    def __init__(self, name, src_data, train_split, valid_split, check=True):
        self.name = name
        if self.name in ['cifar10', 'cifar100']:
            self.feature = deepcopy(src_data)
        else:
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
        if self.name in ['cifar10', 'cifar100']:
            train_image, train_label = self.feature[train_index]
            valid_image, valid_label = self.feature[valid_index]
        else:
            train_image, train_label = self.feature[train_index], self.label[train_index]
            valid_image, valid_label = self.feature[valid_index], self.label[valid_index]
        return train_image, train_label, valid_image, valid_label


# 训练时使用的Dataset类型
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
        train_feature, train_label = self.feature[index], self.label[index]
        return train_feature, train_label


# 获取原始数据
def get_src_dataset(root, name, cutout_length=None):
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
        train_transform, valid_transform = util.get_data_transforms_cifar(name, cutout_length)
        train_data = datasets.CIFAR10(
            root=root, train=True, download=False, transform=train_transform)
        test_data = datasets.CIFAR10(
            root=root, train=False, download=False, transform=valid_transform)
        assert len(train_data) == 50000 and len(test_data) == 10000
    elif name.lower() == 'cifar100':
        train_transform, valid_transform = util.get_data_transforms_cifar(name, cutout_length)
        train_data = datasets.CIFAR100(
            root=root, train=True, download=False, transform=train_transform)
        test_data = datasets.CIFAR100(
            root=root, train=False, download=False, transform=valid_transform)
        assert len(train_data) == 50000 and len(test_data) == 10000
    else:
        raise ValueError
    return train_data, test_data, x_shape[name.lower()], class_num[name.lower()]


# 获取data loader
def get_search_loader(train_data, test_data, name, config_root, workers, batch_size=None):
    if name.lower() in ['hapt', 'uji']:
        if name.lower() == 'hapt':
            config_root += 'HAPT-split.txt'
        else:
            config_root += 'UJI-split.txt'
        dnn_split = load_config('{:}'.format(config_root))
        train_split, valid_split = dnn_split.train, dnn_split.valid
        search_data = SearchDataset(name, train_data, train_split, valid_split)
        search_loader = DataLoader(search_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=workers,
                                   pin_memory=True)
        train_loader = DataLoader(NormalDataset(name, train_data),  # train 改成获取全部数据
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=workers,
                                  drop_last=True,
                                  pin_memory=True)
        test_loader = DataLoader(NormalDataset(name, test_data),
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=workers,
                                 drop_last=True,
                                 pin_memory=True)
    elif name.lower() in ['cifar10', 'cifar100']:
        assert batch_size is not None
        if name.lower() == 'cifar10':
            config_root += 'cifar10-split.txt'
        else:
            config_root += 'cifar100-split.txt'
        cifar_split = load_config(config_root)
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
                                  shuffle=True,
                                  num_workers=workers,
                                  pin_memory=True
                                  )
        test_loader = DataLoader(test_data,
                                 batch_size=batch_size // 4,
                                 shuffle=False,
                                 num_workers=workers,
                                 pin_memory=True)
    else:
        raise ValueError
    return search_loader, train_loader, test_loader
