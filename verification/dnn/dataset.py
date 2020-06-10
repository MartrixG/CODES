from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class NormalDataset(Dataset):
    def __init__(self, src_data):
        self.feature = deepcopy(src_data[0])
        self.label = deepcopy(src_data[1])
        self.length = len(self.feature)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        assert 0 <= index < self.length, 'invalid index = {:}'.format(index)
        chosen_feature, chosen_label = self.feature[index], self.label[index]
        return chosen_feature, chosen_label


def get_dataset(root, name):
    if name == 'HAPT':
        x_train_file = [line.split(' ') for line in open('{:}/HAPT/Train/X_train.txt'.format(root)).readlines()]
        y_train_file = [line for line in open('{:}/HAPT/Train/Y_train.txt'.format(root)).readlines()]
        x_test_file = [line.split(' ') for line in open('{:}/HAPT/Test/X_test.txt'.format(root)).readlines()]
        y_test_file = [line for line in open('{:}/HAPT/Test/Y_test.txt'.format(root)).readlines()]

        x_train_data = torch.tensor([list(map(float, line)) for line in x_train_file], dtype=torch.float32)
        y_train_data = torch.tensor(list(map(int, y_train_file)), dtype=torch.long) - 1

        x_test_data = torch.tensor([list(map(float, line)) for line in x_test_file], dtype=torch.float32)
        y_test_data = torch.tensor(list(map(int, y_test_file)), dtype=torch.long) - 1

        assert len(x_train_data) == 7767 and len(y_train_data) == 7767
        assert len(x_test_data) == 3162 and len(y_test_data) == 3162
    elif name == 'UJI':
        train_file = [line.split(',') for line in open('{:}/UJIndoorLoc/trainingData.csv'.format(root)).readlines()]
        test_file = [line.split(',') for line in open('{:}/UJIndoorLoc/validationData.csv'.format(root)).readlines()]

        x_train_data = torch.tensor([list(map(float, line))[:520] for line in train_file], dtype=torch.float32) / 100
        y_train_data = torch.tensor([int(line[522]) for line in train_file], dtype=torch.long)

        x_test_data = torch.tensor([list(map(float, line))[:520] for line in test_file], dtype=torch.float32) / 100
        y_test_data = torch.tensor([int(line[522]) for line in test_file], dtype=torch.long)

        assert len(x_train_data) == 19937 and len(y_train_data) == 19937
        assert len(x_test_data) == 1111 and len(y_test_data) == 1111
    else:
        raise ValueError
    train_data = (x_train_data, y_train_data)
    test_data = (x_test_data, y_test_data)
    return train_data, test_data


def get_data_loader(train_data, valid_data, workers):
    train_loader = DataLoader(NormalDataset(train_data),
                              batch_size=train_data[0].__len__(),
                              shuffle=True,
                              num_workers=workers
                              )
    valid_loader = DataLoader(NormalDataset(valid_data),
                              batch_size=valid_data[0].__len__(),
                              shuffle=False,
                              num_workers=workers
                              )
    return train_loader, valid_loader
