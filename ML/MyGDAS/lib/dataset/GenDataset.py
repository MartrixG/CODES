import torch
import random
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy


class GenDataset(Dataset):
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