import torch
import random
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy


class GenDataset(Dataset):
    def __init__(self, name, src_data, train_split, valid_split, C_in, C_out, number, check=True):
        self.name = name
        self.feature = deepcopy(src_data[0])
        self.label = deepcopy(src_data[1])
        self.train_split = deepcopy(train_split)
        self.valid_split = deepcopy(valid_split)
        if check:
            intersection = set(train_split).intersection(set(valid_split))
            assert len(intersection) == 0, 'the split train and validation sets should have no intersection'
        self.length = len(self.train_split)
        # import random
        # self.X = torch.randn(number, C_in)
        # class_num = C_out
        # batch_size = number
        # label = torch.LongTensor(batch_size, 1).random_() % class_num
        # self.Y = torch.zeros(batch_size, class_num).scatter_(1, label, 1)
        # self.Y = label.squeeze(1)
        # self.X1 = torch.randn(number, C_in)
        # label1 = torch.LongTensor(batch_size, 1).random_() % class_num
        # self.Y1 = torch.zeros(batch_size, class_num).scatter_(1, label1, 1)
        # self.Y1 = label1.squeeze(1)
        # self.len = number

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        assert 0 <= index < self.length, 'invalid index = {:}'.format(index)
        train_index = self.train_split[index]
        valid_index = random.choice(self.valid_split)
        train_image, train_label = self.feature[train_index], self.label[train_index]
        valid_image, valid_label = self.feature[valid_index], self.label[valid_index]
        return train_image, train_label, valid_image, valid_label
        # return self.X[index], self.Y[index], self.X1[index], self.Y1[index]
