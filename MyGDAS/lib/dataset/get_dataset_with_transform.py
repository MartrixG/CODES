from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader, sampler
from torchvision import datasets
from lib.util.configure_util import load_config
from lib.dataset.SearchDataset import SearchDataset
from lib.dataset.GenDataset import GenDataset, NormalDataset
import torchvision.transforms as transforms

Dataset2Class = {'cifar10': 10,
                 'cifar100': 100,
                 'HAPT': 12}


class CUTOUT(object):
    # make a mask to cover some part of pic
    def __init__(self, length):
        self.length = length

    def __repr__(self):
        return '{name}(length={length})'.format(name=self.__class__.__name__, **self.__dict__)

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def get_dataset(name, root, cutout):
    # read the original data
    if name == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif name == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif name == 'HAPT':
        pass
    else:
        raise TypeError("Unknown dataset : {:}".format(name))

    # Data Argumentation
    if name == 'cifar10' or name == 'cifar100':
        lists = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
                 transforms.Normalize(mean, std)]
        if cutout > 0:
            lists += [CUTOUT(cutout)]
        train_transform = transforms.Compose(lists)
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        xshape = (1, 3, 32, 32)
    elif name == 'HAPT':
        xshape = (1, 561)
    else:
        raise TypeError("Unknown dataset : {:}".format(name))

    if name == 'cifar10':
        train_data = datasets.CIFAR10(root, train=True, transform=train_transform, download=False)
        test_data = datasets.CIFAR10(root, train=False, transform=test_transform, download=True)
        assert len(train_data) == 50000 and len(test_data) == 10000
    elif name == 'cifar100':
        train_data = datasets.CIFAR100(root, train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR100(root, train=False, transform=test_transform, download=True)
        assert len(train_data) == 50000 and len(test_data) == 10000
    elif name == 'HAPT':
        x_train_file = open('{:}Train/X_train.txt'.format(root))
        y_train_file = open('{:}Train/Y_train.txt'.format(root))
        x_test_file = open('{:}Test/X_test.txt'.format(root))
        y_test_file = open('{:}Test/Y_test.txt'.format(root))
        x_train_src = np.array([list(map(np.float32, item.split(' '))) for item in x_train_file.readlines()])
        y_train_src = (np.array(list(map(np.int, y_train_file.readlines()))) - 1)
        x_test_src = np.array([list(map(np.float32, item.split(' '))) for item in x_test_file.readlines()])
        y_test_src = (np.array(list(map(np.int, y_test_file.readlines()))) - 1)
        assert len(x_train_src) == 7767 and len(y_train_src) == 7767
        assert len(x_test_src) == 3162 and len(y_test_src) == 3162
        train_data = (torch.from_numpy(x_train_src), torch.tensor(y_train_src, dtype=torch.int64))
        test_data = (torch.from_numpy(x_test_src), torch.tensor(y_test_src, dtype=torch.int64))
    else:
        raise TypeError("Unknown dataset : {:}".format(name))
    class_num = Dataset2Class[name]
    return train_data, test_data, xshape, class_num


def get_nas_search_loaders(train_data, valid_data, dataset, config_root, batch_size, workers):
    # get search_loader, train_loader, valid_loader
    if isinstance(batch_size, (list, tuple)):
        batch, test_batch = batch_size
    else:
        batch, test_batch = batch_size, batch_size
    if dataset == 'cifar10':
        cifar_split = load_config('{:}/cifar10-split.txt'.format(config_root), None, None)
        train_split, valid_split = cifar_split.train, cifar_split.valid
        # search over the proposed training and validation set
        # To split data
        xvalid_data = deepcopy(train_data)
        if hasattr(xvalid_data, 'transforms'):  # to avoid a print issue
            xvalid_data.transforms = valid_data.transform
        xvalid_data.transform = deepcopy(valid_data.transform)
        search_data = SearchDataset(dataset, train_data, train_split, valid_split)
        # data loader
        search_loader = DataLoader(search_data,
                                   batch_size=batch,
                                   shuffle=True,
                                   num_workers=workers,
                                   pin_memory=True)
        train_loader = DataLoader(train_data,
                                  batch_size=batch,
                                  sampler=sampler.SubsetRandomSampler(train_split),
                                  num_workers=workers, pin_memory=True)
        valid_loader = DataLoader(xvalid_data,
                                  batch_size=test_batch,
                                  sampler=sampler.SubsetRandomSampler(valid_split),
                                  num_workers=workers, pin_memory=True)
    elif dataset == 'cifar100':
        cifar100_test_split = load_config('{:}/cifar100-split.txt'.format(config_root), None, None)
        search_train_data = train_data
        search_valid_data = deepcopy(valid_data)
        search_valid_data.transform = train_data.transform
        search_data = SearchDataset(dataset, [search_train_data, search_valid_data],
                                    list(range(len(search_train_data))), cifar100_test_split.xvalid)
        search_loader = DataLoader(search_data,
                                   batch_size=batch,
                                   shuffle=True,
                                   num_workers=workers,
                                   pin_memory=True)
        train_loader = DataLoader(train_data, batch_size=batch,
                                  shuffle=True,
                                  num_workers=workers,
                                  pin_memory=True)
        valid_loader = DataLoader(valid_data,
                                  batch_size=test_batch,
                                  sampler=sampler.SubsetRandomSampler(cifar100_test_split.xvalid),
                                  num_workers=workers,
                                  pin_memory=True)
    elif dataset == 'HAPT':
        HAPT_split = load_config('{:}HAPT-split.txt'.format(config_root), None, None)
        train_split, valid_split = HAPT_split.train, HAPT_split.valid
        search_data = GenDataset(dataset, train_data, train_split, valid_split)
        search_loader = DataLoader(search_data,
                                   batch_size=batch,
                                   shuffle=True,
                                   num_workers=workers,
                                   pin_memory=True)
        train_loader = DataLoader(NormalDataset(dataset, train_data),
                                  batch_size=batch,
                                  shuffle=True,
                                  num_workers=workers,
                                  pin_memory=True
                                  )
        valid_loader = DataLoader(NormalDataset(dataset, valid_data),
                                  batch_size=batch,
                                  shuffle=False,
                                  num_workers=workers,
                                  pin_memory=True)
    else:
        raise ValueError('invalid dataset : {:}'.format(dataset))
    return search_loader, train_loader, valid_loader
