import os.path as osp
from copy import deepcopy

import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from PIL import Image

Dataset2Class = {'cifar10': 10,
                 'cifar100': 100}


class CUTOUT(object):

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
    if name == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif name == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
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
    else:
        raise TypeError("Unknown dataset : {:}".format(name))
    class_num = Dataset2Class[name]
    return train_data, test_data, xshape, class_num
