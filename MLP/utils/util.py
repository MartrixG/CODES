import json
import logging
import os
import shutil
import sys
import time
from collections import namedtuple
from torch import optim, nn

import numpy as np
import torchvision.transforms as transforms
import torch


class AverageMeter(object):

    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


class Cutout(object):
    def __init__(self, length):
        self.length = length

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


def get_data_transforms_cifar10(cutout_length=None):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if cutout_length is not None:
        train_transform.transforms.append(Cutout(cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def convert_param(original_list):
    assert isinstance(original_list, list), 'The type is not correct : {:}'.format(original_list)
    ctype, value = original_list[0], original_list[1]
    is_list = isinstance(value, list)
    if not is_list:
        value = [value]
    re = []
    for x in value:
        if ctype == 'int':
            x = int(x)
        elif ctype == 'str':
            x = str(x)
        elif ctype == 'bool':
            x = bool(int(x))
        elif ctype == 'float':
            x = float(x)
        elif ctype == 'none':
            assert x == 'None', 'for none type, the value must be None instead of {:}'.format(x)
            x = None
        else:
            raise TypeError('Does not know this type : {:}'.format(ctype))
        re.append(x)
    if not is_list:
        re = re[0]
    return re


def load_config(path, if_dict=False):
    path = str(path)
    with open(path, 'r') as f:
        data = json.load(f)
    content = {k: convert_param(v) for k, v in data.items()}
    if if_dict:
        return content
    Arguments = namedtuple('Configure', ' '.join(content.keys()))
    content = Arguments(**content)
    return content


def log_config(args):
    logging.info("mission type : {:}".format(args.type))
    logging.info('load the dataset : {:}'.format(args.name))
    if args.type == 'search':
        args.genotype_file = args.save + args.genotype_file
        logging.info('save the genotype to {:}'.format(args.genotype_file))
        logging.info('batch_size : {:}'.format(args.batch_size))
        logging.info('epoch : {:}'.format(args.epoch))
        logging.info('in_num : {:}'.format(args.in_num))
        logging.info('num_node : {:}'.format(args.num_node))
        logging.info('-' * 25 + 'weight config' + '-' * 25)
        logging.info('base_optm : {:}'.format(args.base_optm))
        logging.info('base_lr : {:}'.format(args.base_lr))
        logging.info('base_decay : {:}'.format(args.base_decay))
        logging.info('base_scheduler : {:}'.format(args.base_scheduler))
        logging.info('-' * 25 + 'arch config' + '-' * 25)
        logging.info('arch_optm : {:}'.format(args.arch_optm))
        logging.info('arch_lr : {:}'.format(args.arch_lr))
        logging.info('arch_decay : {:}'.format(args.arch_decay))
        logging.info('max_tau : {:}'.format(args.max_tau))
        logging.info('min_tau : {:}'.format(args.min_tau))
    elif args.type == 'train':
        logging.info('get the genotype from {:}'.format(args.genotype_file))
        logging.info('batch_size : {:}'.format(args.batch_size))
        logging.info('epoch : {:}'.format(args.epoch))
        logging.info('optimizer : {:}'.format(args.optimizer))
        logging.info('scheduler : {:}'.format(args.scheduler))
        logging.info('learning rate : {:}'.format(args.lr))
        logging.info('momentum : {:}'.format(args.momentum))
        logging.info('weight_decay : {:}'.format(args.weight_decay))
    logging.info('args:{:}'.format(args))


def get_opt_scheduler(params, optm, lr, decay, scheduler_name, epoch):
    if optm == 'Adam':
        optimizer = optim.Adam(params, lr, weight_decay=decay)
    else:
        raise ValueError
    if scheduler_name == 'PolyScheduler':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch)
    else:
        raise ValueError
    criterion = nn.CrossEntropyLoss()
    return optimizer, scheduler, criterion


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def accuracy(output, target, top_k=(1,)):
    max_k = max(top_k)
    batch_size = target.size(0)

    _, pred = output.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in top_k:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def prepare(args):
    if args.seed == -1:
        seed = np.random.randint(0, 10000)
    else:
        seed = args.seed
    args.save = 'log/{}-{}-seed-{:05d}/'.format(args.type, time.strftime("%Y-%m-%d-%H-%M-%S"), seed)
    create_exp_dir(args.save)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, args.type + '.log'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    return seed


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        # mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        mask = torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob).cuda()
        x.div_(keep_prob)
        x.mul_(mask)
    return x
