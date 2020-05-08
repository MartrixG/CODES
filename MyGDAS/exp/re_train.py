import sys
import time
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

from lib.util.starts import prepare_seed, prepare_logger, save_checkpoint, copy_checkpoint
from lib.util.configure_util import load_config
from lib.util.meter import AverageMeter
from lib.util.time_util import convert_secs2time, time_string
from lib.dataset.get_dataset_with_transform import get_dataset, get_nas_search_loaders
from lib.util.model_evaluate import get_model_infos, obtain_accuracy
from model.search_dnncell import DNNModel
from torch.nn.utils import clip_grad_norm_

from torch.backends import cudnn


def train(xargs):
    # start cudnn
    cudnn.enabled = True
    # make each conv is the same
    cudnn.benchmark = False
    # make sure the same seed has the same result
    cudnn.deterministic = True
    torch.set_num_threads(xargs.workers)
    prepare_seed(xargs.rand_seed)
    logger = prepare_logger(xargs)
    # get original data(cifar10/cifar100/uci)
    train_data, valid_data, xshape, class_num = get_dataset(xargs.dataset, xargs.data_path, -1)
    logger.log('{:}Train Config{:}'.format("-" * 50, "-" * 50))
    opt_config = load_config(xargs.opt_config, {'class_num': class_num, 'xshape': xshape,
                                                'batch_size': xargs.batch_size, 'epochs': xargs.epochs,
                                                'LR': xargs.opt_learning_rate}, logger)
    _, train_loader, _ = get_nas_search_loaders(train_data, valid_data, xargs.dataset,
                                                'config/', opt_config.batch_size, xargs.workers)
    logger.log('dataset: {:} Train-Loader-length={:}, batch size={:}'.format(xargs.dataset, len(train_loader),
                                                                             opt_config.batch_size))
