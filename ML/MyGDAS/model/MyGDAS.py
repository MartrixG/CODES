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
from model.search_dnncell import DNNModel

from torch.backends import cudnn

HAPT_SPACE = ['avg_pool_3x3', 'avg_pool_5x5', 'avg_pool_7x7', 'enhance_avg_pool_3x3', 'enhance_avg_pool_5x5',
            'enhance_avg_pool_7x7',
            'group_dense_2', 'group_dense_3', 'group_dense_4', 'group_dense_5',
            'enhance_group_dense_2', 'enhance_group_dense_3', 'enhance_group_dense_4', 'enhance_group_dense_5']


def train(xargs):
    lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
    if str(lib_dir) not in sys.path:
        sys.path.insert(0, str(lib_dir))

    assert (torch.cuda.is_available(), 'CUDA is not available.')
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
    opt_config = load_config(xargs.opt_config, {'class_num': class_num, 'xshape': xshape}, logger)
    search_loader, _, valid_loader = get_nas_search_loaders(train_data, valid_data, xargs.dataset,
                                                            'config/', config.batch_size, xargs.workers)
    logger.log('||||||| {:10s} ||||||| Search-Loader-Num={:}, batch size={:}'.format(xargs.dataset, len(search_loader),
                                                                                     config.batch_size))
    logger.log('||||||| {:10s} ||||||| Config={:}'.format(xargs.dataset, config))
    arch_config = load_config(xargs.arch_config, {'class_num': class_num,
                                                    'space': HAPT_SPACE,
                                                    'affine': False,
                                                    'track_running_stats': bool(xargs.track_running_stats)}, None)
    if xargs.dataset == 'HAPT':
        search_model = DNNModel(config=arch_config)
    elif xargs.dataset in ('cifar10', 'cifar100'):
        search_model = DNNModel(config=arch_config)
    else:
        raise NameError("datase must be in \"HAPT\", \"cifar100\", \"cifar100\"")
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()
    criterion = criterion.cuda()
    w_optimizer = torch.optim.SGD(search_model.get_weights(),
                                  lr=opt_config.LR,
                                  weight_decay=opt_config.w_decay)
    w_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(w_optimizer, opt_config.epochs, eta_min=0)
    a_optimizer = torch.optim.Adam(search_model.get_alphas(),
                                   lr=opt_config.LR,
                                   betas=(0.5, 0.999),
                                   weight_decay=opt_config.a_decay)
