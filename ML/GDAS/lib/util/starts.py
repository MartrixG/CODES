import numpy as np
import random
import torch
import PIL
import os
import sys

from copy import deepcopy
from torch import cuda


def prepare_seed(rand_seed):
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    cuda.manual_seed(rand_seed)
    cuda.manual_seed_all(rand_seed)


def prepare_logger(xargs):
    args = deepcopy(xargs)
    from lib.log_utils import Logger
    logger = Logger(args.save_dir, args.rand_seed)
    logger.log('Main Function with logger : {:}'.format(logger))
    logger.log('Arguments : -------------------------------')
    for name, value in args._get_kwargs():
        logger.log('{:16} : {:}'.format(name, value))
    logger.log("Python  Version  : {:}".format(sys.version.replace('\n', ' ')))
    logger.log("Pillow  Version  : {:}".format(PIL.__version__))
    logger.log("PyTorch Version  : {:}".format(torch.__version__))
    logger.log("cuDNN   Version  : {:}".format(torch.backends.cudnn.version()))
    logger.log("CUDA available   : {:}".format(torch.cuda.is_available()))
    logger.log("CUDA GPU numbers : {:}".format(torch.cuda.device_count()))
    logger.log("CUDA_VISIBLE_DEVICES : {:}".format(
        os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ else 'None'))
    return logger
