import os
import sys

from classifier_model import search_classifier
from pre_model import pre_model
from train_model import Network
from utils.data_process import get_src_dataset, get_search_loader
from utils.util import count_parameters_in_MB

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import numpy as np
import logging
import random

from torch import cuda, nn, manual_seed
from utils import util
from torch.backends import cudnn


def main(args):
    seed = util.prepare(args)
    if not cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    np.random.seed(seed)
    random.seed(seed)
    manual_seed(seed)
    cuda.manual_seed(seed)
    cuda.set_device(args.gpu)
    cudnn.benchmark = False
    cudnn.deterministic = True
    logging.info('gpu device = %d' % args.gpu)
    logging.info("mission type : {:}".format(args.type))

    train_data, test_data, x_shape, class_num = get_src_dataset(args.data_path, args.name)
    search_loader, train_loader, valid_loader, test_loader = get_search_loader(
        train_data, test_data, args.name, args.split, args.workers, args.batch_size)
    logging.info('load the dataset : {:}\tbatch_size : {:}'.format(args.name, args.batch_size))
    logging.info('save the genotype to {:}'.format(args.genotype))

    model = Network(args.name, x_shape, class_num, args).cuda()
    logging.info('model param : {:}MB'.format(count_parameters_in_MB(model)))
