import os
import random
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import argparse
import logging
import numpy as np
import torch
import dataset

from types import SimpleNamespace
from utils import util
from dnn.model import NetworkDNN as Network
from torch import cuda, nn
from torch.backends import cudnn

parser = argparse.ArgumentParser("uci")
# file paths
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--save', type=str, default='DNN', help='experiment name')
parser.add_argument('--dataset', type=str, default='HAPT')
# model params
parser.add_argument('--c_in', type=int, default=561, help='dnn input dimension')
parser.add_argument('--c_out', type=int, default=12, help='dnn output dimension')
# train params
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--epochs', type=int, default=1000, help='num of training epochs')
parser.add_argument('--report_freq', type=float, default=20, help='report frequency')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
# classifier params
parser.add_argument('--hidden_layers', type=int, default=3, choices=[1, 3, 5, 7, 9],
                    help='hidden layers, default is 3')
parser.add_argument('--first_neurons', type=float, help='Number of neurons in the first hidden layer')
parser.add_argument('--change', type=float, default=1, choices=[1, 2, 4, 1 / 2, 1 / 4],
                    help='Hidden layer neuron gradient')
parser.add_argument('--activate_func', type=str, default='relu', choices=['relu', 'tanh', 'sigmoid'],
                    help='Activation function per layer')
parser.add_argument('--opt', type=str, default='dense_layer',
                    choices=['dense_layer', 'avg_pool', 'enhance_avg_pool', 'group_dense', 'enhance_group_dense'],
                    help='Types of operations between layers')
parser.add_argument('--cross_link', type=bool, default=True, choices=[False, True],
                    help='Whether to use cross-layer links')
parser.add_argument('--fully_cross', type=bool, default=True, choices=[False, True],
                    help='Whether to use cross-layer links')

parse = vars(parser.parse_args())


def main(arg):
    ##################################
    for key in arg:
        parse[key] = arg[key]
    global args
    args = SimpleNamespace(**parse)
    '''
    print('seed:{:}'.format(args.seed))
    print('dataset:{:}'.format(args.dataset))
    print('hidden_layers:{:}'.format(args.hidden_layers))
    print('first_neurons:{:}'.format(args.first_neurons))
    print('cross_link:{:}'.format(args.cross_link))
    print('fully_cross:{:}'.format(args.fully_cross))
    print()
    exit(0)
    '''
    ##################################
    seed = util.prepare(args)
    if not cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    cuda.manual_seed(seed)
    cuda.set_device(args.gpu)
    cudnn.benchmark = False
    cudnn.deterministic = True
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)
    logging.info('hidden_layers:{:}'.format(args.hidden_layers))
    logging.info('first_neurons:{:}'.format(args.first_neurons))
    logging.info('change:{:}'.format(args.change))
    logging.info('activate_func:{:}'.format(args.activate_func))
    logging.info('opt:{:}'.format(args.opt))
    logging.info('cross_link:{:}'.format(args.cross_link))
    logging.info('fully_cross:{:}'.format(args.fully_cross))

    model = Network(args)
    model = model.cuda()
    logging.info("param size = %fMB", util.count_parameters_in_MB(model))
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.7)

    train_data, valid_data = dataset.get_dataset(args.data, args.dataset)
    train_queue, valid_queue = dataset.get_data_loader(train_data, valid_data, 2)

    early_stop = util.EarlyStop(patience=10, delta=0.0001, save_path=args.save + '/best.pt')
    for epoch in range(args.epochs):
        logging.info('epoch %d lr %.6f', epoch, scheduler.get_lr()[0])

        epoch_str = '[{:03d}/{:03d}]'.format(epoch, args.epochs)
        train_acc, train_obj = train(train_queue, model, criterion, optimizer, epoch_str)
        logging.info('train_acc %.2f', train_acc)

        valid_acc, valid_obj = infer(valid_queue, model, criterion, epoch_str)
        logging.info('valid_acc %.2f', valid_acc)

        if early_stop.check(train_obj, valid_acc, model):
            logging.info('Early stopping at {:}'.format(epoch))
            break

        scheduler.step()

    # import winsound
    # winsound.MessageBeep()


def train(train_queue, model, criterion, optimizer, epoch_str):
    loss_meter = util.AvgrageMeter()
    top1 = util.AvgrageMeter()
    top5 = util.AvgrageMeter()
    model.train()

    length = train_queue.__len__()
    for step, (input_data, target) in enumerate(train_queue):
        input_data = input_data.cuda()
        target = target.cuda()

        optimizer.zero_grad()
        res = model(input_data)
        loss = criterion(res, target)
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = util.accuracy(res, target, top_k=(1, 5))
        n = input_data.size(0)
        loss_meter.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('train - epoch:{:}\tbatch:[{:03d}/{:03d}]\tavg_loss:{:.6f}\ttop1_acc:{:.2f}%\ttop5_acc:{:.2f}%'
                         .format(epoch_str, step, length, loss_meter.avg, top1.avg, top5.avg))

    return top1.avg, loss_meter.avg


def infer(valid_queue, model, criterion, epoch_str):
    loss_meter = util.AvgrageMeter()
    top1 = util.AvgrageMeter()
    top5 = util.AvgrageMeter()
    model.eval()
    length = valid_queue.__len__()
    with torch.no_grad():
        for step, (input_data, target) in enumerate(valid_queue):
            input_data = input_data.cuda()
            target = target.cuda()

            res = model(input_data)
            loss = criterion(res, target)

            prec1, prec5 = util.accuracy(res, target, top_k=(1, 5))
            n = input_data.size(0)
            loss_meter.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0:
                logging.info(
                    'valid - epoch:{:}\tbatch:[{:03d}/{:03d}]\tavg_loss:{:.6f}\ttop1_acc:{:.2f}%\ttop5_acc:{:.2f}%'
                    .format(epoch_str, step, length, loss_meter.avg, top1.avg, top5.avg))
    return top1.avg, loss_meter.avg


if __name__ == '__main__':
    main()
