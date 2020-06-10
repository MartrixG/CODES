import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
import random
import genotypes

from utils import util
from model import NetworkCIFAR as Network
from torch.backends import cudnn
from torch import cuda
from torchvision import datasets
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser("cifar")
# file paths
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--save', type=str, default='CNN', help='experiment name')
# train params
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--auxiliary', action='store_true', default=True, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
# arch params
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
# classifier params
parser.add_argument('--hidden_layers', type=int, default=3, choices=[1, 3, 5, 7, 9],
                    help='hidden layers, default is 3')
parser.add_argument('--first_neurons', type=float, default=1, choices=[1, 2, 4, 1 / 2, 1 / 4],
                    help='Number of neurons in the first hidden layer')
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

args = parser.parse_args()


def main():
    seed = util.prepare(args)
    if not cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    CIFAR_CLASSES = 10
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

    genotype = eval("genotypes.%s" % args.arch)
    model = Network(args.init_channels, CIFAR_CLASSES,
                    args.layers, args.auxiliary, genotype, args)
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

    train_transform, valid_transform = util.get_data_transforms_cifar10(args)
    train_data = datasets.CIFAR10(
        root=args.data, train=True, download=False, transform=train_transform)
    valid_data = datasets.CIFAR10(
        root=args.data, train=False, download=False, transform=valid_transform)

    train_queue = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=1)

    valid_queue = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=1)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    best_acc = 0
    for epoch in range(args.epochs):
        logging.info('epoch %d lr %.6f', epoch, scheduler.get_lr()[0])
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        epoch_str = '[{:03d}/{:03d}]'.format(epoch, args.epochs)
        train_acc, train_obj = train(train_queue, model, criterion, optimizer, epoch_str)
        logging.info('train_acc %.2f', train_acc)

        valid_acc, valid_obj = infer(valid_queue, model, criterion, epoch_str)
        logging.info('valid_acc %.2f', valid_acc)

        if valid_acc > best_acc:
            logging.info('find the best model. Save it to {:}'.format(args.save + 'best.pt'))
            util.save(model, os.path.join(args.save, 'best.pt'))
            best_acc = valid_acc
        scheduler.step()
    logging.info('best acc is {:}'.format(best_acc))


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
        res, res_aux = model(input_data)
        loss = criterion(res, target)
        if args.auxiliary:
            loss_aux = criterion(res_aux, target)
            loss += args.auxiliary_weight * loss_aux
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

            res, _ = model(input_data)
            loss = criterion(res, target)

            prec1, prec5 = util.accuracy(res, target, top_k=(1, 5))
            n = input_data.size(0)
            loss_meter.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0:
                logging.info('valid - epoch:{:}\tbatch:[{:03d}/{:03d}]\tavg_loss:{:.6f}\ttop1_acc:{:.2f}%\ttop5_acc:{:.2f}%'
                             .format(epoch_str, step, length, loss_meter.avg, top1.avg, top5.avg))

    return top1.avg, loss_meter.avg


if __name__ == '__main__':
    main()
