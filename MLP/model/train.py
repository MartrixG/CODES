import os
import sys
import time

from model.train_model import Network
from utils.data_process import get_src_dataset, get_search_loader
from utils.flop_becnmark import get_model_infos
from utils.util import log_config, get_opt_scheduler, AverageMeter, accuracy, save

import numpy as np
import logging
import random

from torch import cuda, optim, manual_seed, nn
from utils import util
from torch.backends import cudnn


# 搜索过程中一个epoch执行的操作
def search_train(search_loader, model, criterion, w_optimizer, a_optimizer, epoch_str, print_freq, grad_clip):
    model.train()
    base_top1 = AverageMeter()
    base_top5 = AverageMeter()
    base_loss = AverageMeter()
    arch_top1 = AverageMeter()
    arch_top5 = AverageMeter()
    arch_loss = AverageMeter()

    for step, (base_input, base_target, arch_input, arch_target) in enumerate(search_loader):
        batch = base_input.size(0)
        base_input = base_input.cuda()
        base_target = base_target.cuda()

        w_optimizer.zero_grad()
        if model.name in ['cifar10', 'cifar100']:
            feature, logits_aux = model(base_input)
            w_loss = criterion(feature, base_target)
            if logits_aux is not None:
                loss_aux = criterion(logits_aux, base_target)
                w_loss += model.pre_model.auxiliary_weight * loss_aux
        elif model.name in ['uji', 'hapt']:
            feature = model(base_input)
            w_loss = criterion(feature, base_target)
        else:
            raise ValueError
        w_loss.backward()
        nn.utils.clip_grad_norm(model.get_weights(), grad_clip)
        w_optimizer.step()
        pre1, pre5 = accuracy(feature.data, base_target.data, top_k=(1, 5))
        base_top1.update(pre1.item(), batch)
        base_top5.update(pre5.item(), batch)
        base_loss.update(w_loss.item(), batch)

        arch_input = arch_input.cuda()
        arch_target = arch_target.cuda()

        a_optimizer.zero_grad()
        if model.name in ['cifar10', 'cifar100']:
            feature, logits_aux = model(arch_input)
            a_loss = criterion(feature, arch_target)
            if logits_aux is not None:
                loss_aux = criterion(logits_aux, arch_target)
                a_loss += model.pre_model.auxiliary_weight * loss_aux
        elif model.name in ['uji', 'hapt']:
            feature = model(arch_input)
            a_loss = criterion(feature, arch_target)
        else:
            raise ValueError
        a_loss.backward()
        a_optimizer.step()
        pre1, pre5 = accuracy(feature.data, arch_target.data, top_k=(1, 5))
        arch_top1.update(pre1.item(), batch)
        arch_top5.update(pre5.item(), batch)
        arch_loss.update(a_loss.item(), batch)

        if step % print_freq == 0 or step + 1 == len(search_loader):
            str1 = 'search - epoch:' + epoch_str + ' batch:[' + '{:3d}/{:}]  '.format(step, len(search_loader))
            str2 = 'train data[Loss:{:.6f}  Pre@1:{:.5f}%  Pre@5:{:.5f}%]'.format(base_loss.avg, base_top1.avg,
                                                                                  base_top5.avg)
            str3 = '  Val data[Loss:{:.6f}  Pre@1:{:.5f}%  Pre@5:{:.5f}%]'.format(arch_loss.avg, arch_top1.avg,
                                                                                  arch_top5.avg)
            logging.info(str1 + str2 + str3)

    return base_top1.avg, base_top5.avg, base_loss.avg, arch_top1.avg, arch_top5.avg, arch_loss.avg


# 搜索
def search(args):
    logging.info('start load dataset')
    train_data, test_data, x_shape, class_num = get_src_dataset(args.data_path, args.name)
    x_shape[0] = args.batch_size
    search_loader, _, _ = get_search_loader(
        train_data, test_data, args.name, args.split, args.workers, args.batch_size)
    logging.info('dataset loaded')

    model = Network(args.name, x_shape, class_num, args)
    model = model.cuda()
    flop, param = get_model_infos(model, x_shape)
    logging.info('Params={:.2f} MB, FLOPs={:.2f} M'.format(param, flop))

    w_optimizer, w_scheduler, criterion = get_opt_scheduler(
        model.get_weights(), args.base_optm, args.base_lr, args.base_decay, args.base_scheduler, args.epoch)

    criterion = criterion.cuda()

    if args.arch_optm == 'Adam':
        a_optimizer = optim.Adam(model.get_alphas(), args.arch_lr, weight_decay=args.arch_decay)
    else:
        raise ValueError
    logging.info('w-optimizer : {:}'.format(w_optimizer))
    logging.info('a-optimizer : {:}'.format(a_optimizer))
    logging.info('w-scheduler : {:}'.format(w_scheduler))
    logging.info('criterion   : {:}'.format(criterion))
    logging.info('classifier:\n{:}'.format(model.classifier))

    best_acc = 0
    time_str = ''
    for epoch in range(1, args.epoch + 1):
        new_tau = args.max_tau - (args.max_tau - args.min_tau) * epoch / (args.epoch - 1)
        model.set_tau(new_tau)
        logging.info(
            'epoch:{:} LR:{:.6f} tau:{:.6f} need time {:}'.format(epoch, w_scheduler.get_lr()[0], new_tau, time_str))
        if args.name in ['cifar10', 'cifar100']:
            model.set_drop_path_prob(args.drop_path_prob * epoch / args.epoch)

        epoch_str = '[{:03d}/{:03d}]'.format(epoch, args.epoch)

        # A, B = model.show_alphas()
        # logging.info(A)
        # logging.info(B)
        s_time = time.time()
        base_top1, base_top5, base_loss, arch_top1, arch_top5, arch_loss = search_train(
            search_loader, model, criterion, w_optimizer, a_optimizer, epoch_str, args.print_frequency, args.grad_clip)
        batch_time = (time.time() - s_time) * (args.epoch - epoch)
        m, s = divmod(batch_time, 60)
        h, m = divmod(m, 60)
        time_str = "%d:%02d:%02d" % (h, m, s)
        train_str = 'train set - epoch:' + epoch_str + ' result  Loss:'
        vla_str = ' val  set - epoch:' + epoch_str + ' result  Loss:'
        logging.info(train_str + '{:.6f}  Pre@1 : {:.5f}%  Pre@5:{:.5f}%'.format(base_loss, base_top1, base_top5))
        logging.info(vla_str + '{:.6f}  Pre@1 : {:.5f}%  Pre@5:{:.5f}%'.format(arch_loss, arch_top1, arch_top5))

        if arch_top1 > best_acc:
            best_acc = arch_top1
            logging.info('find the best model. best acc is {:.5f}%'.format(best_acc))
            logging.info('Save it to {:}'.format(args.save + 'best.pt'))
            save(model, os.path.join(args.save, 'best.pt'))
            model.get_genotype()

        w_scheduler.step()

    logging.info('best acc is {:.5f}%'.format(best_acc))


# 训练过程中一个spoch执行的操作
def model_train(train_loader, model, criterion, optimizer, epoch_str, print_freq, grad_clip):
    model.train()
    base_top1 = AverageMeter()
    base_top5 = AverageMeter()
    base_loss = AverageMeter()
    for step, (base_input, base_target) in enumerate(train_loader):
        batch = base_input.size(0)
        base_input = base_input.cuda()
        base_target = base_target.cuda()
        optimizer.zero_grad()
        if model.name in ['cifar10', 'cifar100']:
            feature, logits_aux = model(base_input)
            loss = criterion(feature, base_target)
            if logits_aux is not None:
                loss_aux = criterion(logits_aux, base_target)
                loss += model.pre_model.auxiliary_weight * loss_aux
        elif model.name in ['uji', 'hapt']:
            feature = model(base_input)
            loss = criterion(feature, base_target)
        else:
            raise ValueError
        loss.backward()
        nn.utils.clip_grad_norm(model.get_weights(), grad_clip)
        optimizer.step()
        pre1, pre5 = accuracy(feature.data, base_target.data, top_k=(1, 5))
        base_top1.update(pre1.item(), batch)
        base_top5.update(pre5.item(), batch)
        base_loss.update(loss.item(), batch)

        if step % print_freq == 0 or step + 1 == len(train_loader):
            str1 = 'train - epoch:' + epoch_str + ' batch:[' + '{:3d}/{:}]  '.format(step, len(train_loader))
            str2 = '[Loss:{:.6f}  Pre@1:{:.5f}%  Pre@5:{:.5f}%]'.format(base_loss.avg, base_top1.avg,
                                                                        base_top5.avg)
            logging.info(str1 + str2)

    return base_top1.avg, base_top5.avg, base_loss.avg


# 训练过程中一次评估执行的操作
def mode_val(valid_loader, model, criterion, epoch_str, print_freq):
    model.eval()
    val_top1 = AverageMeter()
    val_top5 = AverageMeter()
    val_loss = AverageMeter()
    for step, (val_input, val_target) in enumerate(valid_loader):
        batch = val_input.size(0)
        val_input = val_input.cuda()
        val_target = val_target.cuda()

        if model.name in ['cifar10', 'cifar100']:
            feature, logits_aux = model(val_input)
            loss = criterion(feature, val_target)
            if logits_aux is not None:
                loss_aux = criterion(logits_aux, val_target)
                loss += model.pre_model.auxiliary_weight * loss_aux
        elif model.name in ['uji', 'hapt']:
            feature = model(val_input)
            loss = criterion(feature, val_target)
        else:
            raise ValueError

        pre1, pre5 = accuracy(feature.data, val_target.data, top_k=(1, 5))
        val_top1.update(pre1.item(), batch)
        val_top5.update(pre5.item(), batch)
        val_loss.update(loss.item(), batch)

        if step % print_freq == 0 or step + 1 == len(valid_loader):
            str1 = 'valid - epoch:' + epoch_str + ' batch:[' + '{:3d}/{:}]  '.format(step, len(valid_loader))
            str2 = '[Loss:{:.6f}  Pre@1:{:.5f}%  Pre@5:{:.5f}%]'.format(val_loss.avg, val_top1.avg,
                                                                        val_top5.avg)
            logging.info(str1 + str2)

    return val_top1.avg, val_top5.avg, val_loss.avg


# 训练
def train(args):
    logging.info('start load dataset')
    train_data, test_data, x_shape, class_num = get_src_dataset(args.data_path, args.name)
    _, train_loader, valid_loader = get_search_loader(
        train_data, test_data, args.name, args.split, args.workers, args.batch_size)
    logging.info('dataset loaded')

    model = Network(args.name, x_shape, class_num, args)
    model = model.cuda()
    flop, param = get_model_infos(model, x_shape)
    logging.info('Params={:.2f} MB, FLOPs={:.2f} M'.format(param, flop))

    optimizer, scheduler, criterion = get_opt_scheduler(
        model.get_weights(), args.optimizer, args.lr, args.weight_decay, args.scheduler, args.epoch)
    criterion = criterion.cuda()

    logging.info('optimizer : {:}'.format(optimizer))
    logging.info('scheduler : {:}'.format(scheduler))
    logging.info('criterion   : {:}'.format(criterion))
    logging.info('classifier:\n{:}'.format(model.classifier))

    need_time = AverageMeter()
    time_str = ''
    best_acc = 0
    for epoch in range(1, args.epoch + 1):
        logging.info('epoch:{:} LR:{:.6f} need time {:}'.format(epoch, scheduler.get_lr()[0], time_str))
        if args.name in ['cifar10', 'cifar100']:
            model.set_drop_path_prob(args.drop_path_prob * epoch / args.epoch)
        epoch_str = '[{:03d}/{:03d}]'.format(epoch, args.epoch)

        s_time = time.time()
        train_top1, train_top5, train_loss = model_train(train_loader, model, criterion, optimizer, epoch_str, args.print_frequency, args.grad_clip)
        val_top1, val_top5, val_loss = mode_val(valid_loader, model, criterion, epoch_str, args.print_frequency)
        need_time.update(time.time() - s_time)

        m, s = divmod(need_time.avg * (args.epoch - epoch), 60)
        h, m = divmod(m, 60)
        time_str = "%d:%02d:%02d" % (h, m, s)
        train_str = 'train set - epoch:' + epoch_str + ' result  Loss:'
        vla_str = ' val  set - epoch:' + epoch_str + ' result  Loss:'
        logging.info(train_str + '{:.6f}  Pre@1 : {:.5f}%  Pre@5:{:.5f}%'.format(train_loss, train_top1, train_top5))
        logging.info(vla_str + '{:.6f}  Pre@1 : {:.5f}%  Pre@5:{:.5f}%'.format(val_loss, val_top1, val_top5))

        if val_top1 > best_acc:
            best_acc = val_top1
            logging.info('find the best model. best acc is {:.5f}%'.format(best_acc))
            logging.info('Save it to {:}'.format(args.save + 'best.pt'))
            save(model, os.path.join(args.save, 'best.pt'))

        scheduler.step()

    logging.info('best acc is {:.5f}%'.format(best_acc))


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
    log_config(args)
    start = time.time()
    if args.type == 'search':
        search(args)
    elif args.type == 'train':
        train(args)
    elif args.type == 'test':
        pass
    else:
        raise ValueError
    tot_time = time.time() - start
    m, s = divmod(tot_time, 60)
    h, m = divmod(m, 60)
    logging.info("total time %d:%02d:%02d" % (h, m, s))