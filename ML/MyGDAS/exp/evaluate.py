import time

import torch
import torch.nn as nn
from torch.backends import cudnn

from lib.util.meter import AverageMeter
from lib.util.model_evaluate import get_model_infos, obtain_accuracy
from model.evaluate_model import simple_dense_net, normal_dense_net
from lib.util.starts import prepare_logger, save_checkpoint, copy_checkpoint, prepare_seed
from lib.util.configure_util import load_config

from lib.dataset.get_dataset_with_transform import get_dataset, get_nas_search_loaders


def evaluate(xargs):
    # start cudnn
    cudnn.enabled = True
    # make each conv is the same
    cudnn.benchmark = False
    # make sure the same seed has the same result
    cudnn.deterministic = True
    torch.set_num_threads(xargs.workers)
    prepare_seed(xargs.rand_seed)
    xargs.save_dir += 'eva/'
    logger = prepare_logger(xargs)
    train_data, valid_data, xshape, class_num = get_dataset(xargs.dataset, xargs.data_path, -1)
    logger.log('Train Config:')
    eva_config = load_config(xargs.eva_config, {'class_num': class_num, 'xshape': xshape, 'tau_max': xargs.tau_max,
                                                'tau_min': xargs.tau_min}, logger)
    search_loader, train_loader, test_loader = get_nas_search_loaders(train_data, valid_data, xargs.dataset,
                                                          'config/', eva_config.batch_size, xargs.workers)
    logger.log('dataset: {:} Train-Loader-length={:}, batch size={:}'.format(xargs.dataset, len(train_loader),
                                                                             eva_config.batch_size))
    eva_dense_net(search_loader, test_loader, eva_config, logger, 'normal')
    # eva_dense_net(train_loader, test_loader, eva_config, logger, 'normal')


def eva_dense_net(train_loader, test_loader, config, logger, dense_type):
    logger.log('-' * 200)
    logger.log(("start training {:} net.".format(dense_type)))
    if dense_type == 'simple':
        network = simple_dense_net()
    elif dense_type == 'normal':
        network = normal_dense_net()
    else:
        return
    logger.log("network model:{:}".format(network))
    logger.log("evaluate config:{:}".format(config))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(network.parameters(),
                                lr=config.LR,
                                weight_decay=config.decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                           T_max=config.epochs,
                                                           eta_min=config.eta_min)
    logger.log('optimizer : {:}'.format(optimizer))
    logger.log('scheduler : {:}'.format(scheduler))
    logger.log('criterion   : {:}'.format(criterion))
    flop, param = get_model_infos(network, config.xshape)
    logger.log('FLOP = {:.6f} M, Params = {:.6f} MB'.format(flop, param))
    network = network.cuda()
    criterion = criterion.cuda()
    model_base_path = logger.model_dir / 'seed-{:}-{:}-basic.pth'.format(logger.seed, dense_type)
    model_best_path = logger.model_dir / 'seed-{:}-{:}-best.pth'.format(logger.seed, dense_type)
    training_acc = {'best': -1}
    start_time = time.time()
    for epoch in range(config.epochs):
        epoch_time = time.time()
        scheduler.step()
        epoch_str = '{:03d}-{:03d}'.format(epoch, config.epochs)
        logger.log('\n[Training the {:}-th epoch]'.format(epoch_str))
        loss, top1, top5 = train(train_loader, network, criterion, optimizer, config)
        epoch_time = time.time() - epoch_time
        logger.log('[{:}] train : time using : {:.2f} secs, loss={:.6f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%'.format(
            epoch_str, epoch_time, loss, top1, top5))
        training_acc[epoch] = top1
        if top1 > training_acc['best']:
            training_acc['best'] = top1
            find_best = True
        else:
            find_best = False
        save_checkpoint({'network': network.state_dict()}, model_base_path, logger)
        if find_best:
            logger.log('<<<--->>> The {:}-th epoch : find the highest validation accuracy : {:.2f}%.'.format(epoch_str,
                                                                                                             top1))
            copy_checkpoint(model_base_path, model_best_path, logger)
        test_top1, test_top5 = test(test_loader, network, C_out=12)
        logger.log("----test_top1 : {:.2f}, test_top5 : {:.2f}".format(test_top1, test_top5))
    end_time = time.time()
    logger.log('\n' + '-' * 100)
    logger.log('GDAS : run {:} epochs, best acc is {:.2f}%, using time is {:.2f} secs.'.format(config.epochs,
                                                                                               training_acc['best'],
                                                                                               end_time - start_time))


def train(data_loader, network, criterion, optimizer, config):
    network.train()
    losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()
    for step, (X, Y, X1, Y1) in enumerate(data_loader):
        inputs, targets = X.cuda(), Y.cuda()
        optimizer.zero_grad()
        output = network(inputs)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        prec1, prec5 = obtain_accuracy(output.data, targets.data, topk=(1, min(5, config.C_out)))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
    return losses.avg, top1.avg, top5.avg


def test(test_loader, network, C_out):
    test_top1, test_top5 = AverageMeter(), AverageMeter()
    network.eval()
    with torch.no_grad():
        for X, Y in test_loader:
            test_inputs, test_targets = X.cuda(), Y.cuda()
            output = network(test_inputs)
            import numpy as np
            # print(np.argmax(np.array(output.data.cpu()), axis=1))
            # print(test_targets.data)
            test_prec1, test_prec5 = obtain_accuracy(output.data, test_targets.data, topk=(1, min(5, C_out)))
            test_top1.update(test_prec1.item(), test_inputs.size(0))
            test_top5.update(test_prec5.item(), test_inputs.size(0))
    return test_top1.avg, test_top5.avg
