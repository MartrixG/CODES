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

HAPT_SPACE = ['none', 'avg_pool_3x3', 'avg_pool_5x5', 'avg_pool_7x7', 'enhance_avg_pool_3x3',
              'enhance_avg_pool_5x5', 'enhance_avg_pool_7x7', 'group_dense_2', 'group_dense_3', 'group_dense_4',
              'group_dense_5', 'enhance_group_dense_2', 'enhance_group_dense_3', 'enhance_group_dense_4',
              'enhance_group_dense_5', 'dense_layer']


def search(arch_config, data_loader, network, criterion, w_optimizer, a_optimizer, print_frequency, epoch_str, logger):
    batch_time = AverageMeter()
    base_losses, base_top1, base_top5 = AverageMeter(), AverageMeter(), AverageMeter()
    arch_losses, arch_top1, arch_top5 = AverageMeter(), AverageMeter(), AverageMeter()
    network.train()
    end = time.time()
    for step, (X, Y, X1, Y1) in enumerate(data_loader):
        base_inputs, base_targets = X.cuda(), Y.cuda()
        arch_inputs, arch_targets = X1.cuda(), Y1.cuda()
        # update the weights
        w_optimizer.zero_grad()
        output = network(base_inputs)
        base_loss = criterion(output, base_targets)
        base_loss.backward()
        clip_grad_norm_(network.parameters(), 5)
        w_optimizer.step()
        # record
        base_prec1, base_prec5 = obtain_accuracy(output.data, base_targets.data, topk=(1, min(5, arch_config.C_out)))
        base_losses.update(base_loss.item(), base_inputs.size(0))
        base_top1.update(base_prec1.item(), base_inputs.size(0))
        base_top5.update(base_prec5.item(), base_inputs.size(0))

        # update the architecture-weight
        a_optimizer.zero_grad()
        output = network(arch_inputs)
        arch_loss = criterion(output, arch_targets)
        arch_loss.backward()
        a_optimizer.step()
        # record
        arch_prec1, arch_prec5 = obtain_accuracy(output.data, arch_targets.data, topk=(1, min(5, arch_config.C_out)))
        arch_losses.update(arch_loss.item(), arch_inputs.size(0))
        arch_top1.update(arch_prec1.item(), arch_inputs.size(0))
        arch_top5.update(arch_prec5.item(), arch_inputs.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        if step % print_frequency == 0 or step + 1 == len(data_loader):
            str1 = "SEARCHING***" + "[{:}][{:}/{:}]".format(epoch_str, step, len(data_loader))
            str2 = "Time now step:{batch_time.val:.2f} avg:{batch_time.avg:.2f}".format(batch_time=batch_time)
            str3 = "Base [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]".format(
                loss=base_losses, top1=base_top1, top5=base_top5)
            str4 = 'Arch [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]'.format(
                loss=arch_losses, top1=arch_top1, top5=arch_top5)
            logger.log(str1 + ' ' + str2 + ' ' + str3 + ' ' + str4)
    return base_losses.avg, base_top1.avg, base_top5.avg, arch_losses.avg, arch_top1.avg, arch_top5.avg


def test(test_loader, network, C_out):
    test_top1, test_top5 = AverageMeter(), AverageMeter()
    network.eval()
    for X, Y in test_loader:
        test_inputs, test_targets = X.cuda(), Y.cuda()
        output = network(test_inputs)
        test_prec1, test_prec5 = obtain_accuracy(output.data, test_targets.data, topk=(1, min(5, C_out)))
        test_top1.update(test_prec1.item(), test_inputs.size(0))
        test_top5.update(test_prec5.item(), test_inputs.size(0))
    print("***TEST result***" + "accuracy@1 : {:.2f}%, accuracy@5 : {:.2f}%".format(test_top1.avg, test_top5.avg))


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
    logger.log('Train Config:')
    opt_config = load_config(xargs.opt_config, {'class_num': class_num, 'xshape': xshape,
                                                'batch_size': xargs.batch_size, 'epochs': xargs.epochs,
                                                'LR': xargs.opt_learning_rate}, logger)
    search_loader, _, valid_loader = get_nas_search_loaders(train_data, valid_data, xargs.dataset,
                                                            'config/', opt_config.batch_size, xargs.workers)
    logger.log('dataset: {:} Search-Loader-length={:}, batch size={:}'.format(xargs.dataset, len(search_loader),
                                                                              opt_config.batch_size))
    logger.log('Arch Config:')
    arch_config = load_config(xargs.arch_config, {'class_num': class_num,
                                                  'space': HAPT_SPACE,
                                                  'affine': False,
                                                  'track_running_stats': bool(xargs.track_running_stats)}, logger)
    if xargs.dataset == 'HAPT':
        search_model = DNNModel(config=arch_config, logger=logger)
    elif xargs.dataset in ('cifar10', 'cifar100'):
        search_model = DNNModel(config=arch_config, logger=logger)
    else:
        raise NameError("dataset must be in \"HAPT\", \"cifar100\", \"cifar100\"")
    if xargs.evaluate == 'test':
        search_model.load_state_dict(torch.load(logger.path('best'))['network'])
        network = search_model.cuda()
        test_loader = valid_loader
        test(test_loader, network, arch_config.C_out)
        return
    logger.log('search-model :\n{:}'.format(search_model))
    logger.log('model-config : {:}'.format(arch_config))
    if opt_config.criterion == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    else:
        raise NameError('unknown loss function {:}'.format(opt_config.criterion))
    # criterion = nn.MSELoss()
    w_optimizer = torch.optim.SGD(params=search_model.get_weights(),
                                  lr=opt_config.LR,
                                  weight_decay=opt_config.w_decay)
    w_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=w_optimizer,
                                                             T_max=opt_config.epochs,
                                                             eta_min=opt_config.eta_min)
    a_optimizer = torch.optim.Adam(params=search_model.get_alphas(),
                                   lr=opt_config.LR,
                                   betas=(0.5, 0.999),
                                   weight_decay=opt_config.a_decay)
    logger.log('w-optimizer : {:}'.format(w_optimizer))
    logger.log('a-optimizer : {:}'.format(a_optimizer))
    logger.log('w-scheduler : {:}'.format(w_scheduler))
    logger.log('criterion   : {:}'.format(criterion))
    flop, param = get_model_infos(search_model, xshape)
    logger.log('FLOP = {:.6f} M, Params = {:.6f} MB'.format(flop, param))
    logger.log('search-space [{:} ops] : {:}'.format(len(HAPT_SPACE), HAPT_SPACE))
    last_info, model_base_path, model_best_path = logger.path('info'), logger.path('model'), logger.path('best')
    # network, criterion = torch.nn.DataParallel(search_model).cuda(), criterion.cuda()
    network = search_model.cuda()
    criterion = criterion.cuda()
    if last_info.exists():
        # automatically resume from previous checkpoint
        logger.log("=> loading checkpoint of the last-info '{:}' start".format(last_info))
        last_info = torch.load(last_info)
        start_epoch = last_info['epoch']
        checkpoint = torch.load(last_info['last_checkpoint'])
        genotypes = checkpoint['genotypes']
        valid_accuracies = checkpoint['valid_accuracies']
        network.load_state_dict(checkpoint['network'])
        w_scheduler.load_state_dict(checkpoint['w_scheduler'])
        w_optimizer.load_state_dict(checkpoint['w_optimizer'])
        a_optimizer.load_state_dict(checkpoint['a_optimizer'])
        logger.log(
            "=> loading checkpoint of the last-info '{:}' start with {:}-th epoch.".format(last_info, start_epoch))
    else:
        logger.log("=> do not find the last-info file : {:}".format(last_info))
        start_epoch, valid_accuracies, genotypes = 0, {'best': -1}, {
            -1: network.genotype(xargs.save_dir + xargs.genotype_file)}
    # start_time, search_time, epoch_time = time.time(), AverageMeter(), AverageMeter()
    total_epoch = opt_config.epochs
    for epoch in range(start_epoch, total_epoch):
        # w_scheduler.update(epoch, 0.0)
        w_scheduler.step()
        # need_time = 'Time Left: {:}'.format(convert_secs2time(epoch_time.val * (total_epoch - epoch), True))
        epoch_str = '{:03d}-{:03d}'.format(epoch, total_epoch)
        network.set_tau(xargs.tau_max - (xargs.tau_max - xargs.tau_min) * epoch / (total_epoch - 1))
        logger.log('\n[Search the {:}-th epoch] tau={:}'.format(epoch_str, network.get_tau()))
        base_losses, base_top1, base_top5, arch_losses, arch_top1, arch_top5 = \
            search(arch_config, search_loader, network, criterion, w_optimizer, a_optimizer,
                   xargs.print_frequency, epoch_str, logger)
        logger.log('[{:}] searching : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%'.format(
            epoch_str, base_losses, base_top1, base_top5))
        logger.log(
            '[{:}] evaluate  : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%'.format(epoch_str, arch_losses,
                                                                                           arch_top1, arch_top5))
        valid_accuracies[epoch] = arch_top1
        if arch_top1 > valid_accuracies['best']:
            valid_accuracies['best'] = arch_top1
            genotypes['best'] = network.genotype(xargs.save_dir + xargs.genotype_file)
            find_best = True
        else:
            find_best = False
        genotypes[epoch] = network.genotype(xargs.save_dir + xargs.genotype_file)
        logger.log('<<<--->>> The {:}-th epoch : {:}'.format(epoch_str, genotypes[epoch]))
        save_path = save_checkpoint({'epoch': epoch + 1,
                                     'args': deepcopy(xargs),
                                     'network': network.state_dict(),
                                     'w_optimizer': w_optimizer.state_dict(),
                                     'a_optimizer': a_optimizer.state_dict(),
                                     'w_scheduler': w_scheduler.state_dict(),
                                     'genotypes': genotypes,
                                     'valid_accuracies': valid_accuracies},
                                    model_base_path, logger)
        last_info = save_checkpoint({
            'epoch': epoch + 1,
            'args': deepcopy(xargs),
            'last_checkpoint': save_path,
        }, logger.path('info'), logger)
        if find_best:
            logger.log('<<<--->>> The {:}-th epoch : find the highest validation accuracy : {:.2f}%.'.format(epoch_str,
                                                                                                             arch_top1))
            copy_checkpoint(model_base_path, model_best_path, logger)
        with torch.no_grad():
            logger.log('{:}'.format(network.show_alphas()))
    logger.log('\n' + '-' * 100)
    # check the performance from the architecture dataset
    logger.log('GDAS : run {:} epochs, last-geno is {:}.'.format(total_epoch, genotypes[total_epoch - 1]))
