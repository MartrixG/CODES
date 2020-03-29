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

HAPT_SPACE = ['avg_pool_3x3', 'avg_pool_5x5', 'avg_pool_7x7', 'enhance_avg_pool_3x3', 'enhance_avg_pool_5x5',
              'enhance_avg_pool_7x7',
              'group_dense_2', 'group_dense_3', 'group_dense_4', 'group_dense_5',
              'enhance_group_dense_2', 'enhance_group_dense_3', 'enhance_group_dense_4', 'enhance_group_dense_5']


def search(arch_config, data_loader, network, criterion, w_optimizer, a_optimizer, print_frequency, epoch_str, logger, record_file, genotype_file):
    batch_time = AverageMeter()
    base_losses, base_top1, base_top5 = AverageMeter(), AverageMeter(), AverageMeter()
    arch_losses, arch_top1, arch_top5 = AverageMeter(), AverageMeter(), AverageMeter()
    network.train()
    end = time.time()
    for step, (X, Y, X1, Y1) in enumerate(data_loader):
        base_inputs, base_targets = X, Y
        arch_inputs, arch_targets = X1, Y1
        # update the weights
        w_optimizer.zero_grad()
        output = network(base_inputs)
        base_loss = criterion(output, base_targets)
        base_loss.backward()
        clip_grad_norm_(network.parameters(), 5)
        w_optimizer.step()
        # record
        base_prec1, base_prec5 = obtain_accuracy(output.data, base_targets.data, topk=(1, min(5, arch_config.C_out)))
        logger.log('base_prec1: {:}'.format(str(base_prec1)))
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
    with open(record_file, "a+") as f:
        f.write("Results for epoch " + str(epoch_str) + "\n")
        f.write("\t" + str([base_losses.avg, base_top1.avg, base_top5.avg, arch_losses.avg, arch_top1.avg, arch_top5.avg]) + "\n")
        f.write(str(output[:3]) + "\n")
        f.write(str(arch_targets[:3]) + "\n")
        f.write("\tsearch_model.genotype: " + str(network.genotype(genotype_file)) + "\n\n")
    return base_losses.avg, base_top1.avg, base_top5.avg, arch_losses.avg, arch_top1.avg, arch_top5.avg


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
                                                            'config/', opt_config.batch_size, xargs.workers)
    logger.log('||||||| {:10s} ||||||| Search-Loader-Num={:}, batch size={:}'.format(xargs.dataset, len(search_loader),
                                                                                     opt_config.batch_size))
    logger.log('||||||| {:10s} ||||||| Config={:}'.format(xargs.dataset, opt_config))
    arch_config = load_config(xargs.arch_config, {'class_num': class_num,
                                                  'space': HAPT_SPACE,
                                                  'affine': False,
                                                  'track_running_stats': bool(xargs.track_running_stats)}, None)
    if xargs.dataset == 'HAPT':
        search_model = DNNModel(config=arch_config)
    elif xargs.dataset in ('cifar10', 'cifar100'):
        search_model = DNNModel(config=arch_config)
    else:
        raise NameError("dataset must be in \"HAPT\", \"cifar100\", \"cifar100\"")
    logger.log('search-model :\n{:}'.format(search_model))
    logger.log('model-config : {:}'.format(arch_config))
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()
    criterion = criterion.cuda()
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
    logger.log('FLOP = {:.2f} M, Params = {:.2f} MB'.format(flop, param))
    logger.log('search-space [{:} ops] : {:}'.format(len(HAPT_SPACE), HAPT_SPACE))
    last_info, model_base_path, model_best_path = logger.path('info'), logger.path('model'), logger.path('best')
    network, criterion = torch.nn.DataParallel(search_model).cuda(), criterion.cuda()
    if last_info.exists():
        # automatically resume from previous checkpoint
        logger.log("=> loading checkpoint of the last-info '{:}' start".format(last_info))
        last_info = torch.load(last_info)
        start_epoch = last_info['epoch']
        checkpoint = torch.load(last_info['last_checkpoint'])
        genotypes = checkpoint['genotypes']
        valid_accuracies = checkpoint['valid_accuracies']
        network.load_state_dict(checkpoint['search_model'])
        w_scheduler.load_state_dict(checkpoint['w_scheduler'])
        w_optimizer.load_state_dict(checkpoint['w_optimizer'])
        a_optimizer.load_state_dict(checkpoint['a_optimizer'])
        logger.log(
            "=> loading checkpoint of the last-info '{:}' start with {:}-th epoch.".format(last_info, start_epoch))
    else:
        logger.log("=> do not find the last-info file : {:}".format(last_info))
        start_epoch, valid_accuracies, genotypes = 0, {'best': -1}, {-1: network.genotype()}
    start_time, search_time, epoch_time, total_epoch = time.time(), AverageMeter(), AverageMeter(), opt_config.epochs
    for epoch in range(start_epoch, total_epoch):
        # w_scheduler.update(epoch, 0.0)
        w_scheduler.step()
        need_time = 'Time Left: {:}'.format(convert_secs2time(epoch_time.val * (total_epoch - epoch), True))
        epoch_str = '{:03d}-{:03d}'.format(epoch, total_epoch)
        network.set_tau(xargs.tau_max - (xargs.tau_max - xargs.tau_min) * epoch / (total_epoch - 1))
        logger.log('\n[Search the {:}-th epoch] {:}, tau={:}'.format(epoch_str, need_time, network.get_tau()))
        base_losses, base_top1, base_top5, arch_losses, arch_top1, arch_top5 = \
            search(arch_config, search_loader, network, criterion, w_optimizer, a_optimizer,
                   xargs.print_frequency, epoch_str, logger, xargs.log + xargs.record_file, xargs.log + xargs.genotype_file)
    logger.log("\n\nFinal Results: " + str([base_losses, base_top1, base_top5, arch_losses, arch_top1, arch_top5]))
