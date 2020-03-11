import sys
import time
from copy import deepcopy
from pathlib import Path

import torch

from lib.util.starts import prepare_seed, prepare_logger, save_checkpoint, copy_checkpoint
from lib.util.configure_util import load_config
from lib.util.optimizers import get_optim_scheduler
from lib.util.model_evaluate import get_model_infos, obtain_accuracy
from lib.util.meter import AverageMeter
from lib.util.time_util import convert_secs2time, time_string
from lib.dataset.get_dataset_with_transform import get_dataset, get_nas_search_loaders
from model import get_cell_based_tiny_net


from torch.backends import cudnn

DARTS_SPACE = ['none',
               'skip_connect',
               'dua_sepc_3x3',
               'dua_sepc_5x5',
               'dil_sepc_3x3',
               'dil_sepc_5x5',
               'avg_pool_3x3',
               'max_pool_3x3']


def search_func(xloader, network, criterion, scheduler, w_optimizer, a_optimizer, epoch_str, print_freq, logger):
    data_time, batch_time = AverageMeter(), AverageMeter()
    base_losses, base_top1, base_top5 = AverageMeter(), AverageMeter(), AverageMeter()
    arch_losses, arch_top1, arch_top5 = AverageMeter(), AverageMeter(), AverageMeter()
    network.train()
    end = time.time()
    for step, (base_inputs, base_targets, arch_inputs, arch_targets) in enumerate(xloader):
        scheduler.update(None, 1.0 * step / len(xloader))
        base_targets = base_targets.cuda(non_blocking=True)
        arch_targets = arch_targets.cuda(non_blocking=True)
        # measure data loading time
        data_time.update(time.time() - end)

        # update the weights
        w_optimizer.zero_grad()
        _, logits = network(base_inputs)
        base_loss = criterion(logits, base_targets)
        base_loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
        w_optimizer.step()
        # record
        base_prec1, base_prec5 = obtain_accuracy(logits.data, base_targets.data, topk=(1, 5))
        base_losses.update(base_loss.item(), base_inputs.size(0))
        base_top1.update(base_prec1.item(), base_inputs.size(0))
        base_top5.update(base_prec5.item(), base_inputs.size(0))

        # update the architecture-weight
        a_optimizer.zero_grad()
        _, logits = network(arch_inputs)
        arch_loss = criterion(logits, arch_targets)
        arch_loss.backward()
        a_optimizer.step()
        # record
        arch_prec1, arch_prec5 = obtain_accuracy(logits.data, arch_targets.data, topk=(1, 5))
        arch_losses.update(arch_loss.item(), arch_inputs.size(0))
        arch_top1.update(arch_prec1.item(), arch_inputs.size(0))
        arch_top5.update(arch_prec5.item(), arch_inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if step % print_freq == 0 or step + 1 == len(xloader):
            Sstr = '*SEARCH* ' + time_string() + ' [{:}][{:03d}/{:03d}]'.format(epoch_str, step, len(xloader))
            Tstr = 'Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})'.format(
                batch_time=batch_time, data_time=data_time)
            Wstr = 'Base [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]'.format(
                loss=base_losses, top1=base_top1, top5=base_top5)
            Astr = 'Arch [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]'.format(
                loss=arch_losses, top1=arch_top1, top5=arch_top5)
            logger.log(Sstr + ' ' + Tstr + ' ' + Wstr + ' ' + Astr)
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
    # get original data
    train_data, valid_data, xshape, class_num = get_dataset(xargs.dataset, xargs.data_path, -1)
    config = load_config(xargs.config_path, {'class_num': class_num, 'xshape': xshape}, logger)
    search_loader, _, valid_loader = get_nas_search_loaders(train_data, valid_data, xargs.dataset,
                                                            'config/', config.batch_size, xargs.workers)
    logger.log('||||||| {:10s} ||||||| Search-Loader-Num={:}, batch size={:}'.format(xargs.dataset, len(search_loader),
                                                                                     config.batch_size))
    logger.log('||||||| {:10s} ||||||| Config={:}'.format(xargs.dataset, config))
    search_space = DARTS_SPACE
    model_config = load_config(xargs.model_config, {'num_classes': class_num,
                                                    'space': search_space,
                                                    'affine': False,
                                                    'track_running_stats': bool(xargs.track_running_stats)}, None)
    search_model = get_cell_based_tiny_net(model_config)
    # logger.log('search-model :\n{:}'.format(search_model))
    # logger.log('model-config : {:}'.format(model_config))

    w_optimizer, w_scheduler, criterion = get_optim_scheduler(search_model.get_weights(), config)
    a_optimizer = torch.optim.Adam(search_model.get_alphas(), lr=xargs.arch_learning_rate, betas=(0.5, 0.999),
                                   weight_decay=xargs.arch_weight_decay)
    # logger.log('w-optimizer : {:}'.format(w_optimizer))
    # logger.log('a-optimizer : {:}'.format(a_optimizer))
    # logger.log('w-scheduler : {:}'.format(w_scheduler))
    # logger.log('criterion   : {:}'.format(criterion))
    # flop, param = get_model_infos(search_model, xshape)
    # logger.log('FLOP = {:.2f} M, Params = {:.2f} MB'.format(flop, param))
    logger.log('search-space [{:} ops] : {:}'.format(len(search_space), search_space))
    if xargs.arch_nas_dataset is None:
        api = None
    else:
        pass
        # api = API(xargs.arch_nas_dataset)
    last_info, model_base_path, model_best_path = logger.path('info'), logger.path('model'), logger.path('best')
    # network, criterion = torch.nn.DataParallel(search_model).cuda(), criterion.cuda()
    network = search_model

    if last_info.exists():
        # automatically resume from previous checkpoint
        logger.log("=> loading checkpoint of the last-info '{:}' start".format(last_info))
        last_info = torch.load(last_info)
        start_epoch = last_info['epoch']
        checkpoint = torch.load(last_info['last_checkpoint'])
        genotypes = checkpoint['genotypes']
        valid_accuracies = checkpoint['valid_accuracies']
        search_model.load_state_dict(checkpoint['search_model'])
        w_scheduler.load_state_dict(checkpoint['w_scheduler'])
        w_optimizer.load_state_dict(checkpoint['w_optimizer'])
        a_optimizer.load_state_dict(checkpoint['a_optimizer'])
        logger.log(
            "=> loading checkpoint of the last-info '{:}' start with {:}-th epoch.".format(last_info, start_epoch))
    else:
        logger.log("=> do not find the last-info file : {:}".format(last_info))
        start_epoch, valid_accuracies, genotypes = 0, {'best': -1}, {-1: search_model.genotype()}
    start_time, search_time, epoch_time, total_epoch = time.time(), AverageMeter(), AverageMeter(), config.epochs + config.warmup
    for epoch in range(start_epoch, total_epoch):
        w_scheduler.update(epoch, 0.0)
        need_time = 'Time Left: {:}'.format(convert_secs2time(epoch_time.val * (total_epoch - epoch), True))
        epoch_str = '{:03d}-{:03d}'.format(epoch, total_epoch)
        search_model.set_tau(xargs.tau_max - (xargs.tau_max - xargs.tau_min) * epoch / (total_epoch - 1))
        logger.log(
            '\n[Search the {:}-th epoch] {:}, tau={:}, LR={:}'.format(epoch_str, need_time, search_model.get_tau(),
                                                                      min(w_scheduler.get_lr())))
        search_w_loss, search_w_top1, search_w_top5, valid_a_loss, valid_a_top1, valid_a_top5 \
            = search_func(search_loader, network, criterion, w_scheduler, w_optimizer, a_optimizer, epoch_str,
                          xargs.print_freq, logger)
        search_time.update(time.time() - start_time)
        logger.log('[{:}] searching : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%, time-cost={:.1f} s'.format(
            epoch_str, search_w_loss, search_w_top1, search_w_top5, search_time.sum))
        logger.log(
            '[{:}] evaluate  : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%'.format(epoch_str, valid_a_loss,
                                                                                           valid_a_top1, valid_a_top5))
        # check the best accuracy
        valid_accuracies[epoch] = valid_a_top1
        if valid_a_top1 > valid_accuracies['best']:
            valid_accuracies['best'] = valid_a_top1
            genotypes['best'] = search_model.genotype()
            find_best = True
        else:
            find_best = False

        genotypes[epoch] = search_model.genotype()
        logger.log('<<<--->>> The {:}-th epoch : {:}'.format(epoch_str, genotypes[epoch]))
        # save checkpoint
        save_path = save_checkpoint({'epoch': epoch + 1,
                                     'args': deepcopy(xargs),
                                     'search_model': search_model.state_dict(),
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
                                                                                                             valid_a_top1))
            copy_checkpoint(model_base_path, model_best_path, logger)
        with torch.no_grad():
            logger.log('{:}'.format(search_model.show_alphas()))
        # if api is not None: logger.log('{:}'.format(api.query_by_arch(genotypes[epoch])))
        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
    logger.log('\n' + '-' * 100)
    # check the performance from the architecture dataset
    logger.log('GDAS : run {:} epochs, cost {:.1f} s, last-geno is {:}.'.format(total_epoch, search_time.sum,
                                                                                genotypes[total_epoch - 1]))
    # if api is not None: logger.log('{:}'.format(api.query_by_arch(genotypes[total_epoch - 1])))
    logger.close()
