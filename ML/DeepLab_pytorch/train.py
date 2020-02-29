import argparse
import torch
import torch.nn as nn
import numpy as np
import logging
import time
from tqdm import tqdm
from dataset import data_process
from model.DeepLabV3 import DeepLabV3ResNet
from model.DeepLabV3 import DeepLabV3NasNet
from utils import util

parser = argparse.ArgumentParser(description='args for read data')
parser.add_argument('--image_folder', '-img', type=str,
                    default='dataset/data/VOC_2012/VOCdevkit/VOC2012/JPEGImages/')

parser.add_argument('--semantic_segmentation_folder', '-seg', type=str,
                    default='dataset/data/VOC_2012/VOCdevkit/VOC2012/SegmentationClass/')

parser.add_argument('--train_list', '-train', type=str,
                    default='dataset/data/VOC_2012/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt')

parser.add_argument('--trainval_list', '-tv', type=str,
                    default='dataset/data/VOC_2012/VOCdevkit/VOC2012/ImageSets/Segmentation/trainval.txt')

parser.add_argument('--val_list', '-val', type=str,
                    default='dataset/data/VOC_2012/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt')

parser.add_argument('--res18', type=str, default='pretrain/resnet18/resnet18-5c106cde.pth')

parser.add_argument('--res34', type=str, default='pretrain/resnet34/resnet34-333f7ec4.pth')

parser.add_argument('--nas', type=str, default='pretrain/nasnet/nasnetalarge-a1897284.pth')

parser.add_argument('--log', '-l', type=str,
                    default='log/')

parser.add_argument('--net', '-n', type=str, default='nas')

parser.add_argument('--resized_width', '-w', type=int, default=512)

parser.add_argument('--lr', type=float, default=0.0001)

parser.add_argument('--classes', '-c', type=int, default=21)

parser.add_argument('--os', type=int, default=16)

parser.add_argument('--res_layers', '-r', default=18)

parser.add_argument('--batch_size', '-b', type=int, default=32)

parser.add_argument('--epoch', '-e', type=int, default=1000)

args = parser.parse_args()


def _train(train_loader, val_img, val_seg):
    if args.net == 'nas':
        net = DeepLabV3NasNet(num_class=args.classes, os=args.os, weight_path=args.nas).cuda()
        logging.info('load nas net as backend')
    elif args.net == 'res':
        net = DeepLabV3ResNet(os=args.os, num_class=args.classes, res_layers=args.res_layers,
                              weight_path18=args.res18, weight_path34=args.res34).cuda()
        logging.info('load res net as backend')
    else:
        logging.error('Wrong net name:%s' % args.net)
        raise RuntimeError('Net type must be nas or res.')
    param_decay = util.add_weight_decay(net, l2_value=0.0001)
    optimizer = torch.optim.Adam(param_decay, lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=255)
    logging.info('lr : %d' % args.lr)
    logging.info('batch size : %d' % args.batch_size)
    logging.info('epoch : %d' % args.epoch)
    logging.info('os : %d' % args.os)
    for epoch in tqdm(range(args.epoch)):
        net.train()
        batch_loss = []
        for train_img, train_seg in train_loader:
            train_img_cu = train_img.cuda()
            train_seg_cu = train_seg.type(torch.LongTensor).cuda()

            outputs = net(train_img_cu)

            loss = loss_fn(outputs, train_seg_cu)
            loss_value = loss.data.cpu().numpy()
            batch_loss.append(loss_value)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 5 == 0:
            logging.info('loss : %f' % np.mean(batch_loss))
            hist = np.zeros((args.classes, args.classes))
            for j in range(len(val_img)):
                img_cu = val_img[j].cuda()
                seg = val_seg[j].numpy()
                pre = net(img_cu).detach().cpu().numpy()
                for k in range(len(pre)):
                    pre_seg = np.argmax(pre[k], axis=0)
                    iou = util.compute_IoU(pre_seg, seg[k], args.classes)
                    hist += iou
            MIoU = np.mean(util.per_class_iu(hist))
            logging.info('MIoU : %f' % MIoU)
            checkpoint_path = 'param/DeepLab_epoch' + str(epoch) + '.pth'
            torch.save(net.state_dict(), checkpoint_path)


if __name__ == '__main__':
    now_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename=args.log + str(now_time) + 'DeepLab.log', level=logging.INFO, format=LOG_FORMAT)

    logging.info('start set train loader')
    loader = data_process.get_loader(args.train_list, args.image_folder, args.semantic_segmentation_folder,
                                     args.resized_width, args.batch_size, 4)
    logging.info('train loader set')
    logging.info('start set val loader')
    val_loader = data_process.get_loader(args.val_list, args.image_folder, args.semantic_segmentation_folder,
                                         args.resized_width, args.batch_size, 4)
    logging.info('val loader set')
    val_img = []
    val_seg = []
    logging.info('start read val images and segmentation labels')
    for image, label in val_loader:
        val_img.append(image)
        val_seg.append(label)
    logging.info('read val images and segmentation labels finished')
    _train(loader, val_img, val_seg)
