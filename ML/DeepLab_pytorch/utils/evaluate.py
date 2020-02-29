import numpy as np
import torch
import torchvision.models.segmentation.segmentation
import matplotlib.pyplot as plt
from utils import util
from dataset import data_process
from model.DeepLabV3 import DeepLabV3NasNet

data_list = '../dataset/data/VOC_2012/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt'
image_folder = '../dataset/data/VOC_2012/VOCdevkit/VOC2012/JPEGImages/'
seg_folder = '../dataset/data/VOC_2012/VOCdevkit/VOC2012/SegmentationClass/'
model_param = '../param/DeepLab_epoch170.pth'
log = '../log/2020-02-23-22-30-18DeepLab.log'

train_loader = data_process.get_loader(data_list, image_folder, seg_folder, 512, 8, 0)
num_class = 21


def test_model(param, loader):
    model = DeepLabV3NasNet(os=16, num_class=num_class, weight_path='../pretrain/nasnet/nasnetalarge-a1897284.pth')
    model.load_state_dict(torch.load(param))
    hist = np.zeros((num_class, num_class))
    j = 0
    for img, seg in loader:
        j += 8
        pre = model.forward(img).detach().numpy()
        seg = seg.numpy()
        for i in range(len(img)):
            pre_seg = np.argmax(pre[i], axis=0)
            iou = util.compute_IoU(pre_seg, seg[i], num_class)
            hist += iou
            plt.imshow(pre_seg, cmap='hot')
            plt.show()
            plt.imshow(seg[i], cmap='hot')
            plt.show()
        if j == 8:
            return util.per_class_iu(hist)


def show_los_MIoU(log_path):
    loss = []
    MIoU = []
    f = open(log_path)
    line = f.readline()
    while line:
        if 'loss' in line:
            los = float(line[-8:])
            loss.append(los)
        elif 'MIoU' in line:
            miou = float(line[-8:])
            MIoU.append(miou)
        line = f.readline()
    f.close()
    x = np.linspace(0, 1000, 200, endpoint=False)
    l1 = plt.plot(x, loss, 'r')
    l2 = plt.plot(x, MIoU, 'g')
    plt.legend()
    plt.show()
    print('loss : %f, epoch : %d' % (np.min(loss), np.argmin(loss)))
    print('mIOU : %f, epoch : %d' % (np.max(MIoU), np.argmax(MIoU)))


if __name__ == '__main__':
    test_model(model_param, train_loader)
