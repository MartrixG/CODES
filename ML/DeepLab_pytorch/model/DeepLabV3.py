import torch
import torch.nn.functional as F
import torch.nn as nn
from model.ResNet import ResNet18_OS8, ResNet18_OS16, ResNet34_OS8, ResNet34_OS16
from model.ASPP import ASPP
from model.NasNet import NasNetLargeOS16


class DeepLabV3ResNet(nn.Module):
    def __init__(self, os, num_class, res_layers, weight_path18, weight_path34):
        super(DeepLabV3ResNet, self).__init__()
        self.num_class = num_class
        self.os = os
        self.res_layers = res_layers
        if self.res_layers == 18 and self.os == 16:
            self.resnet = ResNet18_OS16(weight_path18)
        elif self.res_layers == 18 and self.os == 8:
            self.resnet = ResNet18_OS8(weight_path18)
        elif self.res_layers == 34 and self.os == 16:
            self.resnet = ResNet34_OS16(weight_path34)
        elif self.res_layers == 34 and self.os == 8:
            self.resnet = ResNet34_OS8(weight_path34)
        self.aspp = ASPP(os=self.os, num_class=self.num_class)

    def forward(self, x):
        h = x.size()[2]
        w = x.size()[3]

        feature_map = self.resnet(x)

        predict_map = self.aspp(feature_map)

        out = F.upsample(predict_map, size=(h, w), mode='bilinear')

        return out


class DeepLabV3NasNet(nn.Module):
    def __init__(self, weight_path, num_class=21, os=16):
        super(DeepLabV3NasNet, self).__init__()
        self.num_class = num_class
        self.os = os
        self.nasnet = NasNetLargeOS16(weight_path)
        self.aspp = ASPP(os=self.os, num_class=self.num_class)

    def forward(self, x):
        h = x.size()[2]
        w = x.size()[3]

        feature_map = self.nasnet(x)

        predict_map = self.aspp(feature_map)

        out = F.upsample(predict_map, size=(h, w), mode='bilinear')

        return out
