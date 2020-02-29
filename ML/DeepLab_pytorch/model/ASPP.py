import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPP(nn.Module):
    def __init__(self, os=16, num_class=21):
        super(ASPP, self).__init__()
        self.os = os
        self.conv_0 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv_0_bn = nn.BatchNorm2d(256)
        if self.os == 16:
            self.conv_1 = nn.Conv2d(512, 256, kernel_size=3, padding=6, dilation=6)
            self.conv_1_bn = nn.BatchNorm2d(256)

            self.conv_2 = nn.Conv2d(512, 256, kernel_size=3, padding=12, dilation=12)
            self.conv_2_bn = nn.BatchNorm2d(256)

            self.conv_3 = nn.Conv2d(512, 256, kernel_size=3, padding=18, dilation=18)
            self.conv_3_bn = nn.BatchNorm2d(256)
        elif self.os == 8:
            self.conv_1 = nn.Conv2d(512, 256, kernel_size=3, padding=12, dilation=12)
            self.conv_1_bn = nn.BatchNorm2d(256)

            self.conv_2 = nn.Conv2d(512, 256, kernel_size=3, padding=24, dilation=24)
            self.conv_2_bn = nn.BatchNorm2d(256)

            self.conv_3 = nn.Conv2d(512, 256, kernel_size=3, padding=36, dilation=36)
            self.conv_3_bn = nn.BatchNorm2d(256)
        else:
            raise Exception("os must be in {8, 16}!")
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.pool_conv = nn.Conv2d(512, 256, kernel_size=1)
        self.pool_conv_bn = nn.BatchNorm2d(256)

        self.cat_conv = nn.Conv2d(1280, 256, kernel_size=1)
        self.cat_conv_bn = nn.BatchNorm2d(256)

        self.classify = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, feature_map):
        h = feature_map.size()[2]
        w = feature_map.size()[3]

        b0 = F.relu(self.conv_0_bn(self.conv_0(feature_map)))
        b1 = F.relu(self.conv_1_bn(self.conv_1(feature_map)))
        b2 = F.relu(self.conv_2_bn(self.conv_2(feature_map)))
        b3 = F.relu(self.conv_3_bn(self.conv_3(feature_map)))

        avg_img = self.pool(feature_map)
        avg_img = F.relu(self.pool_conv_bn(self.pool_conv(avg_img)))
        out_img = F.upsample(avg_img, size=(h, w), mode='bilinear')

        out = torch.cat([b0, b1, b2, b3, out_img], 1)
        out = F.relu(self.cat_conv_bn(self.cat_conv(out)))
        out = self.classify(out)

        return out
