import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def make_layer(block, in_channels, channels, num_blocks, stride=1, dilation=1):
    strides = [stride] + [1] * (num_blocks - 1)  # (stride == 2, num_blocks == 4 --> strides == [2, 1, 1, 1])

    blocks = []
    for stride in strides:
        blocks.append(block(in_channels=in_channels, channels=channels, stride=stride, dilation=dilation))
        in_channels = block.expansion * channels

    layer = nn.Sequential(*blocks)

    return layer


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        out_channels = channels * self.expansion
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        if stride != 1 or in_channels != out_channels:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            bn = nn.BatchNorm2d(out_channels)
            self.down_sample = nn.Sequential(conv, bn)
        else:
            self.down_sample = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.down_sample(x)
        out = F.relu(out)
        return out


class ResNetBasicBlockOS16(nn.Module):
    def __init__(self, num_layers, weight_path):
        super(ResNetBasicBlockOS16, self).__init__()
        if num_layers == 18:
            resnet = models.resnet18()
            resnet.load_state_dict(torch.load(weight_path))
            self.resnet = nn.Sequential(*list(resnet.children())[:-3])
            num_blocks = 2
        elif num_layers == 34:
            resnet = models.resnet34()
            resnet.load_state_dict(torch.load(weight_path))
            self.resnet = nn.Sequential(*list(resnet.children())[:-3])
            num_blocks = 3
        else:
            raise Exception("num_layers must be in {18, 34}!")

        self.layer5 = make_layer(BasicBlock, in_channels=256, channels=512, num_blocks=num_blocks, stride=1, dilation=2)

    def forward(self, x):
        out = self.resnet(x)
        out = self.layer5(out)
        return out


class ResNetBasicBlockOS8(nn.Module):
    def __init__(self, num_layers, weight_path):
        super(ResNetBasicBlockOS8, self).__init__()
        if num_layers == 18:
            resnet = models.resnet18()
            resnet.load_state_dict(torch.load(weight_path))
            self.resnet = nn.Sequential(*list(resnet.children())[:-4])
            num_blocks_l4 = 2
            num_blocks_l5 = 2
        elif num_layers == 34:
            resnet = models.resnet34()
            resnet.load_state_dict(torch.load(weight_path))
            self.resnet = nn.Sequential(*list(resnet.children())[:-4])
            num_blocks_l4 = 6
            num_blocks_l5 = 3
        else:
            raise Exception("num_layers must be in {18, 34}!")
        self.layer4 = make_layer(BasicBlock, in_channels=128, channels=256, num_blocks=num_blocks_l4, stride=1, dilation=2)
        self.layer5 = make_layer(BasicBlock, in_channels=256, channels=512, num_blocks=num_blocks_l5, stride=1, dilation=2)

    def forward(self, x):
        out = self.resnet(x)
        out = self.layer4(out)
        out = self.layer5(out)
        return out


def ResNet18_OS16(weight_path):
    return ResNetBasicBlockOS16(num_layers=18, weight_path=weight_path)


def ResNet34_OS16(weight_path):
    return ResNetBasicBlockOS16(num_layers=34, weight_path=weight_path)


def ResNet18_OS8(weight_path):
    return ResNetBasicBlockOS8(num_layers=18, weight_path=weight_path)


def ResNet34_OS8(weight_path):
    return ResNetBasicBlockOS8(num_layers=34, weight_path=weight_path)