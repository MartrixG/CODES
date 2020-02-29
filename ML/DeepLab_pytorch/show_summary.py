import argparse
import numpy as np
from model.DeepLabV3 import DeepLabV3NasNet
from model.DeepLabV3 import DeepLabV3ResNet
from utils import summary

parser = argparse.ArgumentParser(description='args for read data')

parser.add_argument('--classes', '-c', type=int, default=21)

parser.add_argument('--os', type=int, default=16)

parser.add_argument('--res_layers', '-r', type=int, default=18)

parser.add_argument('--res18', type=str, default='../pretrain/resnet18/resnet18-5c106cde.pth')

parser.add_argument('--res34', type=str, default='../pretrain/resnet34/resnet34-333f7ec4.pth')

parser.add_argument('--nas', type=str, default='../pretrain/nasnet/nasnetalarge-a1897284.pth')

parser.add_argument('--net', type=str)

args = parser.parse_args()

net_name = args.net
if net_name == 'nas':
    net = DeepLabV3NasNet(os=args.os, num_class=args.classes, weight_path=args.nas).cuda()
elif net_name == 'res':
    if args.res_layers == 18:
        net = DeepLabV3ResNet(os=args.os, num_class=args.classes, res_layers=18, weight_path=args.res18).cuda()
    elif args.res_layers == 34:
        net = DeepLabV3ResNet(os=args.os, num_class=args.classes, res_layers=34, weight_path=args.res34).cuda()
    else:
        raise RuntimeError('param error.')
else:
    raise RuntimeError('param error.')
# for idx, m in enumerate(net.children()):
    # print(idx, '->', m)
# summary.summary(net, input_size=(3, 512, 512), batch_size=32)


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


print(count_parameters_in_MB(net))



