import torch.nn as nn
import torch.nn.functional as F

OPS = {
    'avg_pool': lambda C_in, C_out, track_running_stats: AvgPOOLING(C_in, C_out, 3,
                                                                    track_running_stats),
    'enhance_avg_pool': lambda C_in, C_out, track_running_stats: EnAvgPOOLING(C_in, C_out, 3,
                                                                              track_running_stats),
    'group_dense': lambda C_in, C_out, track_running_stats: GroupDENSE(C_in, C_out, 3,
                                                                       track_running_stats),
    'enhance_group_dense': lambda C_in, C_out, track_running_stats: EnhanceGroupDENSE(C_in, C_out, 3,
                                                                                      track_running_stats),
    'dense_layer': lambda C_in, C_out, track_running_stats: DenseLayer(C_in, C_out, track_running_stats),
    'linear': lambda C_in, C_out: LinearLayer(C_in, C_out)
}


class AvgPOOLING(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, track_running_stats=True):
        super(AvgPOOLING, self).__init__()
        if kernel_size == 3:
            padding = (0, 1)
            kernel_size = (1, 3)
        elif kernel_size == 5:
            padding = (0, 2)
            kernel_size = (1, 5)
        elif kernel_size == 7:
            padding = (0, 3)
            kernel_size = (1, 7)
        else:
            padding = (0, 0)
            kernel_size = (1, 1)
        self.op = nn.Sequential(
            nn.AvgPool2d(kernel_size, stride=(1, 2), padding=padding, count_include_pad=False)
        )
        self.out_dim = (C_in + 1) // 2
        self.C_out = C_out

    def forward(self, x):
        x = x.reshape(x.size(0), 1, 1, x.size(1))
        x = self.op(x)
        x = x.reshape(x.size(0), x.size(3), 1, 1)
        if x.size(1) < self.C_out:
            added = self.C_out - x.size(1)
            x = F.pad(x, [0, 0, 0, 0, 0, added], "constant", 0)
        return x


class EnAvgPOOLING(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, track_running_stats=True):
        super(EnAvgPOOLING, self).__init__()
        if kernel_size == 3:
            padding = (0, 1)
            kernel_size = (1, 3)
        elif kernel_size == 5:
            padding = (0, 2)
            kernel_size = (1, 5)
        elif kernel_size == 5:
            padding = (0, 3)
            kernel_size = (1, 7)
        else:
            padding = (0, 0)
            kernel_size = (1, 1)
        self.en = nn.Conv2d(C_in, C_in, (1, 1), groups=C_in, bias=False)
        self.op = nn.Sequential(
            nn.AvgPool2d(kernel_size, stride=(1, 2), padding=padding, count_include_pad=False)
        )
        self.out_dim = (C_in + 1) // 2
        self.C_out = C_out

    def forward(self, x):
        x = self.en(x)
        x = x.reshape(x.size(0), 1, 1, x.size(1))
        x = self.op(x)
        x = x.reshape(x.size(0), x.size(3), 1, 1)
        if x.size(1) < self.C_out:
            added = self.C_out - x.size(1)
            x = F.pad(x, [0, 0, 0, 0, 0, added], "constant", 0)
        return x


class GroupDENSE(nn.Module):
    def __init__(self, C_in, C_out, complexity, track_running_stats=True):
        super(GroupDENSE, self).__init__()
        if C_in > C_in // (2 * complexity) * (2 * complexity):
            C_in = (C_in // (2 * complexity) + 1) * (2 * complexity)
        _C_out = C_in // 2
        groups = _C_out // complexity
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, _C_out, kernel_size=(1, 1), groups=groups, bias=False),
            nn.BatchNorm2d(_C_out, affine=False, track_running_stats=track_running_stats),
        )
        self.complexity = complexity
        self.out_dim = _C_out
        self.C_out = C_out

    def forward(self, x):
        if x.size(1) > x.size(1) // (2 * self.complexity) * (2 * self.complexity):
            added = (x.size(1) // (2 * self.complexity) + 1) * (2 * self.complexity) - x.size(1)
            x = F.pad(x, [0, 0, 0, 0, 0, added], "constant", 0)
        x = self.op(x)
        if x.size(1) < self.C_out:
            added = self.C_out - x.size(1)
            x = F.pad(x, [0, 0, 0, 0, 0, added], "constant", 0)
        return x


class EnhanceGroupDENSE(nn.Module):
    def __init__(self, C_in, C_out, complexity, track_running_stats=True):
        super(EnhanceGroupDENSE, self).__init__()
        self.en = nn.Conv2d(C_in, C_in, (1, 1), groups=C_in, bias=False)
        if C_in > C_in // (2 * complexity) * (2 * complexity):
            C_in = (C_in // (2 * complexity) + 1) * (2 * complexity)
        _C_out = C_in // 2
        groups = _C_out // complexity
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, _C_out, kernel_size=(1, 1), groups=groups, bias=False),
            nn.BatchNorm2d(_C_out, affine=False, track_running_stats=track_running_stats),
        )
        self.complexity = complexity
        self.out_dim = _C_out
        self.C_out = C_out

    def forward(self, x):
        # print(x.size())
        # print("+++++++")
        x = self.en(x)
        # print(x.size())
        # print("@@@@@@@@@@")
        if x.size(1) > x.size(1) // (2 * self.complexity) * (2 * self.complexity):
            added = (x.size(1) // (2 * self.complexity) + 1) * (2 * self.complexity) - x.size(1)
            x = F.pad(x, [0, 0, 0, 0, 0, added], "constant", 0)
        x = self.op(x)
        if x.size(1) < self.C_out:
            added = self.C_out - x.size(1)
            x = F.pad(x, [0, 0, 0, 0, 0, added], "constant", 0)
        return x


class DenseLayer(nn.Module):
    def __init__(self, C_in, C_out, track_running_stats=True):
        super(DenseLayer, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(C_out, affine=False, track_running_stats=track_running_stats),
        )
        self.out_dim = (C_in + 1) // 2
        self.C_out = C_out

    def forward(self, x):
        x = self.op(x)
        return x


class LinearLayer(nn.Module):
    def __init__(self, c_in, c_out):
        super(LinearLayer, self).__init__()
        self.op = nn.Linear(c_in, c_out)

    def forward(self, x):
        x = self.op(x)
        return x
