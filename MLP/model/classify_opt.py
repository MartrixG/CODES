import torch.nn as nn
import torch.nn.functional as F

OPS = {
    # 3x3 Avgpooling         (1)
    'avg_pool_3x3': lambda C_in, C_out: AvgPOOLING(C_in, C_out),
    # 3x3 Maxpooling         (1)
    'max_pool_3x3': lambda C_in, C_out: MaxPOOLING(C_in, C_out),
    # Dense+Activation       (3)
    'dense_layer_relu': lambda C_in, C_out: DenseLayerRelu(C_in, C_out),
    'dense_layer_sigmoid': lambda C_in, C_out: DenseLayerSigmoid(C_in, C_out),
    'dense_layer_tanh': lambda C_in, C_out: DenseLayerTanh(C_in, C_out),
    # GroupDense+Activation  (3)
    'group_dense_4_relu': lambda C_in, C_out: GroupDenseRelu(C_in, C_out, 4),
    'group_dense_4_sigmoid': lambda C_in, C_out: GroupDenseSigmoid(C_in, C_out, 4),
    'group_dense_4_tanh': lambda C_in, C_out: GroupDenseTanh(C_in, C_out, 4),
    # noe                    (1)
    'none': lambda C_in, C_out: Zero(C_in, C_out),

    'ReLUConvBN': lambda C_in, C_out: ReLUConvBN(C_in, C_out)
}


class MaxPOOLING(nn.Module):
    def __init__(self, C_in, C_out):
        super(MaxPOOLING, self).__init__()
        if C_in == C_out:
            stride = (1, 1)
        elif C_in // 2 == C_out:
            stride = (1, 2)
        else:
            raise ValueError("C_in : {:} and C_out : {:} not match.".format(C_in, C_out))
        self.op = nn.MaxPool2d((1, 3), stride=stride, padding=(0, 1))

    def forward(self, x):
        x = x.reshape(x.size(0), 1, 1, x.size(1))
        x = self.op(x)
        x = x.reshape(x.size(0), x.size(3), 1, 1)
        return x


class AvgPOOLING(nn.Module):
    def __init__(self, C_in, C_out):
        super(AvgPOOLING, self).__init__()
        if C_in == C_out:
            stride = (1, 1)
        elif C_in // 2 == C_out:
            stride = (1, 2)
        else:
            raise ValueError("C_in : {:} and C_out : {:} not match.".format(C_in, C_out))
        self.op = nn.AvgPool2d((1, 3), stride=stride, padding=(0, 1), count_include_pad=False)

    def forward(self, x):
        x = x.reshape(x.size(0), 1, 1, x.size(1))
        x = self.op(x)
        x = x.reshape(x.size(0), x.size(3), 1, 1)
        return x


class DenseLayer(nn.Module):
    def __init__(self, C_in, C_out, track_running_stats=True):
        super(DenseLayer, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(C_out, affine=False, track_running_stats=track_running_stats),
        )

    def forward(self, x):
        x = self.op(x)
        return x


class DenseLayerRelu(DenseLayer):
    def __init__(self, C_in, C_out, track_running_stats=True):
        super(DenseLayerRelu, self).__init__(C_in, C_out, track_running_stats)
        self.op.add_module('2', nn.ReLU())
    
    def forward(self, x):
        return super(DenseLayerRelu, self).forward(x)


class DenseLayerSigmoid(DenseLayer):
    def __init__(self, C_in, C_out, track_running_stats=True):
        super(DenseLayerSigmoid, self).__init__(C_in, C_out, track_running_stats)
        self.op.add_module('2', nn.Sigmoid())

    def forward(self, x):
        return super(DenseLayerSigmoid, self).forward(x)


class DenseLayerTanh(DenseLayer):
    def __init__(self, C_in, C_out, track_running_stats=True):
        super(DenseLayerTanh, self).__init__(C_in, C_out, track_running_stats)
        self.op.add_module('2', nn.Tanh())

    def forward(self, x):
        return super(DenseLayerTanh, self).forward(x)


class GroupDENSE(nn.Module):
    def __init__(self, C_in, C_out, complexity, track_running_stats=True):
        super(GroupDENSE, self).__init__()
        assert C_in % complexity == 0
        if C_in == C_out:
            self.C_in = C_in
            self.C_out = C_out
        elif C_in // 2 == C_out:
            self.C_in = C_in
            self.C_out = C_in // 2
        else:
            raise ValueError
        groups = self.C_out // complexity
        self.op = nn.Sequential(
            nn.Conv2d(self.C_in, self.C_out, kernel_size=(1, 1), groups=groups, bias=False),
            nn.BatchNorm2d(self.C_out, affine=False, track_running_stats=track_running_stats),
        )

    def forward(self, x):
        x = self.op(x)
        return x


class GroupDenseRelu(GroupDENSE):
    def __init__(self, C_in, C_out, complexity):
        super(GroupDenseRelu, self).__init__(C_in, C_out, complexity)
        self.op.add_module('2', nn.ReLU())

    def forward(self, x):
        return super(GroupDenseRelu, self).forward(x)


class GroupDenseSigmoid(GroupDENSE):
    def __init__(self, C_in, C_out, complexity):
        super(GroupDenseSigmoid, self).__init__(C_in, C_out, complexity)
        self.op.add_module('2', nn.Sigmoid())

    def forward(self, x):
        return super(GroupDenseSigmoid, self).forward(x)


class GroupDenseTanh(GroupDENSE):
    def __init__(self, C_in, C_out, complexity):
        super(GroupDenseTanh, self).__init__(C_in, C_out, complexity)
        self.op.add_module('2', nn.Tanh())

    def forward(self, x):
        return super(GroupDenseTanh, self).forward(x)


class Zero(nn.Module):
    def __init__(self, C_in, C_out):
        super(Zero, self).__init__()
        self.C_out = C_out
        self.out_dim = C_out

    def forward(self, x):
        shape = list(x.shape)
        shape[1] = self.C_out
        zeros = x.new_zeros(shape, dtype=x.dtype, device=x.device)
        return zeros


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(C_out, affine=True, track_running_stats=True)
        )

    def forward(self, x):
        return self.op(x)
