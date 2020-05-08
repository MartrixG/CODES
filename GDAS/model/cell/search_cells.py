from model.cell.cell_operations import OPS
from copy import deepcopy
import torch.nn as nn
import torch


class MixedOp(nn.Module):

    def __init__(self, space, C, stride, affine, track_running_stats):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in space:
            op = OPS[primitive](C, C, stride, affine, track_running_stats)
            self._ops.append(op)

    def forward_gdas(self, x, weights, index):
        return self._ops[index](x) * weights[index]

    def forward_darts(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class NASNetSearchCell(nn.Module):
    def __init__(self, space, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, affine,
                 track_running_stats):
        super(NASNetSearchCell, self).__init__()
        self.reduction = reduction
        self.op_names = deepcopy(space)
        if reduction_prev:
            self.preprocess0 = OPS['skip_connect'](C_prev_prev, C, 2, affine, track_running_stats)
        else:
            self.preprocess0 = OPS['nor_conv_1x1'](C_prev_prev, C, 1, affine, track_running_stats)
        self.preprocess1 = OPS['nor_conv_1x1'](C_prev, C, 1, affine, track_running_stats)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self.edges = nn.ModuleDict()
        for i in range(self._steps):
            for j in range(2 + i):
                node_str = '{:}<-{:}'.format(i, j)
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(space, C, stride, affine, track_running_stats)
                self.edges[node_str] = op
        self.edge_keys = sorted(list(self.edges.keys()))
        self.edge2index = {key: i for i, key in enumerate(self.edge_keys)}
        self.num_edges = len(self.edges)

    def forward_gdas(self, s0, s1, weightss, indexs):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            clist = []
            for j, h in enumerate(states):
                node_str = '{:}<-{:}'.format(i, j)
                op = self.edges[node_str]
                weights = weightss[self.edge2index[node_str]]
                index = indexs[self.edge2index[node_str]].item()
                clist.append(op.forward_gdas(h, weights, index))
            states.append(sum(clist))

        return torch.cat(states[-self._multiplier:], dim=1)

    def forward_darts(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            c_list = []
            for j, h in enumerate(states):
                node_str = '{:}<-{:}'.format(i, j)
                op = self.edges[node_str]
                weight = weights[self.edge2index[node_str]]
                c_list.append(op.forward_darts(h, weight))
            states.append(sum(c_list))

        return torch.cat(states[-self._multiplier:], dim=1)
