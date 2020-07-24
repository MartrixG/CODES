import json
from collections import namedtuple

import torch
from torch import nn
from utils.util import drop_path
from model.operations import FactorizedReduce, ReLUConvBN, Identity
from model.operations import OPS as CNN_OPS
from model.classify_opt import OPS as DNN_OPS

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
DARTS_V2 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])


class Network(nn.Module):
    def __init__(self, name, x_shape, num_class, premodel, genotype, args):
        super(Network, self).__init__()
        self.name = name.lower()
        self.x_shape = x_shape
        self.num_class = num_class
        self.premodel = premodel
        self.genotype = genotype
        if self.name in ['cifar10', 'cifar100']:
            assert self.premodel != "None"
            self.premodel = NetworkCIFAR(args.init_channels, self.num_class,
                                         args.layers, args.auxiliary, DARTS_V2, args)
            self.C_in = self.premodel.out_dim
        elif self.name in ['hapt', 'uji']:
            assert self.premodel == "None"
            self.C_in = self.x_shape
            if self.C_in % 4 != 0:
                _c_in = (self.C_in // 4) * 4
                self.premodel = DNN_OPS['ReLUConvBN'](self.C_in, _c_in)
                self.C_in = _c_in
        else:
            raise ValueError
        with open(self.genotype) as f:
            classifier_arch = json.load(f)['classify']
        self._compile(classifier_arch)

    def _compile(self, classifier_arch):
        normal = classifier_arch['normal']
        cout = None

        self.classifier = nn.ModuleDict()
        self.start_node_ops = {}
        self.num_node = len(normal.keys())
        for i in range(1, self.num_node + 1):
            self.start_node_ops[str(i)] = []

        for target_node in normal.keys():
            for edges in normal[target_node]:
                start_node, op_name = edges.split(',')
                edge_name = start_node + '->' + target_node
                node_num = int(start_node)
                cin = self.C_in // 2 if node_num > 0 else self.C_in
                cout = cin // 2 if node_num < 1 else cin
                self.classifier[edge_name] = DNN_OPS[op_name](cin, cout)
                self.start_node_ops[target_node].append(edge_name)

        acti = classifier_arch['activation']
        if acti == 'relu':
            self.activation = nn.ReLU()
        elif acti == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif acti == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError

        assert cout is not None
        self.last_linear = nn.Linear(cout, self.num_class)

    def forward(self, feature):
        if self.name in ['cifar10', 'cifar100']:
            feature, logits_aux = self.premodel(feature)
        else:
            feature = feature.reshape(feature.size(0), feature.size(1), 1, 1)
            feature = self.premodel(feature)
        states = [feature]
        for i in range(1, self.num_node + 1):
            edges = self.start_node_ops[str(i)]
            c_list = []
            for edge in edges:
                op = self.classifier[edge]
                c_list.append(op(states[int(edge[0])]))
            states.append(sum(c_list) / len(c_list))
        out = sum(states[1:])
        out = self.activation(out)
        out = out.reshape(out.size(0), -1)
        out = self.last_linear(out)
        if self.name in ['cifar10', 'cifar100']:
            return out, logits_aux
        else:
            return out


class Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = CNN_OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


class AuxiliaryHeadCIFAR(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),  # image size = 2 x 2
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class NetworkCIFAR(nn.Module):

    def __init__(self, C, num_classes, layers, auxiliary, genotype, args):
        super(NetworkCIFAR, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        self.drop_path_prob = args.drop_path_prob

        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.out_dim = C_prev
        # self.classifier = nn.Linear(C_prev, num_classes)
        # if args.cross_link:
        #     self.classifier = nn.ModuleList([cross_classifier(C_prev, num_classes)])
        # else:
        #     self.classifier = nn.ModuleList([classifier(C_prev, num_classes, args)])
        # logging.info('classifier:\n{:}'.format(self.classifier))

    def forward(self, feature):
        logits_aux = None
        s0 = s1 = self.stem(feature)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        # logits = self.classifier(out.view(out.size(0), -1))
        # logits = self.classifier[0](out)
        return out, logits_aux
