import json
import logging

import torch
import numpy as np
from torch import nn

from operation.classify_opt import OPS

ops_list = ['avg_pool_3x3',
            'max_pool_3x3',
            'dense_layer_relu',
            'dense_layer_sigmoid',
            'dense_layer_tanh',
            'group_dense_4_relu',
            'group_dense_4_sigmoid',
            'group_dense_4_tanh',
            'none']

acti_list = ['relu', 'sigmoid', 'tanh', 'skip']


class MixedOp(nn.Module):

    def __init__(self, C_in, C_out):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in ops_list:
            op = OPS[primitive](C_in, C_out)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C_out, affine=False))
            self._ops.append(op)

    def forward_gdas(self, x, weights, index):
        return self._ops[index](x) * weights[index]

    def forward_darts(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class LayerOp(nn.Module):
    def __init__(self):
        super(LayerOp, self).__init__()
        self.activation = nn.ModuleList()
        self.activation.append(nn.ReLU())
        self.activation.append(nn.Sigmoid())
        self.activation.append(nn.Tanh())
        self.activation.append(OPS['skip']())

    def forward_gdas(self, x, weights, index):
        out = self.activation[index](x) * weights[index]
        return out


class search_classifier(nn.Module):
    def __init__(self, node_num, in_num, C_in, C_out):
        super(search_classifier, self).__init__()
        self.in_num = in_num
        self.node_num = node_num
        self.C_in = C_in
        self.C_out = C_out
        self.preprocess0 = None
        if self.C_in % 4 != 0:
            _c_in = (self.C_in // 4) * 4
            self.preprocess0 = OPS['ReLUConvBN'](self.C_in, _c_in)
            self.C_in = _c_in

        self.ops = nn.ModuleDict()
        self.node_acti = nn.ModuleDict()
        # 每个node之前加layerOpt
        cout = None
        for i in range(1, self.node_num + 1):
            self.node_acti[str(i)] = LayerOp()
            for j in range(i):
                edge = str(j) + '->' + str(i)
                cin = self.C_in // 2 if j > 0 else self.C_in
                cout = cin // 2 if j < 1 else cin
                op = MixedOp(cin, cout)
                self.ops[edge] = op
                # print(edge)
                # print('in:{:}, out:{:}'.format(cin, cout))
        assert cout is not None
        self.node_acti[str(node_num + 1)] = LayerOp()
        self.final_linear = nn.Linear(cout, self.C_out)

        self.edge_keys = sorted(list(self.ops.keys()))
        self.edge2index = {key: i for i, key in enumerate(self.edge_keys)}
        self.arch_parameters = nn.Parameter(torch.randn(len(self.ops), len(ops_list)))
        self.layer_opt_parameters = nn.Parameter(torch.randn(self.node_num + 1, 4))
        self.tau = 10

    def set_tau(self, tau):
        self.tau = tau

    def get_tau(self):
        return self.tau

    def forward_gdas(self, x, arch_weight, arch_index, layer_weight, layer_index):
        if self.preprocess0 is not None:
            s0 = self.preprocess0(x)
        else:
            s0 = x
        states = [s0]
        for i in range(1, self.node_num + 1):
            clist = []
            for j, h in enumerate(states):
                edge = '{:}->{:}'.format(j, i)
                # print("node_str: "+str(edge))
                # print("edge2index: "+str(self.edge2index[edge]))
                op = self.ops[edge]
                weight = arch_weight[self.edge2index[edge]]
                # print("weight: "+str(weight))
                index = arch_index[self.edge2index[edge]].item()
                # print("index: "+str(index))
                clist.append(op.forward_gdas(h, weight, index))
            clist = sum(clist) / len(clist)
            layer_op = self.node_acti[str(i)]
            weight = layer_weight[i - 1]
            index = layer_index[i - 1].item()
            states.append(layer_op.forward_gdas(clist, weight, index))
        state = sum(states[1:]) / (len(states) - 1)
        final_op = self.node_acti[str(self.node_num + 1)]
        weight = layer_weight[self.node_num]
        index = layer_index[self.node_num]
        state = final_op.forward_gdas(state, weight, index)
        return state

    def forward(self, x, arc_type='gdas'):
        def get_gumbel_prob(xins):
            while True:
                gumbel = -torch.empty_like(xins).exponential_().log()
                logits = (xins.log_softmax(dim=1) + gumbel) / self.tau
                prob = nn.functional.softmax(logits, dim=1)
                index = prob.max(-1, keepdim=True)[1]
                one_h = torch.zeros_like(logits).scatter_(-1, index, 1.0)
                hardwts = one_h - prob.detach() + prob
                if (torch.isinf(gumbel).any()) or (torch.isinf(prob).any()) or (torch.isnan(prob).any()):
                    continue
                else:
                    break
            return hardwts, index

        x = x.reshape(x.size(0), x.size(1), 1, 1)
        arc_hardwts, arc_index = get_gumbel_prob(self.arch_parameters)
        layer_hardwts, layer_index = get_gumbel_prob(self.layer_opt_parameters)
        if arc_type == 'gdas':
            output = self.forward_gdas(x, arc_hardwts, arc_index, layer_hardwts, layer_index)
        else:
            raise ValueError
        output = output.view(output.size(0), -1)
        output = self.final_linear(output)
        return output

    def show_alphas(self):
        with torch.no_grad():
            A = 'arch-parameters :\n{:}'.format(nn.functional.softmax(self.arch_parameters, dim=-1).cpu())
            B = 'activation-parameters : \n{:}'.format(nn.functional.softmax(self.layer_opt_parameters, dim=-1).cpu())
        return A, B

    def get_alphas(self):
        return [self.arch_parameters, self.layer_opt_parameters]

    def get_weights(self):
        x_list = list(self.ops.parameters()) + list(self.node_acti.parameters()) + list(self.final_linear.parameters())
        return x_list

    def genotype(self, genotype_file):
        arch_weight = torch.softmax(self.arch_parameters, dim=-1)
        acti_weight = torch.softmax(self.layer_opt_parameters, dim=-1)
        json_to_write = {'classify': {}}
        json_to_write['classify']['normal'] = {}
        json_to_write['classify']['activation'] = {}
        for i in range(1, self.node_num + 1):
            json_to_write['classify']['normal'][str(i)] = []
        for i in range(1, self.node_num + 1):
            in_num = []
            for j in range(i):
                edges = []
                edge = str(j) + '->' + str(i)
                weights = arch_weight[self.edge2index[edge]]
                for k, op_name in enumerate(ops_list):
                    # if op_name == 'none': continue
                    edges.append((op_name, j, i, float(weights[k])))
                edges = sorted(edges, key=lambda x: -x[-1])
                selected_edges = edges[0]
                in_num.append(selected_edges)
            in_num = sorted(in_num, key=lambda x: -x[-1])
            logging.info('node {:} selectable edges:{:}'.format(i, in_num))
            for j in range(self.in_num):
                if j >= in_num.__len__():
                    break
                json_to_write['classify']['normal'][str(i)].append('{:},{:}'.format(in_num[j][1], in_num[j][0]))

        for i in range(1, self.node_num + 2):
            weights = acti_weight[i - 1]
            opt = []
            for k, acti_name in enumerate(acti_list):
                opt.append((acti_name, float(weights[k])))
            opt = sorted(opt, key=lambda x: -x[-1])
            selected_opt = opt[0]
            logging.info('node {:} selectable active functions:{:}'.format(i, opt))
            json_to_write['classify']['activation'][str(i)] = selected_opt[0]

        with open(genotype_file, 'w') as f:
            json.dump(json_to_write, f, indent=4)


class train_classifier(nn.Module):
    def __init__(self, C_in, num_class, genotype_file):
        super(train_classifier, self).__init__()
        self.C_in = C_in
        self.num_class = num_class
        self.genotype_file = genotype_file
        with open(self.genotype_file) as f:
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
                self.classifier[edge_name] = OPS[op_name](cin, cout)
                self.start_node_ops[target_node].append(edge_name)

        acti = classifier_arch['activation']
        self.acti = nn.ModuleDict()
        for i in range(1, self.num_node + 2):
            acti_name = acti[str(i)]
            if acti_name == 'relu':
                self.acti[str(i)] = nn.ReLU()
            elif acti_name == 'sigmoid':
                self.acti[str(i)] = nn.Sigmoid()
            elif acti_name == 'tanh':
                self.acti[str(i)] = nn.Tanh()
            elif acti_name == 'skip':
                self.acti[str(i)] = OPS['skip']()
            else:
                raise ValueError

        assert cout is not None
        self.last_linear = nn.Linear(cout, self.num_class)

    def get_weights(self):
        return self.parameters()

    def forward(self, feature):
        states = [feature]
        for i in range(1, self.num_node + 1):
            edges = self.start_node_ops[str(i)]
            c_list = []
            for edge in edges:
                op = self.classifier[edge]
                c_list.append(op(states[int(edge[0])]))
            c_list = sum(c_list) / len(c_list)
            states.append(self.acti[str(i)](c_list))
        out = sum(states[1:]) / (len(states) - 1)
        out = self.acti[str(self.num_node + 1)](out)
        out = out.reshape(out.size(0), -1)
        out = self.last_linear(out)
        return out
