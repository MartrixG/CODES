import torch
import numpy as np
from torch import nn

from model.classify_opt import OPS

ops_list = ['avg_pool_3x3',
            'max_pool_3x3',
            'dense_layer_relu',
            'dense_layer_sigmoid',
            'dense_layer_tanh',
            'group_dense_4_relu',
            'group_dense_4_sigmoid',
            'group_dense_4_tanh',
            'none']

acti_list = ['relu', 'sigmoid', 'tanh']


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
    def __init__(self, C_in, C_out):
        super(LayerOp, self).__init__()
        self.activation = nn.ModuleList()
        self.activation.append(nn.ReLU())
        self.activation.append(nn.Sigmoid())
        self.activation.append(nn.Tanh())
        self.C_in = C_in
        self.C_out = C_out
        self.final_linear = nn.Linear(self.C_in, self.C_out)

    def forward_gdas(self, x, weights, index):
        out = self.activation[index[0].item()](x) * weights[0][index]
        return self.final_linear(out)


class Classifier(nn.Module):
    def __init__(self, node_num, in_num, C_in, C_out):
        super(Classifier, self).__init__()
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
        cout = None
        for i in range(1, self.node_num + 1):
            for j in range(i):
                edge = str(j) + '->' + str(i)
                cin = self.C_in // 2 if j > 0 else self.C_in
                cout = cin // 2 if j < 1 else cin
                op = MixedOp(cin, cout)
                self.ops[edge] = op
                # print(edge)
                # print('in:{:}, out:{:}'.format(cin, cout))
        assert cout is not None
        self.final_linear = LayerOp(cout, self.C_out)

        self.edge_keys = sorted(list(self.ops.keys()))
        self.edge2index = {key: i for i, key in enumerate(self.edge_keys)}
        self.arch_parameters = nn.Parameter(torch.randn(len(self.ops), len(ops_list)), requires_grad=True)
        self.activation_parameters = nn.Parameter(torch.randn(1, 3), requires_grad=True)
        self.tau = 10

    def forward_gdas(self, x, arch_weight, arch_index):
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
            states.append(sum(clist) / len(clist))
        return sum(states[1:]) / (len(states) - 1)

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
        if arc_type == 'gdas':
            output = self.forward_gdas(x, arc_hardwts, arc_index)
        else:
            raise ValueError
        output = output.view(output.size(0), -1)
        acti_hardwts, acti_index = get_gumbel_prob(self.activation_parameters)
        output = self.final_linear.forward_gdas(output, acti_hardwts, acti_index)
        return output

    def show_alphas(self):
        with torch.no_grad():
            A = 'arch-parameters :\n{:}\n'.format(nn.functional.softmax(self.arch_parameters, dim=-1).cpu())
            B = 'activation-parameters : \n{:}'.format(nn.functional.softmax(self.activation_parameters, dim=-1).cpu())
        return '{:}'.format(A + B)

    def genotype(self, genotype_file):
        arch_weight = torch.softmax(self.arch_parameters, dim=-1)
        acti_weight = torch.softmax(self.activation_parameters, dim=-1)[0].detach().numpy()
        with open(genotype_file, 'w') as f:
            f.write("classify = {\n\t\"normal\": {\n")
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
                selected_edges = edges[-1]
                in_num.append(selected_edges)
            in_num = sorted(in_num, key=lambda x: -x[-1])
            # log_write : 'node {:} selectable edges:{:}\n'.format(i, in_num)
            with open(genotype_file, 'a+') as f:
                f.write("\t\t\"{:}\": ".format(i))
                gen = "["
                for j in range(self.in_num):
                    if j + 1 > in_num.__len__():
                        break
                    gen += "\"{:},{:}\", ".format(in_num[j][1], in_num[j][0])
                gen = gen[:-2]
                if i == self.node_num:
                    index = np.argmax(acti_weight)
                    gen += "]\n\t},\n\t\"activation\": [\"%s\"]\n}\n" % acti_list[index]
                else:
                    gen += "], \n"
                f.write(gen)
