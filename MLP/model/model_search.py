import torch
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
        for i in range(self.node_num):
            for j in range(i + 1):
                edge = str(j) + '->' + str(i + 1)
                cin = self.C_in // 2 if j > 0 else self.C_in
                cout = cin // 2 if j < 1 else cin
                op = MixedOp(cin, cout)
                self.ops[edge] = op
                # print(edge)
                # print('in:{:}, out:{:}'.format(cin, cout))
        assert cout is not None
        self.final_linear = nn.Linear(cout, self.C_out)

        self.edge_keys = sorted(list(self.ops.keys()))
        self.edge2index = {key: i for i, key in enumerate(self.edge_keys)}
        self.arch_parameters = nn.Parameter(1e-3 * torch.randn(len(self.ops), len(ops_list)), requires_grad=True)
        self.tau = 10

    # 加和还是拼接
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
            states.append(sum(clist))
        return states[-1]

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
        return output

    def genotype(self, genotype_file):
        # genotype需要重
        def _parse(weights):
            gene = []
            for i in range(self.node_num):
                for j in range(i + 1):
                    edges = []
                    edge = str(j) + '->' + str(i + 1)
                    ws = weights[self.edge2index[edge]]
                    for k, op_name in enumerate(ops_list):
                        # if op_name == 'none': continue
                        edges.append((op_name, i, j, ws[k]))
                    with open(genotype_file, "a+") as f:
                        f.write("node_str: " + str(edge) + "\n")
                    edges = sorted(edges, key=lambda x: -x[-1])
                    selected_edges = edges[:1]
                    with open(genotype_file, "a+") as f:
                        f.write("edges: " + str(edges) + "\n")
                        f.write("selected_edges: " + str(selected_edges) + "\n\n\n")
                    gene.append(tuple(selected_edges))
            return gene

        with torch.no_grad():
            gene_arc = _parse(torch.softmax(self.arch_parameters, dim=-1).cpu().numpy())
        return gene_arc
