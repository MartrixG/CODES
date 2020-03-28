from model.cell.cell_operations import OPS, FactorizedReduce
import torch.nn as nn
import torch


class MixedOp(nn.Module):

    def __init__(self, space, layer_in_cell, layer_out, track_running_stats):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in space:
            op = OPS[primitive](layer_in_cell, layer_out, track_running_stats)
            self._ops.append(op)

    def forward_ourgdas(self, x, weights, index):
        return self._ops[index](x) * weights[index]

    def forward_ourdarts(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


def Calculate_OutC(layer_in, track_running_stats, search_space):
    layer_out = 0
    for name in search_space:
        cell = OPS[name](layer_in, None, track_running_stats)
        C_out_cell = cell.out_dim
        if C_out_cell > layer_out:
            layer_out = C_out_cell
    return layer_out


class DNNModel(nn.Module):

    def __init__(self, config, track_running_stats=True):
        super(DNNModel, self).__init__()
        self.C_in = config.C_in
        self.C_out = config.C_out
        layer_number = 0
        C_in_test = self.C_in // 2
        while C_in_test > max(20, self.C_out) and layer_number < 5:
            layer_number += 1
            C_in_test = C_in_test // 2
        self.layer_number = layer_number
        print("number of layers: " + str(layer_number) + "\n")

        self.space = config.search_space
        self.edges = nn.ModuleDict()
        self.reductions = nn.ModuleDict()
        layer_in, layer_out_list = self.C_in, [self.C_in, self.C_in]
        for i in range(2, layer_number + 2):
            layer_out = Calculate_OutC(layer_in, track_running_stats)
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                layer_in_cell = layer_out_list[j]
                if j < i - 1:
                    re_op = FactorizedReduce(layer_in_cell, layer_out_list[i - 1], track_running_stats)
                    self.reductions[node_str] = re_op
                    op = MixedOp(self.space, layer_out_list[i - 1], layer_out, track_running_stats)
                else:
                    op = MixedOp(self.space, layer_in_cell, layer_out, track_running_stats)
                self.edges[node_str] = op

            layer_in = layer_out
            layer_out_list.append(layer_in)
        self.edge_keys = sorted(list(self.edges.keys()))
        self.edge2index = {key: i for i, key in enumerate(self.edge_keys)}
        self.num_edges = len(self.edges)
        self.final_dense = nn.Linear(layer_in, self.C_out)
        self.arch_parameters = nn.Parameter(1e-3 * torch.randn(self.num_edges, len(self.space)))
        self.tau = 10

    def get_weights(self):
        xlist = list(self.reductions.parameters()) + list(self.edges.parameters())
        xlist += list(self.final_dense.parameters())
        return xlist

    def get_alphas(self):
        return [self.arch_parameters]

    def set_tau(self, tau):
        self.tau = tau

    def get_tau(self):
        return self.tau

    def genotype(self):
        def _parse(weights):
            gene = []
            for i in range(2, self.layer_number + 2):
                for j in range(i):
                    edges = []
                    node_str = '{:}<-{:}'.format(i, j)
                    ws = weights[self.edge2index[node_str]]
                    for k, op_name in enumerate(self.space):
                        # if op_name == 'none': continue
                        edges.append((op_name, i, j, ws[k]))
                    with open("genotype.txt", "a+") as f:
                        f.write("node_str: " + str(node_str) + "\n")
                        f.write("edges: " + str(edges) + "\n")
                    edges = sorted(edges, key=lambda x: -x[-1])
                    selected_edges = edges[:1]
                    with open("genotype.txt", "a+") as f:
                        f.write("edges: " + str(edges) + "\n")
                        f.write("selected_edges: " + str(selected_edges) + "\n\n\n")
                    gene.append(tuple(selected_edges))
            return gene

        with torch.no_grad():
            gene_arc = _parse(torch.softmax(self.arch_parameters, dim=-1).cpu().numpy())
        return gene_arc

    def forward_ourgdas(self, x, weights, indexs):
        s0, s1, states = x, x, [x, x]
        for i in range(2, self.layer_number + 2):
            clist = []
            for j, h in enumerate(states):
                node_str = '{:}<-{:}'.format(i, j)
                # print("\tnode_str: "+str(node_str)+"\n")
                # print("\tedge2index: "+str(self.edge2index[node_str])+"\n")
                op = self.edges[node_str]
                weight = weights[self.edge2index[node_str]]
                # print("\tweight: "+str(weight)+"\n")
                index = indexs[self.edge2index[node_str]].item()
                # print("\tindex: "+str(index)+"\n")
                if j < i - 1:
                    # print("\th: "+str(h.size())+"\n")
                    re_h = self.reductions[node_str](h)
                    # print("\tre_h: "+str(re_h.size())+"\n")
                    clist.append(op.forward_ourgdas(re_h, weight, index))
                else:
                    # print("\th: "+str(h.size())+"\n")
                    clist.append(op.forward_ourgdas(h, weight, index))
            states.append(sum(clist))
        return states[-1]

    def forward_ourdarts(self, x, weights):
        s0, s1, states = x, x, [x, x]
        for i in range(2, self.layer_number + 2):
            clist = []
            for j, h in enumerate(states):
                node_str = '{:}<-{:}'.format(i, j)
                op = self.edges[node_str]
                weight = weights[self.edge2index[node_str]]
                if j < i - 1:
                    re_h = self.reductions[node_str](h)
                    clist.append(op.forward_ourdarts(re_h, weights))
                else:
                    clist.append(op.forward_ourdarts(h, weights))
            states.append(sum(clist))
        return states[-1]

    def forward(self, inputs, arctype='ourgdas'):
        def get_gumbel_prob(xins):
            while True:
                # 生成符合指数分布的形状和xins相同的随机数（再取对数底数为e）
                gumbels = -torch.empty_like(xins).exponential_().log()
                # 对8个操作进行log_softmax加上随机数再除以10
                logits = (xins.log_softmax(dim=1) + gumbels) / self.tau
                # 在第二个维度对logits计算softmax
                probs = nn.functional.softmax(logits, dim=1)
                # 从8个操作选出概率最大的
                index = probs.max(-1, keepdim=True)[1]
                # one_hot编码
                one_h = torch.zeros_like(logits).scatter_(-1, index, 1.0)
                hardwts = one_h - probs.detach() + probs
                if (torch.isinf(gumbels).any()) or (torch.isinf(probs).any()) or (torch.isnan(probs).any()):
                    continue
                else:
                    break
            return hardwts, index

        x = inputs
        x = x.reshape(x.size(0), x.size(1), 1, 1)
        arc_hardwts, arc_index = get_gumbel_prob(self.arch_parameters)
        # print("arc_hardwts: "+str(arc_hardwts))
        # print(arc_hardwts.size())
        # print("arc_index: "+str(arc_index))
        # print(arc_index.size())
        # print("@@@@@@@@@@@@@")
        if arctype == 'ourgdas':
            outputs = self.forward_ourgdas(x, arc_hardwts, arc_index)
        else:
            outputs = self.forward_ourdarts(x, arc_hardwts)
        outputs = outputs.view(outputs.size(0), -1)
        logits = self.final_dense(outputs)
        return logits
