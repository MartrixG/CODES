import logging

from torch import nn
from utils import classify_opt


class NetworkDNN(nn.Module):
    def __init__(self, args):
        super(NetworkDNN, self).__init__()
        self.c_out = args.c_out
        self.c_in = args.c_in
        if args.cross_link:
            self.classifier = cross_classifier(self.c_in, self.c_out, args)
        else:
            self.classifier = classifier(self.c_in, self.c_out, args)
        logging.info('classifier:\n{:}'.format(self.classifier))

    def forward(self, feature):
        x = feature
        x = x.reshape(x.size(0), x.size(1), 1, 1)
        x = self.classifier(x)
        return x


class classifier(nn.Module):
    def __init__(self, c_in, c_out, args, track_running_stats=True):
        super(classifier, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.blocks = nn.ModuleList()
        self.opt = classify_opt.OPS[args.opt]
        c_prev = c_in
        # c_curr = int(c_in * args.first_neurons)
        c_curr = args.first_neurons
        for i in range(args.hidden_layers):
            self.blocks.append(nn.Sequential(
                self.opt(c_prev, c_curr, track_running_stats),
                self.activate(args.activate_func)
            ))
            c_prev = c_curr
            c_curr = int(c_curr * args.change)
        self.linear = nn.Linear(c_prev, c_out)

    def activate(self, func_name):
        if func_name == 'relu':
            return nn.ReLU(inplace=True)
        elif func_name == 'tanh':
            return nn.Tanh()
        else:
            return nn.Sigmoid()

    def forward(self, feature):
        x = feature
        for block in self.blocks:
            x = block(x)
        x = x.view(x.size(0), -1)
        out = self.linear(x)
        return out


class cross_classifier(nn.Module):
    def __init__(self, c_in, c_out, args, track_running_stats=True):
        super(cross_classifier, self).__init__()
        self.C_in = c_in
        self.C_out = c_out
        self.blocks = nn.ModuleDict()
        self.fully_cross = args.fully_cross
        self.hidden_layers = args.hidden_layers
        out = {0: c_in}
        for i in range(1, self.hidden_layers + 1):
            out[i] = args.first_neurons
        for i in range(1, self.hidden_layers + 1):
            for j in range(i):
                edge = str(j) + '->' + str(i)
                self.blocks[edge] = nn.Sequential(
                    classify_opt.OPS['dense_layer'](out[j], out[i], track_running_stats),
                    self.activate(args.activate_func)
                )
        self.linear = nn.Linear(args.first_neurons, c_out)
        self.linears = nn.ModuleList()
        if self.fully_cross:
            self.linears.append(nn.Linear(c_in, c_out))
            self.linears.append(nn.Linear(args.first_neurons, c_out))
            self.linears.append(nn.Linear(args.first_neurons, c_out))
            self.linears.append(nn.Linear(args.first_neurons, c_out))

    def activate(self, func_name):
        if func_name == 'relu':
            return nn.ReLU(inplace=True)
        elif func_name == 'tanh':
            return nn.Tanh()
        else:
            return nn.Sigmoid()

    def forward(self, feature):
        blocks = [feature]
        for i in range(1, self.hidden_layers + 1):
            c_list = []
            for j in range(i):
                edge = str(j) + '->' + str(i)
                c_list.append(self.blocks[edge](blocks[j]))
            blocks.append(sum(c_list))
        if self.fully_cross:
            c_list = []
            for i in range(blocks.__len__()):
                c_list.append(self.linears[i](blocks[i].view(blocks[i].size(0), -1)))
            out = sum(c_list)
        else:
            out = blocks[-1].view(blocks[-1].size(0), -1)
            out = self.linear(out)
        return out
