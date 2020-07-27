from torch import nn

from model.classifier_model import search_classifier, train_classifier
from model.pre_model import pre_model


class Network(nn.Module):
    def __init__(self, name, x_shape, num_class, args):
        super(Network, self).__init__()
        self.name = name.lower()
        self.x_shape = x_shape
        self.num_class = num_class
        self.genotype_file = args.genotype_file

        self.pre_model = pre_model(self.name, self.x_shape, self.num_class, args)
        if args.type == 'search':
            self.classifier = search_classifier(args.num_node, args.in_num, self.pre_model.C_in, self.num_class)
        else:
            self.classifier = train_classifier(self.pre_model.C_in, self.num_class, self.genotype_file)

    def forward(self, feature):
        if self.name in ['cifar10', 'cifar100']:
            x, logits_aux = self.pre_model(feature)
            out = self.classifier(x)
            return out, logits_aux
        else:
            x = self.pre_model(feature)
            out = self.classifier(x)
            return out

    def get_genotype(self):
        self.classifier.genotype(self.genotype_file)

    def show_alphas(self):
        return self.classifier.show_alphas()

    def get_weights(self):
        return list(self.pre_model.get_weights()) + list(self.classifier.get_weights())

    def get_alphas(self):
        return self.classifier.get_alphas()

    def set_drop_path_prob(self, drop_path_prob):
        self.pre_model.set_drop_path_prob(drop_path_prob)

    def set_tau(self, tau):
        self.classifier.set_tau(tau)