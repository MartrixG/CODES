from model.search_dnncell import DNNModel
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.backends import cudnn
import torch.nn as nn


class DatasetGen(Dataset):
    def __init__(self, C_in, C_out, number):
        self.X = torch.randn(number, C_in)
        class_num = C_out
        batch_size = number
        label = torch.LongTensor(batch_size, 1).random_() % class_num
        # self.Y = torch.zeros(batch_size, class_num).scatter_(1, label, 1)
        self.Y = label.squeeze(1)
        self.X1 = torch.randn(number, C_in)
        label1 = torch.LongTensor(batch_size, 1).random_() % class_num
        # self.Y1 = torch.zeros(batch_size, class_num).scatter_(1, label1, 1)
        self.Y1 = label1.squeeze(1)
        self.len = number

    def __getitem__(self, index):
        return self.X[index], self.Y[index], self.X1[index], self.Y1[index]

    def __len__(self):
        return self.len


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.count = 0.0
        self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0
        self.reset()

    def reset(self):
        pass

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{name}(val={val}, avg={avg}, count={count})'.format(name=self.__class__.__name__, **self.__dict__)


def obtain_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


batch_size = 200
epochs = 200
data_number = 1000
tau_max, tau_min = 10, 0.1
C_in, C_out = 250, 10

np.random.seed(0)
torch.cuda.set_device(0)
cudnn.benchmark = True
torch.manual_seed(0)
cudnn.enabled = True
torch.cuda.manual_seed(0)

search_model = DNNModel(C_in=C_in, C_out=C_out)
criterion = nn.CrossEntropyLoss()
# criterion = nn.MSELoss()
criterion = criterion.cuda()
w_optimizer = torch.optim.SGD(
    search_model.parameters(),
    lr=0.05, weight_decay=0.0005
)

w_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(w_optimizer, float(epochs), eta_min=0)
a_optimizer = torch.optim.Adam(search_model.get_alphas(), lr=0.05, betas=(0.5, 0.999),
                               weight_decay=1e-3)

data_loader = DataLoader(dataset=DatasetGen(C_in=C_in, C_out=C_out, number=data_number),
                         batch_size=batch_size, shuffle=True)

Records = []
for epoch in range(epochs):
    print("epoch: " + str(epoch))
    w_scheduler.step()
    search_model.set_tau(tau_max - (tau_max - tau_min) * epoch / (epochs - 1))

    base_losses, base_top1, base_top5 = AverageMeter(), AverageMeter(), AverageMeter()
    arch_losses, arch_top1, arch_top5 = AverageMeter(), AverageMeter(), AverageMeter()
    for step, (X, Y, X1, Y1) in enumerate(data_loader):
        base_inputs, base_targets = X, Y
        arch_inputs, arch_targets = X1, Y1
        # update the weights
        w_optimizer.zero_grad()
        logits = search_model(base_inputs)
        base_loss = criterion(logits, base_targets)
        base_loss.backward()
        torch.nn.utils.clip_grad_norm_(search_model.parameters(), 5)
        w_optimizer.step()
        # record
        base_prec1, base_prec5 = obtain_accuracy(logits.data, base_targets.data, topk=(1, min(5, C_out)))
        print("base_prec1: " + str(base_prec1))
        base_losses.update(base_loss.item(), base_inputs.size(0))
        base_top1.update(base_prec1.item(), base_inputs.size(0))
        base_top5.update(base_prec5.item(), base_inputs.size(0))

        # update the architecture-weight
        a_optimizer.zero_grad()
        logits = search_model(arch_inputs)
        arch_loss = criterion(logits, arch_targets)
        arch_loss.backward()
        a_optimizer.step()
        # record
        arch_prec1, arch_prec5 = obtain_accuracy(logits.data, arch_targets.data, topk=(1, min(5, C_out)))
        arch_losses.update(arch_loss.item(), arch_inputs.size(0))
        arch_top1.update(arch_prec1.item(), arch_inputs.size(0))
        arch_top5.update(arch_prec5.item(), arch_inputs.size(0))
    Record = [base_losses.avg, base_top1.avg, base_top5.avg, arch_losses.avg, arch_top1.avg, arch_top5.avg]
    print("Record: " + str(Record) + "\n")
    Records.append(list(Record))
    with open("Records.txt", "a+") as f:
        f.write("Results for epoch " + str(epoch) + "\n")
        f.write("\t" + str(Record) + "\n")
        f.write(str(logits[:3]) + "\n")
        f.write(str(arch_targets[:3]) + "\n")
        f.write("\tsearch_model.genotype: " + str(search_model.genotype()) + "\n\n")
print("\n\nFinal Results: " + str(Record))
