import torch.nn as nn
import torch.nn.functional as F


class simple_dense_net(nn.Module):
    def __init__(self):
        super(simple_dense_net, self).__init__()
        self.fc1 = nn.Linear(561, 12)

    def forward(self, x):
        out = self.fc1(x)
        return out


class normal_dense_net(nn.Module):
    def __init__(self):
        super(normal_dense_net, self).__init__()
        self.fc1 = nn.Linear(561, 561)
        self.fc2 = nn.Linear(561, 561)
        self.fc3 = nn.Linear(561, 12)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
