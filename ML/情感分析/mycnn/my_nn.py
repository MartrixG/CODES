#Python 3.6.8 64-bit (pythorch-gpu)

import torch
import torch.nn as nn
import torch.nn.functional as F


class cnn(nn.Module):
    def __init__(self, args):
        super(cnn, self).__init__()
        self.word_size = args["word_size"]  # 单词数量
        self.dim = args["dim"]  # 词向量维度
        self.n_class = 2
        self.max_len = args["max_len"]  # 文本大小限制（一句话的长度），不足则以空格补齐
        self.embeds = nn.Embedding(self.word_size, self.dim)
        self.embeds.weight.data.copy_(torch.from_numpy(args["embeds"]))
        self.conv1 = nn.Sequential(  # (8,1,20,100)
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=(5, self.dim + 4),
                stride=1,
                padding=2,
            ),  # (8,32,20,100)
            nn.ReLU()
        )
        self.out = nn.Linear(32, 2)

    def forward(self, x):
        x = self.embeds(x)  # (8,20,100)
        x = x.view(x.size(0), 1, self.max_len, self.dim)  # (8,1,20,100)
        x = self.conv1(x)  # (8,32,20,1)
        x = x.view(x.size(0), 32, -1)  # (8,32,20)
        x = F.max_pool2d(x, (1, self.dim))  # (8,32,1)
        x = x.view(x.size(0), -1)  # (8,32)
        output = self.out(x)  # (8,2)
        return output
