#Python 3.6.8 64-bit (pythorch-gpu)

import numpy as np
import preview as pre
import torch.nn as nn
import torch
import my_nn
from tqdm import tqdm

wordEmbeddings, vocab_len, dim, max_len, train_data, test_data = pre.getArgs()

args = {}
args["word_size"] = vocab_len
args["dim"] = dim
args["embeds"] = wordEmbeddings
args["max_len"] = max_len

LR = 0.001

net = my_nn.cnn(args)
net.cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=LR)
loss_function = nn.CrossEntropyLoss()

num_of_senteces = len(train_data)

print("Net loaded.")

def test():
    num = len(test_data)
    right = 0
    tmp = 0
    for i in range(num):
        if tmp == 0:
            y = []
            x = []
        tmp = tmp+1
        x.append(test_data[i][0])
        y.append(test_data[i][1])
        if tmp == 8:
            input = torch.LongTensor(x)
            input = input.cuda()
            y = torch.LongTensor(y)
            y = y.cuda()

            output = net.forward(input)
            for j in range(8):
                flag = 1
                if output[j][0] > output[j][1]:
                    flag = 0
                if(flag == y[j]):
                    right += 1
            tmp = 0
    print(right, num)


for k in range(15):
    tmp = 0
    t = tqdm(range(num_of_senteces), ncols=100, desc="training")
    for i in t:
        if tmp == 0:
            y = []
            x = []
        tmp = tmp+1
        x.append(train_data[i][0])
        y.append(train_data[i][1])
        if tmp == 8:
            input = torch.LongTensor(x)
            input = input.cuda()
            y = torch.LongTensor(y)
            y = y.cuda()

            output = net.forward(input)
            optimizer.zero_grad()
            loss = loss_function(output.view(len(y), -1), y)
            loss.backward()
            optimizer.step()

            if (i+1) % 1000 == 0:
                loss = loss.cpu()
                print_loss = loss.data.numpy()
                t.set_postfix(loss=format(print_loss, '.8f'))
            tmp = 0

test()