#Python 3.7.1 64-bit

import makedata as data
import model as nn
import matplotlib.pyplot as plt
import numpy as np
import functional as F

plt.ion()
plt.show()

#绘图的函数
def view(w, x_rate, y_rate):
    print_x = np.vander(np.linspace(0, 1, 100), order + 1, True)
    print_y = np.dot(print_x, w) * y_rate

    sinx = np.linspace(0, 1, 100)
    sinx = np.sin(2 * np.pi * sinx)
    plt.cla()
    plt.plot(np.linspace(0, 1, 100), print_y.T.A[0], c="r")
    plt.plot(np.linspace(0, 1, 100), sinx, c="g")
    plt.scatter(source_x, source_y, s=10, c="b")
    plt.pause(0.1)


args = dict(
    epoch=1000000,
    num_x=100,
    sigma=0.2,
    num_group=1,
    order=15,
    lamda=0.00001,
    optim="CG",
    LR=0.1
)
epoch = args["epoch"]#训练轮数
order = args["order"]#多项式阶数
LR = args["LR"]#学习率
num_x = args["num_x"]#数据的组数
sigma = args["sigma"]#噪声的方差
group = num_x#一次喂到网络中的数据的组数
lamda = args["lamda"]#正则项的系数

source_x, source_y = data.getSource(num_x, sigma)
#归一化
input = source_x / max(source_x)
input = np.vander(input, order + 1, True)
input = np.mat(input)
y_true = source_y / max(max(source_y), -min(source_y))
y_true = np.mat(y_true).T

net = nn.network(args)

#梯度下降
if args["optim"] == "GD":
    for i in range(epoch):
        output = net.forward(input)#获得预测值
        optimizer = nn.GDoptim(net.parameters(), input, LR)#定义优化器为梯度下降法
        loss_founction = nn.MSELoss(output, y_true)#定义loss函数
        loss_founction.backward(optimizer, net)#loss反向传播

        if i % 2000 == 0:#每隔2000轮画一次图
            view(net.parameters(), max(source_x),
                 max(max(source_y), -min(source_y)))
            loss = loss_founction.loss
            print(loss.A[0]/group)
#共轭梯度法
if args["optim"] == "CG":
    optimizer = nn.CGoptim(net.parameters(), input, y_true)#定义优化器为共轭梯度法
    for i in range(order):
        output = net.forward(input)#获得预测值
        loss_founction = nn.MSELoss(output, y_true)#定义loss函数
        optimizer.nextStep(net)#优化器确定共轭梯度的方向并且向前走一步
        #每走一步画一次图
        view(net.parameters(), max(source_x),
             max(max(source_y), -min(source_y)))
        loss = loss_founction.loss
        print(loss.A[0]/group)
    plt.pause(2)
#不带正则项的解析解
if args["optim"] == "MSE":
    #直接调用计算解析解的函数得到系数矩阵进行画图
    view(F.MSE.cla_W(input, y_true), max(source_x),
         max(max(source_y), -min(source_y)))
    plt.pause(5)
#带正则项的解析解
if args["optim"] == "MSElamda":
    #直接调用计算解析解的函数得到系数矩阵进行画图
    view(F.MSElam.cla_W(input, y_true, lamda, group, order + 1), max(source_x),
         max(max(source_y), -min(source_y)))
    plt.pause(5)
