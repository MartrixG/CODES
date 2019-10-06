#Python 3.7.1 64-bit

import numpy as np


class network(object):
    def __init__(self, args):#初始化
        self.n = args["num_group"]
        self.m = args["order"] + 1
        self.w = np.mat(np.random.rand(self.m, 1))#系数矩阵初始化为随机

    def parameters(self):
        return self.w

    def sync(self, w):
        self.w = w

    def backward(self, gradient):#梯度反向传播
        self.w = self.w + gradient

    def forward(self, x):#矩阵计算直接得到预测值
        return np.dot(x, self.w)


class MSELoss(object):#loss函数
    def __init__(self, y_pre, y_true):
        self.y_pre = np.mat(y_pre)
        self.y_true = np.mat(y_true)
        self.loss = np.dot((self.y_pre-self.y_true).T,
                           (self.y_pre-self.y_true))

    def backward(self, optimizer, net):#调用网络的函数和优化器的函数进行梯度的反向传播
        net.backward(optimizer.grad(self.y_true))


class optim(object):#定义优化器的超类
    def __init__(self, w, x, LR):
        self.w = w
        self.x = x
        self.lr = LR


class GDoptim(optim):#梯度下降法的子类，计算梯度的方法报告已给出
    def __init__(self, w, x, LR):
        super(GDoptim, self).__init__(w, x, LR)

    def grad(self, y_true):
        return - np.dot(self.lr * self.x.T, np.dot(self.x, self.w) - y_true) / (self.x.size/self.w.size)


class CGoptim(optim):#共轭梯度法的子类，确定下一步方向和步长的方法报告中已给出
    def __init__(self, w, x, y_true):
        super(CGoptim, self).__init__(w, x, 1)
        self.y_true = np.dot(x.T, y_true)
        self.x = np.dot(x.T, x)
        self.r = self.y_true - np.dot(self.x, self.w)
        self.p = self.r

    def nextStep(self, net):
        self.a = np.dot(self.r.T, self.r) / \
            np.dot(self.p.T, np.dot(self.x, self.p))
        self.a = np.asarray(self.a)[0][0]
        self.w = self.w + self.a * self.p
        self.rr = self.r
        self.r = self.r - self.a * np.dot(self.x, self.p)
        self.b = np.dot(self.r.T, self.r) / np.dot(self.rr.T, self.rr)
        self.b = np.asarray(self.b)[0][0]
        self.p = self.r + self.b * self.p
        net.sync(self.w)
