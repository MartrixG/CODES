import numpy as np


class network(object):
    def __init__(self, args):#初始化
        self.n = args["num_x"]
        if(args['data'] == 'iris'):
            self.m = 5
        else:
            self.m = 3
        self.w = np.mat(np.random.rand(self.m, 1))#系数矩阵初始化为随机

    def paramaters(self):
        return self.w

    def backward(self, gradient):#梯度反向传播
        self.w = self.w + gradient

    def forward(self, x):#矩阵计算直接得到预测值
        return np.dot(x, self.w)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class binary_crossentropy_Loss(object):#loss函数
    def __init__(self, y_pre, y_true, w, lamda):
        self.y_true = np.mat(y_true)
        self.y_pre  = np.mat(y_pre)
        self.w      = np.mat(w)
        self.lamda  = lamda
        self.loss   = - self.y_true.T.dot(np.log(sigmoid(self.y_pre))) -\
                        (1 - self.y_true.T).dot(np.log(1 - sigmoid(self.y_pre))) +\
                        self.lamda * self.w.T.dot(self.w)
                        

    def backward(self, optimizer, net):#调用网络的函数和优化器的函数进行梯度的反向传播
        net.backward(optimizer.grad(self.y_true))


class GDoptim(object):#梯度下降法的子类，计算梯度的方法报告已给出
    def __init__(self, w, x, LR, lamda):
        self.w = np.mat(w)
        self.x = np.mat(x)
        self.lr = LR
        self.lamda = lamda
    
    def grad(self, y_true):
        return self.lr * (self.x.T.dot(y_true - sigmoid(self.x.dot(self.w))) - self.lamda * self.w)
