#Python 3.7.1 64-bit

import numpy as np


class MSE(object):#不带正则项的解析解求解
    def cla_W(x, y_true):
        #计算过程（报告中已给出）
        A = np.dot(x.T, x)
        A = np.linalg.pinv(A)
        return np.dot(A, np.dot(x.T, y_true))


class MSElam(object):#带正则项的解析解求解
    def cla_W(x, y_true, lamda, n, m):
        #计算过程（报告中已给出）
        lamda = lamda * n / 2
        A = np.dot(x.T, x)
        A = lamda * np.identity(m) + A
        B = np.linalg.pinv(np.dot(A.T, A))
        return np.dot(np.dot(B, A.T), np.dot(x.T, y_true))
