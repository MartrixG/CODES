import numpy as np


class MSE(object):
    def cla_W(x, y_true):
        A = np.dot(x.T, x)
        A = np.linalg.pinv(A)
        return np.dot(A, np.dot(x.T, y_true))


class MSElam(object):
    def cla_W(x, y_true, lamda, n, m):
        lamda = lamda * n / 2
        A = np.dot(x.T, x)
        A = lamda * np.identity(m) + A
        B = np.linalg.pinv(np.dot(A.T, A))
        return np.dot(np.dot(B, A.T), np.dot(x.T, y_true))
