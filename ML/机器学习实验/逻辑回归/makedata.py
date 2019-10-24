import numpy as np
import matplotlib.pyplot as plt
import random
import re

def get_data(num_data, sigma, data):
    if data == 'satisfy':
        num_data = int(num_data / 2)
        X = np.ones((num_data * 2, 2))
        Y = np.ones(num_data * 2)
        for i in range(num_data):
            X[i][0] = 1 + np.random.normal(0, sigma)
            X[i][1] = 1 + np.random.normal(0, sigma)
            Y[i] = 1
            X[i + num_data][0] = 2 + np.random.normal(0, sigma)
            X[i + num_data][1] = 2 + np.random.normal(0, sigma)
            Y[i + num_data] = 0
        Y = Y[:, np.newaxis]
        return X, Y
    if data == 'dissatisfy':
        num_data = int(num_data / 2)
        X = np.ones((num_data * 2, 2))
        Y = np.ones(num_data * 2)
        for i in range(num_data):
            X[i][0] = np.random.normal(0.25, sigma)
            X[i][1] = X[i][0] + np.random.normal(0, sigma)
            Y[i] = 1
            X[i + num_data][0] = np.random.normal(0.75, sigma)
            X[i + num_data][1] = X[i + num_data][0] + np.random.normal(0, sigma)
            Y[i + num_data] = 0
        Y = Y[:, np.newaxis]
        return X, Y
    if data == 'iris':
        X_train = np.zeros((80, 4))
        Y_train = np.ones((40, 1), dtype = float)
        Y_train = np.concatenate((Y_train, np.zeros((40, 1))), axis = 0)
        X_test  = np.zeros((20, 4))
        Y_test  = np.ones((10, 1), dtype = float)
        Y_test  = np.concatenate((Y_test, np.zeros((10, 1))), axis = 0)
        filepath = "D:\LEARNING\CODES\ML\机器学习实验\逻辑回归\data\iris.txt"
        f = open(filepath)
        lines = f.readlines()
        p = re.compile(',')
        for i in range(100):
            l = p.split(lines[i])
            if(i <= 39):
                X_train[i, np.array([0, 1, 2, 3], dtype = 'i')] = [l[0], l[1], l[2], l[3]]
            if(i <= 49 and i > 39):
                X_test[i - 40, np.array([0, 1, 2, 3], dtype = 'i')] = [l[0], l[1], l[2], l[3]]
            if(i <= 89 and i > 49):
                X_train[i - 10, np.array([0, 1, 2, 3], dtype = 'i')] = [l[0], l[1], l[2], l[3]]
            if(i > 89):
                X_test[i - 80, np.array([0, 1, 2, 3], dtype = 'i')] = [l[0], l[1], l[2], l[3]]
        return X_train, Y_train, X_test, Y_test