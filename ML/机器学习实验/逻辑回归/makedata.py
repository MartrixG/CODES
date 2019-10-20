import numpy as np
import matplotlib.pyplot as plt
import random

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