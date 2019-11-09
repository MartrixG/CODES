import numpy as np
import matplotlib.pyplot as plt

def get_data(num_x, num_k, x_mean, y_mean, sigma):
    num = int(num_k * num_x)
    X = np.zeros(num)
    Y = np.zeros(num)
    for i in range(num_k):
        for j in range(num_x):
            X[i * num_x + j] = np.random.normal(x_mean[i], sigma[i])
            Y[i * num_x + j] = np.random.normal(y_mean[i], sigma[i])
    X = X[:, np.newaxis]
    Y = Y[:, np.newaxis]
    return X, Y
def get_blob(num_x, num_k, miu, sigma):
    S = np.mat(sigma[0])
    R = np.linalg.cholesky(S)
    x = np.dot(np.random.randn(num_x, 2), R) + np.ones((num_x, 1)).dot(miu[0])
    for i in range(num_k - 1):
        S = np.mat(sigma[i + 1])
        R = np.linalg.cholesky(S)
        x = np.append(x, np.dot(np.random.randn(num_x, 2), R) + np.ones((num_x, 1)).dot(miu[i + 1]), axis=0)
    return np.array(x[:, 0]), np.array(x[:, 1])