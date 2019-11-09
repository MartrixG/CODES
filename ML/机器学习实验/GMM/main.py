import numpy as np
import makedata as data
import matplotlib.pyplot as plt

args = dict(
    x_mean=[0.2, 0.5, 0.6, 0.3, 0.7],
    y_mean=[0.2, 0.5, 0.2, 0.4, 0.5],
    sigma=[0.055, 0.055, 0.075, 0.065, 0.085],
    miu=[[[1, 1]],
         [[4, 4]],
         [[1, 4]],
         [[4, 1]]],
    Sigma=[[[0.2, -0.1],
            [-0.1, 0.2]],
           [[0.7, -0.4],
            [-0.4, 0.5]],
           [[0.2, 0.1],
            [0.1, 0.2]],
           [[0.3, -0.2],
           [-0.2, 0.2]]],
    C=['ro', 'go', 'bo', 'yo', 'ko'],
    epoch=20,
    k=4,
    num_x=100,
    kind=4,
)
epoch = args['epoch']
num_x = args['num_x']
kind = args['kind']
k = args['k']
x_mean = args['x_mean']
y_mean = args['y_mean']
sigma = args['sigma']
C = args['C']
Sigma = args['Sigma']
miu = args['miu']

X, Y = data.get_data(num_x, kind, x_mean, y_mean, sigma)
X, Y = data.get_blob(num_x, kind, miu, Sigma)
x = np.vstack((X.T, Y.T)).T
num = int(num_x * kind)

def dis(x1, y1, x2, y2):
    return np.square(x1 - x2) + np.square(y1 - y2)

def k_means(view):# num * 1
    if view:
        plt.ion()
    k_x = np.ones((num, 1)).dot(np.random.random((1, k)))#num * k
    k_y = np.ones((num, 1)).dot(np.random.random((1, k)))#num * k
    for i in range(epoch):
        if view:
            plt.cla()
        dis = pow(X - k_x, 2) + pow(Y- k_y, 2)
        group = dis.argmin(axis = 1)
        x_mean = np.zeros((k, 1))
        y_mean = np.zeros((k, 1))
        for i in range(num):
            x_mean[group[i]] += X[i]
            y_mean[group[i]] += Y[i]
        for i in range(k):
            if sum(group == i) == 0:
                x_mean[i] = np.random.rand()
                y_mean[i] = np.random.rand()
            else:
                x_mean[i] /= sum(group == i)
                y_mean[i] /= sum(group == i)
        k_x = np.ones((num, 1)).dot(x_mean.T)
        k_y = np.ones((num, 1)).dot(y_mean.T)
        if view:
            for i in range(k):
                plt.plot(x_mean[i], y_mean[i], 'mo')
            for i in range(num):
                plt.plot(X[i], Y[i], C[group[i]])
            plt.pause(0.3)    
    return (group, x_mean, y_mean)

def get_pi_miu_sig(g, x_mean, y_mean):
    mean = np.append(x_mean, y_mean, axis=1)
    pi = np.zeros(k)
    sig = np.zeros((k, 2, 2))
    for i in range(num):
        pi[group[i]] += 1
    for i in range(k):
        sig[i] = np.cov(X[i * num_x : (i + 1) * num_x].T, Y[i * num_x : (i + 1) * num_x].T)
    return pi/num, mean, sig

def pr_F(x_i, miu_j, sig_j):
    sig_det = np.linalg.det(sig_j)
    x_i = np.mat(x_i)
    miu_j = np.mat(miu_j)
    first = (((2 * np.pi) ** 2) * sig_det) ** 0.5
    point = np.exp((-0.5) * (x_i - miu_j).dot(np.linalg.inv(sig_j)).dot((x_i - miu_j).T))
    return (point / first).A[0]

def E_setp(pi, mean, sig):
    gama = np.zeros((num, k))
    for i in range(num):
        base = 0
        for j in range(k):
            base += pi[j] * pr_F(x[i], mean[j], sig[j])
        for j in range(k):
            gama[i][j] = pi[j] * pr_F(x[i], mean[j], sig[j]) / base
    return gama

def M_step(gama):
    resig = np.zeros((k, 2, 2))
    N = np.ones((2, num)).dot(gama)
    remiu = gama.T.dot(x) / N.T
    repi = np.ones((1, num)).dot(gama)[0] / num
    for i in range(k):
        for j in range(num):
            resig[i] += gama[j][i] * (x[j] - remiu[i])[:, np.newaxis].dot((x[j] - remiu[i])[np.newaxis, :])
        resig[i] /= np.ones((1, num)).dot(gama)[0][i]
    return repi, remiu, resig

group, x_mean, y_mean = k_means(view=True)
pi, mean, sig = get_pi_miu_sig(group, x_mean, y_mean)

def likelyhood(pi, mean, sig):
    re = 0
    sig_det = np.linalg.det(sig)
    for i in range(num):
        tmp = 0
        for j in range(k):
            tmp += pi[j] * pr_F(x[i], mean[j], sig[j])
        re += np.log(tmp)
    return -re/num

for i in range(epoch):
    print("epoch=",i)
    gam = E_setp(pi, mean, sig)
    pi, mean, sig = M_step(gam)
    print(likelyhood(pi, mean, sig))
print("pi =",pi)
print("mean =",mean)
print("sigma =",sig)

plt.ioff()
plt.show()