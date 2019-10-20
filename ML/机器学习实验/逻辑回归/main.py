import numpy as np
import random
import matplotlib.pyplot as plt
import makedata
import model as nn

args = dict(
    epoch = 300,
    num_x = 200,
    sigma = 0.15,
    lamda = 0.,
    LR    = 0.01,
    data  = 'satisfy' # cat, dissatisfy, satisfy
)

num_x = args['num_x']
sigma = args['sigma']
epoch = args['epoch']
lr    = args['LR']
lamda = args['lamda']
data  = args['data']

train_X, train_Y = makedata.get_data(num_x, sigma, data)
x_min = min(train_X[...,0]) - 0.1
x_max = max(train_X[...,0]) + 0.1
y_min = min(train_X[...,1]) - 0.1
y_max = max(train_X[...,1]) + 0.1
x0 = np.ones((1, num_x))
inputs = np.insert(train_X, 2, values = x0, axis = 1)

net = nn.network(args)

plt.ion()
for i in range(epoch):
    output = net.forward(inputs)
    optimizer = nn.GDoptim(net.paramaters(), inputs, lr, lamda)
    loss_function = nn.binary_crossentropy_Loss(output, train_Y, net.paramaters(), lamda) 
    loss_function.backward(optimizer, net)

    loss = loss_function.loss
    print(loss.A[0]/num_x)
    plt.cla()
    w = net.paramaters()
    plt.xlim((x_min, x_max))
    plt.ylim((y_min, y_max))
    for i in range(num_x):
        if(output[i] > 0):
            plt.plot(train_X[i][0], train_X[i][1], 'ro')
        else:
            plt.plot(train_X[i][0], train_X[i][1], 'bo')
    X = np.linspace(min(train_X[...,0]), max(train_X[...,0]), 5)
    Y = - w[0].A[0] / w[1].A[0] * X - w[2].A[0] / w[1].A[0]
    plt.plot(X, Y)
    plt.pause(0.1)

plt.figure()
plt.xlim((x_min, x_max))
plt.ylim((y_min, y_max))
for i in range(num_x):
    if(train_Y[i] > 0):
        plt.plot(train_X[i][0], train_X[i][1], 'ro')
    else:
        plt.plot(train_X[i][0], train_X[i][1], 'bo')
plt.ioff()
plt.show()