import numpy as np
import random
import matplotlib.pyplot as plt
import makedata
import model as nn

args = dict(
    epoch = 10,
    num_x = 200,
    sigma = 0.1,
    lamda = 0.,
    LR    = 0.005,
    data  = 'dissatisfy' # iris, dissatisfy, satisfy
)

num_x = args['num_x']
sigma = args['sigma']
epoch = args['epoch']
lr    = args['LR']
lamda = args['lamda']
data  = args['data']

if(data != 'iris'):
    train_X, train_Y = makedata.get_data(num_x, sigma, data)
else:
    num_x = 80
    train_X, train_Y, test_X, test_Y = makedata.get_data(0, 0, data)
    test_inputs = np.insert(test_X, 2, values = np.ones((1, 20)), axis = 1)
x0 = np.ones((1, num_x))
inputs = np.insert(train_X, 2, values = x0, axis = 1)
net = nn.network(args)
if(data != 'iris'):
    x_min = min(train_X[...,0]) - 0.1
    x_max = max(train_X[...,0]) + 0.1
    y_min = min(train_X[...,1]) - 0.1
    y_max = max(train_X[...,1]) + 0.1
    plt.ion()
for i in range(epoch):
    output = net.forward(inputs)
    optimizer = nn.GDoptim(net.paramaters(), inputs, lr, lamda)
    loss_function = nn.binary_crossentropy_Loss(output, train_Y, net.paramaters(), lamda) 
    loss_function.backward(optimizer, net)

    loss = loss_function.loss
    print(loss.A[0]/num_x)
    if data != "iris":
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

if data != "iris":
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
else:
    w = net.paramaters()
    output = test_inputs.dot(w)
    count = 0
    for i in range(len(output)):
        if(output[i] > 0 and test_Y[i] == 1):
            count = count + 1
        if(output[i] < 0 and test_Y[i] == 0):
            count = count + 1
    print(count / 20)