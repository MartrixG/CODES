import makedata as data
import model as nn
import matplotlib.pyplot as plt
import numpy as np
import functional as F

plt.ion()
plt.show()


def view(w, x_rate, y_rate):
    print_x = np.vander(np.linspace(0, 1, 100), order + 1, True)
    print_y = np.dot(print_x, w) * y_rate
    plt.cla()
    plt.plot(np.linspace(0, 1, 100), print_y.T.A[0], c="r")
    plt.scatter(source_x, source_y, s=20)
    plt.pause(0.1)


args = dict(
    epoch=2000000,
    num_x=100,
    sigma=0.1,
    num_group=7,
    order=4,
    lamda=0.001,
    optim="CG",
    LR=0.1
)
epoch = args["epoch"]
order = args["order"]
LR = args["LR"]
num_x = args["num_x"]
sigma = args["sigma"]
group = num_x
lamda = args["lamda"]

source_x, source_y = data.getSource(num_x, sigma)
input = source_x / max(source_x)
input = np.vander(input, order + 1, True)
input = np.mat(input)
y_true = source_y / max(max(source_y), -min(source_y))
y_true = np.mat(y_true).T

net = nn.network(args)

if args["optim"] == "GD":
    for i in range(epoch):
        output = net.forward(input)
        optimizer = nn.GDoptim(net.parameters(), input, LR)
        loss_founction = nn.MSELoss(output, y_true)
        loss_founction.backward(optimizer, net)

        if i % 2000 == 0:
            view(net.parameters(), max(source_x),
                 max(max(source_y), -min(source_y)))
            loss = loss_founction.loss
            print(loss.A[0]/group)

if args["optim"] == "CG":
    optimizer = nn.CGoptim(net.parameters(), input, y_true)
    for i in range(order):
        output = net.forward(input)
        loss_founction = nn.MSELoss(output, y_true)
        optimizer.nextStep(net)
        view(net.parameters(), max(source_x),
             max(max(source_y), -min(source_y)))
        loss = loss_founction.loss
        print(loss.A[0]/group)
    plt.pause(2)

if args["optim"] == "MSE":
    view(F.MSE.cla_W(input, y_true), max(source_x),
         max(max(source_y), -min(source_y)))
    plt.pause(5)

if args["optim"] == "MSElamda":
    view(F.MSElam.cla_W(input, y_true, lamda, group, order + 1), max(source_x),
         max(max(source_y), -min(source_y)))
    plt.pause(5)
