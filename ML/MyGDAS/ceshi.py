import numpy as np

x = open('data/HAPT/Train/X_train.txt')
y = open('data/HAPT/Train/Y_train.txt')
x_src = np.array([list(map(np.float32, item.split(' '))) for item in x.readlines()])
y_src = np.array(list(map(np.int, y.readlines())))
print(x_src.shape)
