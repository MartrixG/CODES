#Python 3.7.1 64-bit

import numpy as np
import random


def getSource(num_x, sigma):
    x = np.linspace(0, 1, num_x)
    y = np.sin(2*np.pi*x) + np.random.normal(0, sigma, num_x)#添加噪声
    return x, y
