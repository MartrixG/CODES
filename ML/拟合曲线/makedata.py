import numpy as np
import random


def getSource(num_x, sigma):
    x = np.linspace(0, 1, num_x)
    y = np.sin(2*np.pi*x) + np.random.normal(0, sigma, num_x)
    return x, y
