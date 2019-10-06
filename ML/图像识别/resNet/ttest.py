#Python 3.6.9 64-bit(tensorflow-gpu)

import numpy as np
import CIFAR_10process as pre
from keras.utils import np_utils

data, label = pre.load_all_train_data()
label = np_utils.to_categorical(label, 10)
