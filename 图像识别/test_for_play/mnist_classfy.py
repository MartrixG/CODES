# Python 3.6.9 64-bit(tensorflow-gpu)

from keras.datasets import mnist
from keras.optimizers import RMSprop
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.utils import np_utils
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], -1) / 255
x_test = x_test.reshape(x_test.shape[0], -1) / 255

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

model = Sequential([
    Dense(32, input_dim=784),
    Activation('relu'),
    Dense(10),
    Activation('softmax')
])

# model.summary()

rmsprop = RMSprop(lr=0.001)

model.compile(
    optimizer=rmsprop,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=2, batch_size=32)
loss, accuracy = model.evaluate(x_test, y_test)
print(loss, accuracy)
