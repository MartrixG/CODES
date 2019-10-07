#Python 3.6.9 64-bit(tensorflow-gpu)

import numpy as np
import CIFAR_10process as pre

from keras.utils import np_utils
from keras import layers
from keras.layers import Input, Add, Dense, Conv2D, Activation, ZeroPadding2D, BatchNormalization,\
    Flatten, AveragePooling2D,MaxPooling2D, GlobalAveragePooling2D
from keras.models import Model, load_model
from keras.initializers import glorot_uniform

import keras.backend as K
K.set_image_data_format('channels_first')#the first dim is channel
K.set_learning_phase(1)

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'

#identity_block
def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block
    Arguments:
    X -- input tensor of shape(m, C, H, W)
    f -- integer, define the shape of kernel
    filters -- list of integers, defining the number of filters in the CONV layers
    stage -- integer, depending on the position in the network
    block -- string, naming the layers, depend on their position in the network

    Returns:
    X -- output of the identity block, thesor of shape(m, C, H, W)
    """

    # defining name base
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    # Retrieve filters
    f1, f2, f3 = filters

    # save the original value
    X_shortcut = X

    # first component(convolution layer) of this block
    X = Conv2D(
        filters = f1,
        kernel_size = (1, 1),
        strides = (1, 1),
        padding = "valid",
        name = conv_name_base + "2a",
        kernel_initializer = glorot_uniform(seed = 0)
    )(X)
    #valid mean no padding / glorot_uniform equal to Xaiver initialization
    
    X = BatchNormalization(axis = 1, name = bn_name_base + "2a")(X)
    X = Activation('relu')(X)

    # second component(convolution layer) of this block
    X = Conv2D(
        filters = f2,
        kernel_size = (f, f),
        strides = (1, 1),
        padding = "same",
        name = conv_name_base + "2b",
        kernel_initializer = glorot_uniform(seed = 0)
    )(X)
    #same mean padding(add 0 axis)
    
    X = BatchNormalization(axis = 1, name = bn_name_base + "2b")(X)
    X = Activation('relu')(X)

    # third componet(convolution layer) of this block
    X = Conv2D(
        filters = f3,
        kernel_size = (1, 1),
        strides = (1, 1),
        padding = 'valid',
        name = conv_name_base + "2c",
        kernel_initializer = glorot_uniform(seed = 0)
    )(X)

    X = BatchNormalization(axis = 1, name = bn_name_base + "2c")(X)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def convolution_block(X, f, filters, stage, block, s = 2):
    """
    Implementation of the convolutional block

    Arguments:
    X -- input tensor of shape (m, C, W, H)
    f -- integer, define the shape of kernel
    filters --  python list of integers, defining the number of filters in the CONV layers
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (m, C, W, H)
    """

    # defining nase name
    conv_name_base = "res" + str(stage) + block + '_branch'
    bn_name_base = "bn" + str(stage) + block + '_branch'

    # filters
    f1, f2, f3 = filters

    # save the input value
    X_shortcut = X

    X = Conv2D(
        filters = f1,
        kernel_size = (1, 1),
        strides = (s, s),
        name = conv_name_base + "2a",
        padding = "valid",
        kernel_initializer = glorot_uniform(seed = 0)
    )(X)
    X = BatchNormalization(axis = 1, name = bn_name_base + "2a")(X)
    X = Activation('relu')(X)

    X = Conv2D(
        filters = f2,
        kernel_size = (f, f),
        strides = (1, 1),
        name = conv_name_base + "2b",
        padding = "same",
        kernel_initializer = glorot_uniform(seed = 0)
    )(X)
    X = BatchNormalization(axis = 1, name = bn_name_base + "2b")(X)
    X = Activation('relu')(X)

    X = Conv2D(
        filters = f3,
        kernel_size = (1, 1),
        strides = (1, 1),
        name = conv_name_base + "2c",
        padding = "valid",
        kernel_initializer = glorot_uniform(seed = 0)
    )(X)
    X = BatchNormalization(axis = 1, name = bn_name_base + "2c")(X)

    X_shortcut = Conv2D(
        filters = f3,
        kernel_size = (1, 1),
        strides = (s, s),
        name = conv_name_base + "1",
        padding = "valid",
        kernel_initializer = glorot_uniform(seed = 0)
    )(X_shortcut)
    X_shortcut = BatchNormalization(axis = 1, name = bn_name_base + "1")(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def ResNet50(input_shape = (3, 32, 32), classes = 10):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    # define the input as a tensor with the shape input_shape
    X_input = Input(input_shape)

    # zero-padding
    X = ZeroPadding2D((3, 3))(X_input)#3, 32, 32

    X = Conv2D(
        filters = 64,
        kernel_size = (7, 7),
        strides = (2, 2),
        name = "conv",
        kernel_initializer = glorot_uniform(seed = 0)
    )(X)#64, 16, 16
    X = BatchNormalization(axis = 1, name = "bn_conv1")(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size = (3, 3), strides = (2, 2))(X)#64, 8, 8

    X = convolution_block(X, f = 3, filters = [64, 64, 256], stage = 2, block = 'a', s = 1)#64, 8, 8
    X = identity_block(X, f = 3, filters = [64, 64, 256], stage = 2, block = 'b')
    X = identity_block(X, f = 3, filters = [64, 64, 256], stage = 2, block = 'c')

    X = convolution_block(X, f = 3, filters = [128, 128, 512], stage = 3, block = 'a', s = 1)#64, 8, 8
    X = identity_block(X, f = 3, filters = [128, 128, 512], stage = 3, block = 'b')
    X = identity_block(X, f = 3, filters = [128, 128, 512], stage = 3, block = 'c')
    X = identity_block(X, f = 3, filters = [128, 128, 512], stage = 3, block = 'd')

    X = convolution_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block = 'a', s = 2)#64, 4, 4
    X = identity_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block = 'b')
    X = identity_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block = 'c')
    X = identity_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block = 'd')
    X = identity_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block = 'e')
    X = identity_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block = 'f')

    X = convolution_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block = 'a', s = 2)#64, 2, 2
    X = identity_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block = 'b')
    X = identity_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block = 'c')

    X = AveragePooling2D(pool_size = (2, 2), padding = "same")(X)

    X = Flatten()(X)
    X = Dense(classes, activation = "softmax", name = "fc" + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs = X_input, outputs = X, name = "ResNet50")

    return model

model = ResNet50(input_shape = (3, 32, 32), classes = 10)
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#load train data and labels
x_train, y_train = pre.load_train_data(1)
x_test, y_test = pre.load_test_data()

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

x_train = x_train.reshape(10000, 3, 32, 32) / 255
x_test = x_test.reshape(10000, 3, 32, 32) / 255

#print ("number of training examples = " + str(x_train.shape[0]))
#print ("number of test examples = " + str(x_test.shape[0]))
#print ("X_train shape: " + str(x_train.shape))
#print ("Y_train shape: " + str(y_train.shape))
#print ("X_test shape: " + str(x_test.shape))
#print ("Y_test shape: " + str(y_test.shape))

#model.summary()

model.fit(x_train, y_train, epochs = 100, batch_size = 64)
preds = model.evaluate(x_test, y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
print(model.evaluate(x_train, y_train))