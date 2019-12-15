# Python 3.6.9 64-bit(tensorflow-gpu)from keras import layers

from keras.layers import Input, Add, Dense, Conv2D, Activation, BatchNormalization
from keras.initializers import glorot_uniform
from keras.regularizers import l2

# identity_block


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
        filters=f1,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="valid",
        name=conv_name_base + "a",
        kernel_initializer=glorot_uniform(seed=0)
    )(X)
    # valid mean no padding / glorot_uniform equal to Xaiver initialization

    X = BatchNormalization(axis=1, name=bn_name_base + "1")(X)
    X = Activation('relu')(X)

    # second component(convolution layer) of this block
    X = Conv2D(
        filters=f2,
        kernel_size=(f, f),
        strides=(1, 1),
        padding="same",
        name=conv_name_base + "b",
        kernel_initializer=glorot_uniform(seed=0)
    )(X)
    # same mean padding(add 0 axis)

    X = BatchNormalization(axis=1, name=bn_name_base + "2")(X)
    X = Activation('relu')(X)

    # third componet(convolution layer) of this block
    X = Conv2D(
        filters=f3,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        name=conv_name_base + "c",
        kernel_initializer=glorot_uniform(seed=0)
    )(X)

    X = BatchNormalization(axis=1, name=bn_name_base + "3")(X)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def convolution_block(X, f, filters, stage, block, s=2):
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
        filters=f1,
        kernel_size=(1, 1),
        strides=(s, s),
        name=conv_name_base + "a",
        padding="valid",
        kernel_initializer=glorot_uniform(seed=0)
    )(X)
    X = BatchNormalization(axis=1, name=bn_name_base + "1")(X)
    X = Activation('relu')(X)

    X = Conv2D(
        filters=f2,
        kernel_size=(f, f),
        strides=(1, 1),
        name=conv_name_base + "b",
        padding="same",
        kernel_initializer=glorot_uniform(seed=0)
    )(X)
    X = BatchNormalization(axis=1, name=bn_name_base + "2")(X)
    X = Activation('relu')(X)

    X = Conv2D(
        filters=f3,
        kernel_size=(1, 1),
        strides=(1, 1),
        name=conv_name_base + "c",
        padding="valid",
        kernel_initializer=glorot_uniform(seed=0)
    )(X)
    X = BatchNormalization(axis=1, name=bn_name_base + "3")(X)

    X_shortcut = Conv2D(
        filters=f3,
        kernel_size=(1, 1),
        strides=(s, s),
        name=conv_name_base + "shortcut",
        padding="valid",
        kernel_initializer=glorot_uniform(seed=0)
    )(X_shortcut)
    X_shortcut = BatchNormalization(
        axis=1, name=bn_name_base + "shortcut")(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def res_block(inputs, num_filters=16, kernel_size=3, strides=1, activation='relu'):
    X = Conv2D(num_filters, kernel_size=kernel_size, strides=strides, padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(inputs)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(num_filters, kernel_size=kernel_size, strides=1, padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(X)
    X = BatchNormalization()(X)
    return X
