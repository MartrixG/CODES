# Python 3.6.9 64-bit(tensorflow-gpu)

from keras.layers import Conv2D, Concatenate, Add, Activation, MaxPooling2D, Lambda
from keras.regularizers import l2
'''
stageA：包含10个2-way的基础模块
stageB：包含10个poly-3，2-way混合的基础模块（即20个基础模型）
stageC：包含5个poly-3，2-way混合的基础模块（即10个基础模块）
'''


def Inception_block(X, filters, stage, block, padding='same', initial='he_normal', active='relu'):
    (branch2, branch3) = filters

    # 3X3 Conv
    out2 = Conv2D(filters=branch2[0], kernel_size=1, strides=1, padding=padding,
                  kernel_regularizer=l2(1e-4), kernel_initializer=initial, activation=active)(X)
    out2 = Conv2D(filters=branch2[1], kernel_size=3, strides=1, padding=padding,
                  kernel_regularizer=l2(1e-4), kernel_initializer=initial, activation=active)(out2)

    # 5X5 Conv
    out3 = Conv2D(filters=branch3[0], kernel_size=1, strides=1, padding=padding,
                  kernel_regularizer=l2(1e-4), kernel_initializer=initial, activation=active)(X)
    out3 = Conv2D(filters=branch3[1], kernel_size=5, strides=1, padding=padding,
                  kernel_regularizer=l2(1e-4), kernel_initializer=initial, activation=active)(out3)

    output = Concatenate(axis=1)([out2, out3])
    return output


def poly_2(X, filters, beta, stage):
    short_cut_x = X
    f = int(X.shape[1])
    short_cut_F = Inception_block(X, filters, stage, block='poly3_F')
    short_cut_F = Conv2D(filters=f, kernel_size=1, strides=1, padding='same',
                         kernel_regularizer=l2(1e-4), kernel_initializer='he_normal')(short_cut_F)
    short_cut_G = Inception_block(short_cut_F, filters, stage, block='poly3_G')
    short_cut_G = Conv2D(filters=f, kernel_size=1, strides=1, padding='same',
                         kernel_regularizer=l2(1e-4), kernel_initializer='he_normal')(short_cut_G)

    short_cut_F = Lambda(lambda x: beta * x)(short_cut_F)
    short_cut_G = Lambda(lambda x: beta * x)(short_cut_G)

    X = Add()([short_cut_x, short_cut_F, short_cut_G])
    X = Activation('relu')(X)
    return X


def way_2(X, filters, beta, stage):
    short_cut_x = X

    f = int(X.shape[1])
    short_cut_F = Inception_block(X, filters, stage, block='2Way_F')
    short_cut_F = Conv2D(filters=f, kernel_size=1, strides=1, padding='same',
                         kernel_regularizer=l2(1e-4), kernel_initializer='he_normal')(short_cut_F)

    short_cut_G = Inception_block(X, filters, stage, block='2Way_G')
    short_cut_G = Conv2D(filters=f, kernel_size=1, strides=1, padding='same',
                         kernel_regularizer=l2(1e-4), kernel_initializer='he_normal')(short_cut_G)

    short_cut_F = Lambda(lambda x: beta * x)(short_cut_F)
    short_cut_G = Lambda(lambda x: beta * x)(short_cut_G)

    X = Add()([short_cut_x, short_cut_F, short_cut_G])
    X = Activation('relu')(X)

    return X
