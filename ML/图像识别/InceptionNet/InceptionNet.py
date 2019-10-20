#Python 3.6.9 64-bit(tensorflow-gpu)

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'

from keras.layers import Concatenate, Conv2D, AveragePooling2D, Activation, MaxPooling2D, BatchNormalization, Input, Flatten, Dense
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.models import Model
import Data_process as data

num_classes = 10
epochs = 100
batch_size = 64

x_train, y_train, x_test, y_test, x_dev, y_dev = data.load_data()

def Inception_net_block(X, filters, padding = 'same', initial = 'he_normal', active = 'relu'):
    """
    Implementation of inception block

    Arguments:
    X --- input tensor of shape (, C, W, H)
    filters --- list of list of integers, define the number of filters in conv layers

    returns:
    X --- output of the inception block, tensor of shape(, C, W, H)
    """

    (branch1, branch2, branch3, branch4) = filters

    # 1X1 Conv
    out1 = Conv2D(filters = branch1[0], kernel_size = 1, strides = 1, padding = padding,
                    kernel_regularizer = l2(1e-4), kernel_initializer = initial, activation = active)(X)
    # 3X3 Conv
    out2 = Conv2D(filters = branch2[0], kernel_size = 1, strides = 1, padding = padding,
                    kernel_regularizer = l2(1e-4), kernel_initializer = initial, activation = active)(X)
    out2 = Conv2D(filters = branch2[1], kernel_size = 3, strides = 1, padding = padding,
                    kernel_regularizer = l2(1e-4), kernel_initializer = initial, activation = active)(out2)

    # 5X5 Conv
    out3 = Conv2D(filters = branch3[0], kernel_size = 1, strides = 1, padding = padding,
                    kernel_regularizer = l2(1e-4), kernel_initializer = initial, activation = active)(X)
    out3 = Conv2D(filters = branch3[1], kernel_size = 5, strides = 1, padding = padding,
                    kernel_regularizer = l2(1e-4), kernel_initializer = initial, activation = active)(out3)

    # maxpool
    out4 = MaxPooling2D(pool_size = (3, 3), strides = 1, padding = padding)(X)
    out4 = Conv2D(filters = branch4[0], kernel_size = 1, strides = 1, padding = padding,
                    kernel_regularizer = l2(1e-4), kernel_initializer = initial, activation = active)(out4)

    output = Concatenate(axis = 1)([out1, out2, out3, out4])
    return output

def InceptionNet(input_shape = (3, 32, 32), classes = num_classes):
    """
    Implementation of the popular MobileNet

    Arguments:
    input_shape --- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    X_input = Input(input_shape)

    X = X_input
    X = Conv2D(filters = 16, kernel_size = 3, strides = 1, padding = 'same', kernel_regularizer = l2(1e-4), kernel_initializer = 'he_normal')(X)# 64, 32, 32
    X = BatchNormalization()(X)

    X = Inception_net_block(X, filters = [(32,),(8,32),(8,24),(32,)])
    X = Inception_net_block(X, filters = [(32,),(8,32),(8,24),(32,)])
    X = BatchNormalization()(X)
    X = MaxPooling2D(pool_size = (3, 3), strides = 2, padding = 'same')(X)

    X = Inception_net_block(X, filters = [(48,),(16,48),(16,32),(48,)])
    X = Inception_net_block(X, filters = [(48,),(16,48),(16,32),(48,)])
    X = Inception_net_block(X, filters = [(48,),(16,48),(16,32),(48,)])
    X = Inception_net_block(X, filters = [(48,),(16,48),(16,32),(48,)])
    X = Inception_net_block(X, filters = [(48,),(16,48),(16,32),(48,)])
    X = BatchNormalization()(X)
    X = MaxPooling2D(pool_size = 3, strides = 2, padding = 'same')(X)

    X = Inception_net_block(X, filters = [(64,),(32,64),(32,48),(64,)])
    X = Inception_net_block(X, filters = [(64,),(32,64),(32,48),(64,)])
    X = BatchNormalization()(X)
    X = AveragePooling2D(pool_size=(4, 4),strides=2,padding='same')(X)

    X = Flatten()(X)
    X = Dense(classes, activation = 'softmax', kernel_initializer = 'he_normal')(X)

    model = Model(inputs = X_input, outputs = X)
    return model

model = InceptionNet()
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

checkpoint = ModelCheckpoint(filepath = './cifar10_InceptionNet.h5',monitor = 'val_acc', verbose=1,save_best_only = True)
def lr_sch(epoch):
    if epoch <50:
        return 1e-3
    if 50<=epoch<100:
        return 1e-4
    if epoch>=100:
        return 1e-5

#learing rate control
lr_scheduler = LearningRateScheduler(lr_sch)
lr_reducer = ReduceLROnPlateau(monitor = 'val_acc', factor = 0.2, patience = 5, mode = 'max', min_lr = 1e-3)
callbacks = [checkpoint, lr_scheduler, lr_reducer]

#model.fit(x_dev, y_dev, epochs = epochs, batch_size = batch_size)
model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size, validation_data = (x_test,y_test), verbose = 1,callbacks = callbacks)
#model.summary()
preds = model.evaluate(x_test, y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
    