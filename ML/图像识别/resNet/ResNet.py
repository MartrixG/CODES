#Python 3.6.9 64-bit(tensorflow-gpu)

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'

from keras.utils import np_utils
from keras.layers import Input, Dense, Flatten, AveragePooling2D, Add, Activation, Conv2D
from keras.models import Model, load_model
from keras.initializers import glorot_uniform
from keras.datasets import cifar10
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.regularizers import l2
import keras.backend as K
import Res_blocks as block

K.set_image_data_format('channels_first')#the first dim is channel
K.set_learning_phase(1)

num_classes = 10
batch_size = 32
epochs = 200

#load train data and labels
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

#model = ResNet50(input_shape = (3, 32, 32), classes = 10)
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
    # X = ZeroPadding2D((3, 3))(X_input)#3, 32, 32
    X = X_input

    #16, 32, 32
    X = block.convolution_block(X, f = 3, filters = [8, 8, 16], stage = 1, block = 'a', s = 1)
    X = block.identity_block(X, f = 3, filters = [8, 8, 16], stage = 1, block = 'b')
    X = block.identity_block(X, f = 3, filters = [8, 8, 16], stage = 1, block = 'c')
    X = block.identity_block(X, f = 3, filters = [8, 8, 16], stage = 1, block = 'd')
    X = block.identity_block(X, f = 3, filters = [8, 8, 16], stage = 1, block = 'e')
    X = block.identity_block(X, f = 3, filters = [8, 8, 16], stage = 1, block = 'f')

    X = block.convolution_block(X, f = 3, filters = [16, 16, 32], stage = 2, block = 'a', s = 2)#32, 16, 16
    X = block.identity_block(X, f = 3, filters = [16, 16, 32], stage = 2, block = 'b')
    X = block.identity_block(X, f = 3, filters = [16, 16, 32], stage = 2, block = 'c')
    X = block.identity_block(X, f = 3, filters = [16, 16, 32], stage = 2, block = 'd')
    X = block.identity_block(X, f = 3, filters = [16, 16, 32], stage = 2, block = 'e')
    X = block.identity_block(X, f = 3, filters = [16, 16, 32], stage = 2, block = 'f')

    X = block.convolution_block(X, f = 3, filters = [32, 32, 64], stage = 3, block = 'a', s = 2)#64, 8, 8
    X = block.identity_block(X, f = 3, filters = [32, 32, 64], stage = 3, block = 'b')
    X = block.identity_block(X, f = 3, filters = [32, 32, 64], stage = 3, block = 'c')
    X = block.identity_block(X, f = 3, filters = [32, 32, 64], stage = 3, block = 'd')
    X = block.identity_block(X, f = 3, filters = [32, 32, 64], stage = 3, block = 'e')
    X = block.identity_block(X, f = 3, filters = [32, 32, 64], stage = 3, block = 'f')

    X = AveragePooling2D(pool_size = (2, 2), padding = "same")(X)#64, 2, 2

    X = Flatten()(X)
    X = Dense(classes, activation = "softmax", name = "fc" + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)

    model = Model(inputs = X_input, outputs = X, name = "ResNet50")

    return model

#model = ResNet20(input_shape = (3, 32, 32), classes = 10)
def ResNet20(input_shape = (3, 32 ,32), classes = 10):
    inputs = Input(shape = input_shape)

    x = block.res_block(inputs)
    for i in range(6):
        tmp = block.res_block(inputs = x)
        x = Add()([x, tmp])
        x = Activation('relu')(x)
    
    for i in range(6):
        if i == 0:
            tmp = block.res_block(inputs = x, strides = 2, num_filters = 32)
        else:
            tmp = block.res_block(inputs = x, num_filters = 32)
        if i == 0:
            x = Conv2D(32, kernel_size = 3, strides = 2, padding = 'same',
                        kernel_initializer = 'he_normal', kernel_regularizer = l2(1e-4))(x)
        x = Add()([x, tmp])
        x = Activation('relu')(x)

    for i in range(6):
        if i == 0:
            tmp = block.res_block(inputs = x, strides = 2, num_filters = 64)
        else:
            tmp = block.res_block(inputs = x, num_filters = 64)
        if i == 0:
            x = Conv2D(64, kernel_size = 3, strides = 2, padding = 'same',
                        kernel_initializer = 'he_normal', kernel_regularizer = l2(1e-4))(x)
        x = Add()([x, tmp])
        x = Activation('relu')(x)
    x = AveragePooling2D(pool_size = 2)(x)
    x = Flatten()(x)
    x = Dense(num_classes, activation = 'softmax', kernel_initializer = 'he_normal')(x)

    model = Model(inputs = inputs, outputs = x)
    return model

model = ResNet20(input_shape = (3, 32, 32), classes = num_classes)  
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

checkpoint = ModelCheckpoint(filepath='./cifar10_resnet20.h5',monitor='val_acc',
                             verbose=1,save_best_only=True)
def lr_sch(epoch):
    #200 total
    if epoch <50:
        return 1e-3
    if 50<=epoch<100:
        return 1e-4
    if epoch>=100:
        return 1e-5

#learing rate 
lr_scheduler = LearningRateScheduler(lr_sch)
lr_reducer = ReduceLROnPlateau(monitor = 'val_acc', factor = 0.2, patience = 5, mode = 'max', min_lr = 1e-3)
callbacks = [checkpoint, lr_scheduler, lr_reducer]
model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size, validation_data = (x_test,y_test), verbose = 1,callbacks = callbacks)
#model.summary()
preds = model.evaluate(x_test, y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))