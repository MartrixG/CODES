# Python 3.6.9 64-bit(tensorflow gpu)

import Data_process
from keras.layers import BatchNormalization, Input, SeparableConv2D, Flatten, Dense, AveragePooling2D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.models import Model
import keras.backend as K

K.set_image_data_format('channels_first')
# load train,test,dev data
x_train, y_train, x_test, y_test, x_dev, y_dev = Data_process.load_data()

# Hyperparameters
num_class = 10
epochs = 200
batch_size = 32

# MobileNet block
def mobile(input, f, stride = 1):
    X = SeparableConv2D(filters = f, kernel_size = 3, strides = stride, padding = 'same', depthwise_initializer = 'he_normal',
                        pointwise_initializer = 'he_normal', activation = 'relu')(input)
    X = BatchNormalization()(X)
    return X
    

# model = MobileNet(input_shape = (3, 32, 32), classes = 10)
def MobileNet(input_shape = (3, 32, 32), classes = 10):
    """
    Implementation of the popular MobileNet

    Arguments:
    input_shape --- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    X_input = Input(input_shape)
    X = X_input #3 32 32
    X = mobile(X, f = 16)
    X = mobile(X, f = 16)
    X = mobile(X, f = 16)
    X = mobile(X, f = 32, stride = 2)
    X = mobile(X, f = 32)
    X = mobile(X, f = 32)
    X = mobile(X, f = 64, stride = 2)
    X = mobile(X, f = 64)
    X = mobile(X, f = 64)
    X = AveragePooling2D(pool_size = 2)(X)
    X = Flatten()(X)
    X = Dense(classes, activation = 'softmax', kernel_initializer = 'he_normal')(X)

    model = Model(inputs = X_input, outputs = X)
    return model

model = MobileNet(input_shape = (3, 32, 32), classes = num_class)
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

checkpoint = ModelCheckpoint(filepath = './cifar10_mobile_net.h5',monitor = 'val_acc', verbose=1,save_best_only = True)
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

model.fit(x_dev, y_dev, epochs = epochs, batch_size = batch_size)
#model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size, validation_data = (x_test,y_test), verbose = 1,callbacks = callbacks)
#model.summary()
preds = model.evaluate(x_test, y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))