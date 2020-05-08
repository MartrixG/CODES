# Python 3.6.9 64-bit(tensorflow-gpu)

import os
from keras.layers import Conv2D, Concatenate, AveragePooling2D, BatchNormalization, Input, Flatten, Dense, MaxPooling2D
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.models import Model
import PloyNet_block as block
import Data_process as data

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

num_classes = 10
epochs = 200
batch_size = 32

x_train, y_train, x_test, y_test, x_dev, y_dev = data.load_data()


def Poly_net(input_shape, num_class):
    inputs = Input(shape=input_shape)

    X = inputs
    X = Conv2D(filters=32, kernel_size=1, strides=1, padding='same',
               kernel_regularizer=l2(1e-4), kernel_initializer='he_normal')(X)
    X = BatchNormalization()(X)

    for i in range(3):
        X = block.way_2(X, filters=[(8, 24), (8, 16)],
                        beta=0.3, stage='A_'+str(i+1))
    X = BatchNormalization()(X)
    X = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(X)
    X = Conv2D(filters=48, kernel_size=1, strides=1, padding='same',
               kernel_regularizer=l2(1e-4), kernel_initializer='he_normal')(X)
    for i in range(6):
        X = block.poly_2(
            X, filters=[(16, 32), (16, 24)], beta=0.3, stage='B_'+str(i+1))
        X = block.way_2(X, filters=[(16, 32), (16, 24)],
                        beta=0.3, stage='B_'+str(i+1))
    X = Conv2D(filters=48, kernel_size=1, strides=1, padding='same',
               kernel_regularizer=l2(1e-4), kernel_initializer='he_normal')(X)
    X = BatchNormalization()(X)
    X = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(X)

    for i in range(3):
        X = block.poly_2(
            X, filters=[(32, 48), (24, 32)], beta=0.3, stage='C_'+str(i+1))
        X = block.way_2(X, filters=[(32, 48), (24, 32)],
                        beta=0.3, stage='C_'+str(i+1))
    X = AveragePooling2D(pool_size=(4, 4), strides=2, padding='same')(X)

    X = Flatten()(X)
    X = Dense(num_class, activation='softmax',
              kernel_initializer='he_normal')(X)

    model = Model(inputs=inputs, outputs=X)
    return model


model = Poly_net(input_shape=(3, 32, 32), num_class=num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

checkpoint = ModelCheckpoint(filepath='./cifar10_PolyNet.h5', monitor='val_acc',
                             verbose=1, save_best_only=True)


def lr_sch(epoch):
    # 200 total
    if epoch < 50:
        return 1e-3
    if 50 <= epoch < 100:
        return 1e-4
    if epoch >= 100:
        return 1e-5


# learing rate
lr_scheduler = LearningRateScheduler(lr_sch)
lr_reducer = ReduceLROnPlateau(
    monitor='val_acc', factor=0.2, patience=5, mode='max', min_lr=1e-3)
callbacks = [checkpoint, lr_scheduler, lr_reducer]

#model.fit(x_dev, y_dev, epochs = epochs, batch_size = batch_size)
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
          validation_data=(x_test, y_test), verbose=1, callbacks=callbacks)
# model.summary()
preds = model.evaluate(x_test, y_test)
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))
