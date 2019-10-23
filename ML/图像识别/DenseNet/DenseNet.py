#Pyhton 3.6.9 64-bit("tensorflow-gpu")

from keras.layers import Conv2D, Concatenate, AveragePooling2D, BatchNormalization, Input, Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.models import Model
import Data_process as data

import os

num_classes = 10
epochs = 200
batch_size = 64

x_train, y_train, x_test, y_test, x_dev, y_dev = data.load_data()

def Dense_block(X, num_layers, growth_rate, block_name, dropout_rate = None, bottle_rate = None):
    features = [X]
    filters = int(X.shape[1])
    for i in range(num_layers):
        if(bottle_rate != None):
            X = Conv2D(filters = growth_rate * bottle_rate, kernel_size = 1, strides = 1, padding = 'same',
                        kernel_initializer = 'he_normal', kernel_regularizer = l2(1e-4), activation = 'relu',
                        name = "bottle_conv1_" + str(block_name) + "_" + str(i + 1))(X)
        X = Conv2D(filters = growth_rate, kernel_size = 3, strides = 1, padding = 'same', 
                    kernel_initializer = 'he_normal', kernel_regularizer = l2(1e-4), activation = 'relu')(X)
        if(dropout_rate != None):
            X = Dropout(1 - dropout_rate)(X)
        features.append(X)
        X = Concatenate(axis = 1)(features)
        filters += growth_rate
    return X, filters

def transition_block(X, filters, dropout_rate = None):
    X = Conv2D(filters = filters, kernel_size = 1, strides = 1, padding = 'same',
                kernel_initializer = 'he_normal', kernel_regularizer = l2(1e-4), activation = 'relu')(X)
    if(dropout_rate != None):
        X = Dropout(1 - dropout_rate)(X)
    X = AveragePooling2D((2, 2), strides = 2)(X)
    X = BatchNormalization()(X)
    return X

def DenseNet(input_shape = (3, 32, 32), num_class = num_classes, growth_rate = 12, 
            num_block = 3, num_layers =9, num_filter = 16, droup_out = None):
    X_input = Input(input_shape)
    X = Conv2D(filters = num_filter, kernel_size = 3, strides = 1, padding = 'same',
                kernel_initializer = 'he_normal', kernel_regularizer = l2(1e-4), activation = 'relu',
                name = 'pre_conv')(X_input)
    X = BatchNormalization()(X)

    for i in range(num_block - 1):
        X, filters = Dense_block(X, num_layers, growth_rate, i + 1, droup_out, 4)
        filters = int(filters * 0.5)
        X = transition_block(X, filters, droup_out)
    
    X, filters = Dense_block(X, num_layers, growth_rate, num_block, droup_out, 4)
    X = GlobalAveragePooling2D()(X)
    X = Dense(num_class, activation = 'softmax', kernel_initializer = 'he_normal', kernel_regularizer = l2(1e-4))(X)

    model = Model(inputs = X_input, outputs = X)
    return model

model = DenseNet(input_shape = (3, 32, 32), num_class = num_classes)
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

checkpoint = ModelCheckpoint(filepath = './cifar10_DenseNet.h5',monitor = 'val_acc', verbose=1,save_best_only = True)
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

os.system('echo hello world')