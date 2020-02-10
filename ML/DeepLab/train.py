import os
import argparse
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
import numpy as np
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from dataset import data_process
from model import DeepLabV3

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
KTF.set_session(sess)

parser = argparse.ArgumentParser(description='args to define and train model')

parser.add_argument('--classes', '-c', type=int, default=21)

parser.add_argument('--batchsize', '-bs', type=int, default=4)

parser.add_argument('--epoch', '-e', type=int, default=5)

parser.add_argument('--train_record', '-t', type=str, default="dataset/data/VOC_2012/tfrecord/train")

parser.add_argument('--val_record', '-v', type=str, default="dataset/data/VOC_2012/tfrecord/val")

parser.add_argument('--trainval_record', '-tv', type=str, default='dataset/data/VOC_2012/tfrecord/trainval')

parser.add_argument('--train_width', '-tw', type=int, default=512)

parser.add_argument('--summary', '-s', type=str, default=False)

arg = parser.parse_args()

tmp = {
    0: 0,
    14: 1.0,
    19: 2.0,
    33: 3.0,
    37: 4.0,
    38: 5.0,
    52: 6.0,
    57: 7.0,
    72: 8.0,
    75: 9.0,
    89: 10.0,
    94: 11.0,
    108: 12.0,
    112: 13.0,
    113: 14.0,
    128: 15.0,
    132: 16.0,
    147: 17.0,
    150: 18.0,
    220: 19.0
}
model = DeepLabV3()
model.compile(optimizer='adam', loss='categorical_crossentropy')


def lr_sch(epoch):
    if epoch < 50:
        return 1e-3
    if 50 <= epoch < 100:
        return 1e-4
    if epoch >= 100:
        return 1e-5


# learning rate control
lr_scheduler = LearningRateScheduler(lr_sch)
lr_reducer = ReduceLROnPlateau(
    monitor='los', factor=0.2, patience=5, mode='max', min_lr=1e-3)
checkpoint = ModelCheckpoint(filepath='./DeepLabV3.h5',
                             monitor='loss', verbose=1, save_best_only=True)
callbacks = [checkpoint, lr_scheduler, lr_reducer]


def map_func(val, dic):
    return dic[val] if val in dic else val


train_x, train_y = data_process.get_data(arg.train_record, arg.train_width)
func = np.vectorize(map_func)
train_y = func(train_y, tmp)
train_y = to_categorical(train_y, num_classes=21)

if arg.summary == 'True':
    model.summary()
else:
    model.fit(train_x, train_y, epochs=arg.epoch, batch_size=arg.batchsize, verbose=1, callbacks=callbacks)
