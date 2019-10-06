#Python 3.6.9 64-bit(tensorflow-gpu)

import numpy as np
from PIL import Image
import os

path = 'D:\\LEARNING\\CODES\\ML\\图像识别\\data\\cats'
train_path = path + '\\train'
test_path = path + '\\test'
dev_path = path + '\\dev'

switchpath = {'train':'\\train', 'test':'\\test', 'dev':'\\dev'}
switchAnswer = {'train':'\\Answer_train.in', 'test':'\\Answer_test.in', 'dev':'\\Answer_dev.in'}

def getImages(kindOfFile):
    filepath = path + switchpath[kindOfFile]
    answerPath = path + switchAnswer[kindOfFile]
    files = os.listdir(filepath)
    files.sort(key = lambda x:int(x[:-4]))
    images = []
    for file in files:
        image = Image.open(filepath + '\\' + file)
        images.append(np.array(image))
    images = np.array(images)
    f = open(answerPath)
    lines = f.read()
    f.close()
    lines = lines.split()
    answers = np.array(lines[0:], dtype=np.int)
    return images, answers