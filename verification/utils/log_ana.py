import os

import matplotlib.pyplot as plt
import numpy as np

default = {'hidden_layers': '3', 'first_neurons': '561', 'change': '1', 'activate_func': 'relu', 'opt': 'dense_layer'}


def get_file_name(lines):
    flag = 0
    for line in lines:
        if line.find(':'):
            tmp = line.split(':')[-1].strip()
            if line.find('hidden_layers') != -1 and tmp != default['hidden_layers']:
                return 'hidden_layers_' + tmp
            if line.find('first_neurons') != -1 and tmp != default['first_neurons']:
                return 'first_neurons_' + tmp
            if line.find('change') != -1 and tmp != default['change']:
                return 'change_' + tmp
            if line.find('activate_func') != -1 and tmp != default['activate_func']:
                return 'activate_func_' + tmp
            if line.find('opt') != -1 and tmp != default['opt']:
                return 'opt_' + tmp
            if line.find('cross_link') != -1 and tmp == 'True':
                flag = 1
                continue
        if flag == 1:
            tmp = line.split(':')[-1].strip()
            if line.find('fully_cross') != -1 and tmp == 'True':
                return 'fully_cross'
            else:
                return 'cross_link'
        if line.find('classifier') != -1:
            return 'default'


def find_data(lines):
    epoch = []
    train = []
    loss = []
    valid = []
    num_epoch = 1
    for line in lines:
        if line.find('batch:[500/521]') != -1:
            tmp = line.split('\t')
            # train.append(1 - float(tmp[-2].split(':')[1][:-1]) / 100)
            loss.append(float(tmp[-3].split(':')[1]))
            epoch.append(num_epoch)
        if line.find('train_acc') != -1:
            train.append(1 - float(line.split(' ')[-1]) / 100)
        if line.find('valid_acc') != -1:
            valid.append(1 - float(line.split(' ')[-1]) / 100)
    return np.asarray(epoch), np.asarray(train), np.asarray(loss), np.asarray(valid)


def ana():
    file = '../log/figures/readme.txt'
    f_to_write = open(file, 'w')
    path = '../log'
    list_dir = os.listdir(path)
    for log in list_dir:
        if log[0] != 'D':
            continue
        log_path = path + '/' + log + '/log.txt'
        seed = log[-1]
        with open(log_path) as f:
            lines = f.readlines()
            filename = get_file_name(lines[2:]) + '_' + seed
            epoch, train, loss, valid = find_data(lines)
            l1, = plt.plot(train)
            l2, = plt.plot(valid)
            # l3, = plt.plot(loss)
            plt.legend([l1, l2], ['train_acc', 'valid_acc'], loc='upper right')
            # plt.ylim(0, 0.1)
            max_acc_x, max_acc_y = np.argmin(valid), np.min(valid)
            # max_acc_y = valid[-1]
            plt.annotate('epoch:{:}\nerr:{:.2f}%'.format(max_acc_x, max_acc_y*100),
                         xy=(max_acc_x, max_acc_y), xytext=(max_acc_x-5, max_acc_y-0.01),
                         arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
            plt.title(filename)
            plt.savefig('../log/figures/' + filename + '.png')
            plt.show()
            print(filename + ':' + '{:.2f}%'.format(max_acc_y*100), file=f_to_write)


if __name__ == '__main__':
    ana()
