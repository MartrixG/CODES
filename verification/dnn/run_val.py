import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import time
from multiprocessing import Process
from dnn.train import main
import numpy as np


def count_p(p_list):
    cnt = 0
    for pro in p_list:
        if pro.is_alive():
            cnt += 1
    return cnt


def get_process(opt, dataset, m, n):
    process_list = []
    all_arg = []
    seed = opt['seed']
    opt.pop('seed')
    for s in seed:
        for key in opt:
            for j in opt[key]:
                arg = {'seed': s, 'c_in': m, 'c_out': n, 'dataset': dataset}
                if key == 'cross':
                    arg['cross_link'] = j[0]
                    arg['fully_cross'] = j[1]
                else:
                    arg[key] = j
                if arg.__contains__('first_neurons') is False:
                    arg['first_neurons'] = m // 2
                process_list.append(Process(target=main, args=(arg,)))
                all_arg.append(arg)
    return process_list, all_arg


def run_process(process_list, max_process):
    for p in process_list:
        p.start()
        time.sleep(1)
        if count_p(process_list) >= max_process:
            while True:
                time.sleep(1)
                if count_p(process_list) < max_process:
                    break
    for p in process_list:
        p.join()


def get_mn(dataset):
    if dataset == 'HAPT':
        m = 561
        n = 12
    elif dataset == 'UJI':
        m = 520
        n = 5
    else:
        raise ValueError
    return m, n


def all_train(dataset):
    m, n = get_mn(dataset)
    hidden_layers = [1, 3, 5, 7, 9]
    first_neurons = [2 * m, 4 * m, m // 2, m // 4]
    change = [2, 4, 1 / 2, 1 / 4]
    activate_func = ['tanh', 'sigmoid']
    cross = [[True, False], [True, True]]
    seed = [0, 1]
    opt = {'hidden_layers': hidden_layers,
           'first_neurons': first_neurons,
           'change': change,
           'activate_func': activate_func,
           'cross': cross,
           'seed': seed}
    process_list, all_arg = get_process(opt, dataset, m, n)

    return process_list, all_arg


def first_neurons_train(dataset):
    m, n = get_mn(dataset)
    first_neurons = [int(np.sqrt(0.43 * m * n + 0.12 * n * n + 2.54 * m + 0.77 * n + 0.35) + 0.51),
         int(np.sqrt(m * n)),
         int(np.sqrt(m + n) + 6),
         m // 4,
         m // 2,
         m * 2]
    seed = [0, 1]
    opt = {'first_neurons': first_neurons, 'seed': seed}
    return get_process(opt, dataset, m, n)


def cross_hidden_train(dataset):
    m, n = get_mn(dataset)
    hidden_layers = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    seed = [0, 1, 2, 3, 4]
    opt = {'hidden_layers': hidden_layers, 'seed': seed}
    return get_process(opt, dataset, m, n)


def hidden_train(dataset):
    m, n = get_mn(dataset)
    hidden_layers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    seed = [0, 1]
    opt = {'hidden_layers': hidden_layers, 'seed': seed}
    return get_process(opt, dataset, m, n)


def cross_seed(dataset):
    m, n = get_mn(dataset)
    seed = [0, 1, 2, 3, 4]
    cross = [[True, False], [True, True]]
    opt = {'seed': seed, 'cross': cross}
    return get_process(opt, dataset, m, n)


if __name__ == '__main__':
    process, args = cross_hidden_train('UJI')
    run_process(process, 5)
