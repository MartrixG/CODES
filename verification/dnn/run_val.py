import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import time
from multiprocessing import Process
from dnn.train import main
import numpy as np


def fun(x):
    print(x)
    time.sleep(1)


def count_p(p_list):
    cnt = 0
    for pro in p_list:
        if pro.is_alive():
            cnt += 1
    return cnt


if __name__ == '__main__':
    m = 520
    n = 5
    hidden_layers = [3, 1, 5, 7, 9]
    first_neurons = [m, 2 * m, 4 * m, m // 2, m // 4]
    change = [1, 2, 4, 1 / 2, 1 / 4]
    activate_func = ['relu', 'tanh', 'sigmoid']

    opt = [hidden_layers, first_neurons, change, activate_func]
    default = [3, m, 1, 'relu', False, False]
    f = [int(np.sqrt(0.43 * m * n + 0.12 * n * n + 2.54 * m + 0.77 * n + 0.35) + 0.51),
         int(np.sqrt(m * n)),
         int(np.sqrt(m + n) + 6),
         m // 4,
         m // 2,
         m * 2]
    process_list = []
    flag = 0

    for i in range(4):
        for j in opt[i]:
            arg = []
            for k in range(4):
                if k == i:
                    arg.append(j)
                else:
                    arg.append(opt[k][0])
            arg.append(False)
            arg.append(False)
            if arg == default and flag == 1:
                continue
            flag = 1
            p = Process(target=main, args=(arg,))
            p.start()
            process_list.append(p)
            time.sleep(1)
            if count_p(process_list) >= 5:
                while True:
                    time.sleep(1)
                    if count_p(process_list) < 5:
                        break
    default[-2] = True
    p = Process(target=main, args=(default,))
    p.start()
    time.sleep(1)
    default[-1] = True
    p = Process(target=main, args=(default,))
    p.start()

    for p in process_list:
        p.join()
