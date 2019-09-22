import torch
import numpy as np
import time
n = 10000
#for n in range(10000):
#    mat_normal = np.mat(np.random.normal(0, 1, (n, n)))
mat_tor = torch.rand(n, n)
mat_tor = mat_tor.cuda()

#    start = time.clock()
#    mat_normal_inv = np.linalg.inv(mat_normal)
#    elapsed = (time.clock() - start)
#    if(n % 100 == 0):
#        print(n)
#        print("np:", elapsed)

start = time.clock()
mat_tor_inv = torch.inverse(mat_tor)
elapsed = (time.clock() - start)
print(n)
print("cuda:", elapsed)
