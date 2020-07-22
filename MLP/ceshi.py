import torch

from model import classify_opt
from utils.data_process import get_src_dataset

model = classify_opt.OPS['group_dense_4_relu'](512, 256)
print(model.op)
x = torch.rand((2, 512, 1, 1))
print(x.shape)
# print(x[0])
y = model(x)
# print(y)
print(y.shape)
