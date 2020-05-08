import Dataset as data
import numpy as np
#data.view_detail('cifar100', show_image=True)
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(a.reshape(3, 3, 3))