import numpy as np
import torch

np.random.seed(2)

T = 20
L = 1000
N = 100

x = np.empty((N, L), 'int64')
length=np.array(range(L))
random_init=np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
x[:] = length+random_init
data = np.sin(x / 1.0 / T).astype('float64')
torch.save(data, open('traindata.pt', 'wb'))