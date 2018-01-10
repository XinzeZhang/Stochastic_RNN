import numpy as np

a=np.array([2,1])
a=a.reshape((2,1))

b=np.ones((2,3,2))

print(b)

instances=a.shape[0]

for i in range(instances):
    w_i=a[i]
    print(w_i)
    b[i]=b[i]*w_i

print(b)