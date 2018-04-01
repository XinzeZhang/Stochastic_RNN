import numpy as np
import torch
import matplotlib.pyplot as plt
# from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

np.random.seed(0)

Amplifier_Times = 20
left_bound=0
right_bound=1
L = 1000
N = 1

x = np.empty((N, L), 'float64')
length=np.array(range(L))
random_init=np.random.randint(left_bound * Amplifier_Times, right_bound * Amplifier_Times, N).reshape(N, 1)
x[:] = length+random_init
x_input=x / 1.0 / L

y1=0.2*np.exp(-1.0*(np.power((10.0*x_input-4.0),2)))
# plt.figure()
# plt.plot(x_input,y1,'bo')
# plt.show()

y2=0.5*np.exp(-1.0*(np.power(80.0*x_input-40.0,2)))
# plt.figure()
# plt.plot(x_input,y2,'bo')
# plt.show()

y3=0.3*np.exp(-1.0*(np.power(80.0*x_input-20.0,2)))
# plt.figure()
# plt.plot(x_input,y3,'bo')
# plt.show()

data=np.add(y1,y2)
# plt.figure()
# plt.plot(x_input,data,'bo')
# plt.show()

data=np.add(data,y3)
# plt.figure()
# plt.plot(x_input,data,'bo')
# plt.show()
data1=data.T
scaler=MinMaxScaler(feature_range=(0,1))
scaler=scaler.fit(data1)
print(scaler.data_max_)
print(scaler.transform(data1))

data2=scaler.transform(data1)

plt.figure()
plt.plot(x_input,data2.T,'bo')
plt.show()

