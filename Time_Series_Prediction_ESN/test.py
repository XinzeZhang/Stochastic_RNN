import numpy as np
# from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import torch
import matplotlib.pyplot as plt

np.random.seed(0)

Amplifier_Times = 20
left_bound=0
right_bound=1
L = 200
N = 1

x = np.empty((N, L), 'float64')
length=np.array(range(L))
random_init=np.random.randint(left_bound * Amplifier_Times, right_bound * Amplifier_Times, N).reshape(N, 1)
x[:] = length+random_init
x_input=x / 1.0 / L
# scaler=MinMaxScaler(feature_range=(0,1))
# scaler=scaler.fit(x_input)
# x_input_scaler=scaler.transform(x_input)


x1=10.0*x_input-4.0

y2=np.power(x1,2)
y3=-1.0*(y2)
y4=0.2*np.exp(y3)

plt.figure(figsize=(20,4))
plt.plot(x_input,y4,'ro',linewidth=100.0)
plt.show()