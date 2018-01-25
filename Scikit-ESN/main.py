# -*- coding: utf-8 -*-

import scipy.io as sio
from scikit_esn_classifier import SimpleESN
from sklearn.preprocessing import OneHotEncoder
import numpy as np


# 载入数据


def load_matsets(dataset_name):
    # Load dataset
    data = sio.loadmat('./dataset/' + dataset_name + '.mat')
    X = data['X']  # shape is [N,T,V]
    # if len(X.shape) < 3:
    #     X = np.atleast_3d(X)
    Y = data['Y']  # shape is [N,1]
    Xte = data['Xte']
    # if len(Xte.shape) < 3:
    #     Xte = np.atleast_3d(Xte)
    Yte = data['Yte']
    # one-hot encoding for labels
    # onehot_encoder = OneHotEncoder(sparse=False)
    # Y = onehot_encoder.fit_transform(Y)
    # Yte = onehot_encoder.transform(Yte)
    # num_classes = Y.shape[1]
    # return num_classes, x, y, xte and yte
    return X, Y, Xte, Yte


X, Y, Xte, Yte = load_matsets('LIB')

# transform Y by using onehot_encoder
onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoder.fit_transform(Yte)
Yte_encoded = onehot_encoder.transform(Yte)
y_true = np.argmax(Yte_encoded, axis=1)
print(y_true)
# Simple training
# ------ Hyperparameters ------
# Parameters for ESN
hp_n_internal_units = 1000  # size of the reservoir
hp_connectivity = 0.33  # percentage of nonzero connections in the reservoir
hp_weight_scaling = 1.12  # scaling of the reservoir weight
hp_input_scaling = 0.47  # scaling of the input weights
hp_noise_level = 0.07  # noise in the reservoir state update

my_esn1 = SimpleESN(n_components=hp_n_internal_units,
                   input_scaling=hp_input_scaling,
                   weight_scaling=hp_weight_scaling,
                   connectivity=hp_connectivity,
                   noise_level=hp_noise_level)

# echo_train or echo_test 即 BDESN-ESN中reservoir.get_states的返回值
my_esn1.fit(X,Y)

# my_esn.predict() has used the onehot_encoder and returns encoded y_pred
y_pred1=my_esn1.predict(Xte)
print(y_pred1)

my_esn2 = SimpleESN(n_components=hp_n_internal_units,
                   input_scaling=hp_input_scaling,
                   weight_scaling=hp_weight_scaling,
                   connectivity=hp_connectivity,
                   noise_level=hp_noise_level)

# echo_train or echo_test 即 BDESN-ESN中reservoir.get_states的返回值
my_esn2.fit(X,Y)

# my_esn.predict() has used the onehot_encoder and returns encoded y_pred
y_pred2=my_esn2.predict(Xte)
print(y_pred2)


accuracy = my_esn1.accuracy(Xte, Yte)
print('Acc: %.3f' % accuracy)