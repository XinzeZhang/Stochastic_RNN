# -*- coding: utf-8 -*-

import scipy.io as sio
from td_esn import SimpleESN
from sklearn.linear_model import Ridge
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

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

# Simple training
# ------ Hyperparameters ------
# Parameters for ESN
hp_n_internal_units = 1000  # size of the reservoir
hp_connectivity = 0.33  # percentage of nonzero connections in the reservoir
hp_weight_scaling = 1.12  # scaling of the reservoir weight
hp_input_scaling = 0.47  # scaling of the input weights
hp_noise_level = 0.07  # noise in the reservoir state update

my_esn = SimpleESN(n_components=hp_n_internal_units,
                   input_scaling=hp_input_scaling,
                   weight_scaling=hp_weight_scaling,
                   connectivity=hp_connectivity,
                   noise_level=hp_noise_level)

# echo_train or echo_test 即 BDESN-ESN中reservoir.get_states的返回值
my_esn.fit(X,Y)

# y_pred=my_esn.predict(Xte)

# onehot_encoder = OneHotEncoder(sparse=False)
# onehot_encoder.fit_transform(Yte)
# y_true_encoded = onehot_encoder.transform(Yte)
# y_true = np.argmax(y_true_encoded, axis=1)

# # y_true=Yte
# accuracy = accuracy_score(y_true, y_pred)
# print('Acc: %.3f' % accuracy)
# # f1 = f1_score(y_true, y_pred, average='weighted')


# Yte=Yte.reshape((Yte.shape[0],))
# # Yte=Yte.astpye(y_pred.dtype)
# Yte=Yte.astype(y_pred.dtype)
accuracy = my_esn.accuracy(Xte, Yte)
print('Acc: %.3f' % accuracy)