# -*- coding: utf-8 -*-

import scipy.io as sio
from simple_esn import SimpleESN
from sklearn.linear_model import Ridge
import numpy as np
from sklearn.metrics import accuracy_score

# 载入数据
def load_matsets(dataset_name):
    # Load dataset
    data=sio.loadmat('./dataset/'+dataset_name+'.mat')
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
    return  X, Y, Xte, Yte  

X, Y, Xte, Yte=load_matsets('PHAL')

# Simple training
my_esn = SimpleESN(n_readout=1000, n_components=1000,
                    damping = 0.3, weight_scaling = 1.25)

# echo_train or echo_test 即 BDESN-ESN中reservoir.get_states的返回值
echo_train = my_esn.fit_transform(X)
regr = Ridge(alpha = 2.12)
regr.fit(echo_train, Y)
echo_test = my_esn.transform(Xte)

y_true = Yte

logits=regr.predict(echo_test)

# y_pred=np.argmax(logits, axis=1)

accuracy = accuracy_score(y_true, y_pred)
# f1 = f1_score(y_true, y_pred, average='weighted')

print('Acc: %.3f' % accuracy)