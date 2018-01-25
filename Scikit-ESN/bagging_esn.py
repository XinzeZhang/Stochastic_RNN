# -*- coding: utf-8 -*-

import scipy.io as sio
from scikit_esn_classifier import SimpleESN
from sklearn.preprocessing import OneHotEncoder
import numpy as np
# added for bagging
from collections import Counter
from sklearn.metrics import accuracy_score

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


X, Y, Xte, Yte = load_matsets('BLOOD')




# ------ Hyperparameters ------
# Parameters for ESN
hp_n_internal_units = 1000  # size of the reservoir
hp_connectivity = 0.33  # percentage of nonzero connections in the reservoir
hp_weight_scaling = 1.12  # scaling of the reservoir weight
hp_input_scaling = 0.47  # scaling of the input weights
hp_noise_level = 0.07  # noise in the reservoir state update

# Single training
# ------------------------------------------------------------------------------
single_esn = SimpleESN(n_components=hp_n_internal_units,
                    input_scaling=hp_input_scaling,
                    weight_scaling=hp_weight_scaling,
                    connectivity=hp_connectivity,
                    noise_level=hp_noise_level)

# echo_train or echo_test 即 BDESN-ESN中reservoir.get_states的返回值
single_esn.fit(X, Y)
# my_esn.predict() has used the onehot_encoder and returns encoded y_pred
# y_pred = single_esn.predict(Xte)
accuracy = single_esn.accuracy(Xte, Yte)
print('-----------------------------------------')
print('Single training')
print('Acc: %.3f \n' % accuracy)

# transform Yte by using onehot_encoder
onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoder.fit_transform(Yte)
Yte_tranfomed = onehot_encoder.transform(Yte)
Yte_encoded = np.argmax(Yte_tranfomed, axis=1)
# print(Yte_encoded)

# bagging training
numbers_estimators=20
prediction_list=[]

for estimator in range(numbers_estimators):
    estimator=SimpleESN(n_components=hp_n_internal_units,
                    input_scaling=hp_input_scaling,
                    weight_scaling=hp_weight_scaling,
                    connectivity=hp_connectivity,
                    noise_level=hp_noise_level)
    estimator.fit(X,Y)
    y_pred=estimator.predict(Xte)
    prediction_list.append(y_pred)

prediction_list=np.array(prediction_list)

samples=len(Yte_encoded)

bagging_prediction =np.zeros((samples))
for sample in range(samples):
    lable_counts =Counter(prediction_list[:,sample])
    bagging_prediction[sample]=lable_counts.most_common(1)[0][0]

bagging_prediction=bagging_prediction.reshape((samples,1))

onehot_encoder.fit_transform(bagging_prediction)
bagging_prediction_transformed = onehot_encoder.transform(bagging_prediction)
bagging_prediction_encoded = np.argmax(bagging_prediction_transformed, axis=1)

acc=accuracy_score(Yte_encoded,bagging_prediction_encoded)

print('-----------------------------------------')
print('Bagging training')
print('Acc: %.3f' % acc)
