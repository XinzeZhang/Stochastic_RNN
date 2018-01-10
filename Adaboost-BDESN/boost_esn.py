# -*- coding: utf-8 -*-
# general imports
import math
import numpy as np
import tensorflow as tf
import time
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder

# Custom imports
from modules import train_ESN, train_RNN, train_BDESN
from reservoir import Reservoir
from tf_utils import train_tf_model



use_seed = False  # set to False to generate different random initializations at each execution

# ------ Hyperparameters ------
# Parameters for ESN and BDESN
n_internal_units = 1000  # size of the reservoir
connectivity = 0.33  # percentage of nonzero connections in the reservoir
spectral_radius = 1.12  # largest eigenvalue of the reservoir
input_scaling = 0.47  # scaling of the input weights
noise_level = 0.07  # noise in the reservoir state update

# Parameters specific to ESN
w_ridge = 2.12  # Regularization coefficient for ridge regression
embedding_method='identity'

def echo_state_network(data_matrix,label,reservoir):
    # Initialize reservoir
    if reservoir is None:
        reservoir = Reservoir(n_internal_units, spectral_radius, connectivity,
                              input_scaling, noise_level)

    res_states = reservoir.get_states(data_matrix, embedding=embedding_method,
                                      n_dim=None, train=True, bidir=False)

    # Readout training
    readout = Ridge(alpha=w_ridge)
    readout.fit(res_states, label)
    return reservoir, readout

def boosting_classify_esn(data_matrix, reservoir, readout):
    # Compute reservoir states
    res=reservoir
    rout = readout

    res_states = res.get_states(data_matrix, embedding=embedding_method,
                                      n_dim=None, train=True, bidir=False)

    logits = rout.predict(res_states)
    pred_class = np.argmax(logits, axis=1)

    return pred_class


def boosting_train_ESN(X, Y, Xte, Yte, data_weights):
    """
    :param: data_matrix: 测试集, 样本按行排列
    :param: labels: 标注
    :param: data_weights: 训练集样本权重
    # :param: step_number: 迭代次数, 亦即设置阈值每一步的步长

    :return: esn: ESN, 用dict实现
    :return: return_prediction: 预测的标注值
    :return: min_error: 最小损失函数值
    Boosting_ESN训练函数
    输入: 训练集, 训练集权重, 迭代次数
    输出: ESN, 输出值, 最小损失
    """
    num_instances=Xte.shape[0]
    esn={}
    return_prediction = np.zeros((num_instances, 1))
    min_error = np.inf

    # one-hot encoding for labels
    onehot_encoder = OneHotEncoder(sparse=False)
    Y = onehot_encoder.fit_transform(Y)
    Yte = onehot_encoder.transform(Yte)

    num_classes = Y.shape[1]
   
    print('--- Training ESN ---')

    # Initialize timer
    time_tr_start=time.time()

    # Initialize echo state network to get reservoir and readout
    reservoir, readout = echo_state_network(X,Y,reservoir=None)

    # training end
    training_time = (time.time()-time_tr_start)/60

    # Prediction
    pred_class = boosting_classify_esn(Xte, reservoir, readout)
    true_class = np.argmax(Yte, axis=1)
    
    accuracy = accuracy_score(true_class, pred_class)

    if num_classes > 2:
        f1 = f1_score(true_class, pred_class, average='weighted')
    else:
        f1 = f1_score(true_class, pred_class, average='binary')
    
    print('\tTot training time: %.3f' % (training_time))
    print('\tAcc: %.3f, F1: %.3f' % (accuracy, f1))

    is_error = np.ones((num_instances, 1))
    is_error[pred_class == true_class] = 0
    weighted_error = np.dot(data_weights.T, is_error)

    pred_class=pred_class.reshape((num_instances,1))
    if weighted_error < min_error:
                    min_error = weighted_error
                    return_prediction = pred_class.copy()
    esn['reservoir']=reservoir
    esn['readout']=readout
    return  esn,return_prediction, min_error


