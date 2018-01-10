# -*- coding: utf-8 -*-

import math
import numpy as np
import boost_esn as esn

# New added
import tensorflow as tf
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder

# Custom imports
from tf_utils import train_tf_model
from reservoir import Reservoir


onehot_encoder = OneHotEncoder(sparse=False)


class Model:
    """
    :return: model_weights: np数组,弱分类器权重
    :return: model_list: weak model list, 其中每个弱分类器用dict实现
    """

    def __init__(self, size=10):
        self.model_weights = np.zeros((size, 1))
        self.model_list = []


def adaboost_train(X, Y, Xte, Yte, iteration=10):
    """
    :param: data_matrix: (N,T,V) np数组, 训练集, 样本按行排列
    :param: labels: (N,1) np数组 标注
    :param: iteration: int 弱分类器个数
    输入训练集和弱分类器个数, 输出模型
    """
    number = X.shape[0]
    data_weights = np.ones((number, 1)) / number     # 初始化训练集权重为1/number
    # 初始化模型权重为0
    boosting_models = Model(iteration)
    Yte = Yte.reshape((number, 1))

    Y_class = onehot_encoder.fit_transform(Yte)
    num_classes = Y_class.shape[1]

    for i in range(iteration):
                
        #  训练集X乘以对应的权重
        X_weighted=X.copy()
        # for r in range(number):
        #     w_r=data_weights[r]
        #     X_weighted[r]=X[r]*w_r

        i_esn, predictions, weighted_error = esn.boosting_train_ESN(
            X_weighted, Y, Xte, Yte, data_weights)
        boosting_models.model_list.append(i_esn)
        boosting_models.model_weights[i] = (math.log(
            (1.0 - weighted_error) / weighted_error) + math.log(num_classes - 1))/num_classes
        data_weights = data_weights * \
            np.exp(-1.0 * boosting_models.model_weights[i] * Yte * predictions)
        data_weights /= np.sum(data_weights)
    return boosting_models


def adaboost_classify(input_matrix, m):
    """
    :param: data_matrix: (m,n) np数组,测试集, 样本按行排列
    :param: m: 模型
    :return: models_output: (m,1) np数组,强分类器输出值
    ensemble model, 输入训练集, 返回输出结果
    """
    models_output = np.zeros((input_matrix.shape[0], 1))
    for i in range(len(m.model_list)):
        # model_prediction = st.stump_classifier(input_matrix,
        #                                        m.model_list[i]['feature_index'],
        #                                        m.model_list[i]['threshold'],
        #                                        m.model_list[i]['rule'])
        model_prediction = esn.boosting_classify_esn(
            input_matrix, m.model_list[i]['reservoir'], m.model_list[i]['readout'])
        model_prediction = model_prediction.reshape((input_matrix.shape[0], 1))
        models_output += m.model_weights[i] * model_prediction
    return np.around(models_output)


def adaboost_test(Xte, Yte, model):
    """
    :param: data_matrix: 测试集, 样本按行排列
    :param: labels: 标注
    输入测试集和模型, 输出模型参数, 输出结果和正确率, 返回输出结果
    """

    Yte = onehot_encoder.fit_transform(Yte)
    num_classes = Yte.shape[1]
    
    models_output = adaboost_classify(Xte, model)
    models_output=models_output.astype(int)
    true_class = np.argmax(Yte, axis=1)
    true_class=true_class.reshape((true_class.shape[0],1))

    accuracy = accuracy_score(true_class, models_output)

    print('\n')
    print('Model Acc: %.3f' % accuracy)

    # if num_classes > 2:
    #     f1 = f1_score(true_class, models_output, average='weighted')
    # else:
    #     f1 = f1_score(true_class, models_output, average='binary')

    # print('\t Model Acc: %.3f, Model F1: %.3f' % (accuracy, f1))

    return models_output
