from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import numpy as np
from numpy import concatenate

import math

import matplotlib
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataset = np.insert(dataset, [0] * look_back, 0)
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    dataY = np.array(dataY)
    dataY = np.reshape(dataY, (dataY.shape[0], 1))
    dataset = np.concatenate((dataX, dataY), axis=1)
    return dataset

# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
    ori = list()
    for i in range(len(yhat)):
        value=yhat[i]+history[-interval+i]
        ori.append(value)
    return Series(ori).values

# scale train and test data to [-1, 1]
def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled

# inverse scaling for a forecasted value
def invert_scale(scaler, ori_array ,pred_array):
    # reshape the array to 2D
    pred_array=pred_array.reshape(pred_array.shape[0],1)
    ori_array=ori_array.reshape(ori_array.shape[0],ori_array.shape[1])
    # maintain the broadcast shape with scaler
    pre_inverted=concatenate((ori_array, pred_array), axis=1)
    inverted = scaler.inverse_transform(pre_inverted)
    # extraction the pred_array_inverted
    pred_array_inverted=inverted[:,-1]
    return pred_array_inverted

if __name__ == '__main__':
    #------------------------------------------------------------------------
    # load dataset
    series = read_csv('chinese_oil_production.csv', header=0,
                    parse_dates=[0], index_col=0, squeeze=True)

    # transfer the dataset to array
    raw_values = series.values
    ts_values_array=np.array(raw_values)
    set_length=len(ts_values_array)

    # transform data to be stationary
    diff = difference(raw_values, 1)

    # create dataset x,y
    dataset = diff.values
    ts_look_back=12
    dataset = create_dataset(dataset, look_back=ts_look_back)

    # split into train and test sets
    train_size = int(dataset.shape[0] * 0.8)
    diff_length=dataset.shape[0]
    test_size = diff_length - train_size
    train, test = dataset[0:train_size], dataset[train_size:]

    # transform the scale of the data
    scaler, train_scaled, test_scaled = scale(train, test)

    # -----------------------------------------------------------
    input_scaled=train_scaled[:,:-1]

    Y_train=train_scaled[:,-1]
    Y_train=invert_scale(scaler,input_scaled,Y_train)

    # invert differenced train value
    def inverse_train_difference(history, y_train_prediction, look_back=1):
        ori = list()
        # appended the base
        for i in range(look_back):
            ori.append(history[i])
        # appended the inverted diff
        for i in range(len(y_train_prediction)):
            value=y_train_prediction[i]+history[i]
            ori.append(value)
        return Series(ori).values

    Y_train=inverse_train_difference(raw_values,Y_train)

    #----------------------------------------------------------
    test_input_scaled=test_scaled[:, :-1]
    Y_pred=test_scaled[:,-1]
    Y_pred=invert_scale(scaler,test_input_scaled,Y_pred)
    # invert differenced value
    def inverse_test_difference(history, Y_test_prediction, train_size):
        ori = list()
        for i in range(len(Y_test_prediction)):
            value=Y_test_prediction[i]+history[train_size+i]
            ori.append(value)
        return Series(ori).values
    Y_pred=inverse_test_difference(raw_values,Y_pred,train_size)
    # # print forecast
    # for i in range(len(test)):
    #     print('Predicted=%f, Expected=%f' % ( y_pred[i], raw_values[-len(test)+i]))
    

    train_scope=np.arange(train_size+1)
    test_scope=np.arange(train_size+1,set_length)

    plt.figure(figsize=(60,10))
    plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
    plt.xlabel('x', fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    time_period=np.arange(set_length)
    plt.plot(time_period,ts_values_array,color='green',linestyle='-',label='Original')
    plt.plot(train_scope,Y_train,'b:',label='train')
    plt.plot(test_scope,Y_pred,'r:',label='prediction')

    plt.legend(loc='upper left')
    # plt.savefig('Prediction.png')
    plt.show()