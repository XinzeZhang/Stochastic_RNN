from simple_esn import SimpleESN
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from numpy import loadtxt, atleast_2d
import matplotlib.pyplot as plt
from pprint import pprint
import time
import numpy as np
import torch

from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime

from _data_process import *

if __name__ == '__main__':
    # load dataset
    series = read_csv('chinese_oil_production.csv', header=0,
                      parse_dates=[0], index_col=0, squeeze=True)

    # transfer the dataset to array
    raw_values = series.values
    ts_values_array = np.array(raw_values)
    set_length = len(ts_values_array)

    # transform data to be stationary
    dataset_difference = difference(raw_values, 1)

    # creat dataset train, test
    ts_look_back = 12
    using_difference = False
    Diff=''
    if using_difference==True:
        Diff='_Diff'
    if using_difference == True:
            # using dataset_diference for training
        dataset = dataset_difference
    else:
        # using dataset for training
        dataset = ts_values_array

    # split into train and test sets
    train_size = int(len(dataset) * 0.8)
    print('train_size: %i' % train_size)

    datset=atleast_2d(dataset).T
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler=scaler.fit(datset)
    dataset_scaled=scaler.fit_transform(datset)


    train, test = dataset_scaled[0:train_size,:], dataset_scaled[train_size:,:]
    # data shape should be (lens_ts, n_features)
    train_input = train[:-1,:]

    train_target = train[1:,:]

    test_input=test[:-1,:]

    test_target=test[1:,:]


    model_esn = SimpleESN(n_readout=1000, n_components=1000,
                          damping=0.3, weight_scaling=1.25)
    # Initialize timer
    time_tr_start=time.time()

    echo_train_state = model_esn.fit_transform(train_input)
    regr = Ridge(alpha=0.01)
    regr.fit(echo_train_state, train_target)

    # show the train result
    echo_train_target, echo_train_pred= train_target, regr.predict(echo_train_state)

    err_train = mean_squared_error(echo_train_target, echo_train_pred)
    print ("done in %0.3f s" % (time.time()-time_tr_start))

    data_figures = plt.figure(figsize=(12, 4))
    trainplot = data_figures.add_subplot(1, 3, 1)
    trainplot.plot(train_input[:], 'b')
    trainplot.set_title('Training Signal')

    echoplot = data_figures.add_subplot(1, 3, 2)
    # echoplot.plot(echo_train_state[0, :20])
    echoplot.plot(echo_train_state[:, :8])
    echoplot.set_title('Some Reservoir Activation')

    testplot = data_figures.add_subplot(1, 3, 3)
    testplot.plot(train_target[:], 'r', label='Target Signal')
    testplot.plot(echo_train_pred[:], 'g', label='Prediction')
    testplot.set_title('Prediction (MSE %0.8f)' % err_train)

    testplot.legend(loc='upper right')
    plt.tight_layout(0.5)
    plt.savefig('RealWorld_ESN_Train_Prediction.png')

    # show the test result
    echo_test_state=model_esn.transform(test_input)
    echo_test_target, echo_test_pred = test_target, regr.predict(echo_test_state)
    
    err_test = mean_squared_error(echo_test_target, echo_test_pred)

    data_figures = plt.figure(figsize=(12, 4))
    trainplot = data_figures.add_subplot(1, 3, 1)
    trainplot.plot(test_input[:], 'b')
    trainplot.set_title('Test Signal')

    echoplot = data_figures.add_subplot(1, 3, 2)
    echoplot.plot(echo_test_state[:, :8])
    echoplot.set_title('Some reservoir activation')

    testplot = data_figures.add_subplot(1, 3, 3)
    testplot.plot(test_target[:], 'r', label='Target signal')
    testplot.plot(echo_test_pred[:], 'g', label='Prediction')
    testplot.set_title('Prediction (MSE %0.8f)' % err_test)

    testplot.legend(loc='upper right')
    plt.tight_layout(0.5)
    plt.savefig('RealWorld_ESN_Test_Prediction.png')
