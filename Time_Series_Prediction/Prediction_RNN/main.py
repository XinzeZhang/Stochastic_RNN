from model_gpu import *

import torch.nn as nn

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

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import time

from _data_process import *

if __name__ == '__main__':
    #------------------------------------------------------------------------
    # load dataset
    series = read_csv('chinese_oil_production.csv', header=0,
                      parse_dates=[0], index_col=0, squeeze=True)

    # transfer the dataset to array
    raw_values = series.values
    ts_values_array = np.array(raw_values)
    set_length = len(ts_values_array)

    # transform data to be stationary
    dataset = difference(raw_values, 1)

    # creat dataset train, test
    ts_look_back = 24
    dataset = create_dataset(dataset, look_back=ts_look_back)

    # split into train and test sets
    train_size = int(dataset.shape[0] * 0.8)
    diff_length = dataset.shape[0]
    test_size = diff_length - train_size
    train, test = dataset[0:train_size], dataset[train_size:]

    # transform the scale of the data
    scaler, train_scaled, test_scaled = scale(train, test)
    # train_scaled :: shape:[train_num,seq_len] which meams [batch, input_size]
    # test_scaled :: shape:[train_num,seq_len] which meams [batch, input_size]

    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    # ---------------------------------------------------------------------------------------
    # load data and make training set
    train_input_scaled = train_scaled[:, :-1, np.newaxis]
    train_input = Variable(torch.from_numpy(
        train_input_scaled), requires_grad=False)

    train_target_scaled = train_scaled[:, 1:]
    train_target = Variable(torch.from_numpy(
        train_target_scaled), requires_grad=False)

    test_input_scaled = test_scaled[:, :-1, np.newaxis]
    test_input = Variable(torch.from_numpy(
        test_input_scaled), requires_grad=False)

    test_target_scaled = test_scaled[:, 1:]
    test_target = Variable(torch.from_numpy(
        test_target_scaled), requires_grad=False)

    # ========================================================================================
    # hyper parameters
    Num_layers = 1
    Num_iters = 2000
    Hidden_size = 200
    Print_interval = 50
    Plot_interval = 1
    Optim_method='_Adam' # '_SGD' or '_Adam'
    Learning_rate = 0.001
    Cell="GRU"

    GRU_demo = GRUModel(input_dim=1,
                        hidden_size=Hidden_size,
                        output_dim=1,
                        num_layers=Num_layers,
                        cell=Cell,
                        num_iters=Num_iters,
                        optim_method=Optim_method,
                        learning_rate=Learning_rate,
                        print_interval=Print_interval,
                        plot_interval=Plot_interval).cuda()
    # ========================================================================================
    # GRU_demo.fit(train_input, train_target)

    Model_ViewList=GRU_demo.fit_view(train_input, train_target)
    # GRU_demo=Model_ViewList[0]
    Train_ViewList=Model_ViewList[1]
    #---------------------------------------------------------------------------------------
    # begin to forcast
    print('\n------------------------------------------------')
    print('Forecasting Testing Data')
    print('------------------------------------------------')
    

    Y_train = GRU_demo.predict(train_input)
    Y_train = Y_train[:, -1]
    # inverse the train pred
    Y_train = invert_scale(scaler, train_input_scaled, Y_train)
    Y_train = inverse_train_difference(raw_values, Y_train, ts_look_back)

    # get test_result
    Y_pred = GRU_demo.predict(test_input)
    Y_pred = Y_pred[:, -1]
    Y_target = test_target_scaled[:, -1]

    # get prediction loss
    MSE_loss = nn.MSELoss()
    Y_pred_torch = Variable(torch.from_numpy(Y_pred), requires_grad=False)
    Y_target_torch = Variable(torch.from_numpy(Y_target), requires_grad=False)
    MSE_pred = MSE_loss(Y_pred_torch, Y_target_torch)
    MSE_pred = MSE_pred.data.numpy()
    # inverse the test pred
    Y_pred = invert_scale(scaler, test_input_scaled, Y_pred)
    Y_pred = inverse_test_difference(
        raw_values, Y_pred, train_size, ts_look_back)

    # # print forecast
    # for i in range(len(test)):
    #     print('Predicted=%f, Expected=%f' % ( y_pred[i], raw_values[-len(test)+i]))

    plot_result(TS_values=ts_values_array,
                Train_value=Y_train,
                Pred_value=Y_pred,
                Loss_pred=MSE_pred,
                Fig_name='Prediction' + '_L' + str(Num_layers) + '_H' + str(Hidden_size) + '_I' + str(Num_iters))
    
    #---------------------------------------------------------------------------------------
    # show the view of training

    print('\n------------------------------------------------')
    print('Showing the view of train')
    print('------------------------------------------------')
    figure_size=[25,5]
    plot_train(figure_size,train_target,Train_ViewList)
