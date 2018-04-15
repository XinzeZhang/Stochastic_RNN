from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import numpy as np
from numpy import loadtxt, atleast_2d
# from numpy import atleast_2d
from sklearn.metrics import mean_squared_error

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import time

from models.model_gpu import RNNModel
from data_process._data_process import create_dataset, plot_regression_result


if __name__ == '__main__':
    print("Using GPU GTX1070.\n")
    print("--- Training RNN ---")

    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    # load data and make training set
    data = torch.load('real-valued-function.pt')

    # data shape should be (batch, ts_lens) which means input_dim=1
    train_input = atleast_2d(data[0, :-1])[:, :, np.newaxis]
    train_target = atleast_2d(data[0, 1:])[:, :, np.newaxis]
    train_target_plot = data[0, 1:]
    test_input = atleast_2d(data[-1, :-1])[:, :, np.newaxis]
    test_target = atleast_2d(data[-1, 1:])[:, :, np.newaxis]
    test_target_plot = data[-1, 1:]

    # data shape should be (batch, lens_ts, input_dim)
    train_input = Variable(torch.from_numpy(
        train_input).float(), requires_grad=False)
    train_target = Variable(torch.from_numpy(
        train_target).float(), requires_grad=False)
    test_input = Variable(torch.from_numpy(
        test_input).float(), requires_grad=False)
    test_target = Variable(torch.from_numpy(
        test_target).float(), requires_grad=False)

 # ========================================================================================
    # hyper parameters
    Num_layers = 1
    Num_iters = 100000
    Hidden_size = 100
    Print_interval = 50
    Plot_interval = 10
    # 'SGD' or 'Adam' or 'RMSprop' or 'Adadelta' or 'Adagrad' or 'SparseAdam' or 'Adamax' or 'ASGD'
    Optim_method = 'SGD'
    Learning_rate = 0.001
    Cell = "GRU"

    RNN_Demo = RNNModel(input_dim=1,
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
    RNN_Demo.fit(train_input, train_target)

    # RNN_Demo.fit(train_input, train_target,View_interval)
    # save the model
    # model_save_road='./Model/Model' + '_L' + str(Num_layers) + '_H' + str(Hidden_size) + '_I' + str(Num_iters)+Optim_method+'.pkl'
    # torch.save(RNN_Demo,model_save_road)

    # RNN_Demo=Model_ViewList[0]
    # Train_ViewList = Model_ViewList[1]
    # ---------------------------------------------------------------------------------------
    # begin to forcast
    print('\n------------------------------------------------')
    print('Forecasting Testing Data')
    print('------------------------------------------------')

    Y_train = RNN_Demo.predict(train_input)
    Y_train = Y_train[0, :, 0]

    # get test_result
    Y_pred = RNN_Demo.predict(test_input)
    Y_pred = Y_pred[0, :, 0]
    # Y_target = test_target

    # get prediction loss
    MSE_loss = nn.MSELoss()
    Y_pred_torch = Variable(torch.from_numpy(
        Y_pred).float(), requires_grad=False)
    Y_target_torch = test_target
    MSE_pred = MSE_loss(Y_pred_torch, Y_target_torch)
    MSE_pred = MSE_pred.data.numpy()

    # # print forecast
    # for i in range(len(test)):
    #     print('Predicted=%f, Expected=%f' % ( y_pred[i], raw_values[-len(test)+i]))

    plot_fig_name = './Results/'+Cell + '_L' + \
        str(Num_layers) + '_H' + str(Hidden_size) + \
        '_I' + str(Num_iters)+'_'+Optim_method
    plot_regression_result(Train_target=train_target_plot,
                           Test_target=test_target_plot,
                           Train_pred=Y_train,
                           Test_pred=Y_pred,
                           Loss_pred=MSE_pred,
                           Fig_name=plot_fig_name)
