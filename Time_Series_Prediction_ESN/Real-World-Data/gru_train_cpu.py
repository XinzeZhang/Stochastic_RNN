
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

from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime

from _data_process import *
# use_cuda=torch.cuda.is_available()

class RNN_Model(nn.Module):
    def __init__(self):
        super(RNN_Model, self).__init__()
        self.rnn = nn.GRUCell(1, 1000)
        # self.rnn2 = nn.rnnCell(1000, 1000)
        self.linear = nn.Linear(1000, 1)

    def forward(self, input, future = 0):
        outputs = []
        h_t = Variable(torch.zeros(input.size(0), 1000).double(), requires_grad=False)
        h_t2 = Variable(torch.zeros(input.size(0), 1000).double(), requires_grad=False)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t= self.rnn(input_t, h_t)
            # h_t2 = self.gru2(h_t, h_t2)
            output = self.linear(h_t)
            outputs += [output]
        for i in range(future):# if we should predict the future
            h_t= self.rnn(output, h_t)
            # h_t2 = self.gru2(h_t, h_t2)
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs



if __name__ == '__main__':
    print("Using CPU I7-7700K.\n")
    print("--- Training GRU ---")
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
    # ts_look_back = 12
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

    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)

    # data shape should be (batch, lens_ts)
    input = Variable(torch.from_numpy(train_input.T), requires_grad=False)
    target = Variable(torch.from_numpy(train_target.T), requires_grad=False)
    test_input = Variable(torch.from_numpy(test_input.T), requires_grad=False)
    test_target = Variable(torch.from_numpy(test_target.T), requires_grad=False)
    # build the model
    seq = RNN_Model()
    seq.double()
    criterion = nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(seq.parameters(), lr=0.8, history_size=50)
    # optimizer = optim.Adam(seq.parameters(), lr=0.01)
    
    # Initialize timer
    time_tr_start=time.time()
    rnn_loss=[]
    loss_iter=0
    def closure():
        optimizer.zero_grad()
        out = seq(input)
        global loss_iter
        loss = criterion(out, target)
        # record train time
        # training_time = time.time()-time_tr_start
        # print('STEP: %i \t MSE: %.10f \t Total time: %.3f' % (i, loss.data[0], training_time))
        # rnn_loss.append(loss.data[0])
        loss_iter=loss.data[0]
        loss.backward()
        return loss
    #begin to train
    for i in range(10):
        optimizer.step(closure)
        # record train time
        training_time = time.time()-time_tr_start
        print('STEP: %i \t MSE: %.10f \t Total time: %.3f' % (i, loss_iter, training_time))
        rnn_loss.append(loss_iter)

    
    plt.figure(figsize=(10,10))
    plt.plot(rnn_loss)
    plt.title('Loss of Training RNN \n (Final MSE: %0.8f, Total Time: %i s)' % (rnn_loss[-1], time.time()-time_tr_start))
    plt.savefig('RealWorld_Loss_GRU.png')
    # plt.show()

    # begin to predict
    future = 0
    pred = seq(test_input, future = future)
    # loss = criterion(pred[:, :-future], test_target)
    # print('test loss:', loss.data[0])
    y_pred = pred.cpu().data.numpy()
    err = mean_squared_error(test_target.data.numpy(), y_pred)

    # draw the result
    plt.figure(figsize=(20,10))
    plt.title('Predict future values for time sequences\n(MSE: %0.8f)' % err)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xticks()
    plt.yticks()
    def draw(yi, color, lable_name):
        plt.plot(np.arange(len(yi)), yi, color,label=lable_name, linewidth = 1.0)
        # plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth = 2.0)
    draw(test_target.data.numpy()[0], 'r','Test Target')
    draw(y_pred[0], 'g','Test Prediction')
    plt.legend(loc='upper right')
    # draw(y[2], 'b')
    # plt.show()
    plt.savefig('RealWorld_GRU.png')
    # plt.close()}