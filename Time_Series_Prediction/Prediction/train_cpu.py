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
    ts_look_back = 12
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
    train_input_scaled = train_scaled[:, :-1,np.newaxis]
    train_input = Variable(torch.from_numpy(
        train_input_scaled), requires_grad=False).cuda()
    train_target_scaled = train_scaled[:, 1:,np.newaxis]
    train_target = Variable(torch.from_numpy(
        train_target_scaled), requires_grad=False).cuda()

    test_input_scaled = test_scaled[:, :-1,np.newaxis]
    test_input = Variable(torch.from_numpy(
        test_input_scaled), requires_grad=False).cuda()
    test_target_scaled = test_scaled[:, 1:,np.newaxis]
    test_target = Variable(torch.from_numpy(
        test_target_scaled), requires_grad=False).cuda()
    
    # ========================================================================================
    # hyper parameters
    n_iters = 6000
    hidden_size=500
    print_every = 50
    plot_every = 1
    learning_rate = 0.1

    seq=GRUModel(inputDim=1,Hidden_size=hidden_size,outputDim=1,layerNum=1,cell="GRU")
    # define the loss
    criterion = nn.MSELoss()
    # use LBFGS/SGD as optimizer since we can load the whole data to train
    # optimizer = optim.LBFGS(seq.parameters(), lr=learning_rate)
    optimizer = optim.SGD(seq.parameters(), lr=learning_rate)

        # Initialize timer
    time_tr_start = time.time()

    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    # compute the MSE and record the loss
    def closure():
        optimizer.zero_grad()
        out = seq(train_input)
        loss = criterion(out, train_target)
        global plot_loss_total
        global print_loss_total
        plot_loss_total += loss.data[0]
        print_loss_total += loss.data[0]
        loss.backward()
        return loss

    # begin to train
    for iter in range(1, n_iters + 1):
        optimizer.step(closure)
        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.8f' % (timeSince(time_tr_start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    plot_loss(plot_losses, Fig_name='L1_H'+str(hidden_size)+'_I'+str(n_iters)+'_Loss')
