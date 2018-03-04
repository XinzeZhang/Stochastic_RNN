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


class Sequence(nn.Module):
    def __init__(self,Input_size=1,Hidden_size=150,Output_size=1):
        super(Sequence, self).__init__()
        self.input_size=Input_size
        self.hidden_size=Hidden_size
        self.output_size=Output_size
        
        self.gru1 = nn.GRUCell(self.input_size, self.hidden_size)
        
        self.gru2=nn.GRUCell(self.hidden_size,self.hidden_size)
        # self.gru2 = nn.GRUCell(500, 500)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, future=0):
        outputs = []
        h_t = Variable(torch.zeros(input.size(0), self.hidden_size).double(),
                       requires_grad=False).cuda()
        h_t2 = Variable(torch.zeros(input.size(0), self.hidden_size).double(),
                       requires_grad=False).cuda()
        # h_t2 = Variable(torch.zeros(input.size(0), 50).double(), requires_grad=False).cuda()

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            #input (batch, input_size): tensor containing input features
            #hidden (batch, hidden_size): tensor containing the initial hidden state for each element in the batch.
            h_t= nn.functional.relu((self.gru1(input_t, h_t)))
            # h_t2 = self.gru2(h_t, h_t2)
            output = self.linear(h_t)
            outputs += [output]
        for i in range(future):  # if we should predict the future
            h_t= self.gru1(input_t, h_t)
            # h_t2 = self.gru2(h_t, h_t2)
            output = self.linear(h_t)
            outputs += [output]
        outputs_T=torch.stack(outputs, 1).cuda()
        outputs = outputs_T.squeeze(2).cuda()
        return outputs
    
    # def initHidden(self,input):
    #     result = Variable(torch.zeros(input.size(0), self.hidden_size).double(),
    #                    requires_grad=False)
    #     return result.cuda()


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
    train_input_scaled = train_scaled[:, :-1]
    train_input = Variable(torch.from_numpy(
        train_input_scaled), requires_grad=False).cuda()
    train_target_scaled = train_scaled[:, 1:]
    train_target = Variable(torch.from_numpy(
        train_target_scaled), requires_grad=False).cuda()

    test_input_scaled = test_scaled[:, :-1]
    test_input = Variable(torch.from_numpy(
        test_input_scaled), requires_grad=False).cuda()
    test_target_scaled = test_scaled[:, 1:]
    test_target = Variable(torch.from_numpy(
        test_target_scaled), requires_grad=False).cuda()

    # ========================================================================================
    # hyper parameters
    n_iters = 6000
    hidden_size=500
    print_every = 50
    plot_every = 1
    learning_rate = 0.1

    # build the model
    seq = Sequence(Hidden_size=hidden_size).cuda()
    seq.double()
    # define the loss
    criterion = nn.MSELoss()
    # use LBFGS/SGD as optimizer since we can load the whole data to train
    # optimizer = optim.LBFGS(seq.parameters(), lr=learning_rate)
    optimizer = optim.SGD(seq.parameters(), lr=learning_rate)

    print("Using CPU i7-7700k & GPU GTX1070 ! \n")
    print("--- Training GRUs ---")

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

    #---------------------------------------------------------------------------------------
    # begin to forcast
    print('Forecasting Testing Data')

    # make a one-step forecast
    def forecast(input, future_step):
        pre = seq(input, future=future_step)
        pre = pre.cpu()
        pre = pre.data.numpy()
        return pre

    # get train_result
    Y_train = forecast(input=train_input, future_step=0)
    Y_train = Y_train[:, -1]
    # inverse the train pred
    Y_train = invert_scale(scaler, train_input_scaled, Y_train)
    Y_train = inverse_train_difference(raw_values, Y_train, ts_look_back)

    # get test_result
    Y_pred = forecast(input=test_input, future_step=0)
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

    plot_result(TS_values=ts_values_array, Train_value=Y_train,
                Pred_value=Y_pred, Loss_pred=MSE_pred, Fig_name='L1_H'+str(hidden_size)+'_I'+str(n_iters)+'_Prediction')
