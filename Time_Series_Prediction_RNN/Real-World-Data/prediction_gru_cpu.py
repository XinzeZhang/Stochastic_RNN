from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import numpy as np
from numpy import concatenate

# from math import sqrt

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import time

import matplotlib
matplotlib.use('Agg')
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
    ori_array=ori_array.reshape(ori_array.shape[0],1)
    # maintain the broadcast shape with scaler
    pre_inverted=concatenate((ori_array, pred_array), axis=1)
    inverted = scaler.inverse_transform(pre_inverted)
    # extraction the pred_array_inverted
    pred_array_inverted=inverted[:,-1]
    return pred_array_inverted

class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.gru1 = nn.GRUCell(1, 51)
        self.gru2 = nn.GRUCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future = 0):
        outputs = []
        h_t = Variable(torch.zeros(input.size(0), 51).double(), requires_grad=False)
        h_t2 = Variable(torch.zeros(input.size(0), 51).double(), requires_grad=False)

        for i,input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t= self.gru1(input_t, h_t)
            h_t2 = self.gru2(h_t, h_t2)
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future):# if we should predict the future
            h_t= self.gru1(output, h_t)
            h_t2 = self.gru2(h_t, h_t2)
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs

if __name__ == '__main__':
    # load dataset
    series = read_csv('chinese_oil_production.csv', header=0,
                    parse_dates=[0], index_col=0, squeeze=True)

    raw_values = series.values

    # transform data to be stationary
    diff = difference(raw_values, 1)

    # create dataset x,y
    dataset = diff.values
    dataset = create_dataset(dataset, look_back=1)

    # split into train and test sets
    train_size = int(dataset.shape[0] * 0.8)
    test_size = dataset.shape[0] - train_size
    train, test = dataset[0:train_size], dataset[train_size:]

    # transform the scale of the data
    scaler, train_scaled, test_scaled = scale(train, test)

    # Initialize timer
    time_tr_start=time.time()

    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    # load data and make training set
    input_scaled=train_scaled[:,:1]
    input = Variable(torch.from_numpy(input_scaled),requires_grad=False)
    target_scaled=train_scaled[:,1:]
    target= Variable(torch.from_numpy(target_scaled),requires_grad=False)
    test_input_scaled=test_scaled[:, :1]
    test_input = Variable(torch.from_numpy(test_input_scaled), requires_grad=False)
    test_target_scaled=test_scaled[:, 1:]
    test_target = Variable(torch.from_numpy(test_target_scaled), requires_grad=False)

    # build the model
    seq = Sequence()
    seq.double()
    criterion = nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(seq.parameters(), lr=0.8)
    
    print("Using CPU i7-7700k! \n")
    print("--- Training GRUs ---")
    #begin to train
    for i in range(15):
        print('STEP: ', i)
        def closure():
            optimizer.zero_grad()
            out = seq(input)
            loss = criterion(out, target)
            # record train time
            training_time = time.time()-time_tr_start
            print('MSE: %.10f \t Total time: %.3f' % (loss.data.numpy()[0], training_time))
            loss.backward()
            return loss
        optimizer.step(closure)
    
    # begin to forcast
    print('Forecasting Testing Data')
    # make a one-step forecast
    def forecast(input, future_step):
        pre=seq(input,future=future_step)
        pre=pre.data.numpy()
        return pre

    y_pred=forecast(input=test_input,future_step=0)
    y_pred=y_pred[:,-1]
    y_pred=invert_scale(scaler,test_input_scaled,y_pred)
    # invert differencing
    y_pred=inverse_difference(raw_values,y_pred,len(test_scaled)+1)
    # print forecast
    for i in range(len(test)):
        print('Predicted=%f, Expected=%f' % ( y_pred[i], raw_values[-len(test)+i]))



  