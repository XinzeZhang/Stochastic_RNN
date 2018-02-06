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
    ori_array=ori_array.reshape(ori_array.shape[0],1)
    # maintain the broadcast shape with scaler
    pre_inverted=concatenate((ori_array, pred_array), axis=1)
    inverted = scaler.inverse_transform(pre_inverted)
    # extraction the pred_array_inverted
    pred_array_inverted=inverted[:,-1]
    return pred_array_inverted

# show loss
def lossPlot(points):
    plt.figure()
    # fig, ax = plt.subplots()
    # # this locator puts ticks at regular intervals
    # loc = ticker.MultipleLocator(base=0.2)
    # ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig('loss.png')
    plt.close()

class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.gru1 = nn.GRUCell(1, 50)
        self.gru2 = nn.GRUCell(50, 50)
        self.linear = nn.Linear(50, 1)

    def forward(self, input, future = 0):
        outputs = []
        h_t = Variable(torch.zeros(input.size(0), 50).double(), requires_grad=False).cuda()
        h_t2 = Variable(torch.zeros(input.size(0), 50).double(), requires_grad=False).cuda()

        for i,input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t= self.gru1(input_t, h_t)
            h_t2 = self.gru2(h_t, h_t2)
            output = self.linear(h_t2).cuda()
            outputs += [output]
        for i in range(future):# if we should predict the future
            h_t= self.gru1(output, h_t)
            h_t2 = self.gru2(h_t, h_t2)
            output = self.linear(h_t2).cuda()
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2).cuda()
        return outputs

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))



if __name__ == '__main__':
    #------------------------------------------------------------------------
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

    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    # load data and make training set
    input_scaled=train_scaled[:,:1]
    input = Variable(torch.from_numpy(input_scaled),requires_grad=False).cuda()
    target_scaled=train_scaled[:,1:]
    target= Variable(torch.from_numpy(target_scaled),requires_grad=False).cuda()
    test_input_scaled=test_scaled[:, :1]
    test_input = Variable(torch.from_numpy(test_input_scaled), requires_grad=False).cuda()
    test_target_scaled=test_scaled[:, 1:]
    test_target = Variable(torch.from_numpy(test_target_scaled), requires_grad=False).cuda()
    # ----------------------------------------------------------------------------------------
    # hyper parameters
    n_iters=2000
    print_every=100
    plot_every=1
    learning_rate=0.001

    # build the model
    seq = Sequence().cuda()
    seq.double()
    # define the loss
    criterion = nn.MSELoss()
    # use LBFGS/SGD as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(seq.parameters(), lr=learning_rate)
    
    print("Using CPU i7-7700k! \n")
    print("--- Training GRUs ---")
    
    # Initialize timer
    time_tr_start=time.time()
    
    plot_losses=[]
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    
    #begin to train
    for iter in range(1, n_iters+1):
        # compute the MSE and record the loss
        def closure():
            optimizer.zero_grad()
            out = seq(input)
            loss = criterion(out, target)
            global plot_loss_total
            global print_loss_total
            plot_loss_total+=loss.data[0]
            print_loss_total+=loss.data[0]
            loss.backward()
            return loss
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

    # lossPlot(plot_losses)

    #---------------------------------------------------------------------------------------
    # begin to forcast
    print('Forecasting Testing Data')

    # make a one-step forecast
    def forecast(input, future_step):
        pre=seq(input,future=future_step)
        pre=pre.cpu()
        pre=pre.data.numpy()
        return pre

    y_pred=forecast(input=test_input,future_step=0)
    y_pred=y_pred[:,-1]
    y_pred=invert_scale(scaler,test_input_scaled,y_pred)
    # invert differencing
    y_pred=inverse_difference(raw_values,y_pred,len(test_scaled)+1)
    # # print forecast
    # for i in range(len(test)):
    #     print('Predicted=%f, Expected=%f' % ( y_pred[i], raw_values[-len(test)+i]))
    raw_values_array=np.array(raw_values)
    trian_length=len(train)
    expected=raw_values_array[trian_length+1:].copy()

    plt.figure(figsize=(30,10))
    plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
    plt.xlabel('x', fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    time_period=np.arange(len(y_pred))
    plt.plot(time_period,y_pred,color='blue',linestyle='--',label='Prediction')
    plt.plot(time_period,expected,color='green',linestyle='-',label='Original')
    plt.legend(loc='upper left')
    plt.savefig('Prediction.png')
    plt.show()