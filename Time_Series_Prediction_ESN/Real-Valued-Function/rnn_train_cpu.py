
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

# use_cuda=torch.cuda.is_available()

class RNN_Model(nn.Module):
    def __init__(self):
        super(RNN_Model, self).__init__()
        self.rnn = nn.RNNCell(1, 1000)
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
    print("--- Training GRUs ---")
    
    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    # load data and make training set
    data = torch.load('real-valued-function.pt')

    # data shape should be (lens_ts, n_features)
    train_input = atleast_2d(data[0, :-1])
    train_target = atleast_2d(data[0, 1:])
    test_input = atleast_2d(data[-1, :-1])
    test_target = atleast_2d(data[-1, 1:])

    # data shape should be (batch, lens_ts)
    input = Variable(torch.from_numpy(train_input), requires_grad=False)
    target = Variable(torch.from_numpy(train_target), requires_grad=False)
    test_input = Variable(torch.from_numpy(test_input), requires_grad=False)
    test_target = Variable(torch.from_numpy(test_target), requires_grad=False)
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
    plt.savefig('RealValued_Loss_RNN.png')
    # plt.show()

    # begin to predict
    future = 0
    pred = seq(test_input, future = future)
    # loss = criterion(pred[:, :-future], test_target)
    # print('test loss:', loss.data[0])
    y_pred = pred.data.numpy()
    err = mean_squared_error(test_target.data.numpy(), y_pred)

    # draw the result
    plt.figure(figsize=(20,10))
    plt.title('Predict future values for time sequences\n(MSE: %0.8f)' % err)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xticks()
    plt.yticks()
    def draw(yi, color, lable_name):
        plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color,label=lable_name, linewidth = 1.0)
        # plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth = 2.0)
    draw(test_target.data.numpy()[0], 'r','Test Target')
    draw(y_pred[0], 'g','Test Prediction')
    plt.legend(loc='upper right')
    # draw(y[2], 'b')
    # plt.show()
    plt.savefig('RealValued_RNN.png')
    # plt.close()}