
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

use_cuda=torch.cuda.is_available()

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

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t= self.gru1(input_t, h_t)
            # h_t2 = self.gru2(h_t, h_t2)
            output = self.linear(h_t).cuda()
            outputs += [output]
        for i in range(future):# if we should predict the future
            h_t= self.gru1(output, h_t)
            h_t2 = self.gru2(h_t, h_t2)
            output = self.linear(h_t2).cuda()
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2).cuda()
        return outputs



if __name__ == '__main__':

    # check cuda
    if use_cuda:
        print("cuda is available! Using GPU GTX1070.\n")
        print("--- Training GRUs ---")
        
        # set random seed to 0
        np.random.seed(0)
        torch.manual_seed(0)
        # load data and make training set
        # data = torch.load('real-valued-function.pt')

        X = loadtxt('MackeyGlass_t17.txt')
        X = atleast_2d(X).T
        train_length = 2000
        test_length = 2000
        # data shape should be (lens_ts, n_features)
        train_input = X[:train_length]
        train_target = X[1:train_length+1]
        test_input = X[train_length:train_length+test_length]
        test_target = X[train_length+1:train_length+test_length+1]

        # data shape should be (batch, lens_ts)
        input = Variable(torch.from_numpy(train_input.T), requires_grad=False).cuda()
        target = Variable(torch.from_numpy(train_target.T), requires_grad=False).cuda()
        test_input = Variable(torch.from_numpy(test_input.T), requires_grad=False).cuda()
        test_target = Variable(torch.from_numpy(test_target.T), requires_grad=False).cuda()
        # build the model
        seq = Sequence().cuda()
        seq.double()
        criterion = nn.MSELoss()
        # use LBFGS as optimizer since we can load the whole data to train
        optimizer = optim.LBFGS(seq.parameters(), lr=0.8)
        # optimizer = optim.SGD(seq.parameters(), lr=0.001)
        
        # Initialize timer
        time_tr_start=time.time()

        #begin to train
        for i in range(15):
            print('STEP: ', i)
            def closure():
                optimizer.zero_grad()
                out = seq(input)
                loss = criterion(out, target)
                # record train time
                training_time = time.time()-time_tr_start
                print('MSE: %.10f \t Total time: %.3f' % (loss.data[0], training_time))
                loss.backward()
                return loss
            optimizer.step(closure)
            # begin to predict
            future = 0
            pred = seq(test_input, future = future)
            # loss = criterion(pred[:, :-future], test_target)
            # print('test loss:', loss.data[0])
            y_pred = pred.cpu().data.numpy()
            err = mean_squared_error(test_target.cpu().data.numpy(), y_pred)

            # draw the result
            plt.figure(figsize=(30,10))
            plt.title('Predict future values for time sequences\n(MSE %0.8f)' % err, fontsize=30)
            plt.xlabel('x', fontsize=20)
            plt.ylabel('y', fontsize=20)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            def draw(yi, color, lable_name):
                plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color,label=lable_name, linewidth = 2.0)
                # plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth = 2.0)
            draw(test_target.cpu().data.numpy()[0], 'r','Test Target')
            draw(y_pred[0], 'g','Test Prediction')
            plt.legend(loc='upper right')
            # draw(y[2], 'b')
            plt.savefig('./GRU_Results/predict%d.png'%i)
            plt.close()