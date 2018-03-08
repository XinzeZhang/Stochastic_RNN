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

import matplotlib.pyplot as plt
from matplotlib import animation

# 模型基类，主要是用于指定参数和cell类型
class BaseModel(nn.Module):

    def __init__(self, input_dim=1, hidden_size=150, output_dim=1, num_layers=1, cell="GRU"):

        super(BaseModel, self).__init__()
        self.Input_dim = input_dim
        self.Output_dim = output_dim
        self.Hidden_Size = hidden_size
        self.Num_layers = num_layers
        if cell == "RNN":
            self.Cell = nn.RNN(input_size=self.Input_dim, hidden_size=self.Hidden_Size,
                        num_layers=self.Num_layers, dropout=0.0,
                         nonlinearity="tanh", batch_first=True,)
        if cell == "LSTM":
            self.Cell = nn.LSTM(input_size=self.Input_dim, hidden_size=self.Hidden_Size,
                               num_layers=self.Num_layers, dropout=0.0,
                               batch_first=True, )
        if cell == "GRU":
            self.Cell = nn.GRU(input_size=self.Input_dim, hidden_size=self.Hidden_Size,
                                num_layers=self.Num_layers, dropout=0.0,
                                 batch_first=True,)
        print('================================================')
        print(self.Cell)
        print('================================================\n')
        self.fc = nn.Linear(self.Hidden_Size, self.Output_dim)

# GRU模型
class GRUModel(BaseModel):

    def __init__(self, input_dim, hidden_size, output_dim, num_layers, cell,num_iters,optim_method,learning_rate=0.1,print_interval=50,plot_interval=1):
        super(GRUModel, self).__init__(input_dim, hidden_size, output_dim, num_layers, cell)
        self.Print_interval=print_interval
        self.Plot_interval=plot_interval
        self.Num_iters=num_iters
        self.Optim_method=optim_method
        self.Learn_rate=learning_rate

    def forward(self, input, h_state): 
        # input: shape[batch,time_step,input_dim]
        # h_state: shape[layer_num*direction,batch,hidden_size]
        # rnn_output: shape[batch,time_sequence_length,hidden_size]
        GRU_Output, h_state_n = self.Cell(input, h_state)  
        FC_Outputs=[] # save all predictions

        for time_step in range(GRU_Output.size(1)):
            GRU_Output_time_step=GRU_Output[:,time_step,:]
            FC_Output_time_step=self.fc(GRU_Output_time_step)
            FC_Outputs.append(FC_Output_time_step)
        
        Outputs=torch.stack(FC_Outputs,dim=1)
        # Outputs = Outputs_T.squeeze(2).cuda()

        return Outputs, h_state_n
    
    def initHidden(self,input):
        batchSize=input.size(0)
        result = Variable(torch.zeros(self.Num_layers * 1, batchSize, self.Hidden_Size)).cuda()
        return result
    
    def fit(self, input, target):
        input=input.cuda()
        target=target.cuda()
        GRU_h_state=self.initHidden(input)
        criterion = nn.MSELoss()
        if self.Optim_method=='_SGD':
            optimizer = optim.SGD(self.parameters(), lr=self.Learn_rate)
        if self.Optim_method=='_Adam':
            optimizer = optim.Adam(self.parameters(), lr=self.Learn_rate)

        # Initialize timer
        time_tr_start = time.time()

        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every
        
        # plt.figure(1,figsize=(30,5))# continuously plot
        # plt.ion() # continuously plot

        # input_size=input.size(0)# continuously plot
        # time_period=np.arange(input_size)# continuously plot

        # begin to train
        for iter in range(1, self.Num_iters + 1): 
            # input: shape[batch,time_step,input_dim]
            # h_state: shape[layer_num*direction,batch,hidden_size]
            # rnn_output: shape[batch,time_sequence_length,hidden_size]
            prediction, GRU_h_state = self.forward(input,GRU_h_state)
            GRU_h_state=Variable(GRU_h_state.data).cuda()
            loss = criterion(prediction, target)
            plot_loss_total += loss.data[0]
            print_loss_total += loss.data[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # target_view=input[:,-1,:].data.numpy()# continuously plot
            # prediction_view=prediction[:,-1,:].data.numpy()
            # plt.plot(time_period,target_view.flatten(),'r-')
            # plt.plot(time_period,prediction_view.flatten(),'b-')
            # plt.draw();plt.pause(0.05)

            if iter % self.Print_interval == 0:
                print_loss_avg = print_loss_total / self.Print_interval
                print_loss_total = 0
                print('%s (%d %d%%) %.8f' % (timeSince(time_tr_start, iter / self.Num_iters),
                                            iter, iter / self.Num_iters * 100, print_loss_avg))
            if iter % self.Plot_interval == 0:
                plot_loss_avg = plot_loss_total / self.Plot_interval
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

        # Plot loss figure
        plot_loss(plot_losses, Fig_name='Loss'+'_L'+str(self.Num_layers)+'_H'+str(self.Hidden_Size)+'_I'+str(self.Num_iters)+self.Optim_method)
        print('\n------------------------------------------------')
        print('GRU Model finished fitting')
        print('------------------------------------------------')

        return self

    def predict(self, input):
        y_pred = self._predict(input)
        return y_pred

    def _predict(self, input):
        input=input.cuda()
        predict_h_state=self.initHidden(input)
        y_pred,predict_h_state=self.forward(input,predict_h_state)
        y_pred=y_pred.cpu().data.numpy()
        return y_pred
    
    def fit_view(self, input, target):
        input=input.cuda()
        target=target.cuda()
        GRU_h_state=self.initHidden(input)
        criterion = nn.MSELoss()
        if self.Optim_method=='_SGD':
            optimizer = optim.SGD(self.parameters(), lr=self.Learn_rate)
        if self.Optim_method=='_Adam':
            optimizer = optim.Adam(self.parameters(), lr=self.Learn_rate)

        # Initialize timer
        time_tr_start = time.time()

        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        Predict_ViewList=[]
        # begin to train
        for iter in range(1, self.Num_iters + 1):
            # input: shape[batch,time_step,input_dim]
            # h_state: shape[layer_num*direction,batch,hidden_size]
            # rnn_output: shape[batch,time_sequence_length,hidden_size]
            prediction, GRU_h_state = self.forward(input,GRU_h_state)
            GRU_h_state=Variable(GRU_h_state.data).cuda()
            loss = criterion(prediction, target)
            plot_loss_total += loss.data[0]
            print_loss_total += loss.data[0]
            optimizer.zero_grad()
            loss.backward()
            Predict_ViewList.append(prediction[:,-1,:].cpu().data)
            optimizer.step()

            if iter % self.Print_interval == 0:
                print_loss_avg = print_loss_total / self.Print_interval
                print_loss_total = 0
                print('%s (%d %d%%) %.8f' % (timeSince(time_tr_start, iter / self.Num_iters),
                                            iter, iter / self.Num_iters * 100, print_loss_avg))
            if iter % self.Plot_interval == 0:
                plot_loss_avg = plot_loss_total / self.Plot_interval
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

        # Plot loss figure
        plot_loss(plot_losses, Fig_name='Loss'+'_L'+str(self.Num_layers)+'_H'+str(self.Hidden_Size)+'_I'+str(self.Num_iters)+self.Optim_method)
        print('\n------------------------------------------------')
        print('GRU Model finished fitting')
        print('------------------------------------------------')

        return self,Predict_ViewList