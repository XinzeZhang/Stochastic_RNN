import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim
from torch.autograd import Variable

# 模型基类，主要是用于指定参数和cell类型
class BaseModel(nn.Module):

    def __init__(self, inputDim=1, Hidden_size=150, outputDim=1, layerNum=1, cell="GRU"):

        super(BaseModel, self).__init__()
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.hidden_size = Hidden_size
        self.layerNum = layerNum
        if cell == "RNN":
            self.cell = nn.RNN(input_size=self.inputDim, hidden_size=self.hidden_size,
                        num_layers=self.layerNum, dropout=0.0,
                         nonlinearity="tanh", batch_first=True,)
        if cell == "LSTM":
            self.cell = nn.LSTM(input_size=self.inputDim, hidden_size=self.hidden_size,
                               num_layers=self.layerNum, dropout=0.0,
                               batch_first=True, )
        if cell == "GRU":
            self.cell = nn.GRU(input_size=self.inputDim, hidden_size=self.hidden_size,
                                num_layers=self.layerNum, dropout=0.0,
                                 batch_first=True,)
        print(self.cell)
        self.fc = nn.Linear(self.hidden_size, self.outputDim)

# GRU模型
class GRUModel(BaseModel):

    def __init__(self, inputDim, Hidden_size, outputDim, layerNum, cell):
        super(GRUModel, self).__init__(inputDim, Hidden_size, outputDim, layerNum, cell)

    def forward(self, input): #input: shape[batch,time_step,input_dim]
        batchSize=input.size(0)
        h_state = Variable(torch.zeros(self.layerNum * 1, batchSize, self.hidden_size)double(),
                       requires_grad=False).cuda() # h_state (n_layers * num_direction, batch, hidden_size)                      
        rnnOutput, h_state = (self.cell(x, h_state)  
        h_state = h_state.view(batchSize, self.hidden_size)
        fcOutputs = self.fc(h_state)
        
        Outputs_T=torch.stack(fcOutputs,dim=1)
        Outputs = Outputs_T.squeeze(2)

        return Outputs