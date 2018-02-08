from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import numpy as np
from numpy import concatenate

import matplotlib.ticker as ticker
# matplotlib.use('Agg')
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
    # diff.append(0)
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

if __name__ == '__main__':
    # load dataset
    series = read_csv('chinese_oil_production.csv', header=0,
                    parse_dates=[0], index_col=0, squeeze=True)
    #transfer the dataset to array
    raw_values = series.values
    tf_values_array=np.array(raw_values)
    set_length=len(tf_values_array)

    # transform data to be stationary
    diff = difference(raw_values, 1)

    # create differece_dataset x,y
    dataset = diff.values
    dataset = create_dataset(dataset, look_back=1)

    # split into train and test sets
    train_size = int(dataset.shape[0] * 0.8)
    train_scope=np.arange(train_size)
    # test_size = dataset.shape[0] - train_size
    test_scope= np.arange(train_size,set_length)
    
    # divide the tf_values to train set and test set
    tf_train=tf_values_array[:train_size].copy()
    tf_expect=tf_values_array[train_size:].copy()
    #divide the tf_values_diff to train set and test set
    train, test = dataset[0:train_size], dataset[train_size:]
    tf_train_diff=train[:,:1]
    tf_test_diff=test[:,:1]
    tf_test_diff=np.insert(tf_test_diff,[-1],dataset[-1,-1])

    #prepare the data plot
    def draw(scope,date, date_color,date_label):
        plt.plot(scope, date,date_color,label=date_label,linewidth = 1.0)
    # plt.figure(figsize=(90,10))
    plt.figure(figsize=(50,60))    
    # locate the sublabel
    # draw the train set and test set
    ax1=plt.subplot(311)
    ax1.set_xticks(np.arange(0,set_length,10))
    plt.plot(train_scope, tf_train,'k',label='tf_train',linewidth = 1.0)
    plt.plot(test_scope, tf_expect,'k:',label='tf_expect',linewidth = 1.0)
    # plt.step(ax1.get_xticklabels(),fontsize=5)
    # plt.minorticks_on()
    plt.grid()
    plt.legend(loc='upper right')
    ax1.set_title('values for time sequences')
    plt.xlabel('Time Sequence' )
    plt.ylabel('Value')
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # plt.show()

    ax2=plt.subplot(312,sharex=ax1)
    ax2.plot(train_scope, tf_train_diff,'r',label='tf_train_diff',linewidth = 1.0)
    ax2.plot(test_scope, tf_test_diff,'r:',label='tf_expect_diff',linewidth = 1.0)
    ax2.minorticks_on()
    # tick_spacing=ticker.MultipleLocator(base=5.0)
    # ax.xaxis.set_major_locator=(tick_spacing)
    ax2.grid(which='both')
    plt.legend(loc='upper right')
    ax2.set_title('values_difference for time sequences')
    plt.xlabel('Time Sequence')
    plt.ylabel('Difference')
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)

    plt.subplots_adjust(hspace=0.5)
    plt.show()
    #



    # # transform the scale of the data
    # scaler, train_scaled, test_scaled = scale(train, test)

    # # divided the train_set and test_set
    # train_input_scaled=train_scaled[:,:1]
    # train_target_scaled=train_scaled[:,1:]
    # test_input_scaled=test_scaled[:, :1]
    # test_target_scaled=test_scaled[:, 1:]






    # time_period=np.arange(len(y_pred))
    # plt.plot(time_period,y_pred,color='blue',linestyle='--',label='Prediction')
    # plt.plot(time_period,expected,color='green',linestyle='-',label='Original')

    # plt.savefig('Prediction.png')
    # plt.show()
