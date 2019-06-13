# -*- coding: utf-8 -*-
from datetime import datetime
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import pandas as pd
from sklearn.metrics import r2_score
# load data

def _series_to_supervised(values, n_in, n_out,  col_names = None):
    """
    values: dataset scaled values
    n_in: number of time lags (intervals) to use in each neuron, 与多少个之前的time_step相关,和后面的n_intervals是一样
    n_out: number of time-steps in future to predict，预测未来多少个time_step
    dropnan: whether to drop rows with NaN values after conversion to supervised learning
    col_names: name of columns for dataset
    verbose: whether to output some debug data
    """

    n_vars = 1 if type(values) is list else values.shape[1]
    if col_names is None: col_names = ["var%d" % (j + 1) for j in range(n_vars)]
    df = DataFrame(values)
    cols, names = list(), list()

    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [("%s(t-%d)" % (col_names[j], i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))  # 这里循环结束后cols是个列表，每个列表都是一个shift过的矩阵
        if i == 0:
            names += [("%s(t)" % (col_names[j])) for j in range(n_vars)]
        else:
            names += [("%s(t+%d)" % (col_names[j], i)) for j in range(n_vars)]

    # put it all together
    agg = concat(cols,
                 axis=1)  # 将cols中的每一行元素一字排开，连接起来，vala t-n_in, valb t-n_in ... valta t, valb t... vala t+n_out-1, valb t+n_out-1
    agg.columns = names
    return agg

def ProData(df_tran):
    df_testN=(df_tran-df_tran.min())/(df_tran.max()-df_tran.min())
    # print('--------归一化后数据--------')
    # print(df_testN)
    return df_testN
def parse(x):
    return datetime.strptime(x, '%Y %m %d %H')
pd.set_option('display.max_columns',1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth',1000)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
df = pd.read_excel(u"D:/a-算法项目/气候课题2019-03-22/联合大表_fare.xls")
df = df [['日期','急诊人次','AQI指数','PM2.5','PM10','So2','No2','Co','O3','最高气温','最低气温']]
df.set_index('日期', inplace=True)
values = df.values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
reframed = _series_to_supervised(scaled, 7, 1)
#reframed.drop(reframed.columns[[0,10,20,30,40,50,60,70,80,90,101,102,103,104,105,106,107,108,109]], axis=1, inplace=True)
#reframed.drop(reframed.columns[[0,10,20,30,40,50,60,70,80,91,92,93,94,95,96,97,98,99]], axis=1, inplace=True)
#reframed.drop(reframed.columns[[0,10,20,30,40,50,60,70,81,82,83,84,85,86,87,88,89]], axis=1, inplace=True)
reframed.drop(reframed.columns[[0,10,20,30,40,50,60,71,72,73,74,75,76,77,78,79]], axis=1, inplace=True)
#reframed.drop(reframed.columns[[0,10,20,30,40,50,61,62,63,64,65,66,67,68,69]], axis=1, inplace=True)
#reframed.drop(reframed.columns[[0,10,20,30,40,51,52,53,54,55,56,57,58,59]], axis=1, inplace=True)
#reframed.drop(reframed.columns[[0,10,20,30,41,42,43,44,45,46,47,48,49]], axis=1, inplace=True)
#reframed.drop(reframed.columns[[0,10,20,31,32,33,34,35,36,37,38,39]], axis=1, inplace=True)
reframed.dropna(inplace=True)
values = reframed.values
###33-398 是2016全年，399-764是2017全年,765-1120是2018全年
train = values[764:1090 :]
test = values[1090:1120 :]
# # split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# # eshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
history = model.fit(train_X, train_y, epochs=400, batch_size=4, validation_data=(test_X, test_y), verbose=2, shuffle=False)
yhat = model.predict(test_X)
# print(test_X)
yhat = pd.DataFrame(yhat)
yhat.columns = ['本月']
yhat = yhat['本月']
test_y = pd.DataFrame(test_y)
test_y.columns = ['本月']
test_y = test_y['本月']
r2 = r2_score(test_y,yhat)
print(r2)
yhat.plot(Color = 'g')
test_y.plot(Color = 'r')
pyplot.show()
#反归一化,若无进行归一化操作则屏蔽
# test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# # invert scaling for forecast
# inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
# inv_yhat = scaler.inverse_transform(inv_yhat)
# inv_yhat = inv_yhat[:,0]
# # invert scaling for actual
# test_y = test_y.reshape((len(test_y), 1))
# inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
# inv_y = scaler.inverse_transform(inv_y)
# inv_y = inv_y[:,0]
# # calculate RMSE
# rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
# print('Test RMSE: %.3f' % rmse)
# inv_yhat = pd.DataFrame(inv_yhat)
# inv_yhat.columns = ['本月']
# inv_yhat = inv_yhat['本月']
# inv_y = pd.DataFrame(inv_y)
# inv_y.columns = ['本月']
# inv_y = inv_y['本月']
# inv_y.plot(Color = 'r')
# inv_yhat.plot(Color = 'g')
# pyplot.show()



