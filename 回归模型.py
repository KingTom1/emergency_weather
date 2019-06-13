# -*- coding: utf-8 -*-
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
#from statsmodels.tsa.vector_ar.var_model import VAR
import statsmodels.api as sm
from sklearn.metrics import r2_score
from sklearn.cross_validation import train_test_split
from statsmodels.tsa.stattools import  adfuller as ADF
from statsmodels.stats.diagnostic import  acorr_ljungbox
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARMA
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import warnings
import seaborn as sns
from pylab import *
from sklearn.decomposition import PCA
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
pd.set_option('display.max_columns',1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth',1000)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
def ProData(df_tran):
    df_testN=(df_tran-df_tran.min())/(df_tran.max()-df_tran.min())
    return df_testN
#7代表窗口为7
df=pd.read_excel(u"D:/a-算法项目/气候课题2019-03-22/联合大表_fare.xls")
df1 = df[['日期',u'AQI指数','PM2.5','PM10','So2','No2','Co','O3',u'最高气温',u'最低气温',u'急诊人次']]
df1.set_index('日期', inplace=True)
#箱线图
#df1.boxplot()
df = df1.ix['2017-12-01':'2018-12-01']
df_value= df1.ix['2017-03-01':'2017-06-30']
df = df.reset_index()
df =  df[['AQI指数','PM2.5','PM10','So2','No2','Co','O3','最高气温','最低气温','急诊人次']]
x = df[['AQI指数','PM2.5','PM10','So2','No2','Co','O3','最高气温','最低气温']]
#多重共线性去除
# x = df[['AQI指数','So2','No2','O3','最低气温']]
#数据压缩归一化
# #x = ProData(x)
y = df['急诊人次']
# #y = ProData(y)
#划分训练集,测试集
df_train_feature,df_test_feature, df_train_goal, df_test_goal = train_test_split(x, y, test_size=0.33, random_state=42)
#回归模型算法
#clf=GradientBoostingRegressor( learning_rate= 0.01,n_estimators=300)
#clf = LinearRegression()
clf = RandomForestRegressor(n_estimators=400)
#clf = Ridge(alpha=0.03)
#clf = Lasso()
#clf =SVR()
#神经网络,双隐层,一层28神经元,二层16神经元
# model = Sequential()
# model.add(Dense(28,activation='relu',input_dim=10))
# model.add(Dense(16,activation='relu'))
# model.add(Dense(1))
# model.compile(optimizer='adam',loss='mse')
# model.fit(df_train_feature,df_train_goal,epochs=1000,batch_size=4,verbose=2)
clf.fit(df_train_feature, df_train_goal)
#验证数据集
x_value = df_value[['AQI指数','PM2.5','PM10','So2','No2','Co','O3','最高气温','最低气温']]
#x_value = df_value[['AQI指数','So2','No2','O3','最低气温']]
y_value = df_value['急诊人次']
#开始进行数据预测
prediction = clf.predict(x_value)
prediction = pd.DataFrame(prediction)
#处理验证集数据
y_value= y_value.reset_index()
y_value = y_value['急诊人次']
#评价指标
r2 = r2_score(y_value,prediction)
print(r2)
df_wucha = pd.concat([prediction,y_value],axis =1)
df_wucha.columns = ['预测急诊人次','实际急诊人次']
df_wucha['误差'] = abs(df_wucha['实际急诊人次']-df_wucha['预测急诊人次'])
mean_value = df_wucha['误差'].mean()
print(mean_value)#
names = ['AQI指数','PM2.5','PM10','So2','No2','Co','O3','最高气温','最低气温']
#names = ['AQI指数','So2','No2','O3','最低气温']
#输出各因素权重
print(sorted(zip(map(lambda x:round(x,4),clf.feature_importances_),names),reverse=True))
prediction.plot(Color = 'g')
y_value.plot(Color ='r')
#df_wucha['误差'] .plot(Color='y')
plt.show()


