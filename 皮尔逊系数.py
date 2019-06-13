# -*- coding: utf-8 -*-
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
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
import matplotlib
from sklearn.decomposition import PCA
pd.set_option('display.max_columns',1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth',1000)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
import seaborn as sns
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
def ProData(df_tran):
    df_testN=(df_tran-df_tran.min())/(df_tran.max()-df_tran.min())
    # print('--------归一化后数据--------')
    # print(df_testN)
    return df_testN
#df=pd.read_excel('D:\a-算法项目\气候课题2019-03-22\联合大表_fare.xls')
df = pd.read_excel(u"D:/a-算法项目/气候课题2019-03-22/联合大表_fare.xls")
df = df[['日期',u'AQI指数','PM2.5','PM10','So2','No2','Co','O3',u'最高气温',u'最低气温',u'急诊人次']]
df.set_index('日期', inplace=True)
#df = ProData(df)
df = df.ix['2017-03-01':'2017-06-30']
print(df)
dfData = df.corr()
plt.subplots(figsize=(9, 9)) # 设置画面大小
sns.heatmap(dfData, annot=True, vmax=1, square=True, cmap="Blues")
plt.show()
print(df.corr())