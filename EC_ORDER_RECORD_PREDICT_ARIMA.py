#!/usr/bin/env python
# coding: utf-8
import numpy as np 
import pandas as pd 
import sklearn 
import pymysql
import sqlalchemy
from sqlalchemy import create_engine
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from sqlalchemy.types import NVARCHAR, Integer, Float,VARCHAR

pd.set_option('display.max_columns', None) #設定jupyter note字元顯示寬度linux 環境不用 
pd.set_option('display.max_rows', None)
## get data from db
engine = create_engine('mysql+pymysql://root:****************')
sql = '''
select * from db02.df_daily_order order by LOAD_TIME asc;
'''
df = pd.read_sql_query(sql, engine)
print(df)


## 確認整體數據性質趨勢（是否存在季節性或隨機性）
'''import statsmodels.api as sm
decomposition = sm.tsa.seasonal_decompose(df,model = 'additive', extrapolate_trend='freq')
plt.rc('figure',figsize = (12,8))
fig = decomposition.plot()
plt.show()'''

## 決定訓練/測試數據要用的區間
## filter data after 2021-12
df = df['count(DISTINCT PER_TRNO)']
df_train = df.iloc[328:449]
df_test = df.iloc[449:478]

#訓練數據整理part2 只是不想改df_order 名稱...
#df_train_filter
df_filter= df_train.reset_index(drop=True)
df_order =df_filter
df_order
 
## acf plot 
plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})

# Import data : Internet Usage per Minute
df = df_order
# Original Series
fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(df_order); axes[0, 0].set_title('Original Series')
plot_acf(df_order, ax=axes[0, 1])

# 1st Differencing
axes[1, 0].plot(df_order.diff()); axes[1, 0].set_title('1st Order Differencing')
plot_acf(df_order.diff().dropna(), ax=axes[1, 1])

# 2nd Differencing
axes[2, 0].plot(df_order.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(df_order.diff().diff().dropna(), ax=axes[2, 1])
#plt.show()

## pacf plot
# Original Series
fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(df_order); axes[0, 0].set_title('Original Series')
plot_pacf(df_order, ax=axes[0, 1])

# 1st Differencing
axes[1, 0].plot(df_order.diff()); axes[1, 0].set_title('1st Order Differencing')
plot_pacf(df_order.diff().dropna(), ax=axes[1, 1])

# 2nd Differencing
axes[2, 0].plot(df_order.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
plot_pacf(df_order.diff().diff().dropna(), ax=axes[2, 1])
#plt.show()

## 從acf 和pacf 圖『主觀地』決定我們arima 的d和p要設置多少
import statsmodels.api as sm                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
fig = plt.figure(figsize=(12,8))
 
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df_order, lags=20,ax=ax1)
ax1.xaxis.set_ticks_position('bottom')
fig.tight_layout()
 
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df_order, lags=20, ax=ax2)
ax2.xaxis.set_ticks_position('bottom')
fig.tight_layout()
#plt.show()

#from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA

# 搭建ARIMA Model
model = sm.tsa.arima.ARIMA(df_order, order=(25,1,1))## p 設定和前n筆資料趨勢有關，d設定為1，q設定為1
model_fit = model.fit()
#print(model_fit.summary())
## 重點關注指標，1.P值，2. coef 

# Plot residual errors
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(2,1)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
#plt.show()

# Actual vs Fitted
#df_test = df[(df["LOAD_TIME"]>= '2022/4/1')& (df['LOAD_TIME']<='2022/4/30')]
#print(df_filter) #121
#print (df_test) #477
#print ('==================================================')

### 輸出模型預測結果
prediction = model_fit.predict(1,150,dynamic=False)
#print(type(prediction)) #150

## 組裝訓練＋期望預測數據
df_filter_test_2_2_1 = df_order.append(df_test,ignore_index=True)
#print(type(df_filter_test_2_2_1)) #150

#combine the columns 
arima_result = pd.merge(df_filter_test_2_2_1, prediction, left_index=True, right_index= True)
#print(arima_result)
arima_result['date'] = pd.date_range(start= '2021-12-01', periods= len(arima_result), freq= 'D')
print(arima_result)

  
## 視覺化
#visualize 
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(prediction,label = 'prediction',linestyle='--',color = 'red')
ax.plot(df_filter_test_2_2_1[120:150],label = 'real_order_202204', color = 'green')
ax.plot(df_order,label = 'real_history_data', color = 'gray',linestyle=':')
ax.set_xlabel('timestamp')  # Add an x-label to the axes.
ax.set_ylabel('order count')  # Add a y-label to the axes.
ax.set_title("order_predict")  # Add a title to the axes.
ax.legend();  # Add a legend.


## prediction 寫入db 
dtypedict = {'object':VARCHAR(length = 255),'int':Integer(),'float':Float()}
MYSQL_engine = create_engine('mysql+pymysql://root:*********', encoding='utf_8_sig')
#for l in range(len(df_finalll)):
## 使用try except 避開PK 重複寫入問題
try:
	arima_result.to_sql('arima_result',MYSQL_engine,index = False, if_exists = 'append',dtype = dtypedict)
	print ("INSERT_SUCCESS!!!")

except sqlalchemy.exc.IntegrityError:
	print ("ALL DATA WAS UPDATED !!")
	pass
