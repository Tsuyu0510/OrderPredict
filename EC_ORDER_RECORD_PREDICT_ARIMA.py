#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import sklearn 


# In[2]:


pd.set_option('display.max_columns', None) # 設定jupyter note字元顯示寬度linux 環境不用 
pd.set_option('display.max_rows', None)
df = pd.read_csv('order_record.csv')
df


# In[21]:


## 確認整體數據性質趨勢（是否存在季節性或隨機性）
'''import statsmodels.api as sm
decomposition = sm.tsa.seasonal_decompose(df,model = 'additive', extrapolate_trend='freq')
plt.rc('figure',figsize = (12,8))
fig = decomposition.plot()
plt.show()'''


# In[22]:


## 決定訓練/測試數據要用的區間
## filter data after 2021-12
df = df['count(DISTINCT PER_TRNO)']
df_train = df.iloc[328:449]
df_test = df.iloc[449:478]




# In[23]:



#訓練數據整理part2 只是不想改df_order 名稱...
#df_train_filter
df_filter= df_train.reset_index(drop=True)
df_order =df_filter
df_order



# In[24]:


import numpy as np, pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
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

plt.show()


# In[25]:


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

plt.show()


# In[26]:


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
plt.show()


# In[ ]:





# In[27]:


#from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA


# 搭建ARIMA Model
model = sm.tsa.arima.ARIMA(df_order, order=(25,1,1))## p 設定和前n筆資料趨勢有關，d設定為1，q設定為1
model_fit = model.fit()
print(model_fit.summary())
## 重點關注指標，1.P值，2. coef 


# In[28]:


# Plot residual errors
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(2,1)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()


# In[43]:


import pickle
with open('ARIMAmodel.pickle', 'wb') as f:
    pickle.dump(model_fit, f)


# In[44]:


import gzip
with gzip.GzipFile('ARIMAmodel.pgz', 'w') as f:
    pickle.dump(model_fit, f)


# In[31]:


# Actual vs Fitted
#df_test = df[(df["LOAD_TIME"]>= '2022/4/1')& (df['LOAD_TIME']<='2022/4/30')]
#df_filter
df_test


# In[45]:


#import pickle
#import gzip

#讀取Model
with gzip.open('ARIMAmodel.pgz', 'r') as f:
    ARIMAmodel = pickle.load(f)
    pred=ARIMAmodel.predict(1,150,dynamic=False)
    print(pred)


# In[40]:


### 輸出模型預測結果
prediction = model_fit.predict(1,151,dynamic=False)
prediction


# In[41]:


## 組裝訓練＋期望預測數據
df_filter_test_2_2_1 = df_order.append(df_test,ignore_index=True)
df_filter_test_2_2_1


# In[ ]:





# In[42]:


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


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




