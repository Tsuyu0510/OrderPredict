#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from apyori import apriori
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv('20220111_o2o.csv')
df = pd.DataFrame(df)


# In[4]:


df_11 = pd.read_csv('EC_O2O_0112_0114.csv')
df_22 = pd.read_csv('EC_O2O_0115_0117.csv')


# In[6]:


df_11 = pd.DataFrame(df_11)
df_22 = pd.DataFrame(df_22)


# In[4]:


df


# In[43]:


df_1 = df.drop([0],axis = 0)## 刪掉第一行row 不需要的資訊
df_1 = df_1[['RE51取貨編號','RE51日翊商品代號']]
df_1


# In[52]:


df_11[["PER_TRNO","ITEM"]]
df_22[["PER_TRNO","ITEM"]]
df_11_1= df_11[["ITEM"]]
df_22_1=df_22[["ITEM"]]

df_33 = pd.concat([df_11,df_22],ignore_index = True)
df_33[["PER_TRNO","ITEM"]]


# In[ ]:





# In[ ]:





# In[6]:


## group by 每筆取貨編號都買了哪些商品 
## 找出有併買行為的取貨編號（單筆訂單只購買一件商品的先篩選掉）
productorder = df_1.groupby('RE51取貨編號').count()

indexname = productorder[(productorder['RE51日翊商品代號'] <= 1)].index
productorder_drop = productorder.drop(indexname,inplace=False)
productorder_drop.sort_values('RE51日翊商品代號',ascending=False) 


# In[ ]:





# In[54]:


df_1 = df.drop([0],axis = 0)
df_2 = df_1['RE51日翊商品代號']
df_2


# In[62]:


## pd.concat([obj1, obj2], axis=1)的
df2_append = pd.concat([df_11_1,df_22_1],axis = 1)
df2_append.dropna()


# In[65]:


## 運用迴圈把商品編號全部整理成一個list 裡面包含各商品的名稱並用lost of list 去呈現
records = []
for i in range(0,33815):
    records.append([str(df2_append.values[i])])


# In[66]:


records


# In[ ]:





# In[67]:


## 這是另一個approach 是以商品的角度不分訂單做的關聯，但因為現在只有一天的數據，等之後累積其他時間的數據可以重啟這邊的CODE 來試試看


# In[68]:


## min_support -> 4訂/天 除以 總數9690 筆data 
association_rules = apriori(records,min_support = 0.0004,min_confidence = 0.01,min_lift = 1, min_length =2)


# In[69]:


association_rules = list(association_rules)


# In[70]:


print(len(association_rules))


# In[14]:


print (association_rules)


# In[15]:


for item in association_rules:
    pair = item[0]
    items = [x for x in pair]
    print ("Rule:" + items[0]+ "->" )
    print ("Support:" + str(item[1]))
    print ("Confidence:"+ str(item[2][0][2]))
    print ("Lift:"+ str(item[2][0][3]))
    print("====================================")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[45]:


## 把df_1 表裡面的'RE51取貨編號'取出，運用for loop 找出同一筆訂單下重複訂購的商品
recordss = []
order_no = np.unique(df_1['RE51取貨編號'])
order_no
for i in order_no:
    cart = df_1[df_1['RE51取貨編號']==i]['RE51日翊商品代號'].values
    recordss.append(cart)
    #print(cart)
recordss 


# In[46]:


association_rules = apriori(recordss,min_support = 0.001,min_confidence = 0.1,min_lift = 3, min_length =3)


# In[47]:


association_rules = list(association_rules) 


# In[48]:


print (len(association_rules))


# In[49]:


print (association_rules)


# In[50]:


for item in association_rules:
   pair = item[0] 
   items = [x for x in pair]
   print("Rule: " + items[0] + " -> " + items[1])
   print("Support: " + str(item[1]))
   print("Confidence: " + str(item[2][0][2]))
   print("Lift: " + str(item[2][0][3]))
   print("=====================================")


# In[22]:


## 支持度是表示在所有數據中出現該商品機率。
##confidence 是Ａ商品出現的同時同時包含Ｂ商品機率。
##最後 lift 是表示在買A商品同時B商品被購買的倍數。


# In[ ]:




