#!/usr/bin/env python
# coding: utf-8

# ## 資料前處理

# In[1]:


# 導入套件
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter


# In[2]:


# 讀入作業的csv檔並存到df
df = pd.read_csv('HW2data.csv')
df


# In[3]:


# 看一下df的資料型態以及有無空值
df.info()


# In[4]:


# 取df中income以外的欄位並存到df1
df1 = df.iloc[:, 0:-1]
df1


# In[5]:


# 取出df中的income欄位並存到Y
Y = df.iloc[:, -1]
Y


# In[6]:


# 看一下df1有哪些特徵
df1.columns


# In[7]:


# 看一下df1中'age'和 'workclass'這一行內ㄉ內的資料型態
print(df1['age'].dtypes)
print(df1['workclass'].dtypes)


# In[8]:


# 試試看能不能使用這個判斷式，因為想用迴圈取出需要做預處理的特徵
df1['workclass'].dtypes == 'object'


# In[9]:


# 取出需要做預處理的特徵
unprocess_feature = []
for i in df1.columns:
    if df1[i].dtypes == 'object':
        unprocess_feature.append(i)
print(unprocess_feature)


# In[10]:


# 看一下需要做預處理的特徵之內容
for i in unprocess_feature:
    print(f'{i}這一欄')
    print(Counter(df1[i]), end='\n\n')


# In[11]:


# 取出不需要做預處理的特徵
# 取出特徵下的內容存成X0
process_feature = []
for i in df1.columns:
    if df1[i].dtypes == np.int64:
        process_feature.append(i)
print(process_feature)

X0 = df1[process_feature]
X0


# In[12]:


# 從df1取出age這個欄位，方便合併表格，之後會剔除(因為會重複)
X_temp1 = df1['age'] 
# 做one-hot-encoding，並將結果存到X_temp1
for i in unprocess_feature:
    X_temp2 = pd.get_dummies(df1[i], prefix=i)
    X_temp1 = pd.concat([X_temp1, X_temp2], axis=1)
X_temp1


# In[13]:


# 將處理好的特徵做合併，接著存到X
X = pd.concat([X0, X_temp1.iloc[:, 1:]], axis=1)
X


# In[14]:


# 看一下X的資料型態，發現已無Object型態，代表處理成功
X.info()


# In[15]:


# 將X, Y做合併存到XY，等等要放入我們做好的函示中
XY = pd.concat([X, Y], axis=1)
XY


# ## k-fold cross-validation

# In[16]:


# 建立k-fold cross-validation的函示，並用print(XXXX.index)來看看使用出錯
def K_fold_and_RF(data, num):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(max_depth=10)
    score = []
    s = int(len(data)/num)
    for i in range(1, num+1):
        train_index = []
        X_test = data.iloc[(i-1)*s:(i*s-1)]
        X_test = X_test.drop(['income'], axis=1)
        Y_test = data['income'].iloc[X_test.index]
        for i in data.index:
            if i not in X_test.index:
                train_index.append(i)
        X_train = data.iloc[train_index]
        X_train = X_train.drop(['income'], axis=1)
        Y_train = data['income'].iloc[train_index]
        
        print(X_train.index)
        print(X_test.index)
        print(Y_train.index)
        print(Y_test.index)
        print('---------------------------------------------------------------------------')


# In[17]:


# 舉3-fold為例子，發現運作正確
K_fold_and_RF(XY, 3)


# ## k-fold cross-validation&隨機森林

# In[18]:


# 延長剛剛的函示，放入隨機森林(最大深度限制為10層)，並用迴圈算出每次的準確率然後求平均
def K_fold_and_RF(data, num):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(max_depth=10)
    scores = []
    sumofscores = 0
    s = int(len(data)/num)
    for i in range(1, num+1):
        train_index = []
        X_test = data.iloc[(i-1)*s:(i*s-1)]
        X_test = X_test.drop(['income'], axis=1)
        Y_test = data['income'].iloc[X_test.index]
        for i in data.index:
            if i not in X_test.index:
                train_index.append(i)
        X_train = data.iloc[train_index]
        X_train = X_train.drop(['income'], axis=1)
        Y_train = data['income'].iloc[train_index]
        
        clf.fit(X_train, Y_train)
        scores.append(clf.score(X_test, Y_test))
        
    print(f'這{num}次跑出來的準確率分別為：\n{scores}')
    for i in scores:
        sumofscores += i
    print('算出來的平均準確率為{}'.format(sumofscores/num))


# In[19]:


# 計算作業所要求的10-fold的準確率
K_fold_and_RF(XY, 10)


# In[ ]:




